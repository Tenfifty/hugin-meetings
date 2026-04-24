#!/usr/bin/env python3
"""Compare diarization pipelines on one or more recordings.

Reuses the existing Hugin ASR and merge logic so diarization is the main
variable being compared.
"""

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import transcribe as base
from .config import load_config

_cfg = load_config()
COMPARISON_DIR = _cfg.state_dir / "comparisons"
PIPELINES = ("current", "pyannote_direct", "nemo_msdd_telephonic")


@dataclass
class MeetingArtifacts:
    ts: str
    out_dir: Path
    mic_result: dict
    sys_result: dict | None
    mic_path: Path
    sys_path: Path | None


def _annotation_to_df(annotation) -> pd.DataFrame:
    rows = []
    for segment, track, speaker in annotation.itertracks(yield_label=True):
        rows.append(
            {
                "segment": segment,
                "label": track,
                "speaker": speaker,
                "start": float(segment.start),
                "end": float(segment.end),
            }
        )
    return pd.DataFrame(rows)


def _audio_input_for_pyannote(audio_path: Path):
    import torch
    import whisperx

    audio = whisperx.load_audio(str(audio_path))
    return {
        "waveform": torch.from_numpy(audio[None, :]),
        "sample_rate": 16000,
        "uri": audio_path.stem,
    }


def _maybe_name_map_from_embeddings(embeddings, labels) -> dict[str, str]:
    if embeddings is None:
        return {}

    speaker_embeddings = {}
    try:
        for idx, speaker in enumerate(labels):
            emb = embeddings[idx]
            if hasattr(emb, "detach"):
                emb = emb.detach().cpu().numpy()
            elif hasattr(emb, "cpu"):
                emb = emb.cpu().numpy()
            elif hasattr(emb, "tolist"):
                emb = emb.tolist()
            speaker_embeddings[speaker] = emb
    except Exception:
        return {}

    return base.match_speakers(speaker_embeddings)


def _rename_result_speakers(result: dict, name_map: dict[str, str]) -> dict:
    if not name_map:
        return result
    for seg in result.get("segments", []):
        spk = seg.get("speaker")
        if spk in name_map:
            seg["speaker"] = name_map[spk]
        for word in seg.get("words", []):
            word_spk = word.get("speaker")
            if word_spk in name_map:
                word["speaker"] = name_map[word_spk]
    return result


def _write_pipeline_outputs(
    meeting: MeetingArtifacts,
    pipeline_name: str,
    mic_result: dict,
    sys_result: dict | None,
) -> dict:
    entries = base.merge_channels(mic_result, sys_result, meeting.ts)
    out_json = meeting.out_dir / f"{pipeline_name}.json"
    out_md = meeting.out_dir / f"{pipeline_name}.md"
    out_json.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    out_md.write_text(base.format_transcript(entries, meeting.ts))
    return {
        "entries": len(entries),
        "speakers": sorted({f"{e['channel']}:{e['speaker']}" for e in entries}),
        "json": out_json,
        "md": out_md,
    }


def _load_cached_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_cached_result(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))


def _ensure_pcm_wav(audio_path: Path, out_dir: Path) -> Path:
    wav_path = out_dir / f"{audio_path.stem}.wav"
    if wav_path.exists():
        return wav_path

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return wav_path


def _force_nemo_clustering_to_cuda(diarizer, device: str) -> None:
    import torch

    if device != "cuda":
        return

    torch_device = torch.device(device)
    clus = diarizer.clustering_embedding.clus_diar_model

    # NeMo 2.7.2 leaves the speaker embedding model on CPU even when the
    # diarizer is loaded with map_location="cuda". Clustering uses
    # _speaker_model.device to decide whether to run GPU eigendecomposition.
    clus._speaker_model = clus._speaker_model.to(torch_device)


def prepare_meeting(mic_path: Path, model, device: str, force: bool = False) -> MeetingArtifacts:
    ts = mic_path.stem.removeprefix("mic-")
    out_dir = COMPARISON_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    mic_asr_cache = out_dir / "aligned-mic.json"
    sys_asr_cache = out_dir / "aligned-sys.json"

    mic_result = None if force else _load_cached_result(mic_asr_cache)
    if mic_result is None:
        print(f"  Transcribing mic: {mic_path.name}")
        mic_result = base.transcribe(mic_path, model, device)
        _write_cached_result(mic_asr_cache, mic_result)
    else:
        print(f"  Reusing cached mic ASR: {mic_asr_cache.name}")

    sys_path = base.find_pair(mic_path)
    sys_result = None
    if sys_path:
        if base.is_silent(sys_path):
            print(f"  System audio is silent, skipping: {sys_path.name}")
        else:
            sys_result = None if force else _load_cached_result(sys_asr_cache)
            if sys_result is None:
                print(f"  Transcribing sys: {sys_path.name}")
                sys_result = base.transcribe(sys_path, model, device)
                _write_cached_result(sys_asr_cache, sys_result)
            else:
                print(f"  Reusing cached sys ASR: {sys_asr_cache.name}")
    else:
        print("  No matching sys file found")

    return MeetingArtifacts(
        ts=ts,
        out_dir=out_dir,
        mic_result=mic_result,
        sys_result=sys_result,
        mic_path=mic_path,
        sys_path=sys_path,
    )


def run_current(meetings: list[MeetingArtifacts], device: str, hf_token: str) -> dict[str, dict]:
    import whisperx
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    import torch

    diarizer = DiarizationPipeline(token=hf_token, device=device)
    summaries = {}

    for meeting in meetings:
        print(f"\n[{meeting.ts}] current")
        mic_result = copy.deepcopy(meeting.mic_result)
        sys_result = copy.deepcopy(meeting.sys_result)

        audio = whisperx.load_audio(str(meeting.mic_path))
        diarize_df, embeddings = diarizer(audio, return_embeddings=True)
        mic_result = assign_word_speakers(diarize_df, mic_result)
        mic_result = _rename_result_speakers(mic_result, base.match_speakers(embeddings or {}))

        if sys_result and meeting.sys_path:
            audio = whisperx.load_audio(str(meeting.sys_path))
            diarize_df, embeddings = diarizer(audio, return_embeddings=True)
            sys_result = assign_word_speakers(diarize_df, sys_result)
            sys_result = _rename_result_speakers(sys_result, base.match_speakers(embeddings or {}))

        summaries[meeting.ts] = _write_pipeline_outputs(meeting, "current", mic_result, sys_result)

    del diarizer
    torch.cuda.empty_cache()
    return summaries


def run_pyannote_direct(meetings: list[MeetingArtifacts], device: str, hf_token: str) -> dict[str, dict]:
    import torch
    from pyannote.audio import Pipeline
    from whisperx.diarize import assign_word_speakers

    diarizer = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    ).to(torch.device(device))
    summaries = {}

    for meeting in meetings:
        print(f"\n[{meeting.ts}] pyannote_direct")
        mic_result = copy.deepcopy(meeting.mic_result)
        sys_result = copy.deepcopy(meeting.sys_result)

        output = diarizer(_audio_input_for_pyannote(meeting.mic_path))
        annotation = getattr(output, "exclusive_speaker_diarization", None) or output.speaker_diarization
        mic_result = assign_word_speakers(_annotation_to_df(annotation), mic_result)
        name_map = _maybe_name_map_from_embeddings(
            getattr(output, "speaker_embeddings", None),
            list(output.speaker_diarization.labels()),
        )
        mic_result = _rename_result_speakers(mic_result, name_map)

        if sys_result and meeting.sys_path:
            output = diarizer(_audio_input_for_pyannote(meeting.sys_path))
            annotation = getattr(output, "exclusive_speaker_diarization", None) or output.speaker_diarization
            sys_result = assign_word_speakers(_annotation_to_df(annotation), sys_result)
            name_map = _maybe_name_map_from_embeddings(
                getattr(output, "speaker_embeddings", None),
                list(output.speaker_diarization.labels()),
            )
            sys_result = _rename_result_speakers(sys_result, name_map)

        summaries[meeting.ts] = _write_pipeline_outputs(
            meeting, "pyannote_direct", mic_result, sys_result
        )

    del diarizer
    torch.cuda.empty_cache()
    return summaries


def run_nemo_msdd(meetings: list[MeetingArtifacts], device: str) -> dict[str, dict]:
    import torch
    from nemo.collections.asr.models import NeuralDiarizer
    from whisperx.diarize import assign_word_speakers

    diarizer = NeuralDiarizer.from_pretrained(
        model_name="diar_msdd_telephonic",
        vad_model_name="vad_multilingual_marblenet",
        map_location=device,
        verbose=False,
    )
    _force_nemo_clustering_to_cuda(diarizer, device)
    summaries = {}

    for meeting in meetings:
        print(f"\n[{meeting.ts}] nemo_msdd_telephonic")
        mic_result = copy.deepcopy(meeting.mic_result)
        sys_result = copy.deepcopy(meeting.sys_result)

        mic_wav = _ensure_pcm_wav(meeting.mic_path, meeting.out_dir)
        annotation = diarizer(str(mic_wav))
        mic_result = assign_word_speakers(_annotation_to_df(annotation), mic_result)

        if sys_result and meeting.sys_path:
            sys_wav = _ensure_pcm_wav(meeting.sys_path, meeting.out_dir)
            annotation = diarizer(str(sys_wav))
            sys_result = assign_word_speakers(_annotation_to_df(annotation), sys_result)

        summaries[meeting.ts] = _write_pipeline_outputs(
            meeting, "nemo_msdd_telephonic", mic_result, sys_result
        )

    del diarizer
    torch.cuda.empty_cache()
    return summaries


def write_index(meetings: list[MeetingArtifacts], all_summaries: dict[str, dict[str, dict]]) -> Path:
    index_path = COMPARISON_DIR / "README.md"
    lines = [
        "# Diarization Comparisons",
        "",
        "Pipelines:",
        "- `current`: current WhisperX diarization wrapper",
        "- `pyannote_direct`: direct `pyannote.audio` with `exclusive_speaker_diarization`",
        "- `nemo_msdd_telephonic`: NeMo `diar_msdd_telephonic` + multilingual MarbleNet VAD",
        "",
    ]

    for meeting in meetings:
        lines.extend(
            [
                f"## {meeting.ts}",
                "",
                f"- Mic: `{meeting.mic_path}`",
                f"- Sys: `{meeting.sys_path}`" if meeting.sys_path else "- Sys: none",
                "",
            ]
        )
        for pipeline_name in PIPELINES:
            summary = all_summaries[pipeline_name][meeting.ts]
            lines.append(f"### {pipeline_name}")
            lines.append("")
            lines.append(f"- Transcript: `{summary['md']}`")
            lines.append(f"- JSON: `{summary['json']}`")
            lines.append(f"- Entries: {summary['entries']}")
            lines.append(f"- Speakers: {', '.join(summary['speakers']) if summary['speakers'] else '(none)'}")
            lines.append("")

    index_path.write_text("\n".join(lines))
    return index_path


def resolve_mic_paths(args_files: list[str]) -> list[Path]:
    if args_files:
        paths = []
        for file_name in args_files:
            path = Path(file_name)
            if not path.is_absolute():
                candidate = base.AUDIO_DIR / path
                path = candidate if candidate.exists() else (Path.cwd() / path)
            if not path.exists():
                raise FileNotFoundError(path)
            paths.append(path)
        return paths

    return sorted(base.AUDIO_DIR.glob("mic-*.opus"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare diarization pipelines")
    parser.add_argument("files", nargs="*", help="Mic recordings to process")
    parser.add_argument("--today", action="store_true", help="Process today's mic recordings only")
    parser.add_argument("--force", action="store_true", help="Recompute aligned ASR cache")
    args = parser.parse_args()

    if args.today:
        prefix = f"mic-{base.datetime.now().strftime('%Y%m%d')}"
        mic_paths = sorted(base.AUDIO_DIR.glob(f"{prefix}-*.opus"))
    else:
        mic_paths = resolve_mic_paths(args.files)

    if not mic_paths:
        print("No recordings found.", file=sys.stderr)
        sys.exit(1)

    hf_token = base.get_hf_token()
    if not hf_token:
        print("No HuggingFace token found. Run `huggingface-cli login`.", file=sys.stderr)
        sys.exit(1)

    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ASR model: {base.MODEL}")
    model = whisperx.load_model(base.MODEL, device, compute_type="float16" if device == "cuda" else "int8")

    meetings = []
    for mic_path in mic_paths:
        print(f"\nPreparing {mic_path.name}")
        meetings.append(prepare_meeting(mic_path, model, device, force=args.force))

    del model
    torch.cuda.empty_cache()

    all_summaries = {
        "current": run_current(meetings, device, hf_token),
        "pyannote_direct": run_pyannote_direct(meetings, device, hf_token),
        "nemo_msdd_telephonic": run_nemo_msdd(meetings, device),
    }

    index_path = write_index(meetings, all_summaries)
    print(f"\nComparison index written to {index_path}")


if __name__ == "__main__":
    main()
