#!/usr/bin/env python3
"""Transcribe and diarize a mic/sys recording session, merge into a unified transcript.

Usage:
    transcribe.py mic-20260408-213541-p01.opus      # process whole session
    transcribe.py 20260408-213541                   # process whole session
    transcribe.py --all                             # process all unprocessed sessions
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import gc
import json
import os
import site
import subprocess
import sys
import sysconfig
import tempfile
import time
import traceback
from pathlib import Path

from .cli_utils import get_hf_token
from .pipeline import (
    SPEAKER_RE,
    audio_duration,
    extract_timestamp,
    is_backchannel,
    parse_raw_audio_part,
    raw_audio_session,
    scan_raw_audio_sessions,
    transcript_json_path,
)
from .config import load_config

AUDIO_DIR = load_config().raw_audio_dir
WAV_CACHE_DIR = load_config().wav_cache_dir
TRANSCRIPT_DIR = load_config().transcripts_dir
SPEAKERS_DIR = load_config().speakers_dir
MODEL = load_config().whisper_model

# whisperx ships built-in default wav2vec2 alignment models for a fixed set of
# languages, but recent releases dropped several (incl. Swedish), turning a
# missing default into a hard ``ValueError: No default align-model for
# language: <lang>``. Supplement whisperx's defaults here so the languages we
# actually transcribe keep aligning. Override or extend via
# ``meetings.transcribe_align_models: {<lang>: <hf-model>}`` in config.
DEFAULT_ALIGN_MODELS = {
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
}
ALIGN_MODELS = {
    **DEFAULT_ALIGN_MODELS,
    **(load_config().raw.get("meetings", {}).get("transcribe_align_models") or {}),
}

DEFAULT_DIARIZER = "nemo"
# NeMo's default MSDD inference batch (25) OOMs on a 65 min part; 4 fits with
# ~1.4 GB spare and diarizes 6.3x faster than the CPU fallback it replaces.
# Measured identical output to batch 8, so there is no reason to run hotter.
NEMO_MSDD_INFER_BATCH_SIZE = 4
SILENCE_THRESHOLD_DB = -40
SILENCE_MIN_DURATION = 0.99  # fraction of total duration that must be silent
SPEAKER_MATCH_THRESHOLD = 0.5  # cosine similarity threshold for speaker matching
MIN_ID_SEGMENT_DURATION = 2.5
MAX_ID_SEGMENT_DURATION = 30.0
MIN_ID_WORDS = 4


def _is_oom(exc: BaseException) -> bool:
    """Detect CUDA OOM across the several exception types torch/cuBLAS raise."""
    import torch

    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return isinstance(exc, RuntimeError) and "out of memory" in msg


def _release_oom_traceback(exc: BaseException) -> None:
    """Detach an OOM traceback so the CUDA models it pins can be collected.

    Every CPU fallback below runs inside its ``except`` block, where the live
    exception still owns the traceback of the call that failed. Those frames
    hold the CUDA whisper/align/diarizer models as locals, so ``del model`` +
    ``empty_cache()`` free nothing and the GPU stays pinned for the entire slow
    CPU run — measured at 7.6 GB of 8 GB retained with the GPU fully idle.
    """
    pending = [exc]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        current.__traceback__ = None
        pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)


@contextmanager
def _trusted_model_load_context():
    """Load bundled/trusted WhisperX/PyAnnote checkpoints with legacy metadata."""
    key = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _cuda_library_dirs() -> list[Path]:
    """Find pip-installed NVIDIA runtime library dirs for CUDA model backends."""
    site_package_dirs = []
    for value in (
        sysconfig.get_paths().get("purelib"),
        sysconfig.get_paths().get("platlib"),
        *getattr(site, "getsitepackages", lambda: [])(),
        site.getusersitepackages(),
    ):
        if value:
            path = Path(value)
            if path not in site_package_dirs:
                site_package_dirs.append(path)
    nvidia_packages = (
        "cublas",
        "cuda_nvrtc",
        "cudnn",
        "cufft",
        "curand",
        "cusolver",
        "cusparse",
        "nccl",
        "nvjitlink",
    )
    candidates = [
        site_package_dir / "nvidia" / package / "lib"
        for site_package_dir in site_package_dirs
        for package in nvidia_packages
    ]
    return [path for path in candidates if path.exists()]


def _with_cuda_library_path(env: dict[str, str] | None = None) -> dict[str, str]:
    """Prepend venv/user-site NVIDIA libs so CTranslate2 can dlopen CUDA deps."""
    merged = dict(os.environ if env is None else env)
    cuda_dirs = [str(path) for path in _cuda_library_dirs()]
    if not cuda_dirs:
        return merged

    existing = [part for part in merged.get("LD_LIBRARY_PATH", "").split(":") if part]
    merged["LD_LIBRARY_PATH"] = ":".join([*cuda_dirs, *existing])
    return merged


def _activate_cuda_library_path() -> None:
    os.environ.update(_with_cuda_library_path())


def is_silent(path: Path) -> bool:
    """Check if an audio file is effectively silent."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", str(path),
                "-af", f"silencedetect=noise={SILENCE_THRESHOLD_DB}dB:d=1",
                "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=60,
        )
        stderr = result.stderr

        # Get total duration
        duration = None
        for line in stderr.split("\n"):
            if "Duration:" in line:
                parts = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = parts.split(":")
                duration = float(h) * 3600 + float(m) * 60 + float(s)
                break

        if not duration or duration < 1:
            return True

        # Sum silence durations
        silence_total = 0.0
        for line in stderr.split("\n"):
            if "silence_duration:" in line:
                dur = float(line.split("silence_duration:")[1].strip())
                silence_total += dur

        return (silence_total / duration) >= SILENCE_MIN_DURATION

    except Exception as e:
        print(f"  Warning: silence detection failed: {e}", file=sys.stderr)
        return False


def _cuda_memory_stats() -> str:
    """Torch peaks exclude CTranslate2; driver free memory covers both."""
    import torch

    try:
        if not torch.cuda.is_available():
            return "cuda=unavailable"
        free, total = torch.cuda.mem_get_info()
        values = {
            "free": free, "total": total,
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
        }
        return " ".join(f"{key}_MiB={value / 2**20:.0f}" for key, value in values.items())
    except Exception as exc:
        return f"cuda_stats_unavailable={exc}"


def _log_oom(exc: BaseException, phase: str) -> None:
    # Log before detaching frames, including NeMo's internal failing stage.
    print(f"    OOM phase={phase} {_cuda_memory_stats()}", file=sys.stderr, flush=True)
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    _release_oom_traceback(exc)


def _clear_model_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe(
    audio_path: Path, model, device: str, language: str,
    *, batch_state: dict | None = None,
) -> dict:
    """Run ASR only; align after all tracks have released Whisper and VAD.

    Keep a working reduced GPU batch for subsequent tracks in this part.
    The verified context supplies the language, never Whisper's first-window
    language detection.
    """
    import whisperx

    audio = whisperx.load_audio(str(audio_path))
    batch_state = {} if batch_state is None else batch_state
    start_batch = batch_state.get("batch_size", 4)
    batches = [batch for batch in (4, 2, 1) if batch <= start_batch] if device == "cuda" else [8]
    print(f"    ASR input={audio_path.name} device={device} batch_size={batches[0]}", flush=True)
    for batch_size in batches:
        try:
            result = model.transcribe(audio, batch_size=batch_size, language=language)
            if device == "cuda":
                batch_state["batch_size"] = batch_size
            return result
        except Exception as exc:
            if device != "cuda" or not _is_oom(exc) or batch_size == batches[-1]:
                raise
            _log_oom(exc, f"asr:{audio_path.name}:batch={batch_size}")
            print(f"    Retrying ASR with batch_size={batch_size // 2}")
            _clear_model_memory()
    raise AssertionError("No ASR batch attempted")


def load_alignment_model(device: str, language: str):
    import whisperx

    return whisperx.load_align_model(
        language_code=language, device=device,
        model_name=ALIGN_MODELS.get(language),
    )


def align_transcription(audio_path: Path, result: dict, device: str, alignment_model) -> dict:
    """Align raw ASR, preserving the original segments for a CPU retry."""
    import whisperx

    audio = whisperx.load_audio(str(audio_path))
    align_model, metadata = alignment_model
    return whisperx.align(
        copy.deepcopy(result["segments"]), align_model, metadata, audio, device,
        return_char_alignments=False,
    )


def _run_model_stage(
    phase: str, tracks: dict, device: str, load_model, run,
    *, skip_load_errors: bool = False, skip_empty: bool = False,
) -> dict:
    """Load one stage, reuse its model, and retry only that stage on CPU.

    Exceptions are detached before releasing a failed GPU model. Each call
    receives the preferred device independently of earlier stages' fallbacks.
    """
    model = None
    results = {}
    started = time.monotonic()
    if device == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()
    print(f"  Stage {phase}: device={device} {_cuda_memory_stats()}", flush=True)
    try:
        for channel, (path, previous) in tracks.items():
            if skip_empty and not previous.get("segments"):
                results[channel] = previous
                print(f"    {phase}: skipping {path.name}, no transcript segments")
                continue
            print(f"    {phase}: starting {channel} {path.name}", flush=True)
            while True:
                operation = "load" if model is None else channel
                try:
                    if model is None:
                        model = load_model(device)
                    operation = channel
                    results[channel] = run(path, previous, model, device)
                    if isinstance(results[channel], dict):
                        print(f"    {phase}: completed {path.name}, segments={len(results[channel].get('segments', []))}", flush=True)
                    break
                except Exception as exc:
                    oom = _is_oom(exc)
                    if skip_load_errors and operation == "load" and isinstance(exc, RuntimeError) and not oom:
                        print(f"  Skipping {phase} ({exc})")
                        return {name: results.get(name, previous) for name, (_, previous) in tracks.items()}
                    if device != "cuda" or not oom:
                        raise
                    _log_oom(exc, f"{phase}:{operation}:{path.name}")
                    model = None
                    _clear_model_memory()
                    print(f"    {phase} falling back to CPU; completed tracks retained.", flush=True)
                    device = "cpu"
        return results
    finally:
        model = None
        _clear_model_memory()
        print(f"  Stage {phase}: elapsed_s={time.monotonic() - started:.1f} {_cuda_memory_stats()}", flush=True)


def _annotation_to_df(annotation) -> pd.DataFrame:
    import pandas as pd

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


def _ensure_pcm_wav(audio_path: Path) -> Path:
    WAV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = WAV_CACHE_DIR / f"{audio_path.stem}.wav"
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


def _cleanup_pcm_wav(audio_path: Path | None) -> None:
    if audio_path is None:
        return
    wav_path = WAV_CACHE_DIR / f"{audio_path.stem}.wav"
    wav_path.unlink(missing_ok=True)


def _set_nemo_clustering_device(diarizer, device: str) -> None:
    import torch

    clus = diarizer.clustering_embedding.clus_diar_model
    clus._speaker_model = clus._speaker_model.to(torch.device(device))


def _set_nemo_msdd_batch_size(diarizer, batch_size: int) -> None:
    """Shrink MSDD's inference batch so long parts fit on an 8 GB card.

    NeMo defaults ``infer_batch_size`` to 25, which OOMs in the MSDD forward
    pass (``conv_scale_weights`` -> relu) on a 65 min part — the length the
    recorder rotates at, so every long meeting produces one. Note the OOM is
    *not* in clustering: that stage handles 12940 segments unchunked without
    complaint, so ``embeddings_per_chunk`` is the wrong knob here.

    ``transfer_diar_params_to_model_params`` already copied the config value
    into ``test_ds`` at construction, so set both; ``setup_test_data`` re-reads
    ``test_ds.batch_size`` at inference time, which is what actually takes
    effect.
    """
    diarizer._cfg.diarizer.msdd_model.parameters.infer_batch_size = batch_size
    diarizer.msdd_model.cfg.test_ds.batch_size = batch_size


def load_pyannote_embedding_model(hf_token: str, device: str):
    """Load pyannote's speaker embedding model for post-hoc speaker naming."""
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    ).to(torch.device(device))
    return pipeline._embedding


def load_diarizer(diarizer_name: str, device: str, hf_token: str | None):
    """Load the requested diarization backend."""
    if diarizer_name == "nemo":
        from nemo.collections.asr.models import NeuralDiarizer

        diarizer = NeuralDiarizer.from_pretrained(
            model_name="diar_msdd_telephonic",
            vad_model_name="vad_multilingual_marblenet",
            map_location=device,
            verbose=False,
        )
        # NeMo 2.7.2 otherwise leaves clustering on CPU even when loaded on CUDA.
        if device == "cuda":
            _set_nemo_clustering_device(diarizer, device)
            _set_nemo_msdd_batch_size(diarizer, NEMO_MSDD_INFER_BATCH_SIZE)
        return diarizer

    if diarizer_name == "whisperx":
        from whisperx.diarize import DiarizationPipeline

        if not hf_token:
            raise RuntimeError("WhisperX diarization requires a HuggingFace token.")
        return DiarizationPipeline(token=hf_token, device=device)

    raise ValueError(f"Unknown diarizer: {diarizer_name}")


def _filter_segments_for_identification(segments: list[dict], speaker_label: str) -> list[dict]:
    candidates = []
    for seg in segments:
        if seg.get("speaker") != speaker_label:
            continue
        duration = seg.get("end", 0.0) - seg.get("start", 0.0)
        if duration < MIN_ID_SEGMENT_DURATION or duration > MAX_ID_SEGMENT_DURATION:
            continue
        text = seg.get("text", "").strip()
        if len(text.split()) < MIN_ID_WORDS:
            continue
        if is_backchannel(text):
            continue
        candidates.append(seg)
    return candidates


def extract_segment_embeddings(
    audio_path: Path, segments: list[dict], emb_model, device: str
) -> np.ndarray:
    """Extract one pyannote embedding per segment."""
    import numpy as np
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    embeddings = []
    for seg in segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = waveform[:, start_sample:end_sample]
        if chunk.shape[1] < emb_model.min_num_samples:
            continue
        chunk = chunk.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = emb_model(chunk)
        if isinstance(emb, np.ndarray):
            embeddings.append(emb.squeeze())
        else:
            embeddings.append(emb.squeeze().cpu().numpy())

    return np.array(embeddings) if embeddings else np.empty((0, 256))


def build_speaker_centroids_from_result(
    audio_path: Path, result: dict, emb_model, device: str
) -> dict[str, list[float]]:
    """Build one pyannote centroid per anonymous speaker label in the result."""
    by_speaker = {}
    for seg in result.get("segments", []):
        speaker = seg.get("speaker")
        if not speaker or speaker == "unknown":
            continue
        by_speaker.setdefault(speaker, []).append(seg)

    centroids = {}
    for speaker, segments in by_speaker.items():
        candidates = _filter_segments_for_identification(segments, speaker)
        if not candidates:
            continue
        embeddings = extract_segment_embeddings(audio_path, candidates, emb_model, device)
        if len(embeddings) == 0:
            continue
        centroids[speaker] = embeddings.mean(axis=0).tolist()
    return centroids


def _rename_result_speakers(result: dict, name_map: dict[str, str]) -> dict:
    """Rename speaker labels in both segment and word annotations."""
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


def apply_enrolled_speaker_names(
    audio_path: Path,
    result: dict | None,
    emb_model,
    device: str,
) -> dict | None:
    if result is None:
        return None
    embeddings = build_speaker_centroids_from_result(audio_path, result, emb_model, device)
    if not embeddings:
        return result
    name_map = match_speakers(embeddings)
    if name_map:
        print(f"    Speaker matches: {name_map}")
        result = _rename_result_speakers(result, name_map)
    return result


def diarize(
    audio_path: Path,
    result: dict,
    device: str,
    diarizer_name: str,
    diarizer_model,
    hf_token: str | None,
    speaker_id_model=None,
) -> dict:
    """Add speaker labels to transcription result using the selected diarizer."""
    if not result.get("segments"):
        print("    Skipping diarization: transcript has no segments")
        return result

    if diarizer_name == "nemo":
        from whisperx.diarize import assign_word_speakers

        wav_path = _ensure_pcm_wav(audio_path)
        print(
            f"    NeMo input={audio_path.name} duration_s={audio_duration(wav_path):.1f} "
            f"device={device} {_cuda_memory_stats()}", flush=True,
        )
        annotation = diarizer_model(str(wav_path))
        print(f"    NeMo output speakers={len(annotation.labels())}", flush=True)
        result = assign_word_speakers(_annotation_to_df(annotation), result)
        if speaker_id_model is not None:
            embeddings = build_speaker_centroids_from_result(
                wav_path, result, speaker_id_model, device
            )
            if embeddings:
                name_map = match_speakers(embeddings)
                if name_map:
                    print(f"    Speaker matches: {name_map}")
                    result = _rename_result_speakers(result, name_map)
        return result

    if diarizer_name != "whisperx":
        raise ValueError(f"Unknown diarizer: {diarizer_name}")

    from whisperx.diarize import assign_word_speakers
    import whisperx

    audio = whisperx.load_audio(str(audio_path))
    diarize_segments, embeddings = diarizer_model(audio, return_embeddings=True)
    result = assign_word_speakers(diarize_segments, result)

    # Match anonymous speakers against enrolled voices
    if embeddings:
        name_map = match_speakers(embeddings)
        if name_map:
            print(f"    Speaker matches: {name_map}")
            for seg in result.get("segments", []):
                spk = seg.get("speaker", "")
                if spk in name_map:
                    seg["speaker"] = name_map[spk]

    return result


def resegment_by_speaker(segments: list[dict]) -> list[dict]:
    """Split segments at word-level speaker boundaries.

    WhisperX assigns speakers per word but keeps Whisper's original segments
    (split on pauses). When two speakers talk without a pause, the whole segment
    gets the dominant speaker's label. This function splits such segments so each
    contiguous run of words from one speaker becomes its own segment.
    """
    new_segments = []
    for seg in segments:
        words = seg.get("words", [])
        if not words:
            new_segments.append(seg)
            continue

        # Group consecutive words by speaker
        current_speaker = None
        current_words = []
        for word in words:
            word_speaker = word.get("speaker", seg.get("speaker", "unknown"))
            if word_speaker != current_speaker and current_words:
                new_segments.append(_words_to_segment(current_words, current_speaker))
                current_words = []
            current_speaker = word_speaker
            current_words.append(word)

        if current_words:
            new_segments.append(_words_to_segment(current_words, current_speaker))

    return new_segments


def _words_to_segment(words: list[dict], speaker: str) -> dict:
    """Build a segment dict from a list of word dicts."""
    text = " ".join(w.get("word", "") for w in words).strip()
    start = words[0].get("start", words[0].get("end", 0.0))
    end = words[-1].get("end", words[-1].get("start", 0.0))
    return {
        "start": start,
        "end": end,
        "text": text,
        "speaker": speaker,
        "words": words,
    }


AMBIGUITY_GAP = 0.1  # reject if gap between best and second-best < this


def match_speakers(embeddings: dict) -> dict[str, str]:
    """Match diarization speaker embeddings against enrolled speakers.
    Returns a map of SPEAKER_XX -> enrolled name.
    Rejects ambiguous matches where two enrolled speakers are close."""
    import numpy as np

    if not SPEAKERS_DIR.exists():
        return {}

    # Load centroids for ready speakers
    enrolled = {}
    for d in SPEAKERS_DIR.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        centroid_path = d / "centroid.npy"
        if not meta_path.exists() or not centroid_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if not meta.get("ready", False):
            continue
        enrolled[meta.get("display_name", d.name)] = np.load(centroid_path)

    if not enrolled:
        return {}

    name_map = {}
    for spk_label, spk_emb in embeddings.items():
        spk_vec = np.array(spk_emb)

        # Compute similarity to all enrolled speakers
        sims = {}
        for name, centroid in enrolled.items():
            sim = np.dot(spk_vec, centroid) / (
                np.linalg.norm(spk_vec) * np.linalg.norm(centroid) + 1e-8
            )
            sims[name] = float(sim)

        if not sims:
            continue

        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        best_name, best_sim = ranked[0]

        # Must exceed threshold
        if best_sim < SPEAKER_MATCH_THRESHOLD:
            continue

        # Must not be ambiguous with second-best
        if len(ranked) > 1:
            _, second_sim = ranked[1]
            if best_sim - second_sim < AMBIGUITY_GAP:
                continue

        name_map[spk_label] = best_name

    return name_map


def _text_similarity(a: str, b: str) -> float:
    """Rough word-overlap similarity between two strings (Jaccard on words)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


DEDUP_TIME_WINDOW = 10.0   # seconds: max offset between echo and original
DEDUP_SIMILARITY = 0.5     # Jaccard threshold to consider it a duplicate


def _dedup_echo(mic_entries: list[dict], sys_entries: list[dict]) -> list[dict]:
    """Remove mic segments that are echoes of sys (remote speaker bleeding
    through the laptop speaker into the mic). Sys is the clean digital
    capture, so when both contain similar text near the same timestamp,
    drop the mic version."""
    if not sys_entries:
        return mic_entries

    kept = []
    for mic_seg in mic_entries:
        is_echo = False
        for sys_seg in sys_entries:
            # Check time proximity
            time_diff = abs(mic_seg["start"] - sys_seg["start"])
            if time_diff > DEDUP_TIME_WINDOW:
                continue
            # Check text similarity
            if _text_similarity(mic_seg["text"], sys_seg["text"]) >= DEDUP_SIMILARITY:
                is_echo = True
                break
        if not is_echo:
            kept.append(mic_seg)
    return kept


def merge_channels(mic_result: dict, sys_result: dict | None, ts: str) -> list[dict]:
    """Merge mic and sys transcripts into a single timeline, removing echo duplicates."""
    mic_entries = []
    for seg in mic_result.get("segments", []):
        mic_entries.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "speaker": seg.get("speaker", "unknown"),
            "channel": "mic",
        })

    sys_entries = []
    if sys_result:
        for seg in sys_result.get("segments", []):
            sys_entries.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "speaker": seg.get("speaker", "unknown"),
                "channel": "sys",
            })

    # Remove mic echoes of remote speech
    if sys_entries:
        before = len(mic_entries)
        mic_entries = _dedup_echo(mic_entries, sys_entries)
        dropped = before - len(mic_entries)
        if dropped:
            print(f"    Dedup: removed {dropped} mic echo segment(s)")

    entries = mic_entries + sys_entries
    entries.sort(key=lambda e: e["start"])
    return entries


def format_transcript(entries: list[dict], ts: str) -> str:
    """Format merged entries as readable markdown."""
    lines = [f"# Transcript {ts}", ""]

    prev_speaker = None
    prev_channel = None
    for e in entries:
        tag = f"{e['channel']}:{e['speaker']}"
        if tag != f"{prev_channel}:{prev_speaker}":
            start_fmt = _fmt_time(e["start"])
            lines.append(f"\n**[{start_fmt}] {tag}**\n")
            prev_speaker = e["speaker"]
            prev_channel = e["channel"]
        lines.append(e["text"])

    return "\n".join(lines) + "\n"


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _describe_parts(paths: list[Path]) -> str:
    if not paths:
        return "0 parts"
    if len(paths) == 1:
        return paths[0].name
    return f"{len(paths)} parts ({paths[0].name} .. {paths[-1].name})"


def _audio_duration(path: Path | None) -> float:
    return audio_duration(path)


def _offset_entries(entries: list[dict], offset_seconds: float) -> list[dict]:
    if not offset_seconds:
        return entries
    shifted: list[dict] = []
    for entry in entries:
        shifted.append(
            {
                **entry,
                "start": entry["start"] + offset_seconds,
                "end": entry["end"] + offset_seconds,
            }
        )
    return shifted


def _relabel_anonymous_entries(
    entries: list[dict],
    *,
    part_index: int,
    use_part_suffix: bool,
) -> list[dict]:
    relabeled: list[dict] = []

    for entry in entries:
        speaker = entry.get("speaker", "unknown")
        match = SPEAKER_RE.match(str(speaker))
        if match:
            speaker = f"SPEAKER_{match.group(1)}"
            if use_part_suffix:
                speaker = f"{speaker}_p{part_index:02d}"
        relabeled.append({**entry, "speaker": speaker})

    return relabeled


def process_part(
    mic_part: Path,
    sys_part: Path | None,
    *,
    part_index: int,
    use_part_suffix: bool,
    language: str,
    do_diarize: bool = True,
    diarizer_name: str = DEFAULT_DIARIZER,
) -> list[dict]:
    _activate_cuda_library_path()

    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = get_hf_token() if do_diarize else None
    tracks = {"mic": (mic_part, None)}
    if sys_part:
        if is_silent(sys_part):
            print(f"  Part p{part_index:02d}: system audio is silent, skipping {sys_part.name}")
        else:
            tracks["sys"] = (sys_part, None)
    else:
        print(f"  Part p{part_index:02d}: no matching sys file found")

    def _load_whisper(dev: str):
        with _trusted_model_load_context():
            return whisperx.load_model(
                MODEL, dev,
                compute_type="float16" if dev == "cuda" else "int8",
                language=language,
            )

    def _with_results(results):
        return {channel: (path, results[channel]) for channel, (path, _) in tracks.items()}

    batch_state = {}
    print(f"  Loading ASR model: {MODEL} (language: {language})", flush=True)
    try:
        results = _run_model_stage(
            "asr", tracks, device, _load_whisper,
            lambda path, previous, model, dev: transcribe(
                path, model, dev, language, batch_state=batch_state,
            ),
        )
        # _run_model_stage has dropped Whisper AND its VAD before any aligner
        # is loaded. Reload audio here instead of retaining both PCM tracks.
        results = _run_model_stage(
            "alignment", _with_results(results), device,
            lambda dev: load_alignment_model(dev, language),
            lambda path, previous, model, dev: align_transcription(path, previous, dev, model),
            skip_empty=True,
        )
        if do_diarize:
            results = _run_model_stage(
                f"diarization:{diarizer_name}", _with_results(results), device,
                lambda dev: load_diarizer(diarizer_name, dev, hf_token),
                lambda path, previous, model, dev: diarize(
                    path, previous, dev, diarizer_name, model, hf_token,
                ),
                skip_load_errors=True, skip_empty=True,
            )

        if do_diarize and diarizer_name == "nemo" and hf_token and SPEAKERS_DIR.exists():
            results = _run_model_stage(
                "speaker-naming", _with_results(results), device,
                lambda dev: load_pyannote_embedding_model(hf_token, dev),
                lambda path, previous, model, dev: apply_enrolled_speaker_names(
                    path, previous, model, dev,
                ),
            )

        part_entries = merge_channels(results["mic"], results.get("sys"), mic_part.stem)
        return _relabel_anonymous_entries(
            part_entries, part_index=part_index, use_part_suffix=use_part_suffix,
        )
    finally:
        if diarizer_name == "nemo":
            _cleanup_pcm_wav(mic_part)
            _cleanup_pcm_wav(sys_part)


def process_session(session_id: str, do_diarize: bool = True, diarizer_name: str = DEFAULT_DIARIZER):
    """Process a single recording session, possibly spanning multiple rotated parts."""
    from . import context as meeting_context

    session = raw_audio_session(session_id)
    if session is None or not session.mic_parts:
        raise RuntimeError(f"No mic recording parts found for session {session_id}")

    # The context is the contract: it decides the language, and a person has
    # normally confirmed it. Transcribing without one means guessing again.
    context = meeting_context.load_context(session_id)
    if context is None:
        raise RuntimeError(
            f"No context for {session_id} — run hugin-meet-context {session_id} first"
        )
    language = context.language_value

    from .pipeline import year_subdir

    out_json = transcript_json_path(session_id)
    out_md = TRANSCRIPT_DIR / year_subdir(session_id) / f"transcript-{session_id}.md"

    if out_json.exists():
        print(f"  Already processed: {out_json.name}")
        return

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    sys_parts_by_index = {
        parse_raw_audio_part(path).part: path
        for path in session.sys_parts
        if parse_raw_audio_part(path) is not None
    }
    session_entries: list[dict] = []
    session_offset = 0.0
    use_part_suffix = len(session.mic_parts) > 1 or len(session.sys_parts) > 1

    for mic_part in session.mic_parts:
        mic_info = parse_raw_audio_part(mic_part)
        if mic_info is None:
            continue
        sys_part = sys_parts_by_index.get(mic_info.part)
        with tempfile.NamedTemporaryFile(
            prefix=f"transcribe-part-{session_id}-p{mic_info.part:02d}-",
            suffix=".json",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        try:
            cmd = [
                sys.executable,
                "-m",
                "hugin_meetings.transcribe_part",
                "--mic",
                str(mic_part),
                "--part-index",
                str(mic_info.part),
                "--json-out",
                str(tmp_path),
                "--diarizer",
                diarizer_name,
                "--language",
                language,
            ]
            if sys_part is not None:
                cmd.extend(["--sys", str(sys_part)])
            if use_part_suffix:
                cmd.append("--use-part-suffix")
            if not do_diarize:
                cmd.append("--no-diarize")

            subprocess.run(cmd, check=True, env=_with_cuda_library_path())
            part_entries = json.loads(tmp_path.read_text())
            part_entries = _offset_entries(part_entries, session_offset)
            session_entries.extend(part_entries)
            session_offset += max(_audio_duration(mic_part), _audio_duration(sys_part))
        finally:
            tmp_path.unlink(missing_ok=True)

    out_json.write_text(json.dumps(session_entries, indent=2, ensure_ascii=False))
    print(f"  Wrote {out_json}")

    out_md.write_text(format_transcript(session_entries, session_id))
    print(f"  Wrote {out_md}")

    meeting_context.record_applied(session_id, language=language)


def find_unprocessed() -> list[str]:
    """Find recording sessions that don't have a corresponding transcript."""
    json_dir = load_config().transcript_json_dir
    processed = {
        p.stem.removeprefix("transcript-")
        for p in json_dir.rglob("transcript-*.json")
        if not p.name.endswith(".customer.json")
    }
    sessions = scan_raw_audio_sessions()
    return [
        session_id
        for session_id in sorted(sessions)
        if session_id not in processed and sessions[session_id].mic_parts
    ]


def main():
    parser = argparse.ArgumentParser(description="Transcribe mic/sys recording sessions")
    parser.add_argument("file", nargs="?", help="Session id or raw opus file from the session")
    parser.add_argument("--all", action="store_true", help="Process all unprocessed sessions")
    parser.add_argument("--no-diarize", action="store_true", help="Skip diarization")
    parser.add_argument(
        "--diarizer",
        choices=("nemo", "whisperx"),
        default=DEFAULT_DIARIZER,
        help=f"Diarization backend (default: {DEFAULT_DIARIZER})",
    )
    args = parser.parse_args()

    if args.all:
        unprocessed = find_unprocessed()
        if not unprocessed:
            print("Nothing to process.")
            return
        print(f"Found {len(unprocessed)} unprocessed recording session(s)")
        for session_id in unprocessed:
            print(f"\nProcessing: {session_id}")
            process_session(
                session_id,
                do_diarize=not args.no_diarize,
                diarizer_name=args.diarizer,
            )
    elif args.file:
        input_path = Path(args.file)
        resolved_path = None
        if input_path.is_absolute() and input_path.exists():
            resolved_path = input_path
        elif (AUDIO_DIR / input_path).exists():
            resolved_path = AUDIO_DIR / input_path
        elif (Path.cwd() / input_path).exists():
            resolved_path = Path.cwd() / input_path

        session_id = extract_timestamp(resolved_path.name if resolved_path else args.file)
        if not session_id:
            print(f"Could not determine recording session from: {args.file}", file=sys.stderr)
            sys.exit(1)
        print(f"Processing: {session_id}")
        process_session(
            session_id,
            do_diarize=not args.no_diarize,
            diarizer_name=args.diarizer,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
