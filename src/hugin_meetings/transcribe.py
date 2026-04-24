#!/usr/bin/env python3
"""Transcribe and diarize a mic/sys recording session, merge into a unified transcript.

Usage:
    transcribe.py mic-20260408-213541-p01.opus      # process whole session
    transcribe.py 20260408-213541                   # process whole session
    transcribe.py --all                             # process all unprocessed sessions
"""

import argparse
import gc
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from .pipeline import (
    extract_timestamp,
    parse_raw_audio_part,
    raw_audio_session,
    scan_raw_audio_sessions,
    transcript_json_path,
)
from .config import load_config

_cfg = load_config()

AUDIO_DIR = _cfg.raw_audio_dir
WAV_CACHE_DIR = _cfg.wav_cache_dir
TRANSCRIPT_DIR = _cfg.transcripts_dir
SPEAKERS_DIR = _cfg.speakers_dir
MODEL = _cfg.raw.get("meetings", {}).get("transcribe_model", "KBLab/kb-whisper-large")
DEFAULT_DIARIZER = "nemo"
SILENCE_THRESHOLD_DB = -40
SILENCE_MIN_DURATION = 0.99  # fraction of total duration that must be silent
SPEAKER_MATCH_THRESHOLD = 0.5  # cosine similarity threshold for speaker matching
MIN_ID_SEGMENT_DURATION = 2.5
MAX_ID_SEGMENT_DURATION = 30.0
MIN_ID_WORDS = 4
BACKCHANNEL_WORDS = {
    "mm", "mhm", "mmm", "ja", "jo", "yes", "yeah", "ok", "okej", "okay",
    "aha", "haha", "hm", "hmm", "nej", "nä", "no", "jaha", "japp",
}
ANON_SPEAKER_RE = re.compile(r"^(?:speaker|SPEAKER)_(\d+)$")


def _is_oom(exc: BaseException) -> bool:
    """Detect CUDA OOM across the several exception types torch/cuBLAS raise."""
    import torch

    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return isinstance(exc, RuntimeError) and "out of memory" in msg


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


def transcribe(audio_path: Path, model, device: str) -> dict:
    """Transcribe a single audio file. Returns whisperx result dict."""
    import torch
    import whisperx

    audio = whisperx.load_audio(str(audio_path))

    for batch_size in (8, 4, 2):
        try:
            result = model.transcribe(audio, batch_size=batch_size)
            break
        except RuntimeError as e:
            if "out of memory" in str(e) and batch_size > 2:
                print(f"    OOM at batch_size={batch_size}, retrying with {batch_size // 2}...")
                torch.cuda.empty_cache()
            else:
                raise

    # Word-level alignment
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device,
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )
    del align_model
    torch.cuda.empty_cache()
    return result


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
        return diarizer

    if diarizer_name == "whisperx":
        from whisperx.diarize import DiarizationPipeline

        if not hf_token:
            raise RuntimeError("WhisperX diarization requires a HuggingFace token.")
        return DiarizationPipeline(token=hf_token, device=device)

    raise ValueError(f"Unknown diarizer: {diarizer_name}")


def _is_backchannel(text: str) -> bool:
    words = text.lower().strip().split()
    return bool(words) and all(w.strip(".,!?") in BACKCHANNEL_WORDS for w in words)


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
        if _is_backchannel(text):
            continue
        candidates.append(seg)
    return candidates


def extract_segment_embeddings(
    audio_path: Path, segments: list[dict], emb_model, device: str
) -> np.ndarray:
    """Extract one pyannote embedding per segment."""
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
    if diarizer_name == "nemo":
        from whisperx.diarize import assign_word_speakers

        wav_path = _ensure_pcm_wav(audio_path)
        annotation = diarizer_model(str(wav_path))
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


def get_hf_token() -> str | None:
    """Try to get HuggingFace token from standard locations."""
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass

    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()

    return None


def _describe_parts(paths: list[Path]) -> str:
    if not paths:
        return "0 parts"
    if len(paths) == 1:
        return paths[0].name
    return f"{len(paths)} parts ({paths[0].name} .. {paths[-1].name})"


def _audio_duration(path: Path | None) -> float:
    if path is None:
        return 0.0
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip() or "0")


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
        match = ANON_SPEAKER_RE.match(str(speaker))
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
    do_diarize: bool = True,
    diarizer_name: str = DEFAULT_DIARIZER,
) -> list[dict]:
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = get_hf_token() if do_diarize else None
    sys_result = None
    mic_result = None
    diarization_device = device

    def _load_whisper(dev: str):
        return whisperx.load_model(
            MODEL,
            dev,
            compute_type="float16" if dev == "cuda" else "int8",
        )

    def _run_whisper(audio_part: Path):
        nonlocal model, device, diarization_device
        try:
            return transcribe(audio_part, model, device)
        except Exception as exc:
            if not _is_oom(exc) or device != "cuda":
                raise
            print(f"    Whisper OOM on CUDA for {audio_part.name}; falling back to CPU/int8 (slow).")
            try:
                del model
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            device = "cpu"
            diarization_device = "cpu"
            model = _load_whisper(device)
            return transcribe(audio_part, model, device)

    print(f"  Loading model: {MODEL}")
    try:
        try:
            model = _load_whisper(device)
        except Exception as exc:
            if not _is_oom(exc) or device != "cuda":
                raise
            print("    Whisper OOM loading on CUDA; falling back to CPU/int8 (slow).")
            gc.collect()
            torch.cuda.empty_cache()
            device = "cpu"
            diarization_device = "cpu"
            model = _load_whisper(device)

        print(f"  Part p{part_index:02d}: transcribing mic {mic_part.name}")
        mic_result = _run_whisper(mic_part)
        print(
            f"    {len(mic_result.get('segments', []))} segments, "
            f"language: {mic_result.get('language', '?')}"
        )

        if sys_part:
            if is_silent(sys_part):
                print(f"  Part p{part_index:02d}: system audio is silent, skipping {sys_part.name}")
            else:
                print(f"  Part p{part_index:02d}: transcribing sys {sys_part.name}")
                sys_result = _run_whisper(sys_part)
                print(f"    {len(sys_result.get('segments', []))} segments")
        else:
            print(f"  Part p{part_index:02d}: no matching sys file found")

        del model
        if device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        diarizer_model = None
        if do_diarize:
            try:
                diarizer_model = load_diarizer(diarizer_name, diarization_device, hf_token)
            except RuntimeError as exc:
                print(f"  Skipping diarization ({exc})")
                diarizer_model = None

        if diarizer_model is not None:
            print(f"  Part p{part_index:02d}: diarizing mic with {diarizer_name}...")
            try:
                mic_result = diarize(
                    mic_part,
                    mic_result,
                    diarization_device,
                    diarizer_name,
                    diarizer_model,
                    hf_token,
                )
            except Exception as exc:
                if not _is_oom(exc) or diarization_device != "cuda":
                    raise
                print(f"    {diarizer_name} diarization OOM on CUDA for mic, retrying with a fresh CPU diarizer...")
                diarizer_model = None
                gc.collect()
                torch.cuda.empty_cache()
                diarization_device = "cpu"
                diarizer_model = load_diarizer(diarizer_name, diarization_device, hf_token)
                mic_result = diarize(
                    mic_part,
                    mic_result,
                    diarization_device,
                    diarizer_name,
                    diarizer_model,
                    hf_token,
                )
            if device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
            if sys_result is not None and sys_part is not None:
                print(f"  Part p{part_index:02d}: diarizing sys with {diarizer_name}...")
                try:
                    sys_result = diarize(
                        sys_part,
                        sys_result,
                        diarization_device,
                        diarizer_name,
                        diarizer_model,
                        hf_token,
                    )
                except Exception as exc:
                    if not _is_oom(exc) or diarization_device != "cuda":
                        raise
                    print(f"    {diarizer_name} diarization OOM on CUDA, retrying with a fresh CPU diarizer...")
                    diarizer_model = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    diarization_device = "cpu"
                    diarizer_model = load_diarizer(diarizer_name, diarization_device, hf_token)
                    sys_result = diarize(
                        sys_part,
                        sys_result,
                        diarization_device,
                        diarizer_name,
                        diarizer_model,
                        hf_token,
                    )
                if device == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()

        diarizer_model = None
        if device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        if do_diarize and diarizer_name == "nemo" and hf_token and SPEAKERS_DIR.exists():
            speaker_id_device = device
            print("  Loading pyannote embedding model for speaker naming...")
            speaker_id_model = load_pyannote_embedding_model(hf_token, speaker_id_device)

            def _name_speakers(audio_path, result):
                nonlocal speaker_id_model, speaker_id_device
                try:
                    return apply_enrolled_speaker_names(
                        audio_path, result, speaker_id_model, speaker_id_device
                    )
                except Exception as exc:
                    if not _is_oom(exc) or speaker_id_device != "cuda":
                        raise
                    print("    Speaker-naming OOM on CUDA; reloading embedding model on CPU.")
                    speaker_id_model = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    speaker_id_device = "cpu"
                    speaker_id_model = load_pyannote_embedding_model(hf_token, speaker_id_device)
                    return apply_enrolled_speaker_names(
                        audio_path, result, speaker_id_model, speaker_id_device
                    )

            mic_result = _name_speakers(mic_part, mic_result)
            if sys_result is not None and sys_part is not None:
                sys_result = _name_speakers(sys_part, sys_result)
            speaker_id_model = None
            if device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

        part_entries = merge_channels(mic_result, sys_result, mic_part.stem)
        return _relabel_anonymous_entries(
            part_entries,
            part_index=part_index,
            use_part_suffix=use_part_suffix,
        )
    finally:
        if diarizer_name == "nemo":
            _cleanup_pcm_wav(mic_part)
            _cleanup_pcm_wav(sys_part if sys_result is not None else None)


def process_session(session_id: str, do_diarize: bool = True, diarizer_name: str = DEFAULT_DIARIZER):
    """Process a single recording session, possibly spanning multiple rotated parts."""
    session = raw_audio_session(session_id)
    if session is None or not session.mic_parts:
        raise RuntimeError(f"No mic recording parts found for session {session_id}")

    out_json = transcript_json_path(session_id)
    out_md = TRANSCRIPT_DIR / f"transcript-{session_id}.md"

    if out_json.exists():
        print(f"  Already processed: {out_json.name}")
        return

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
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
            ]
            if sys_part is not None:
                cmd.extend(["--sys", str(sys_part)])
            if use_part_suffix:
                cmd.append("--use-part-suffix")
            if not do_diarize:
                cmd.append("--no-diarize")

            subprocess.run(cmd, check=True)
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


def find_unprocessed() -> list[str]:
    """Find recording sessions that don't have a corresponding transcript."""
    processed = {
        p.stem.removeprefix("transcript-")
        for p in transcript_json_path("").parent.glob("transcript-*.json")
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
