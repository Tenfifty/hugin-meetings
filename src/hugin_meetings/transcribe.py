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
import re
import site
import subprocess
import sys
import sysconfig
import tempfile
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Iterable

from .cli_utils import get_hf_token
from .pipeline import (
    SPEAKER_RE,
    audio_duration,
    extract_timestamp,
    parse_raw_audio_part,
    raw_audio_session,
    scan_in_progress_parts,
    scan_raw_audio_sessions,
    transcript_json_path,
    transcript_md_path,
)
from .config import load_config

AUDIO_DIR = load_config().raw_audio_dir
WAV_CACHE_DIR = load_config().wav_cache_dir
SPEAKERS_DIR = load_config().speakers_dir
MODEL = load_config().whisper_model

# ``whisper_model`` is the default for every language; these override it per
# language. KB-Whisper (KBLab, the National Library of Sweden) is large-v3
# fine-tuned on Swedish and roughly halves its word error rate there. Until
# 2026-06-04 it ran for every language, whatever ``whisper_model`` said. The
# fix made large-v3 the model for every language instead, Swedish included,
# until 2026-10-27. Override or extend via
# ``meetings.whisper_models: {<lang>: <model>}`` in config.
DEFAULT_WHISPER_MODELS = {
    "sv": "KBLab/kb-whisper-large",
}
WHISPER_MODELS = {
    **DEFAULT_WHISPER_MODELS,
    **(load_config().raw.get("meetings", {}).get("whisper_models") or {}),
}


def whisper_model_for(language: str) -> str:
    return WHISPER_MODELS.get(language, MODEL)


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

SILENCE_THRESHOLD_DB = -40
SILENCE_MIN_DURATION = 0.99  # fraction of total duration that must be silent
SPEAKER_MATCH_THRESHOLD = 0.6  # cosine to an enrolled centroid; two enrolled women sit 0.53 apart, so 0.5 did not separate same-sex speakers


def _exception_chain(exc: BaseException):
    """``exc`` and everything in its cause/context chain, each once.

    Guards against cycles in library exception chains.
    """
    pending = [exc]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)


def _is_oom(exc: BaseException) -> bool:
    """Detect CUDA OOM across the several exception types torch/cuBLAS raise.

    Torch writes its OOM message with NVML's list of GPU processes. When NVML
    cannot initialize, as when a driver upgrade has replaced the userspace
    libraries but not the loaded kernel module (until the next boot), the
    allocator raises an internal assert from that report instead of
    ``OutOfMemoryError``. On 2026-10-27 that killed a part that would have
    finished on CPU.
    """
    import torch

    # Pyannote wraps CUDA OOM in a plain MemoryError about VAD batch size.
    # Inspect the original exception, rather than treating every host-memory
    # failure as a GPU OOM.
    for current in _exception_chain(exc):
        if isinstance(current, torch.OutOfMemoryError):
            return True
        if isinstance(current, RuntimeError):
            msg = str(current)
            if "out of memory" in msg.lower() or ("nvmlInit" in msg and "CUDACachingAllocator" in msg):
                return True
    return False


def _release_oom_traceback(exc: BaseException) -> None:
    """Detach an OOM traceback so the CUDA models it pins can be collected.

    Every CPU fallback below runs inside its ``except`` block, where the live
    exception still owns the traceback of the call that failed. Those frames
    hold the CUDA whisper/align/diarizer models as locals, so ``del model`` +
    ``empty_cache()`` free nothing and the GPU stays pinned for the entire slow
    CPU run — measured at 7.6 GB of 8 GB retained with the GPU fully idle.
    """
    for current in _exception_chain(exc):
        current.__traceback__ = None


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


def _with_cuda_library_path() -> dict[str, str]:
    """Prepend venv/user-site NVIDIA libs so CTranslate2 can dlopen CUDA deps.

    For a child's environment: ld.so reads LD_LIBRARY_PATH only at process
    start, so setting it in a running process changes nothing.
    """
    merged = dict(os.environ)
    cuda_dirs = [str(path) for path in _cuda_library_dirs()]
    if not cuda_dirs:
        return merged

    existing = [part for part in merged.get("LD_LIBRARY_PATH", "").split(":") if part]
    merged["LD_LIBRARY_PATH"] = ":".join([*cuda_dirs, *existing])
    return merged


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


def _gpu_processes() -> list[str]:
    """Every process holding GPU memory, largest first, graphics ones included.

    Free memory alone cannot say who took the card. On 2026-10-26 ASR ran out
    with about 1 GB more in use than at noon, held by something that had let
    go two minutes later and could not be named afterwards. ``-q`` prints the
    full command line, where the process table truncates it.
    """
    try:
        report = subprocess.run(
            ["nvidia-smi", "-q", "-d", "PIDS"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"gpu_processes_unavailable={exc}"]
    processes = []
    for line in report.splitlines():
        key, _, value = (part.strip() for part in line.partition(":"))
        if key == "Process ID":
            processes.append({"pid": value})
        elif processes and key in ("Type", "Name", "Used GPU Memory"):
            processes[-1][key] = value

    def used_mib(process: dict) -> int:
        amount = process.get("Used GPU Memory", "").removesuffix("MiB").strip()
        return int(amount) if amount.isdigit() else -1

    return [
        f"gpu pid={p['pid']} type={p.get('Type', '?')} used_MiB={used_mib(p)} "
        f"{(p.get('Name') or '?').split()[0]}"
        for p in sorted(processes, key=used_mib, reverse=True)
    ]


def _log_oom(exc: BaseException, phase: str) -> None:
    # Log before detaching frames, including NeMo's internal failing stage.
    print(f"    OOM phase={phase} {_cuda_memory_stats()}", file=sys.stderr, flush=True)
    for line in _gpu_processes():
        print(f"      {line}", file=sys.stderr, flush=True)
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    _release_oom_traceback(exc)


def _clear_model_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_vad_cache(model) -> None:
    """Hand the VAD's torch cache back to the driver before Whisper decodes.

    WhisperX runs pyannote's VAD in torch, then decodes in CTranslate2, which
    allocates outside torch and cannot use torch's cached blocks. The VAD left
    about 400 MiB reserved for the whole decode, and every batch retry reran it.
    WhisperX dispatches on ``Vad`` subclasses, so the wrapper is one.
    """
    import torch
    from whisperx.vads import Vad

    class CacheReleasingVad(Vad):
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, *args, **kwargs):
            try:
                return self.inner(*args, **kwargs)
            finally:
                torch.cuda.empty_cache()

        def preprocess_audio(self, audio):
            return self.inner.preprocess_audio(audio)

        def merge_chunks(self, *args, **kwargs):
            return self.inner.merge_chunks(*args, **kwargs)

    model.vad_model = CacheReleasingVad(model.vad_model)


# Both models learned from subtitled video and leave its traces in near
# silence, each its own kind:
# - large-v3 writes the subtitlers' credits. 53 such segments in 18 sessions,
#   all from 2026-08-18 to 2026-10-20, while Swedish ran on large-v3; English
#   still does. Each was a whole segment on its own, and real speech mentions
#   the same words ("Textning ska vara längre ...", "... så prenumererar man
#   på det"), so only a segment that is nothing but a credit goes.
# - KB-Whisper writes subtitle markup, mostly entities without their
#   semicolon, inside real speech and in loops: "M&amp,A", "Smith &ampphew",
#   "Nja, så &lt&gt&gt&lt&gt..." (73 hits in 2 of 14 meetings in the
#   2026-10-27 benchmark, vault digitalt/taligenkänning.md; 79 segments in the
#   KB-Whisper sessions before 2026-06-04 and on 2026-11-02). "&amp" becomes
#   "&"; other entities and tags go, since speech never holds "<". A segment
#   with no speech left once its markup is gone goes.
SUBTITLE_CREDIT_RE = re.compile(
    r"(svensk)?textning\.nu"
    r"|undertexter från amara\.org-gemenskapen"
    r"|tack för att (du|ni) tittade på den här videon",
    re.IGNORECASE,
)
MARKUP_TAG_RE = re.compile(r"</?[a-zA-Z]+\s*/?>")
MARKUP_ENTITY_RE = re.compile(r"&(amp|lt|gt|quot|apos|nbsp|#\d+);?", re.IGNORECASE)


def _strip_markup(text: str) -> str:
    # Until nothing changes: "&amp;amp," is escaped twice.
    while True:
        stripped = MARKUP_ENTITY_RE.sub(
            lambda m: "&" if m.group(1).lower() == "amp" else "", MARKUP_TAG_RE.sub("", text)
        )
        if stripped == text:
            return text
        text = stripped


def drop_subtitle_artifacts(result: dict) -> dict:
    kept, dropped, cleaned = [], 0, 0
    for segment in result.get("segments", []):
        text = segment.get("text", "")
        plain = _strip_markup(text)
        had_markup = plain != text
        credit = SUBTITLE_CREDIT_RE.fullmatch(plain.strip().rstrip(".!"))
        if credit or (had_markup and not any(c.isalnum() for c in plain)):
            dropped += 1
            continue
        if had_markup:
            segment = {**segment, "text": plain}
            cleaned += 1
        kept.append(segment)
    if dropped or cleaned:
        print(f"    Subtitle artifacts: dropped {dropped} segment(s), cleaned markup from {cleaned}")
    return {**result, "segments": kept}


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
            return drop_subtitle_artifacts(result)
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
    *, skip_empty: bool = False,
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
                    if device != "cuda" or not _is_oom(exc):
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


def _segments_to_df(segments) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(
        [{"start": float(a), "end": float(b), "speaker": label} for a, b, label in segments],
        columns=["start", "end", "speaker"],
    )



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


def cancel_echo(mic_part: Path, sys_part: Path, out_path: Path):
    """The mic with sys cancelled out (see echo.py), or None without echo."""
    from .echo import cancel_echo as _cancel_echo

    return _cancel_echo(mic_part, sys_part, out_path)


def _cleanup_pcm_wav(audio_path: Path | None) -> None:
    if audio_path is None:
        return
    wav_path = WAV_CACHE_DIR / f"{audio_path.stem}.wav"
    wav_path.unlink(missing_ok=True)



# Sortformer finds turns; sortformer.py re-decides who is who by voice. It
# replaced NeMo's MSDD, which NeMo 3 removed and which OOMed on long parts.
def load_diarizer(device: str):
    from .sortformer import SortformerDiarizer

    return SortformerDiarizer(device)


def build_speaker_centroids_from_result(
    audio_path: Path, result: dict, emb_model, device: str
) -> dict[str, list[float]]:
    """Build one pyannote centroid per anonymous speaker label in the result."""
    from .speakers import embed_waveform_segments, enrollable, load_waveform

    by_speaker = {}
    for seg in result.get("segments", []):
        speaker = seg.get("speaker")
        if not speaker or speaker == "unknown" or not enrollable(seg):
            continue
        by_speaker.setdefault(speaker, []).append(seg)
    if not by_speaker:
        return {}

    waveform = load_waveform(audio_path)
    centroids = {}
    for speaker, segments in by_speaker.items():
        embeddings = embed_waveform_segments(waveform, segments, emb_model, device)
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
    allowed_speakers: Iterable[str],
) -> dict | None:
    if result is None:
        return None
    embeddings = build_speaker_centroids_from_result(audio_path, result, emb_model, device)
    if not embeddings:
        return result
    name_map = match_speakers(embeddings, allowed_speakers)
    if name_map:
        print(f"    Speaker matches: {name_map}")
        result = _rename_result_speakers(result, name_map)
    return result


def diarize(audio_path: Path, result: dict, device: str, diarizer_model) -> dict:
    """Add Sortformer's speaker labels to the words of a transcription result."""
    if not result.get("segments"):
        print("    Skipping diarization: transcript has no segments")
        return result

    from whisperx.diarize import assign_word_speakers

    wav_path = _ensure_pcm_wav(audio_path)
    print(
        f"    sortformer input={audio_path.name} duration_s={audio_duration(wav_path):.1f} "
        f"device={device} {_cuda_memory_stats()}", flush=True,
    )
    segments = diarizer_model(str(wav_path))
    if not segments:
        # The diarizer heard no speech, so what Whisper produced for this
        # track is a hallucination on noise ("Undertexter från
        # Amara.org-gemenskapen"). Log the dropped text as evidence.
        dropped = [seg.get("text", "").strip() for seg in result["segments"]]
        print(
            f"    sortformer found no speech in {audio_path.name}; "
            f"dropping {len(dropped)} ASR segment(s): {dropped}", flush=True,
        )
        return {**result, "segments": [], "word_segments": []}
    diarization = _segments_to_df(segments)
    print(f"    sortformer output speakers={diarization['speaker'].nunique()}", flush=True)
    return assign_word_speakers(diarization, result)


AMBIGUITY_GAP = 0.1  # reject if gap between best and second-best < this


def _ready_speaker_metas() -> list[tuple[Path, dict]]:
    """(dir, meta) for every enrolled speaker whose profile is ready."""
    if not SPEAKERS_DIR.exists():
        return []
    out = []
    for d in sorted(SPEAKERS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if not meta_path.exists() or not (d / "centroid.npy").exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        if meta.get("ready", False):
            out.append((d, meta))
    return out


def name_token_list(text: str) -> list[str]:
    """Lower-cased, accent-folded word tokens in order: 'Pär Blixt' -> ['par', 'blixt'].

    Emails are cut at '@' first so 'charlotte.eriksson@x.se' tokenizes to the
    name it stands for. 'ö' folds to 'o', so 'Öhlin' and 'ohlin' agree.
    """
    text = text.split("@", 1)[0]
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return "".join(ch if ch.isalnum() else " " for ch in folded.lower()).split()


def name_tokens(text: str) -> set[str]:
    """``name_token_list`` as a set, for order-free comparison ('Öhlin, Johannes')."""
    return set(name_token_list(text))


# Dropped from an attendee before the run-together comparison below.
NAME_HONORIFICS = {"dr", "prof", "mr", "mrs", "ms"}


def attendee_matches_speaker(attendee: str, speaker_name: str) -> bool:
    """Does an attendee entry (display name or email) denote this enrolled speaker?

    Two-plus-token attendees must equal the full name. A single token (a bare
    first name, or an email like 'amer@…') matches if it is one of the
    speaker's name tokens — loose, but the voice still has to match on top.

    Email local parts often fuse or pad a name: 'anna.vonhallen',
    'per.h.berg', 'dr.erik.lind'. Until 2026-10-07 addresses like these kept
    three enrolled people out of 17 meetings they were invited to. So a multi-token attendee also
    matches when, with initials and honorifics dropped, its tokens run together
    spell the name run together, in order.
    """
    a = name_tokens(attendee)
    s = name_tokens(speaker_name)
    if not a or not s:
        return False
    if len(a) >= 2:
        if a == s:
            return True
        core = [t for t in name_token_list(attendee) if len(t) > 1 and t not in NAME_HONORIFICS]
        return len(core) >= 2 and "".join(core) == "".join(name_token_list(speaker_name))
    return a <= s


def event_attendee_names(event: dict) -> list[str]:
    """Attendee display names / emails from a stored calendar event, organizer included."""
    names = []
    for attendee in event.get("attendees") or []:
        name = attendee.get("displayName") or attendee.get("email")
        if name:
            names.append(name)
    organizer = event.get("organizer") or {}
    name = organizer.get("displayName") or organizer.get("email")
    if name and name not in names:
        names.append(name)
    return names


def context_attendees(context) -> list[str]:
    """Attendees of the session's verified calendar event; [] without one."""
    from . import context as meeting_context

    candidate = meeting_context.best_candidate(context) if context is not None else None
    return event_attendee_names(candidate.event) if candidate else []


def speaker_allowlist(context) -> list[str]:
    """Enrolled names a transcript of this session may be labelled with.

    Voice matching alone cannot say "nobody I know": an unknown voice lands
    on whichever enrolled centroid is nearest, and with few speakers of the
    same sex enrolled that clears the 0.5 threshold routinely. Between April
    and September 2026 eight meetings had a woman labelled with the one
    enrolled female name although she was in none of their invitations. So
    the calendar decides who is a candidate, and the voice only decides
    which of them it is.

    Speakers whose meta has ``always_candidate: true`` (David) are candidates
    regardless. With no attendee list to go on, only those are.
    """
    always = []
    others = []
    for _, meta in _ready_speaker_metas():
        name = meta.get("display_name")
        if not name:
            continue
        (always if meta.get("always_candidate") else others).append(name)

    attendees = context_attendees(context)
    if not attendees:
        return always

    allowed = list(always)
    for name in others:
        if any(attendee_matches_speaker(a, name) for a in attendees):
            allowed.append(name)
    return allowed


def match_speakers(embeddings: dict, allowed_speakers: Iterable[str]) -> dict[str, str]:
    """Match diarization speaker embeddings against enrolled speakers.
    Returns a map of SPEAKER_XX -> enrolled name.
    Rejects ambiguous matches where two enrolled speakers are close.

    ``allowed_speakers`` limits the candidates (see ``speaker_allowlist``);
    an empty list means nobody. There is deliberately no "everyone": an
    unknown voice would land on whichever enrolled centroid is nearest."""
    import numpy as np

    from .speakers import cosine_similarities

    allowed = set(allowed_speakers)

    # Load centroids for ready speakers
    enrolled = {}
    for d, meta in _ready_speaker_metas():
        name = meta.get("display_name", d.name)
        if name not in allowed:
            continue
        enrolled[name] = np.load(d / "centroid.npy")

    if not enrolled:
        return {}

    names = list(enrolled)
    centroids = np.stack([enrolled[name] for name in names])
    name_map = {}
    for spk_label, spk_emb in embeddings.items():
        # Similarity to all enrolled speakers
        sims = cosine_similarities(centroids, np.asarray(spk_emb))
        ranked = sorted(zip(names, sims.tolist()), key=lambda x: x[1], reverse=True)
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


def merge_channels(mic_result: dict, sys_result: dict | None) -> list[dict]:
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
    allowed_speakers: Iterable[str],
    do_diarize: bool = True,
) -> list[dict]:
    import torch
    import whisperx
    # WhisperX imports its ASR module on first use, and through pyannote that
    # pulls in optuna, which keeps the ImportError of an optional backend
    # (redis) with its traceback. Imported from inside _load_whisper, the
    # traceback's frames pinned that function's frame, and the Whisper model
    # in it, for the rest of the process: 3.9 GB that Sortformer then lacked
    # on an 84-minute part (2026-10-27). Imported here, the pinned frames hold
    # no model.
    import whisperx.asr  # noqa: F401

    from .speakers import load_embedding_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Only speaker naming needs it (pyannote's embedding model); Sortformer does not.
    hf_token = get_hf_token() if do_diarize else None
    # Without a token the naming stage used to be skipped silently, leaving every
    # enrolled speaker anonymous in the transcript. Fail before the expensive part.
    if do_diarize and not hf_token and _ready_speaker_metas():
        raise RuntimeError(
            f"Enrolled speakers in {SPEAKERS_DIR} need a HuggingFace token to be named, "
            "and none was found: log in with `huggingface-cli login`, or pass --no-diarize."
        )
    model_name = whisper_model_for(language)
    tracks = {"mic": (mic_part, None)}
    if sys_part:
        if is_silent(sys_part):
            print(f"  Part p{part_index:02d}: system audio is silent, skipping {sys_part.name}")
        else:
            tracks["sys"] = (sys_part, None)
    else:
        print(f"  Part p{part_index:02d}: no matching sys file found")

    # Everything after ASR (alignment, diarization, speaker naming) reads the
    # mic from tracks, so the cancelled track replaces it for all of them.
    aec_path = None
    if "sys" in tracks:
        candidate = WAV_CACHE_DIR / f"{mic_part.stem}.aec.wav"
        report = cancel_echo(mic_part, sys_part, candidate)
        if report is None:
            print(f"  Part p{part_index:02d}: no echo of sys on the mic, transcribing it as recorded")
        else:
            aec_path = candidate
            tracks["mic"] = (aec_path, None)
            print(f"  Part p{part_index:02d}: cancelled sys out of the mic, {report.describe()}", flush=True)

    def _load_whisper(dev: str):
        with _trusted_model_load_context():
            model = whisperx.load_model(
                model_name, dev,
                compute_type="float16" if dev == "cuda" else "int8",
                language=language,
            )
        if dev == "cuda":
            _release_vad_cache(model)
        return model

    def _with_results(results):
        return {channel: (path, results[channel]) for channel, (path, _) in tracks.items()}

    batch_state = {}
    print(f"  Loading ASR model: {model_name} (language: {language})", flush=True)
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
                "diarization:sortformer", _with_results(results), device,
                load_diarizer,
                lambda path, previous, model, dev: diarize(path, previous, dev, model),
                skip_empty=True,
            )

        if do_diarize and hf_token and SPEAKERS_DIR.exists():
            results = _run_model_stage(
                "speaker-naming", _with_results(results), device,
                lambda dev: load_embedding_model(hf_token, dev),
                lambda path, previous, model, dev: apply_enrolled_speaker_names(
                    path, previous, model, dev, allowed_speakers,
                ),
            )

        part_entries = merge_channels(results["mic"], results.get("sys"))
        return _relabel_anonymous_entries(
            part_entries, part_index=part_index, use_part_suffix=use_part_suffix,
        )
    finally:
        _cleanup_pcm_wav(mic_part)
        _cleanup_pcm_wav(sys_part)
        # The cancelled mic is itself in the WAV cache, under its own stem.
        _cleanup_pcm_wav(aec_path)


def ensure_speaker_cache() -> None:
    """Bring ``speakers/`` up to date with the enrollment events before it is read.

    The update runs as ``hugin-meet-enroll --update-cache`` in a subprocess,
    like each part's transcription, so its model leaves the GPU when it is
    done. It only embeds when the events or the embedding model changed; a
    failure stops the transcription rather than name speakers from a cache
    that no longer matches the events.
    """
    from . import speakers

    status, reason = speakers.cache_status(SPEAKERS_DIR, load_config().enrollments_dir)
    if status == "stale":
        print(f"  Speaker cache out of date ({reason}); updating", flush=True)
        subprocess.run(
            [sys.executable, "-m", "hugin_meetings.enroll", "--update-cache"],
            check=True, env=_with_cuda_library_path(),
        )


def process_session(session_id: str, do_diarize: bool = True):
    """Process a single recording session, possibly spanning multiple rotated parts."""
    from . import context as meeting_context

    session = raw_audio_session(session_id)
    if session is None or not session.mic_parts:
        raise RuntimeError(f"No mic recording parts found for session {session_id}")
    # Transcribing the archived parts alone would cache a transcript of half a meeting.
    in_progress = scan_in_progress_parts().get(session_id)
    if in_progress:
        raise RuntimeError(
            f"{session_id} has {len(in_progress)} part(s) still in {in_progress[0].parent} "
            "(recording, or interrupted); not transcribing until they are archived"
        )

    # The context is the contract: it decides the language, and a person has
    # normally confirmed it. Transcribing without one means guessing again.
    context = meeting_context.load_context(session_id)
    if context is None:
        raise RuntimeError(
            f"No context for {session_id} — run hugin-meet-context {session_id} first"
        )
    language = context.language_value

    out_json = transcript_json_path(session_id)
    out_md = transcript_md_path(session_id)

    if out_json.exists():
        print(f"  Already processed: {out_json.name}")
        return

    if do_diarize:
        ensure_speaker_cache()
    allowed_speakers = speaker_allowlist(context)
    print(
        "  Speaker candidates from calendar: "
        + (", ".join(allowed_speakers) if allowed_speakers else "(none)")
    )

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
                "--language",
                language,
                "--allowed-speakers",
                json.dumps(allowed_speakers, ensure_ascii=False),
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
            session_offset += max(audio_duration(mic_part), audio_duration(sys_part))
        finally:
            tmp_path.unlink(missing_ok=True)

    out_json.write_text(json.dumps(session_entries, indent=2, ensure_ascii=False))
    print(f"  Wrote {out_json}")

    out_md.write_text(format_transcript(session_entries, session_id))
    print(f"  Wrote {out_md}")

    meeting_context.record_applied(
        session_id, language=language, whisper_model=whisper_model_for(language),
    )


def find_unprocessed() -> list[str]:
    """Find recording sessions that don't have a corresponding transcript."""
    json_dir = load_config().transcript_json_dir
    processed = {
        p.stem.removeprefix("transcript-")
        for p in json_dir.rglob("transcript-*.json")
    }
    sessions = scan_raw_audio_sessions()
    in_progress = scan_in_progress_parts()
    for session_id in sorted(set(in_progress) - processed):
        print(f"Skipping {session_id}: part(s) still in {in_progress[session_id][0].parent}", file=sys.stderr)
    return [
        session_id
        for session_id in sorted(sessions)
        if session_id not in processed and sessions[session_id].mic_parts and session_id not in in_progress
    ]


def main():
    parser = argparse.ArgumentParser(description="Transcribe mic/sys recording sessions")
    parser.add_argument("file", nargs="?", help="Session id or raw opus file from the session")
    parser.add_argument("--all", action="store_true", help="Process all unprocessed sessions")
    parser.add_argument("--no-diarize", action="store_true", help="Skip diarization")
    args = parser.parse_args()

    if args.all:
        unprocessed = find_unprocessed()
        if not unprocessed:
            print("Nothing to process.")
            return
        print(f"Found {len(unprocessed)} unprocessed recording session(s)")
        for session_id in unprocessed:
            print(f"\nProcessing: {session_id}")
            process_session(session_id, do_diarize=not args.no_diarize)
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
        process_session(session_id, do_diarize=not args.no_diarize)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
