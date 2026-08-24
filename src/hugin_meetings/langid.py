#!/usr/bin/env python3
"""Session language identification from raw audio.

Whisper's own detection reads the first 30 seconds of a file, which on a
recording that opens with people connecting is 30 seconds of empty room. That
produced ``nl (0.17)`` for a Swedish meeting on 2026-08-24, and the whole part
then decoded — and aligned — as Dutch.

This module answers the same question from evidence instead: strip silence with
VAD, sample windows from the middle of the *speech*, and vote. Sampling starts
at 25% because a meeting's opening minutes are routinely small talk in the local
language while everyone waits for a guest who will switch the room to English.

One language per session, not per channel: mic and sys carry the same
conversation, so sys windows join the same vote rather than deciding separately.

Only the largest part of each channel is probed. Switching audio device at the
start of a meeting leaves short junk parts behind, and they carry no useful
speech. And only the sampled regions are decoded — running VAD over a whole
51 min recording cost 10.5s of the 18s per session, to place three 30s windows.

Deliberately a small CPU model (``base``, ~142 MB): the language question is far
easier than transcription, and this runs before the GPU stack is loaded.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .config import load_config

if TYPE_CHECKING:
    import numpy as np

# Fractions of the speech-only timeline to sample. Never 0.0 — see module docstring.
SAMPLE_POINTS = (0.25, 0.50, 0.75)
DEFAULT_MODEL = "base"
DEFAULT_COMPUTE_TYPE = "int8"
# Languages we actually hold meetings in. A top hit outside this set means the
# probe is confused, not that the meeting is in Javanese.
DEFAULT_LANGUAGES = ("sv", "en")
# faster-whisper's own early-exit threshold. Probes below it don't get a vote.
DEFAULT_THRESHOLD = 0.5
# Audio decoded around each sample point. Wide enough that a pause at the point
# itself still yields speech, narrow enough to stay cheap.
REGION_SECONDS = 180.0
# A region yielding less speech than this is silence or noise, not a sample.
MIN_PROBE_SPEECH_SECONDS = 10.0
SAMPLE_RATE = 16000
# Probes are independent: decode, VAD and encode all release the GIL, so the six
# of a two-channel session run as threads against one shared model.
DEFAULT_JOBS = len(SAMPLE_POINTS) * 2


def _settings() -> dict[str, Any]:
    """Overridable via ``meetings.langid_*`` in config, like the transcribe knobs."""
    meetings = load_config().raw.get("meetings", {}) or {}
    return {
        "model": meetings.get("langid_model", DEFAULT_MODEL),
        "compute_type": meetings.get("langid_compute_type", DEFAULT_COMPUTE_TYPE),
        "languages": tuple(meetings.get("langid_languages", DEFAULT_LANGUAGES)),
        "threshold": float(meetings.get("langid_threshold", DEFAULT_THRESHOLD)),
        "jobs": int(meetings.get("langid_jobs", DEFAULT_JOBS)),
    }


@dataclass
class LanguageProbe:
    """One 30s window, scored."""

    channel: str
    at_fraction: float
    language: str
    probability: float
    top: list[tuple[str, float]] = field(default_factory=list)

    @property
    def counts(self) -> bool:
        cfg = _settings()
        return self.language in cfg["languages"] and self.probability >= cfg["threshold"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "at": self.at_fraction,
            "language": self.language,
            "probability": round(self.probability, 4),
            "top": [[lang, round(prob, 4)] for lang, prob in self.top],
        }


@dataclass
class LanguageResult:
    """What the session should be transcribed as, and why."""

    value: str
    confidence: float
    source: str  # "langid" | "fallback"
    probes: list[LanguageProbe] = field(default_factory=list)
    note: str = ""

    @property
    def votes(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for probe in self.probes:
            if probe.counts:
                tally[probe.language] = tally.get(probe.language, 0) + 1
        return tally

    @property
    def label(self) -> str:
        counted = sum(self.votes.values())
        if self.source != "langid":
            return f"{self.value} ({self.source}: {self.note})"
        return f"{self.value} ({counted}/{len(self.probes)} windows, {self.confidence:.2f})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "note": self.note,
            "votes": self.votes,
            "probes": [probe.to_dict() for probe in self.probes],
        }


def decide_language(probes: list[LanguageProbe], fallback: str) -> LanguageResult:
    """Majority vote over the probes that clear the allowlist and threshold.

    Ties break on the highest single probability. With nothing to count we fall
    back to the configured language rather than guessing — the failure mode we
    are here to prevent is a confident-sounding wrong answer.
    """
    counted = [probe for probe in probes if probe.counts]
    if not counted:
        best = max(probes, key=lambda probe: probe.probability, default=None)
        note = (
            f"no probe cleared the allowlist/threshold (best: {best.language} {best.probability:.2f})"
            if best
            else "no probes"
        )
        return LanguageResult(fallback, 0.0, "fallback", probes, note)

    tally: dict[str, list[float]] = {}
    for probe in counted:
        tally.setdefault(probe.language, []).append(probe.probability)
    language = max(tally, key=lambda lang: (len(tally[lang]), max(tally[lang])))
    note = "split vote" if len(tally) > 1 else ""
    return LanguageResult(language, max(tally[language]), "langid", probes, note)


@lru_cache(maxsize=1)
def _load_model():
    import os

    from faster_whisper import WhisperModel

    cfg = _settings()
    jobs = max(1, cfg["jobs"])
    # num_workers gives ctranslate2 one replica per calling thread; cpu_threads
    # is then split between them so the replicas do not fight over the cores.
    cores = os.cpu_count() or 1
    return WhisperModel(
        cfg["model"],
        device="cpu",
        compute_type=cfg["compute_type"],
        num_workers=jobs,
        cpu_threads=max(1, cores // jobs),
    )


def _speech_only(audio: "np.ndarray") -> "np.ndarray":
    """Drop silence so a window is 30s of speech, not 30s of empty room."""
    import numpy as np
    from faster_whisper.vad import collect_chunks, get_speech_timestamps

    timestamps = get_speech_timestamps(audio, None)
    if not timestamps:
        return np.empty(0, dtype=audio.dtype)
    chunks, _ = collect_chunks(audio, timestamps)
    return np.concatenate(chunks, axis=0)


def _decode_region(path: Path, start: float, seconds: float) -> "np.ndarray":
    """Decode one slice of an audio file to 16 kHz mono float32."""
    import subprocess

    import numpy as np

    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-threads", "0",
            "-ss", f"{max(0.0, start):.3f}",
            "-t", f"{seconds:.3f}",
            "-i", str(path),
            "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


def largest_part(paths: list[Path]) -> Path | None:
    """The part worth listening to.

    Changing audio device mid-setup rotates the recording, leaving a few
    seconds of junk in its own part. The meeting is in the long one.
    """
    from .pipeline import audio_duration

    if not paths:
        return None
    return max(paths, key=audio_duration)


def _probe_region(path: Path, channel: str, fraction: float, duration: float) -> LanguageProbe | None:
    """Decode, de-silence and score one sample point. None if it holds no speech."""
    model = _load_model()
    # Keep the region inside the recording, so a short part still samples audio
    # rather than trailing silence.
    start = min(duration * fraction, max(0.0, duration - REGION_SECONDS))
    speech = _speech_only(_decode_region(path, start, REGION_SECONDS))
    if speech.size < MIN_PROBE_SPEECH_SECONDS * SAMPLE_RATE:
        return None
    language, probability, all_probs = model.detect_language(
        speech[: model.feature_extractor.n_samples],
        vad_filter=False,
        language_detection_segments=1,
    )
    top = sorted(all_probs, key=lambda item: -item[1])[:3]
    return LanguageProbe(channel, fraction, language, probability, top)


def _run_probes(tasks: list[tuple[Path, str, float, float]]) -> list[LanguageProbe]:
    """Run sample points concurrently, returning them in a stable order."""
    from concurrent.futures import ThreadPoolExecutor

    if not tasks:
        return []
    _load_model()  # load once here, not racing inside the workers
    workers = max(1, min(_settings()["jobs"], len(tasks)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        probes = list(pool.map(lambda task: _probe_region(*task), tasks))
    return sorted(
        (probe for probe in probes if probe is not None),
        key=lambda probe: (probe.channel != "mic", probe.at_fraction),
    )


def channel_tasks(paths: list[Path], channel: str) -> list[tuple[Path, str, float, float]]:
    """One task per sample point of the channel's largest part."""
    from .pipeline import audio_duration

    path = largest_part(paths)
    if path is None:
        return []
    duration = audio_duration(path)
    if duration <= 0:
        return []
    return [(path, channel, fraction, duration) for fraction in SAMPLE_POINTS]


def probe_channel(paths: list[Path], channel: str) -> list[LanguageProbe]:
    """Score SAMPLE_POINTS of one channel's largest part."""
    return _run_probes(channel_tasks(paths, channel))


def identify_session(mic_parts: list[Path], sys_parts: list[Path]) -> LanguageResult:
    """Identify the language of one recording session.

    Both channels feed a single vote: they carry the same conversation, so sys
    is extra evidence for one decision, never a second decision.
    """
    tasks = channel_tasks(mic_parts, "mic") + channel_tasks(sys_parts, "sys")
    return decide_language(_run_probes(tasks), load_config().language)


def main() -> int:
    from . import pipeline

    parser = argparse.ArgumentParser(description="Identify the spoken language of recording sessions")
    parser.add_argument("session", nargs="*", help="Session ids (default: the latest -n)")
    parser.add_argument("-n", type=int, default=1, help="Number of latest sessions to probe")
    parser.add_argument("--json", action="store_true", help="Emit the full probe detail as JSON")
    args = parser.parse_args()

    sessions = pipeline.scan_raw_audio_sessions()
    wanted = args.session or sorted(sessions, reverse=True)[: args.n]

    results = {}
    for session_id in wanted:
        session = sessions.get(session_id)
        if session is None:
            print(f"{session_id}: no raw audio", file=sys.stderr)
            continue
        result = identify_session(session.mic_parts, session.sys_parts)
        results[session_id] = result
        if not args.json:
            print(f"{session_id}  {result.label}")

    if args.json:
        import json

        print(json.dumps({sid: res.to_dict() for sid, res in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
