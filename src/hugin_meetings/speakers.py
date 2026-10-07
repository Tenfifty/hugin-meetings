"""Enrolled speaker profiles: events in the archive are the truth, ``speakers/`` is a cache.

Every enrollment, and every correction to one, is a small JSON event written
once into ``cfg.enrollments_dir`` (archive, CONVENTIONS.md "The archive
root") and never edited. An enroll event lists the segments that were
*accepted* for a person, as ``{start, end}`` seconds in the session timeline
that ``session_audio_path`` produces (the parts of one channel concatenated).
Diarization labels are recorded for information only: they change whenever the
diarizer does, the time spans do not.

``speakers/<slug>/`` is folded from those events: each person's segment set,
minus removals, embedded with the current model, centroid = mean. The fold
applies no outlier filtering (that was decided once, at enrollment, and the
accepted spans are explicit), so the cache is a function of the event set.
``cache.json`` records the embedding model and a fingerprint of the event set;
when either changes the cache is stale and is brought up to date, per person
where the model is unchanged. It is safe to wipe.

The on-disk format is what ``transcribe`` reads: ``meta.json``
(``display_name``, ``ready``, ``always_candidate``) and ``centroid.npy``.
``segments.json`` lists the rows of ``embeddings.npy`` so an update only embeds
what is new.

Event types (``schema`` 1):

- ``enroll``: ``name``, ``slug``, ``session``, ``channel``, ``segments``,
  ``embedding_model``, ``diarization_label``.
- ``remove_person``: everything enrolled for ``slug`` before this event, and
  its ``always_candidate`` flag, is dropped. A later enroll starts afresh.
- ``remove_segments``: drops the person's segments from ``session``
  (``channel`` if given, else both) that overlap any of ``segments``; with no
  ``segments``, the whole session.
- ``always_candidate``: ``value`` true or false; the latest one wins.

Events are applied in filename order, which starts with the UTC time of
writing, so the order is fixed by the set itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np

SCHEMA = 1
EMBEDDING_MODEL = "pyannote/speaker-diarization-community-1"
MIN_READY_SEGMENTS = 20       # don't use for matching until we have this many
CACHE_FILE = "cache.json"
EVENT_TYPES = ("enroll", "remove_person", "remove_segments", "always_candidate")
CHANNELS = ("mic", "sys")
SESSION_RE = re.compile(r"^\d{8}-\d{6}$")


class EnrollmentError(RuntimeError):
    """An event, or the audio behind it, is not what the cache needs it to be."""


class LegacySpeakerStore(EnrollmentError):
    """``speakers/`` holds profiles written before enrollment events existed."""


# --- Events ---

def slugify(name: str) -> str:
    """Directory name for a person. The same rule the pre-event store used."""
    return name.strip().lower().replace(" ", "_")


@dataclass(frozen=True, order=True)
class Segment:
    session: str
    channel: str
    start: float
    end: float

    def overlaps(self, start: float, end: float) -> bool:
        return self.start < end and start < self.end


def utc_stamp(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _require(cond: bool, where: str, message: str) -> None:
    if not cond:
        raise EnrollmentError(f"{where}: {message}")


def _check_spans(spans, where: str) -> None:
    _require(isinstance(spans, list) and spans, where, "segments must be a non-empty list")
    for span in spans:
        ok = (
            isinstance(span, dict)
            and isinstance(span.get("start"), (int, float))
            and isinstance(span.get("end"), (int, float))
            and 0 <= span["start"] < span["end"]
        )
        _require(ok, where, f"bad segment {span!r}")


def validate_event(event: dict, where: str = "event") -> None:
    """Raise ``EnrollmentError`` unless ``event`` is a well-formed event."""
    _require(isinstance(event, dict), where, "not a JSON object")
    _require(event.get("schema") == SCHEMA, where, f"unknown schema {event.get('schema')!r}")
    kind = event.get("type")
    _require(kind in EVENT_TYPES, where, f"unknown event type {kind!r}")
    name = event.get("name")
    _require(isinstance(name, str) and name.strip(), where, "missing name")
    _require(event.get("slug") == slugify(name), where, f"slug does not match name {name!r}")
    _require("/" not in name and not name.startswith("."), where, f"unusable name {name!r}")
    if kind in ("enroll", "remove_segments"):
        _require(SESSION_RE.match(str(event.get("session", ""))) is not None, where, "bad session id")
    if kind == "enroll":
        _require(event.get("channel") in CHANNELS, where, f"bad channel {event.get('channel')!r}")
        _check_spans(event.get("segments"), where)
        _require(isinstance(event.get("embedding_model"), str), where, "missing embedding_model")
    elif kind == "remove_segments":
        _require(event.get("channel") in (*CHANNELS, None), where, f"bad channel {event.get('channel')!r}")
        if event.get("segments") is not None:
            _check_spans(event["segments"], where)
    elif kind == "always_candidate":
        _require(isinstance(event.get("value"), bool), where, "value must be true or false")


def write_event(events_dir: Path, event: dict, now: datetime | None = None) -> Path:
    """Write one event into the archive. Never replaces an existing file.

    Written under ``.<name>.tmp`` in the same directory, then hard-linked to
    its final name, which fails atomically if that name exists.
    """
    stamp = utc_stamp(now)
    event = {"schema": SCHEMA, **event, "created": stamp}
    validate_event(event)
    events_dir.mkdir(parents=True, exist_ok=True)
    name = f"{stamp}-{event['slug']}.json"
    target = events_dir / name
    if target.exists():
        raise FileExistsError(f"Enrollment event already exists, not replacing it: {target}")
    tmp = events_dir / f".{name}.tmp"
    tmp.write_text(json.dumps(event, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        os.link(tmp, target)
    finally:
        tmp.unlink()
    return target


def event_files(events_dir: Path) -> list[Path]:
    if not events_dir.exists():
        return []
    return sorted(
        p for p in events_dir.iterdir()
        if p.is_file() and p.suffix == ".json" and not p.name.startswith(".")
    )


def load_events(events_dir: Path) -> list[dict]:
    """Every event, validated, in the order they apply."""
    events = []
    for path in event_files(events_dir):
        try:
            event = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EnrollmentError(f"{path}: not valid JSON ({exc})") from exc
        validate_event(event, str(path))
        events.append(event)
    return events


def events_fingerprint(events_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in event_files(events_dir):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


# --- Fold ---

@dataclass
class Person:
    slug: str
    name: str
    segments: set[Segment] = field(default_factory=set)
    always_candidate: bool = False

    def sorted_segments(self) -> list[Segment]:
        return sorted(self.segments)


def fold(events: Iterable[dict]) -> dict[str, Person]:
    """People with at least one segment, after applying ``events`` in order."""
    people: dict[str, Person] = {}
    always: dict[str, bool] = {}
    for event in events:
        slug, kind = event["slug"], event["type"]
        if kind == "enroll":
            person = people.setdefault(slug, Person(slug, event["name"]))
            person.name = event["name"]
            person.segments.update(
                Segment(event["session"], event["channel"], float(s["start"]), float(s["end"]))
                for s in event["segments"]
            )
        elif kind == "remove_person":
            people.pop(slug, None)
            always.pop(slug, None)
        elif kind == "remove_segments" and slug in people:
            spans = event.get("segments")
            person = people[slug]
            person.segments = {
                seg for seg in person.segments
                if not (
                    seg.session == event["session"]
                    and event.get("channel") in (None, seg.channel)
                    and (spans is None or any(seg.overlaps(s["start"], s["end"]) for s in spans))
                )
            }
        elif kind == "always_candidate":
            always[slug] = event["value"]
    result = {}
    for slug, person in sorted(people.items()):
        if person.segments:
            person.always_candidate = always.get(slug, False)
            result[slug] = person
    return result


def resolve_person(people: dict[str, Person], name: str) -> Person:
    """Find a person by display name or slug, case-insensitively."""
    wanted = slugify(name)
    for person in people.values():
        if person.slug == wanted or slugify(person.name) == wanted:
            return person
    known = ", ".join(p.name for p in people.values()) or "nobody"
    raise EnrollmentError(f"No enrolled speaker '{name}' (enrolled: {known})")


# --- Embedding ---

class Embedder(Protocol):
    model_id: str

    def embed(self, session: str, channel: str, spans: list[tuple[float, float]]) -> np.ndarray:
        """One row per span, in order. Raises rather than skip a span."""
        ...


def load_embedding_model(hf_token: str, device: str):
    """The speaker embedding model inside pyannote's diarization pipeline."""
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(EMBEDDING_MODEL, token=hf_token).to(torch.device(device))
    return pipeline._embedding


def load_waveform(audio_path: Path):
    """16 kHz mono waveform, shape (1, samples)."""
    import torchaudio

    waveform, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def embed_waveform_segments(waveform, segments: list[dict], emb_model, device: str) -> np.ndarray:
    """One embedding per segment that is long enough; shorter ones are skipped."""
    import torch

    sr = 16000
    embeddings = []
    for seg in segments:
        chunk = waveform[:, int(seg["start"] * sr):int(seg["end"] * sr)]
        if chunk.shape[1] < emb_model.min_num_samples:
            continue
        # Model expects (batch, channel, samples)
        with torch.no_grad():
            emb = emb_model(chunk.unsqueeze(0).to(device))
        if isinstance(emb, np.ndarray):
            embeddings.append(emb.squeeze())
        else:
            embeddings.append(emb.squeeze().cpu().numpy())
    return np.array(embeddings) if embeddings else np.empty((0, 256))


def session_audio_path(session_id: str, channel: str, tmp_dir: Path) -> Path | None:
    """One channel of a session as a single file: the part itself, or its parts concatenated."""
    from .pipeline import raw_audio_session

    session = raw_audio_session(session_id)
    if session is None:
        return None

    parts = session.mic_parts if channel == "mic" else session.sys_parts
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    output_path = tmp_dir / f"{channel}-{session_id}.opus"
    list_path = tmp_dir / f"{channel}-{session_id}.txt"
    lines = []
    for path in parts:
        escaped = str(path).replace("'", r"'\''")
        lines.append(f"file '{escaped}'\n")
    list_path.write_text("".join(lines), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return output_path


class PyannoteEmbedder:
    """Embeds spans of archived session audio. Loads the model on first use.

    Remembers what it has embedded, so an enrollment followed by a cache
    update embeds each segment once. Use as a context manager: concatenated
    multi-part audio lives in a temp dir until exit.
    """

    model_id = EMBEDDING_MODEL

    def __init__(self, hf_token: str | None = None, device: str | None = None):
        self._hf_token = hf_token
        self._device = device
        self._model = None
        self._tmp = tempfile.TemporaryDirectory(prefix="hugin-speakers-")
        self._waveform_key: tuple[str, str] | None = None
        self._waveform = None
        self._memo: dict[tuple[str, str, float, float], np.ndarray] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._tmp.cleanup()

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def model(self):
        if self._model is None:
            from .cli_utils import get_hf_token

            token = self._hf_token or get_hf_token()
            if not token:
                raise EnrollmentError("No HuggingFace token found. Run: huggingface-cli login")
            print(f"Loading embedding model {self.model_id} on {self.device}...", flush=True)
            try:
                self._model = load_embedding_model(token, self.device)
            except Exception as exc:
                raise EnrollmentError(f"Could not load {self.model_id}: {exc}") from exc
        return self._model

    def _waveform_for(self, session: str, channel: str):
        if self._waveform_key != (session, channel):
            audio = session_audio_path(session, channel, Path(self._tmp.name))
            if audio is None or not audio.exists():
                raise EnrollmentError(f"No {channel} audio in the archive for session {session}")
            self._waveform = load_waveform(audio)
            self._waveform_key = (session, channel)
        return self._waveform

    def embed(self, session: str, channel: str, spans: list[tuple[float, float]]) -> np.ndarray:
        missing = [s for s in spans if (session, channel, *s) not in self._memo]
        if missing:
            waveform = self._waveform_for(session, channel)
            rows = embed_waveform_segments(
                waveform, [{"start": a, "end": b} for a, b in missing], self.model(), self.device
            )
            if len(rows) != len(missing):
                duration = waveform.shape[1] / 16000
                raise EnrollmentError(
                    f"{session} {channel}: {len(missing) - len(rows)} of {len(missing)} segment(s) "
                    f"too short or past the end of the audio ({duration:.1f} s)"
                )
            for span, row in zip(missing, rows):
                self._memo[(session, channel, *span)] = row
        if not spans:
            return np.empty((0, 256))
        return np.stack([self._memo[(session, channel, *s)] for s in spans])


def cosine_similarities(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Cosine similarity between each embedding and a centroid."""
    norms_e = np.linalg.norm(embeddings, axis=1) + 1e-8
    norm_c = np.linalg.norm(centroid) + 1e-8
    return (embeddings @ centroid) / (norms_e * norm_c)


# --- Cache ---

def is_legacy_store(speakers_dir: Path) -> bool:
    """True if ``speakers_dir`` holds a profile that was not folded from events."""
    if not speakers_dir.exists():
        return False
    return any(
        d.is_dir() and not d.name.startswith(".")
        and (d / "meta.json").exists() and not (d / "segments.json").exists()
        for d in speakers_dir.iterdir()
    )


def cache_status(speakers_dir: Path, events_dir: Path, model_id: str = EMBEDDING_MODEL) -> tuple[str, str]:
    """``("current" | "stale" | "legacy", reason)``. Reads no audio and loads no model."""
    if is_legacy_store(speakers_dir):
        return "legacy", f"{speakers_dir} holds profiles from before enrollment events"
    try:
        stored = json.loads((speakers_dir / CACHE_FILE).read_text())
    except FileNotFoundError:
        return "stale", "no cache yet"
    except json.JSONDecodeError:
        return "stale", f"unreadable {CACHE_FILE}"
    if stored.get("embedding_model") != model_id:
        return "stale", f"embedding model changed ({stored.get('embedding_model')} -> {model_id})"
    if stored.get("events_fingerprint") != events_fingerprint(events_dir):
        return "stale", "enrollment events changed"
    return "current", ""


def _cached_rows(person_dir: Path, model_id: str) -> dict[Segment, np.ndarray]:
    """Embeddings already in the cache for this person, if made with ``model_id``."""
    try:
        meta = json.loads((person_dir / "meta.json").read_text())
        if meta.get("embedding_model") != model_id:
            return {}
        rows = json.loads((person_dir / "segments.json").read_text())
        embeddings = np.load(person_dir / "embeddings.npy")
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {}
    if len(rows) != len(embeddings):
        return {}
    return {Segment(**row): emb for row, emb in zip(rows, embeddings)}


def _write_person(speakers_dir: Path, person: Person, segments: list[Segment],
                  embeddings: np.ndarray, model_id: str) -> None:
    sources: dict[tuple[str, str], int] = {}
    for seg in segments:
        sources[(seg.session, seg.channel)] = sources.get((seg.session, seg.channel), 0) + 1
    meta = {
        "display_name": person.name,
        "total_segments": len(segments),
        "ready": len(segments) >= MIN_READY_SEGMENTS,
        "always_candidate": person.always_candidate,
        "embedding_model": model_id,
        "sources": [
            {"session": s, "channel": c, "segments": n} for (s, c), n in sorted(sources.items())
        ],
    }
    tmp = speakers_dir / f".{person.slug}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.save(tmp / "embeddings.npy", embeddings)
    np.save(tmp / "centroid.npy", embeddings.mean(axis=0))
    (tmp / "segments.json").write_text(
        json.dumps([seg.__dict__ for seg in segments], indent=1) + "\n"
    )
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    final = speakers_dir / person.slug
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)


def _real_audio_check() -> Callable[[str, str], bool]:
    from .pipeline import scan_raw_audio_sessions

    sessions = scan_raw_audio_sessions()

    def has_audio(session: str, channel: str) -> bool:
        found = sessions.get(session)
        return bool(found and (found.mic_parts if channel == "mic" else found.sys_parts))

    return has_audio


def update_cache(
    speakers_dir: Path,
    events_dir: Path,
    embedder: Embedder,
    *,
    force: bool = False,
    has_audio: Callable[[str, str], bool] | None = None,
) -> dict[str, Person]:
    """Bring ``speakers_dir`` in line with the events. Returns the people folded.

    Reuses a person's cached embeddings for segments it already holds, unless
    ``force`` or the model changed. Every segment's session audio must exist
    in the archive, cached or not: a session deleted from the archive needs a
    ``remove_segments`` event, not silence.
    """
    if is_legacy_store(speakers_dir):
        raise LegacySpeakerStore(
            f"{speakers_dir} holds speaker profiles written before enrollment events. "
            "Move it aside (see README, 'Enrolled speakers') before building the cache."
        )
    fingerprint = events_fingerprint(events_dir)
    people = fold(load_events(events_dir))
    has_audio = has_audio or _real_audio_check()
    for person in people.values():
        for session, channel in sorted({(s.session, s.channel) for s in person.segments}):
            if not has_audio(session, channel):
                raise EnrollmentError(
                    f"{person.name}: enrolled from {channel} of session {session}, "
                    "which is not in the archive"
                )

    speakers_dir.mkdir(parents=True, exist_ok=True)
    # Stale until the last person is written, so an interrupted update is redone.
    (speakers_dir / CACHE_FILE).unlink(missing_ok=True)
    for person in people.values():
        segments = person.sorted_segments()
        cached = {} if force else _cached_rows(speakers_dir / person.slug, embedder.model_id)
        missing: dict[tuple[str, str], list[Segment]] = {}
        for seg in segments:
            if seg not in cached:
                missing.setdefault((seg.session, seg.channel), []).append(seg)
        for (session, channel), segs in sorted(missing.items()):
            rows = embedder.embed(session, channel, [(s.start, s.end) for s in segs])
            if len(rows) != len(segs):
                raise EnrollmentError(
                    f"{person.name}: embedder returned {len(rows)} rows for {len(segs)} segments "
                    f"of {channel} {session}"
                )
            cached.update(zip(segs, rows))
        embeddings = np.stack([cached[seg] for seg in segments])
        _write_person(speakers_dir, person, segments, embeddings, embedder.model_id)

    for d in speakers_dir.iterdir():
        if d.is_dir() and (d.name.startswith(".") or d.name not in people):
            shutil.rmtree(d)
    (speakers_dir / CACHE_FILE).write_text(json.dumps({
        "schema": SCHEMA,
        "embedding_model": embedder.model_id,
        "events_fingerprint": fingerprint,
    }, indent=2) + "\n")
    return people


def load_cached_centroid(speakers_dir: Path, slug: str) -> np.ndarray | None:
    path = speakers_dir / slug / "centroid.npy"
    return np.load(path) if path.exists() else None
