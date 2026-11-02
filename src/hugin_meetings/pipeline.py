#!/usr/bin/env python3
"""Helpers for Hugin's meeting audio pipeline and metadata."""

from __future__ import annotations

import json
import os
import re
import subprocess
from urllib.parse import unquote
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .config import load_config

# Output directories (written to the user's vault/knowledge base)
TRANSCRIPT_DIR = load_config().transcripts_dir
SUMMARY_DIR = load_config().summaries_dir

# Finished raw audio: the write-once archive. Everything reads audio from here.
RAW_AUDIO_DIR = load_config().raw_audio_dir
# Parts still being recorded, or left by a recorder that died. Only shown, never read.
RECORDING_DIR = load_config().recording_dir

# State / cache directories
WAV_CACHE_DIR = load_config().wav_cache_dir
TRANSCRIPT_JSON_DIR = load_config().transcript_json_dir

# Project/customer matching. Projects dir may be None if not configured.
# Internal naming retains "CUSTOMERS_" for on-disk backward compatibility;
# the user-facing concept is "project" (see config.project_matcher).
CUSTOMERS_DIR = load_config().project_matcher.projects_dir

CALENDAR_METADATA_START = "<!-- calendar-metadata:start -->"
CALENDAR_METADATA_END = "<!-- calendar-metadata:end -->"
TS_RE = re.compile(r"(\d{8}-\d{6})")
RAW_AUDIO_RE = re.compile(
    r"^(?P<prefix>mic|sys)-(?P<ts>\d{8}-\d{6})(?:-p(?P<part>\d{2}))?\.opus$"
)
SPEAKER_RE = re.compile(r"^(?:speaker|SPEAKER)_(\d+)(?:_p\d{2})?$")

BACKCHANNEL_WORDS = {
    "mm", "mhm", "mmm", "ja", "jo", "yes", "yeah", "ok", "okej", "okay",
    "aha", "haha", "hm", "hmm", "nej", "nä", "no", "jaha", "japp",
}


def is_backchannel(text: str) -> bool:
    """True if ``text`` is non-empty and consists only of backchannel words."""
    words = text.lower().strip().split()
    return bool(words) and all(w.strip(".,!?") in BACKCHANNEL_WORDS for w in words)


def _frontmatter_line_count(lines: list[str]) -> int:
    """Return lines occupied by a leading YAML frontmatter block, or 0 if none."""
    if not lines or lines[0].strip() != "---":
        return 0
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return i + 1
    return 0  # unclosed — treat as absent


def read_project_frontmatter(path: Path) -> dict[str, str]:
    """Parse simple key: value YAML frontmatter from a project markdown file."""
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    fm_end = _frontmatter_line_count(lines)
    if fm_end == 0:
        return {}
    result: dict[str, str] = {}
    for line in lines[1 : fm_end - 1]:
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip().strip('"').strip("'")
    return result


FIELD_RE = re.compile(r"^- ([^:]+):\s*(.*)$")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
SUMMARY_HEADER_RE = re.compile(
    rf"^({re.escape(load_config().summary_header)})(.*)$", re.MULTILINE
)
PERSONAL_SECTION_HEADER = load_config().personal_section_header
PERSONAL_SECTION_RE = (
    re.compile(
        rf"^{re.escape(PERSONAL_SECTION_HEADER)}\s*$.*?(?=^### |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    if PERSONAL_SECTION_HEADER
    else None
)
TRANSCRIPT_LINK_RE = re.compile(r"^\[Transcript\]\([^)]+\)\s*$", re.MULTILINE)
CUSTOMER_ENTRY_HEADER_RE = re.compile(r"^## <\d{4}-\d{2}-\d{2} [^>]+>\s*$", re.MULTILINE)

CustomerAction = Literal["link_existing", "suggest_new", "no_match"]
Confidence = Literal["high", "medium", "low"]

DEFAULT_CUSTOMER_MODEL = load_config().project_matcher.model
INACTIVE_DIR_NAMES = set(load_config().project_matcher.inactive_dir_names)
MAX_CANDIDATE_PREVIEW = 700


@dataclass
class CustomerNote:
    name: str
    path: Path
    is_active: bool
    preview: str


@dataclass
class CustomerState:
    action: CustomerAction
    confidence: Confidence
    rationale: str
    model: str
    verified: bool = False
    customer_name: str | None = None
    customer_path: Path | None = None
    suggested_name: str | None = None
    source: str = "auto"

    @property
    def label(self) -> str:
        if self.action == "link_existing":
            base = self.customer_name or "-"
        elif self.action == "suggest_new":
            base = self.suggested_name or "(new)"
        else:
            base = "(none)"
        return base if self.verified else f"??{base}??"


@dataclass
class MeetingStatus:
    timestamp: str
    mic_parts: tuple[Path, ...]
    sys_parts: tuple[Path, ...]
    transcript_json: Path | None
    transcript_md: Path | None
    summary_md: Path | None
    calendar_fields: dict[str, str]
    anonymous_speakers: list[str]
    context: Any | None = None
    # Parts of this session still in the recording dir, and whether they are
    # being written ("recording") or were left behind ("interrupted").
    in_progress_parts: tuple[Path, ...] = ()
    recording_state: str = ""

    @property
    def has_transcript(self) -> bool:
        return self.transcript_json is not None and self.transcript_md is not None

    @property
    def has_calendar_metadata(self) -> bool:
        return bool(self.calendar_fields)

    @property
    def has_summary(self) -> bool:
        return self.summary_md is not None and self.summary_md.exists() and bool(
            self.summary_md.read_text().strip()
        )

    @property
    def customer_state(self) -> CustomerState | None:
        """The customer as the context holds it: a guess, or a person's decision.

        The context is the only place a customer is stored. Until 2026-11-02 a
        decision was also written to ``transcript-{ts}.customer.json``, and two
        sources had to be reconciled on every read; sessions verified before
        the context stage existed were given a minimal verified context then.
        """
        if self.context is None or not self.context.customer:
            return None
        return deserialize_customer_state(self.context.customer)

    @property
    def is_verified(self) -> bool:
        """A person has confirmed what this meeting is."""
        return self.context is not None and self.context.verified

    @property
    def needs_verification(self) -> bool:
        return not self.is_verified

    @property
    def needs_link(self) -> bool:
        """Summarized and bound to a customer, but not published into the note yet."""
        if not self.has_summary:
            return False
        state = self.customer_state
        if state is None or not state.verified or state.action != "link_existing":
            return False
        return summary_customer_link(self.summary_md) is None

    @property
    def pipeline_steps(self) -> list[bool]:
        """The stages in order: verified context, transcript, calendar, summary, link."""
        return [
            self.is_verified,
            self.has_transcript,
            self.has_calendar_metadata,
            self.has_summary,
            self.has_summary and not self.needs_link,
        ]

    @property
    def pipeline_steps_complete(self) -> int:
        return sum(self.pipeline_steps)

    @property
    def pipeline_total_steps(self) -> int:
        return len(self.pipeline_steps)

    @property
    def needs_pipeline(self) -> bool:
        return not all(self.pipeline_steps)

    @property
    def ready_for_pipeline(self) -> bool:
        """Nothing expensive runs on a session a person has not confirmed,
        or on one whose parts are not all in the archive yet."""
        return self.is_verified and self.needs_pipeline and not self.in_progress_parts

    @property
    def raw_part_count(self) -> int:
        return max(len(self.mic_parts), len(self.sys_parts))

    @property
    def title(self) -> str:
        event = self.calendar_fields.get("Event")
        if event and event != "-":
            return event
        # Before transcription there is no metadata block yet, but the context
        # already knows which meeting this was — same source, earlier.
        if self.context is not None and self.context.event_title:
            return self.context.event_title
        if self.summary_md and self.summary_md.exists():
            excerpt = summary_excerpt(self.summary_md)
            if excerpt:
                return excerpt[:90]
        return self.timestamp

    @property
    def short_status(self) -> str:
        return "".join(
            letter if done else "."
            for letter, done in zip("VTCSL", self.pipeline_steps)
        )


def extract_timestamp(name: str) -> str | None:
    match = TS_RE.search(name)
    return match.group(1) if match else None


def year_subdir(name_or_ts: str) -> str:
    """Return the YYYY subdir for a session timestamp or any name containing one.

    Per-session files (transcripts, summaries, audio, json) live under
    ``<dir>/YYYY/`` where YYYY is the year embedded in the timestamp.
    Single source of truth so write and migration code agree.
    """
    ts = extract_timestamp(name_or_ts) or name_or_ts
    if len(ts) < 4 or not ts[:4].isdigit():
        raise ValueError(f"No year in {name_or_ts!r}")
    return ts[:4]


def relative_link(target: Path, base_dir: Path) -> str:
    """A relative markdown link target. Spaces are encoded so Obsidian sees a link.

    Obsidian ignores `[x](a b.md)`, so it neither renders nor updates such a link
    when the target moves. Read link targets back with `unquote`.
    """
    return os.path.relpath(target, base_dir).replace(" ", "%20")


def audio_duration(path: Path | None) -> float:
    """Wall-clock seconds of an audio file, via ffprobe. 0.0 for a missing path."""
    if path is None:
        return 0.0
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip() or "0")


def transcript_json_path(ts: str) -> Path:
    return TRANSCRIPT_JSON_DIR / year_subdir(ts) / f"transcript-{ts}.json"


def transcript_md_path(ts: str) -> Path:
    return TRANSCRIPT_DIR / year_subdir(ts) / f"transcript-{ts}.md"


def summary_md_path(ts: str) -> Path:
    return SUMMARY_DIR / year_subdir(ts) / f"summary-{ts}.md"


def context_path(ts: str) -> Path:
    return TRANSCRIPT_JSON_DIR / year_subdir(ts) / f"context-{ts}.json"


def transcript_json_for_markdown(path: Path) -> Path | None:
    ts = extract_timestamp(path.name)
    if not ts:
        return None
    return transcript_json_path(ts)


@dataclass(frozen=True)
class RawAudioPart:
    prefix: str
    session_id: str
    part: int
    path: Path


@dataclass
class RawAudioSession:
    session_id: str
    mic_parts: list[Path]
    sys_parts: list[Path]


def parse_raw_audio_part(path: Path | str) -> RawAudioPart | None:
    raw_path = Path(path)
    match = RAW_AUDIO_RE.match(raw_path.name)
    if not match:
        return None

    return RawAudioPart(
        prefix=match.group("prefix"),
        session_id=match.group("ts"),
        part=int(match.group("part") or "1"),
        path=raw_path,
    )


def scan_raw_audio_sessions() -> dict[str, RawAudioSession]:
    sessions: dict[str, RawAudioSession] = {}

    for path in sorted(RAW_AUDIO_DIR.rglob("*.opus")):
        part = parse_raw_audio_part(path)
        if part is None:
            continue

        session = sessions.setdefault(
            part.session_id,
            RawAudioSession(session_id=part.session_id, mic_parts=[], sys_parts=[]),
        )
        target = session.mic_parts if part.prefix == "mic" else session.sys_parts
        target.append(part.path)

    for session in sessions.values():
        session.mic_parts.sort(key=_raw_part_sort_key)
        session.sys_parts.sort(key=_raw_part_sort_key)

    return sessions


def raw_audio_session(session_id: str) -> RawAudioSession | None:
    return scan_raw_audio_sessions().get(session_id)


def scan_in_progress_parts() -> dict[str, list[Path]]:
    """Parts in the recording dir by session. Not finished, so not for reading."""
    sessions: dict[str, list[Path]] = {}
    if not RECORDING_DIR.exists():
        return sessions
    for path in sorted(RECORDING_DIR.glob("*.opus")):
        part = parse_raw_audio_part(path)
        if part is not None:
            sessions.setdefault(part.session_id, []).append(path)
    return sessions


def recording_state(session_id: str, parts: list[Path]) -> str:
    """'recording' while a recorder holds the session, else 'interrupted'.

    A recorder that stops normally moves every part into the archive, so a
    part left here by a session nobody holds was cut off: a crash, a power
    loss, or a failed move. It is left for a person to archive or delete.
    """
    from .recording import session_is_recording

    if not parts:
        return ""
    return "recording" if session_is_recording(RECORDING_DIR, session_id) else "interrupted"


def _raw_part_sort_key(path: Path) -> int:
    part = parse_raw_audio_part(path)
    return part.part if part else 0


def meeting_artifact_paths(rec: MeetingStatus) -> list[Path]:
    """Return timestamp-scoped files that make up a meeting entry."""
    paths: list[Path] = []

    def add(path: Path | None) -> None:
        if path is not None and path not in paths:
            paths.append(path)

    # The WAV glob also catches the echo-cancelled *.aec.wav beside each part.
    for prefix in ("mic", "sys"):
        for path in sorted(RAW_AUDIO_DIR.rglob(f"{prefix}-{rec.timestamp}*.opus")):
            add(path)
        for path in sorted(WAV_CACHE_DIR.rglob(f"{prefix}-{rec.timestamp}*.wav")):
            add(path)

    add(transcript_json_path(rec.timestamp))
    add(context_path(rec.timestamp))
    add(transcript_md_path(rec.timestamp))
    add(summary_md_path(rec.timestamp))

    return [path for path in paths if path.exists()]


def delete_meeting_entry(rec: MeetingStatus) -> list[Path]:
    """Delete a meeting entry without touching project/customer notes."""
    deleted: list[Path] = []
    for path in meeting_artifact_paths(rec):
        if path.is_dir() and not path.is_symlink():
            raise RuntimeError(f"Refusing to delete directory: {path}")
        path.unlink(missing_ok=True)
        deleted.append(path)
    return deleted


def extract_metadata_block(text: str, start: str, end: str) -> str | None:
    pattern = re.compile(
        rf"{re.escape(start)}\n?(.*?)\n?{re.escape(end)}",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


def metadata_block_re(start: str, end: str) -> re.Pattern[str]:
    """A whole marked block with the blank lines after it."""
    return re.compile(rf"{re.escape(start)}.*?{re.escape(end)}\n*", re.DOTALL)


def remove_metadata_block(text: str, start: str, end: str) -> str:
    updated = metadata_block_re(start, end).sub("", text)
    return re.sub(r"\n{3,}", "\n\n", updated).strip() + "\n"


def parse_key_value_block(block: str | None) -> dict[str, str]:
    if not block:
        return {}
    fields: dict[str, str] = {}
    for line in block.splitlines():
        match = FIELD_RE.match(line.strip())
        if match:
            fields[match.group(1).strip()] = match.group(2).strip()
    return fields


def parse_markdown_link(value: str) -> tuple[str, Path] | None:
    match = LINK_RE.search(value)
    if not match:
        return None
    return match.group(1), Path(unquote(match.group(2)))


def calendar_metadata_fields(text: str) -> dict[str, str]:
    """The calendar block in ``text`` as key/value pairs, {} without one."""
    return parse_key_value_block(
        extract_metadata_block(text, CALENDAR_METADATA_START, CALENDAR_METADATA_END)
    )


def parse_calendar_metadata(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    return calendar_metadata_fields(path.read_text())


def summary_customer_link(path: Path | None) -> tuple[str, Path] | None:
    """The customer a summary's header links to, as (name, note path)."""
    if not path or not path.exists():
        return None
    header_match = SUMMARY_HEADER_RE.search(path.read_text())
    link = parse_markdown_link(header_match.group(2)) if header_match else None
    if link is None:
        return None
    name, rel_path = link
    return name, current_customer_path(name, (path.parent / rel_path).resolve())


def serialize_customer_state(state: CustomerState) -> dict[str, str | bool | None]:
    return {
        "action": state.action,
        "confidence": state.confidence,
        "rationale": state.rationale,
        "model": state.model,
        "verified": state.verified,
        "customer_name": state.customer_name,
        "customer_path": str(state.customer_path) if state.customer_path else None,
        "suggested_name": state.suggested_name,
        "source": state.source,
    }


def deserialize_customer_state(payload: dict) -> CustomerState:
    confidence = str(payload.get("confidence", "low")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    action = str(payload.get("action", "no_match"))
    if action not in {"link_existing", "suggest_new", "no_match"}:
        action = "no_match"

    customer_path_value = payload.get("customer_path")
    customer_path = current_customer_path(
        payload.get("customer_name") or None,
        Path(customer_path_value).resolve() if customer_path_value else None,
    )

    return CustomerState(
        action=action,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
        rationale=str(payload.get("rationale", "")),
        model=str(payload.get("model", DEFAULT_CUSTOMER_MODEL)),
        verified=bool(payload.get("verified", False)),
        customer_name=payload.get("customer_name") or None,
        customer_path=customer_path,
        suggested_name=payload.get("suggested_name") or None,
        source=str(payload.get("source", "auto")),
    )


def summary_excerpt(path: Path, max_len: int = 120) -> str:
    text = remove_metadata_block(path.read_text(), CALENDAR_METADATA_START, CALENDAR_METADATA_END)
    text = _strip_customer_header_link(text)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:max_len]
    return ""


def _customer_note_paths() -> list[tuple[Path, bool]]:
    """Project notes: files directly in CUSTOMERS_DIR or in one of its inactive dirs.

    Any other subdirectory holds material for a project (`AKA/`, `Notana Care/`),
    not projects of its own, so it is not a match candidate. Returns each path
    with whether it is active.
    """
    if not CUSTOMERS_DIR or not CUSTOMERS_DIR.is_dir():
        return []
    paths = list(CUSTOMERS_DIR.glob("*.md"))
    for name in INACTIVE_DIR_NAMES:
        paths.extend((CUSTOMERS_DIR / name).glob("*.md"))
    return [(path, path.parent == CUSTOMERS_DIR) for path in sorted(paths)]


def list_customer_notes() -> list[CustomerNote]:
    """The project notes with a preview of each, for the matcher prompt."""
    notes: list[CustomerNote] = []
    for path, is_active in _customer_note_paths():
        name = path.stem
        text = path.read_text()
        preview = re.sub(r"\s+", " ", text).strip()[:MAX_CANDIDATE_PREVIEW]
        notes.append(
            CustomerNote(
                name=name,
                path=path,
                is_active=is_active,
                preview=preview,
            )
        )
    return notes


def load_anonymous_speakers(path: Path | None) -> list[str]:
    if not path or not path.exists():
        return []
    try:
        entries = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    if not isinstance(entries, list):
        return []

    speakers = {
        f"{entry.get('channel', '?')}:{entry.get('speaker', '?')}"
        for entry in entries
        if isinstance(entry, dict) and SPEAKER_RE.match(str(entry.get("speaker", "")))
    }
    return sorted(speakers)


def scan_recordings() -> list[MeetingStatus]:
    # Imported here: context builds on pipeline, so the dependency only runs
    # one way at import time.
    from . import context as meeting_context

    raw_sessions = scan_raw_audio_sessions()
    in_progress = scan_in_progress_parts()
    timestamps: set[str] = set(raw_sessions) | set(in_progress)

    for paths in (
        TRANSCRIPT_DIR.rglob("transcript-*.md"),
        TRANSCRIPT_JSON_DIR.rglob("transcript-*.json"),
        SUMMARY_DIR.rglob("summary-*.md"),
    ):
        for path in paths:
            ts = extract_timestamp(path.name)
            if ts:
                timestamps.add(ts)

    meetings: list[MeetingStatus] = []
    for ts in sorted(timestamps, reverse=True):
        raw_session = raw_sessions.get(ts)
        mic_parts = tuple(raw_session.mic_parts) if raw_session else ()
        sys_parts = tuple(raw_session.sys_parts) if raw_session else ()
        transcript_md = transcript_md_path(ts)
        transcript_json = transcript_json_path(ts)
        summary_md = summary_md_path(ts)

        meetings.append(
            MeetingStatus(
                timestamp=ts,
                mic_parts=mic_parts,
                sys_parts=sys_parts,
                transcript_json=transcript_json if transcript_json.exists() else None,
                transcript_md=transcript_md if transcript_md.exists() else None,
                summary_md=summary_md if summary_md.exists() else None,
                calendar_fields=parse_calendar_metadata(transcript_md if transcript_md.exists() else None),
                anonymous_speakers=load_anonymous_speakers(transcript_json if transcript_json.exists() else None),
                context=meeting_context.load_context(ts),
                in_progress_parts=tuple(in_progress.get(ts, ())),
                recording_state=recording_state(ts, in_progress.get(ts, [])),
            )
        )
    return meetings


def resolve_customer_path(name: str | None, candidates: list[CustomerNote]) -> Path | None:
    if not name:
        return None

    for note in candidates:
        if note.name.lower() == name.lower():
            return note.path

    # By filename only: reading every note for its preview made each lookup
    # cost tens of milliseconds, and stale stored paths look up on every load.
    for path, _ in _customer_note_paths():
        if path.stem.lower() == name.lower():
            return path

    return None


def _folder_names() -> set[str]:
    """Casefolded names of every directory under CUSTOMERS_DIR: material folders and inactive dirs."""
    if not CUSTOMERS_DIR or not CUSTOMERS_DIR.is_dir():
        return set()
    return {path.name.casefold() for path in CUSTOMERS_DIR.rglob("*") if path.is_dir()}


def _is_folder_name(name: str | None) -> bool:
    """True when `name` names a directory under CUSTOMERS_DIR and no project note of that name exists.

    A material folder shares its name with its note (`AKA/` next to `AKA.md`), so
    naming it still resolves to the note. Anything else that names a folder is not
    a project and must never be linked to or created.
    """
    return bool(name) and name.strip().casefold() in _folder_names() and resolve_customer_path(name, []) is None


def current_customer_path(name: str | None, path: Path | None) -> Path | None:
    """The stored path if it still exists, otherwise the project note of the same name.

    Paths are stored absolute, so a note moved to another folder (or the projects
    dir moved as a whole) leaves stale paths in old meeting state. Looking the
    note up by name again keeps those meetings linked. When no note of that name
    exists, the stale path is returned unchanged so the failure stays visible.
    """
    if path is None or path.exists() or not name:
        return path
    return resolve_customer_path(name, []) or path


def _customer_note_filename(name: str) -> str:
    cleaned = re.sub(r"[\\/]+", "-", name).strip().strip(".")
    if not cleaned:
        raise ValueError("Customer name must not be empty.")
    return f"{cleaned}.md"


def ensure_customer_note(customer_name: str) -> Path:
    existing = resolve_customer_path(customer_name, [])
    if existing:
        return existing

    if not CUSTOMERS_DIR:
        raise ValueError("No projects_dir is configured, so there is nowhere to create a customer note.")
    CUSTOMERS_DIR.mkdir(parents=True, exist_ok=True)
    path = CUSTOMERS_DIR / _customer_note_filename(customer_name)
    # A folder name that did not resolve above is a material or inactive dir,
    # never a project; creating a note for it would give one project two names.
    if _is_folder_name(customer_name):
        raise ValueError(
            f"{customer_name!r} is a folder in {CUSTOMERS_DIR}, not a project note"
        )
    if not path.exists():
        path.write_text(f"# {customer_name}\n", encoding="utf-8")
    return path


def materialize_verified_customer_state(state: CustomerState) -> CustomerState:
    if not state.verified:
        return state

    if state.action == "link_existing":
        if not state.customer_name:
            return state
        customer_path = state.customer_path or ensure_customer_note(state.customer_name)
        state.customer_path = customer_path
        return state

    if state.action == "suggest_new" and state.suggested_name:
        customer_path = ensure_customer_note(state.suggested_name)
        state.action = "link_existing"
        state.customer_name = state.suggested_name
        state.customer_path = customer_path
        state.suggested_name = None
        return state

    return state


def _strip_customer_header_link(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        base = match.group(1)
        rest = match.group(2)
        cleaned = LINK_RE.sub("", rest)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).rstrip()
        return f"{base}{cleaned}"

    return SUMMARY_HEADER_RE.sub(repl, text, count=1)


def _strip_transcript_link(text: str) -> str:
    lines = [line for line in text.splitlines() if not TRANSCRIPT_LINK_RE.match(line.strip())]
    cleaned = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"


def _summary_transcript_path(summary_path: Path) -> Path:
    ts = extract_timestamp(summary_path.name)
    if not ts:
        raise ValueError(f"Could not determine timestamp from {summary_path.name}")
    return transcript_md_path(ts)


def ensure_summary_transcript_link(summary_path: Path) -> None:
    transcript_path = _summary_transcript_path(summary_path)
    transcript_rel = relative_link(transcript_path, summary_path.parent)
    text = _strip_transcript_link(summary_path.read_text())
    summary_path.write_text(text.rstrip() + f"\n\n[Transcript]({transcript_rel})\n")


def _weekday_label(dt: datetime) -> str:
    return ["mån", "tis", "ons", "tor", "fre", "lör", "sön"][dt.weekday()]


def format_log_timestamp(ts: str) -> str:
    dt = datetime.strptime(ts, "%Y%m%d-%H%M%S")
    return f"<{dt:%Y-%m-%d} {_weekday_label(dt)} {dt:%H:%M}>"


def summary_body(summary_path: Path) -> str:
    """The summary text without its header line and its customer and transcript links."""
    text = _strip_transcript_link(_strip_customer_header_link(summary_path.read_text())).strip()
    lines = text.splitlines()
    if lines and lines[0].startswith(load_config().summary_header):
        lines = lines[1:]
    return "\n".join(lines).strip()


def summary_lead(body: str) -> str:
    """Text before the first ### subsection."""
    lines: list[str] = []
    for line in body.splitlines():
        if line.strip().startswith("### "):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def personal_section(body: str) -> str:
    """The configured personal follow-up section, header included, or ""."""
    match = PERSONAL_SECTION_RE.search(body) if PERSONAL_SECTION_RE else None
    return match.group(0).strip() if match else ""


def without_personal_section(body: str) -> str:
    if PERSONAL_SECTION_RE is None:
        return body
    return re.sub(r"\n{3,}", "\n\n", PERSONAL_SECTION_RE.sub("", body)).strip()


def _summary_body_for_customer_note(summary_path: Path) -> str:
    body = summary_body(summary_path)
    if not body:
        return ""
    parts = [part for part in (summary_lead(body), personal_section(body)) if part]
    return "\n\n".join(parts).strip() + ("\n" if parts else "")


def _customer_note_links(customer_path: Path, summary_path: Path) -> tuple[str, str]:
    transcript_path = _summary_transcript_path(summary_path)
    return (
        relative_link(summary_path, customer_path.parent),
        relative_link(transcript_path, customer_path.parent),
    )


FULL_SUMMARY_LINK_RE = re.compile(r"\[Full summary\]\(([^)]+)\)")


def _find_customer_note_entry_bounds(text: str, summary_name: str) -> tuple[int, int] | None:
    """The entry in a project note whose "Full summary" link points at `summary_name`.

    Matched on the file name, which is unique per meeting, rather than on the
    whole relative path: the path changes whenever the vault is reorganised or
    Obsidian rewrites the link, and a missed match would duplicate the entry.
    """
    idx = next(
        (
            match.start()
            for match in FULL_SUMMARY_LINK_RE.finditer(text)
            if Path(unquote(match.group(1))).name == summary_name
        ),
        -1,
    )
    if idx < 0:
        return None

    header_matches = list(CUSTOMER_ENTRY_HEADER_RE.finditer(text))
    start = 0
    for match in header_matches:
        if match.start() <= idx:
            start = match.start()
        else:
            break

    end = len(text)
    for match in header_matches:
        if match.start() > idx:
            end = match.start()
            break

    return start, end


def sync_customer_note(customer_path: Path, summary_path: Path) -> None:
    customer_text = customer_path.read_text()
    summary_rel, transcript_rel = _customer_note_links(customer_path, summary_path)
    body = _summary_body_for_customer_note(summary_path).rstrip()
    ts = extract_timestamp(summary_path.name)
    if not ts:
        raise ValueError(f"Could not determine timestamp from {summary_path.name}")

    entry_lines = [
        f"## {format_log_timestamp(ts)}",
        "",
        body,
        "",
        f"[Full summary]({summary_rel})",
        f"[Transcript]({transcript_rel})",
        "",
    ]
    entry = "\n".join(line for line in entry_lines if line is not None)
    entry = re.sub(r"\n{3,}", "\n\n", entry).strip() + "\n\n"

    bounds = _find_customer_note_entry_bounds(customer_text, summary_path.name)
    if bounds is not None:
        start, end = bounds
        updated = customer_text[:start].rstrip() + "\n\n" + entry + customer_text[end:].lstrip()
    else:
        lines = customer_text.splitlines()
        fm_end = _frontmatter_line_count(lines)
        insert_at = fm_end
        while insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1
        if insert_at < len(lines) and lines[insert_at].startswith("# "):
            insert_at += 1
            while insert_at < len(lines) and not lines[insert_at].strip():
                insert_at += 1
        prefix = "\n".join(lines[:insert_at]).rstrip()
        suffix = "\n".join(lines[insert_at:]).lstrip()
        updated = f"{prefix}\n\n{entry}{suffix}"

    customer_path.write_text(re.sub(r"\n{3,}", "\n\n", updated).rstrip() + "\n")


def remove_customer_note_link(customer_path: Path, summary_path: Path) -> None:
    if not customer_path.exists():
        return
    text = customer_path.read_text()
    bounds = _find_customer_note_entry_bounds(text, summary_path.name)
    if bounds is None:
        return
    start, end = bounds
    updated = text[:start].rstrip() + "\n\n" + text[end:].lstrip()
    customer_path.write_text(re.sub(r"\n{3,}", "\n\n", updated).rstrip() + "\n")


def write_customer_metadata(summary_path: Path, customer_name: str, customer_path: Path) -> None:
    """Link the summary's header to its customer note, and publish it there."""
    previous = summary_customer_link(summary_path)
    text = summary_path.read_text()
    updated = _strip_customer_header_link(text)

    link = f"[{customer_name}]({relative_link(customer_path, summary_path.parent)})"

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)} {link}"

    if SUMMARY_HEADER_RE.search(updated):
        updated = SUMMARY_HEADER_RE.sub(repl, updated, count=1)
    else:
        updated = f"{load_config().summary_header} {link}\n\n{updated.lstrip()}"

    summary_path.write_text(updated)
    ensure_summary_transcript_link(summary_path)

    if previous and previous[1] != customer_path:
        remove_customer_note_link(previous[1], summary_path)
    sync_customer_note(customer_path, summary_path)


def clear_customer_metadata(summary_path: Path) -> None:
    previous = summary_customer_link(summary_path)
    summary_path.write_text(_strip_customer_header_link(summary_path.read_text()))
    ensure_summary_transcript_link(summary_path)
    if previous:
        remove_customer_note_link(previous[1], summary_path)


def manual_customer_state(*, customer_name: str, customer_path: Path) -> CustomerState:
    return CustomerState(
        action="link_existing",
        confidence="high",
        rationale="Manual selection from audio-tui.",
        model="manual",
        verified=True,
        customer_name=customer_name,
        customer_path=customer_path,
        source="manual",
    )


def manual_new_customer_state(*, suggested_name: str) -> CustomerState:
    return CustomerState(
        action="suggest_new",
        confidence="high",
        rationale="Manual free-text selection from audio-tui.",
        model="manual",
        verified=True,
        suggested_name=suggested_name,
        source="manual",
    )
