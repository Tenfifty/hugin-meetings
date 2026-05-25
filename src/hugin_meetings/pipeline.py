#!/usr/bin/env python3
"""Helpers for Hugin's meeting audio pipeline and metadata."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from hugin.llm import run_prompt
from hugin.prompts import resolve_prompt

from . import summarize as summarize_tool
from .config import load_config

# Output directories (written to the user's vault/knowledge base)
TRANSCRIPT_DIR = load_config().transcripts_dir
SUMMARY_DIR = load_config().summaries_dir

# State / cache directories
RAW_AUDIO_DIR = load_config().raw_audio_dir
STATE_AUDIO_DIR = load_config().state_dir
WAV_CACHE_DIR = load_config().wav_cache_dir
TRANSCRIPT_JSON_DIR = load_config().transcript_json_dir

# Project/customer matching. Projects dir may be None if not configured.
# Internal naming retains "CUSTOMERS_" for on-disk backward compatibility;
# the user-facing concept is "project" (see config.project_matcher).
CUSTOMERS_DIR = load_config().project_matcher.projects_dir
INTERNAL_PROJECT = load_config().project_matcher.internal_project

def _relative_or_absolute(path: Path) -> str:
    """Return path relative to vault if possible, else absolute."""
    if load_config().vault_path:
        try:
            return str(path.relative_to(load_config().vault_path))
        except ValueError:
            pass
    return str(path)

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
FIELD_RE = re.compile(r"^- ([^:]+):\s*(.*)$")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
SUMMARY_HEADER_RE = re.compile(
    rf"^({re.escape(load_config().summary_header)})(.*)$", re.MULTILINE
)
PERSONAL_SECTION_HEADER = load_config().personal_section_header
TRANSCRIPT_LINK_RE = re.compile(r"^\[Transcript\]\([^)]+\)\s*$", re.MULTILINE)
CUSTOMER_ENTRY_HEADER_RE = re.compile(r"^## <\d{4}-\d{2}-\d{2} [^>]+>\s*$", re.MULTILINE)

CustomerAction = Literal["link_existing", "suggest_new", "no_match"]
CustomerStatus = Literal["linked", "suggested_new", "no_match"]
Confidence = Literal["high", "medium", "low"]

DEFAULT_CUSTOMER_MODEL = load_config().project_matcher.model
DEFAULT_CUSTOMER_EFFORT = load_config().project_matcher.effort
CUSTOMER_JSON_SYSTEM_PROMPT = load_config().project_matcher.json_system_prompt
INACTIVE_DIR_NAMES = set(load_config().project_matcher.inactive_dir_names)
MAX_CANDIDATE_PREVIEW = 700
MAX_MEETING_TEXT = 12000
_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class CustomerNote:
    name: str
    path: Path
    is_active: bool
    preview: str


@dataclass
class CustomerDecision:
    action: CustomerAction
    confidence: Confidence
    rationale: str
    customer_name: str | None = None
    customer_path: Path | None = None
    suggested_name: str | None = None
    model: str = DEFAULT_CUSTOMER_MODEL


@dataclass
class CustomerMetadata:
    status: CustomerStatus
    confidence: Confidence
    rationale: str
    transcript_path: Path
    model: str
    customer_name: str | None = None
    customer_path: Path | None = None
    suggested_name: str | None = None


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
    def status(self) -> CustomerStatus:
        if self.action == "link_existing":
            return "linked"
        if self.action == "suggest_new":
            return "suggested_new"
        return "no_match"

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
    mic_path: Path | None
    sys_path: Path | None
    mic_parts: tuple[Path, ...]
    sys_parts: tuple[Path, ...]
    transcript_json: Path | None
    transcript_md: Path | None
    summary_md: Path | None
    calendar_fields: dict[str, str]
    customer_state: CustomerState | None
    anonymous_speakers: list[str]

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
    def has_customer_guess(self) -> bool:
        return self.customer_state is not None

    @property
    def has_customer_link(self) -> bool:
        return bool(
            self.customer_state
            and self.customer_state.verified
            and self.customer_state.action == "link_existing"
        )

    @property
    def pipeline_steps_complete(self) -> int:
        return sum(
            [
                self.has_transcript,
                self.has_calendar_metadata,
                self.has_summary,
                self.has_customer_guess,
            ]
        )

    @property
    def pipeline_total_steps(self) -> int:
        return 4

    @property
    def needs_pipeline(self) -> bool:
        return not (
            self.has_transcript
            and self.has_calendar_metadata
            and self.has_summary
            and self.has_customer_guess
        )

    @property
    def needs_enrollment(self) -> bool:
        return bool(self.anonymous_speakers)

    @property
    def raw_part_count(self) -> int:
        return max(len(self.mic_parts), len(self.sys_parts))

    @property
    def title(self) -> str:
        event = self.calendar_fields.get("Event")
        if event and event != "-":
            return event
        if self.summary_md and self.summary_md.exists():
            excerpt = summary_excerpt(self.summary_md)
            if excerpt:
                return excerpt[:90]
        return self.timestamp

    @property
    def short_status(self) -> str:
        flags = [
            "T" if self.has_transcript else ".",
            "C" if self.has_calendar_metadata else ".",
            "S" if self.has_summary else ".",
            "G" if self.has_customer_guess else ".",
        ]
        return "".join(flags)


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
    return os.path.relpath(target, base_dir)


def transcript_json_path(ts: str) -> Path:
    return TRANSCRIPT_JSON_DIR / year_subdir(ts) / f"transcript-{ts}.json"


def transcript_json_for_markdown(path: Path) -> Path | None:
    ts = extract_timestamp(path.name)
    if not ts:
        return None
    return transcript_json_path(ts)


def customer_state_path(ts: str) -> Path:
    return TRANSCRIPT_JSON_DIR / year_subdir(ts) / f"transcript-{ts}.customer.json"


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


def _raw_part_sort_key(path: Path) -> int:
    part = parse_raw_audio_part(path)
    return part.part if part else 0


def meeting_artifact_paths(rec: MeetingStatus) -> list[Path]:
    """Return timestamp-scoped files that make up a meeting entry."""
    paths: list[Path] = []

    def add(path: Path | None) -> None:
        if path is not None and path not in paths:
            paths.append(path)

    for path in (*rec.mic_parts, *rec.sys_parts):
        add(path)
        add(WAV_CACHE_DIR / f"{path.stem}.wav")

    for prefix in ("mic", "sys"):
        for path in sorted(RAW_AUDIO_DIR.rglob(f"{prefix}-{rec.timestamp}*.opus")):
            add(path)
            add(WAV_CACHE_DIR / f"{path.stem}.wav")
        for path in sorted(WAV_CACHE_DIR.rglob(f"{prefix}-{rec.timestamp}*.wav")):
            add(path)

    yyyy = year_subdir(rec.timestamp)
    add(rec.transcript_json)
    add(transcript_json_path(rec.timestamp))
    add(customer_state_path(rec.timestamp))
    add(rec.transcript_md)
    add(TRANSCRIPT_DIR / yyyy / f"transcript-{rec.timestamp}.md")
    add(rec.summary_md)
    add(SUMMARY_DIR / yyyy / f"summary-{rec.timestamp}.md")

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


def remove_metadata_block(text: str, start: str, end: str) -> str:
    pattern = re.compile(
        rf"{re.escape(start)}.*?{re.escape(end)}\n*",
        re.DOTALL,
    )
    updated = pattern.sub("", text)
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
    return match.group(1), Path(match.group(2))


def parse_calendar_metadata(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    text = path.read_text()
    block = extract_metadata_block(text, CALENDAR_METADATA_START, CALENDAR_METADATA_END)
    return parse_key_value_block(block)


def parse_customer_metadata(path: Path | None) -> CustomerMetadata | None:
    if not path or not path.exists():
        return None

    text = path.read_text()
    header_match = SUMMARY_HEADER_RE.search(text)
    if header_match:
        header_links = LINK_RE.findall(header_match.group(2))
        if header_links:
            name, rel_path = header_links[0]
            ts = extract_timestamp(path.name)
            transcript_path = TRANSCRIPT_DIR / f"transcript-{ts}.md"
            return CustomerMetadata(
                status="linked",
                confidence="high",
                rationale="Inferred from compact summary customer link.",
                transcript_path=transcript_path,
                model="summary",
                customer_name=name,
                customer_path=(path.parent / rel_path).resolve(),
            )
    return None


def state_from_summary_metadata(path: Path | None) -> CustomerState | None:
    metadata = parse_customer_metadata(path)
    if metadata is None:
        return None

    if metadata.status == "linked":
        action: CustomerAction = "link_existing"
    elif metadata.status == "suggested_new":
        action = "suggest_new"
    else:
        action = "no_match"

    return CustomerState(
        action=action,
        confidence=metadata.confidence,
        rationale=metadata.rationale,
        model=metadata.model,
        verified=True,
        customer_name=metadata.customer_name,
        customer_path=metadata.customer_path,
        suggested_name=metadata.suggested_name,
        source="summary",
    )


def state_from_decision(
    decision: CustomerDecision,
    *,
    verified: bool = False,
    source: str = "auto",
) -> CustomerState:
    return CustomerState(
        action=decision.action,
        confidence=decision.confidence,
        rationale=decision.rationale,
        model=decision.model,
        verified=verified,
        customer_name=decision.customer_name,
        customer_path=decision.customer_path,
        suggested_name=decision.suggested_name,
        source=source,
    )


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
    customer_path = Path(customer_path_value).resolve() if customer_path_value else None

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


def load_customer_state(ts: str, summary_path: Path | None = None) -> CustomerState | None:
    state_path = customer_state_path(ts)
    if state_path.exists():
        try:
            return deserialize_customer_state(json.loads(state_path.read_text()))
        except Exception:
            pass
    return state_from_summary_metadata(summary_path)


def save_customer_state(ts: str, state: CustomerState) -> Path:
    path = customer_state_path(ts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialize_customer_state(state), indent=2, ensure_ascii=False) + "\n")
    return path


def clear_customer_state(ts: str) -> None:
    customer_state_path(ts).unlink(missing_ok=True)


def summary_excerpt(path: Path, max_len: int = 120) -> str:
    text = remove_metadata_block(path.read_text(), CALENDAR_METADATA_START, CALENDAR_METADATA_END)
    text = _strip_customer_header_link(text)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:max_len]
    return ""


def list_customer_notes() -> list[CustomerNote]:
    notes: list[CustomerNote] = []
    for path in sorted(CUSTOMERS_DIR.rglob("*.md")):
        name = path.stem
        is_active = not any(part in INACTIVE_DIR_NAMES for part in path.parts)
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
    raw_sessions = scan_raw_audio_sessions()
    timestamps: set[str] = set(raw_sessions)

    for pattern, prefix in (
        (TRANSCRIPT_DIR.rglob("transcript-*.md"), "transcript-"),
        (TRANSCRIPT_JSON_DIR.rglob("transcript-*.json"), "transcript-"),
        (SUMMARY_DIR.rglob("summary-*.md"), "summary-"),
    ):
        for path in pattern:
            if path.name.endswith(".customer.json"):
                continue
            stem = path.stem
            if stem.endswith(".md"):
                stem = Path(stem).stem
            ts = stem.removeprefix(prefix)
            if ts:
                timestamps.add(ts)

    meetings: list[MeetingStatus] = []
    for ts in sorted(timestamps, reverse=True):
        raw_session = raw_sessions.get(ts)
        mic_parts = tuple(raw_session.mic_parts) if raw_session else ()
        sys_parts = tuple(raw_session.sys_parts) if raw_session else ()
        yyyy = year_subdir(ts)
        transcript_md = TRANSCRIPT_DIR / yyyy / f"transcript-{ts}.md"
        transcript_json = transcript_json_path(ts)
        summary_md = SUMMARY_DIR / yyyy / f"summary-{ts}.md"

        meetings.append(
            MeetingStatus(
                timestamp=ts,
                mic_path=mic_parts[0] if mic_parts else None,
                sys_path=sys_parts[0] if sys_parts else None,
                mic_parts=mic_parts,
                sys_parts=sys_parts,
                transcript_json=transcript_json if transcript_json.exists() else None,
                transcript_md=transcript_md if transcript_md.exists() else None,
                summary_md=summary_md if summary_md.exists() else None,
                calendar_fields=parse_calendar_metadata(transcript_md if transcript_md.exists() else None),
                customer_state=load_customer_state(ts, summary_md if summary_md.exists() else None),
                anonymous_speakers=load_anonymous_speakers(transcript_json if transcript_json.exists() else None),
            )
        )
    return meetings


def _load_customer_prompt_template() -> str:
    path = resolve_prompt(
        base="project_matcher",
        language=load_config().language,
        explicit=load_config().project_matcher.prompt_path,
        package_dir=_PROMPTS_DIR,
    )
    return path.read_text(encoding="utf-8")


def _render_customer_prompt_template(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def build_customer_prompt(summary_path: Path, model: str) -> tuple[str, list[CustomerNote]]:
    summary_text = summary_path.read_text()
    summary_body = _strip_customer_header_link(summary_text)
    summary_body = summary_body.strip()[:MAX_MEETING_TEXT]

    ts = extract_timestamp(summary_path.name)
    transcript_path = TRANSCRIPT_DIR / f"transcript-{ts}.md"
    calendar_fields = parse_calendar_metadata(transcript_path if transcript_path.exists() else None)

    calendar_lines = "\n".join(
        f"- {key}: {value}" for key, value in calendar_fields.items()
    ) or "- (no calendar metadata)"

    all_notes = list_customer_notes()

    candidate_context = "\n\n".join(
        [
            f"Customer: {note.name}\nPath: {_relative_or_absolute(note.path)}\nNotes: {note.preview}"
            for note in all_notes
        ]
    )

    internal_name = INTERNAL_PROJECT or "(internal)"
    internal_rules = (
        (
            f"- A meeting can still belong to a project even if all attendees are internal {internal_name} people.\n"
            f'- Choose "{internal_name}" only for genuinely internal meetings with no clear project/customer focus, such as admin, staffing, company process, internal tooling, internal strategy without a specific project, or similar operational topics.\n'
            f'- If a known project is discussed concretely, prefer that project over "{internal_name}".\n'
        )
        if INTERNAL_PROJECT
        else ""
    )

    prompt = _render_customer_prompt_template(
        _load_customer_prompt_template(),
        {
            "internal_rules": internal_rules,
            "candidate_context": candidate_context,
            "calendar_lines": calendar_lines,
            "summary_body": summary_body,
        },
    )

    return prompt, all_notes


def extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("No JSON object found in model output")


def run_remote_json_prompt(model: str, prompt: str, effort: str | None = None) -> dict:
    return extract_json_object(run_prompt(load_config().llm, model, prompt, effort=effort))


def run_local_json_prompt(model: str, prompt: str) -> dict:
    llm = summarize_tool.load_local_model(model)
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": CUSTOMER_JSON_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=600,
        temperature=0.2,
    )
    text = response["choices"][0]["message"]["content"]
    return extract_json_object(text)


def resolve_customer_path(name: str | None, candidates: list[CustomerNote]) -> Path | None:
    if not name:
        return None

    for note in candidates:
        if note.name.lower() == name.lower():
            return note.path

    for note in list_customer_notes():
        if note.name.lower() == name.lower():
            return note.path

    return None


def _customer_note_filename(name: str) -> str:
    cleaned = re.sub(r"[\\/]+", "-", name).strip().strip(".")
    if not cleaned:
        raise ValueError("Customer name must not be empty.")
    return f"{cleaned}.md"


def ensure_customer_note(customer_name: str) -> Path:
    existing = resolve_customer_path(customer_name, [])
    if existing:
        return existing

    CUSTOMERS_DIR.mkdir(parents=True, exist_ok=True)
    path = CUSTOMERS_DIR / _customer_note_filename(customer_name)
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


def suggest_customer_link(
    summary_path: Path,
    model: str = DEFAULT_CUSTOMER_MODEL,
    effort: str = DEFAULT_CUSTOMER_EFFORT,
) -> CustomerDecision:
    prompt, candidates = build_customer_prompt(summary_path, model)
    if model in summarize_tool.LOCAL_MODELS:
        payload = run_local_json_prompt(model, prompt)
    else:
        payload = run_remote_json_prompt(model, prompt, effort=effort)

    action = str(payload.get("action", "no_match")).strip()
    if action not in {"link_existing", "suggest_new", "no_match"}:
        action = "no_match"

    confidence = str(payload.get("confidence", "low")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    customer_name = payload.get("customer_name") or None
    suggested_name = payload.get("suggested_name") or None
    rationale = str(payload.get("rationale", "")).strip()

    customer_path = resolve_customer_path(customer_name, candidates)
    if action == "link_existing" and not customer_path:
        action = "suggest_new" if suggested_name else "no_match"

    return CustomerDecision(
        action=action,
        confidence=confidence,  # type: ignore[arg-type]
        rationale=rationale or "No clear rationale returned.",
        customer_name=customer_name,
        customer_path=customer_path,
        suggested_name=suggested_name,
        model=model,
    )


def metadata_from_decision(
    summary_path: Path,
    decision: CustomerDecision,
) -> CustomerMetadata:
    ts = extract_timestamp(summary_path.name)
    transcript_path = TRANSCRIPT_DIR / f"transcript-{ts}.md"

    if decision.action == "link_existing":
        return CustomerMetadata(
            status="linked",
            confidence=decision.confidence,
            rationale=decision.rationale,
            transcript_path=transcript_path,
            model=decision.model,
            customer_name=decision.customer_name,
            customer_path=decision.customer_path,
        )

    if decision.action == "suggest_new":
        return CustomerMetadata(
            status="suggested_new",
            confidence=decision.confidence,
            rationale=decision.rationale,
            transcript_path=transcript_path,
            model=decision.model,
            suggested_name=decision.suggested_name,
        )

    return CustomerMetadata(
        status="no_match",
        confidence=decision.confidence,
        rationale=decision.rationale,
        transcript_path=transcript_path,
        model=decision.model,
    )


def metadata_from_state(
    summary_path: Path,
    state: CustomerState,
) -> CustomerMetadata:
    ts = extract_timestamp(summary_path.name)
    transcript_path = TRANSCRIPT_DIR / f"transcript-{ts}.md"

    if state.action == "link_existing":
        return CustomerMetadata(
            status="linked",
            confidence=state.confidence,
            rationale=state.rationale,
            transcript_path=transcript_path,
            model=state.model,
            customer_name=state.customer_name,
            customer_path=state.customer_path,
        )

    if state.action == "suggest_new":
        return CustomerMetadata(
            status="suggested_new",
            confidence=state.confidence,
            rationale=state.rationale,
            transcript_path=transcript_path,
            model=state.model,
            suggested_name=state.suggested_name,
        )

    return CustomerMetadata(
        status="no_match",
        confidence=state.confidence,
        rationale=state.rationale,
        transcript_path=transcript_path,
        model=state.model,
    )


def render_customer_metadata(summary_path: Path, metadata: CustomerMetadata) -> str:
    if metadata.status == "linked" and metadata.customer_name and metadata.customer_path:
        customer_rel = relative_link(metadata.customer_path, summary_path.parent)
        return f"[{metadata.customer_name}]({customer_rel})"
    return ""


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
    return TRANSCRIPT_DIR / f"transcript-{ts}.md"


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


def _summary_body_for_customer_note(summary_path: Path) -> str:
    text = _strip_transcript_link(_strip_customer_header_link(summary_path.read_text())).strip()
    lines = text.splitlines()
    if lines and lines[0].startswith(load_config().summary_header):
        lines = lines[1:]

    body = "\n".join(lines).strip()
    if not body:
        return ""

    lead_lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            break
        lead_lines.append(line)

    lead = "\n".join(lead_lines).strip()

    personal = ""
    if PERSONAL_SECTION_HEADER:
        match = re.search(
            rf"(^{re.escape(PERSONAL_SECTION_HEADER)}\s*$.*?)(?=^### |\Z)",
            body,
            re.MULTILINE | re.DOTALL,
        )
        personal = match.group(1).strip() if match else ""

    parts = [part for part in (lead, personal) if part]
    return "\n\n".join(parts).strip() + ("\n" if parts else "")


def _customer_note_links(customer_path: Path, summary_path: Path) -> tuple[str, str]:
    transcript_path = _summary_transcript_path(summary_path)
    return (
        relative_link(summary_path, customer_path.parent),
        relative_link(transcript_path, customer_path.parent),
    )


def _find_customer_note_entry_bounds(text: str, summary_rel: str) -> tuple[int, int] | None:
    marker = f"[Full summary]({summary_rel})"
    idx = text.find(marker)
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

    bounds = _find_customer_note_entry_bounds(customer_text, summary_rel)
    if bounds is not None:
        start, end = bounds
        updated = customer_text[:start].rstrip() + "\n\n" + entry + customer_text[end:].lstrip()
    else:
        lines = customer_text.splitlines()
        insert_at = 0
        if lines and lines[0].startswith("# "):
            insert_at = 1
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
    summary_rel, _ = _customer_note_links(customer_path, summary_path)
    bounds = _find_customer_note_entry_bounds(text, summary_rel)
    if bounds is None:
        return
    start, end = bounds
    updated = text[:start].rstrip() + "\n\n" + text[end:].lstrip()
    customer_path.write_text(re.sub(r"\n{3,}", "\n\n", updated).rstrip() + "\n")


def write_customer_metadata(summary_path: Path, metadata: CustomerMetadata) -> None:
    previous = parse_customer_metadata(summary_path)
    text = summary_path.read_text()
    updated = _strip_customer_header_link(text)

    link = render_customer_metadata(summary_path, metadata)
    if not link:
        summary_path.write_text(updated)
        return

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)} {link}"

    if SUMMARY_HEADER_RE.search(updated):
        updated = SUMMARY_HEADER_RE.sub(repl, updated, count=1)
    else:
        updated = f"{load_config().summary_header} {link}\n\n{updated.lstrip()}"

    summary_path.write_text(updated)
    ensure_summary_transcript_link(summary_path)

    if previous and previous.customer_path and previous.customer_path != metadata.customer_path:
        remove_customer_note_link(previous.customer_path, summary_path)
    if metadata.customer_path:
        sync_customer_note(metadata.customer_path, summary_path)


def clear_customer_metadata(summary_path: Path) -> None:
    previous = parse_customer_metadata(summary_path)
    summary_path.write_text(_strip_customer_header_link(summary_path.read_text()))
    ensure_summary_transcript_link(summary_path)
    if previous and previous.customer_path:
        remove_customer_note_link(previous.customer_path, summary_path)


def manual_customer_state(
    *,
    customer_name: str,
    customer_path: Path,
    rationale: str = "Manual selection from audio-tui.",
    confidence: Confidence = "high",
    verified: bool = True,
) -> CustomerState:
    return CustomerState(
        action="link_existing",
        confidence=confidence,
        rationale=rationale,
        model="manual",
        verified=verified,
        customer_name=customer_name,
        customer_path=customer_path,
        source="manual",
    )


def manual_new_customer_state(
    *,
    suggested_name: str,
    rationale: str = "Manual free-text selection from audio-tui.",
    confidence: Confidence = "high",
    verified: bool = True,
) -> CustomerState:
    return CustomerState(
        action="suggest_new",
        confidence=confidence,
        rationale=rationale,
        model="manual",
        verified=verified,
        suggested_name=suggested_name,
        source="manual",
    )


def state_for_summary(summary_path: Path) -> CustomerState | None:
    ts = extract_timestamp(summary_path.name)
    if not ts:
        return None
    return load_customer_state(ts, summary_path)


def guess_customer_state(summary_path: Path, model: str = DEFAULT_CUSTOMER_MODEL) -> CustomerState:
    decision = suggest_customer_link(summary_path, model)
    state = state_from_decision(decision, verified=False, source="auto")
    ts = extract_timestamp(summary_path.name)
    if not ts:
        raise ValueError(f"Could not determine timestamp from {summary_path.name}")
    save_customer_state(ts, state)
    clear_customer_metadata(summary_path)
    return state


def verify_customer_state(summary_path: Path) -> CustomerState:
    ts = extract_timestamp(summary_path.name)
    if not ts:
        raise ValueError(f"Could not determine timestamp from {summary_path.name}")
    state = load_customer_state(ts, summary_path)
    if state is None:
        raise ValueError("No cached customer suggestion to verify.")
    state.verified = True
    state = materialize_verified_customer_state(state)
    save_customer_state(ts, state)
    write_customer_metadata(summary_path, metadata_from_state(summary_path, state))
    return state


def save_customer_state_and_sync_summary(
    summary_path: Path,
    state: CustomerState,
) -> CustomerState:
    ts = extract_timestamp(summary_path.name)
    if not ts:
        raise ValueError(f"Could not determine timestamp from {summary_path.name}")
    state = materialize_verified_customer_state(state)
    save_customer_state(ts, state)
    if state.verified:
        write_customer_metadata(summary_path, metadata_from_state(summary_path, state))
    else:
        clear_customer_metadata(summary_path)
    return state


def remove_customer_link(summary_path: Path) -> None:
    ts = extract_timestamp(summary_path.name)
    if ts:
        clear_customer_state(ts)
    clear_customer_metadata(summary_path)
