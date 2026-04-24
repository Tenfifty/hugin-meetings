"""Scheduled meeting and reminder helpers shared by recorder frontends."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

START_PROMPT_GRACE_SECONDS = 10 * 60
MAX_MEETING_DURATION = timedelta(hours=4)

SECTION_HEADER_RE = re.compile(r"^\s*##\s+<(?P<date>\d{4}-\d{2}-\d{2})\b")
BRACE_TIME_RE = re.compile(
    r"\*(?P<ignored>~)?\{(?P<start>\d{1,2}[:.]\d{2})(?:\s*-\s*(?P<end>\d{1,2}[:.]\d{2}))?\}\*"
)
LEGACY_TIME_RE = re.compile(
    r"\[(?P<start>\d{1,2}[:.]\d{2})(?:\s*-\s*(?P<end>\d{1,2}[:.]\d{2}))?\]"
)
AGENDA_ITEM_RE = re.compile(r"^- \[[ xX]\]\s*(?P<body>.+)$")


@dataclass(frozen=True)
class ScheduledMeeting:
    key: str
    title: str
    start_at: datetime
    end_at: datetime | None
    source_line: str

    @property
    def time_label(self) -> str:
        if not self.end_at:
            return self.start_at.strftime("%H:%M")
        return f"{self.start_at.strftime('%H:%M')} - {self.end_at.strftime('%H:%M')}"


def _parse_clock(value: str):
    normalized = value.replace(".", ":")
    return datetime.strptime(normalized, "%H:%M").time()


def _strip_time_markup(text: str) -> str:
    stripped = BRACE_TIME_RE.sub("", text)
    stripped = LEGACY_TIME_RE.sub("", stripped)
    return " ".join(stripped.split()).strip()


def load_todays_journal_meetings(
    journal_path: Path | None,
    today: date,
    *,
    max_meeting_duration: timedelta = MAX_MEETING_DURATION,
) -> list[ScheduledMeeting]:
    if journal_path is None or not journal_path.exists():
        return []

    lines = journal_path.read_text(encoding="utf-8").splitlines()
    in_today_section = False
    meetings: list[ScheduledMeeting] = []

    for raw_line in lines:
        header_match = SECTION_HEADER_RE.match(raw_line)
        if header_match:
            in_today_section = header_match.group("date") == today.isoformat()
            continue

        if not in_today_section:
            continue

        item_match = AGENDA_ITEM_RE.match(raw_line)
        if not item_match:
            continue

        body = item_match.group("body").strip()
        time_match = BRACE_TIME_RE.search(body)
        ignored = False
        if time_match:
            ignored = bool(time_match.group("ignored"))
        else:
            time_match = LEGACY_TIME_RE.search(body)

        if not time_match or ignored:
            continue

        start_time = _parse_clock(time_match.group("start"))
        end_token = time_match.group("end")
        end_at = None
        start_at = datetime.combine(today, start_time)
        if end_token:
            end_time = _parse_clock(end_token)
            end_at = datetime.combine(today, end_time)
            if end_at <= start_at:
                continue
            if end_at - start_at > max_meeting_duration:
                continue

        title = _strip_time_markup(body)
        if not title:
            continue

        key = f"{start_at.isoformat()}::{title}"
        meetings.append(
            ScheduledMeeting(
                key=key,
                title=title,
                start_at=start_at,
                end_at=end_at,
                source_line=raw_line,
            )
        )

    meetings.sort(key=lambda meeting: meeting.start_at)
    return meetings


def default_reminder_state(today: date | None = None) -> dict[str, Any]:
    return {
        "date": today.isoformat() if today else None,
        "prompted_start": [],
        "prompted_stop": [],
        "recording_meeting_key": None,
    }


def normalize_reminder_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": state.get("date"),
        "prompted_start": list(state.get("prompted_start", [])),
        "prompted_stop": list(state.get("prompted_stop", [])),
        "recording_meeting_key": state.get("recording_meeting_key"),
    }


def load_reminder_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_reminder_state()

    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logging.exception("Failed to read reminder state from %s", path)
        return default_reminder_state()

    return normalize_reminder_state(state if isinstance(state, dict) else {})


def save_reminder_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalize_reminder_state(state), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def reset_reminder_state_for_today(
    state: dict[str, Any],
    today: date | None = None,
) -> tuple[dict[str, Any], bool]:
    today = today or date.today()
    today_iso = today.isoformat()
    if state.get("date") == today_iso:
        return normalize_reminder_state(state), False
    return default_reminder_state(today), True


def mark_prompted(state: dict[str, Any], kind: str, meeting_key: str) -> tuple[dict[str, Any], bool]:
    state = normalize_reminder_state(state)
    state_key = f"prompted_{kind}"
    prompted = set(state[state_key])
    if meeting_key in prompted:
        return state, False
    prompted.add(meeting_key)
    state[state_key] = sorted(prompted)
    return state, True


def set_recording_meeting(state: dict[str, Any], meeting_key: str | None) -> dict[str, Any]:
    state = normalize_reminder_state(state)
    state["recording_meeting_key"] = meeting_key
    return state


def associate_current_recording(
    meetings: list[ScheduledMeeting],
    state: dict[str, Any],
    now: datetime,
    *,
    is_recording: bool,
    grace_seconds: int = START_PROMPT_GRACE_SECONDS,
) -> ScheduledMeeting | None:
    if not is_recording or state.get("recording_meeting_key"):
        return None

    candidates = [
        meeting
        for meeting in meetings
        if 0 <= (now - meeting.start_at).total_seconds() <= grace_seconds
        and (meeting.end_at is None or now <= meeting.end_at)
    ]
    return candidates[0] if len(candidates) == 1 else None


def start_reminder_candidate(
    meetings: list[ScheduledMeeting],
    state: dict[str, Any],
    now: datetime,
    *,
    is_recording: bool,
    grace_seconds: int = START_PROMPT_GRACE_SECONDS,
) -> ScheduledMeeting | None:
    if is_recording:
        return None

    prompted = set(state["prompted_start"])
    for meeting in meetings:
        if meeting.key in prompted:
            continue
        age_seconds = (now - meeting.start_at).total_seconds()
        if 0 <= age_seconds <= grace_seconds:
            return meeting
    return None


def stop_reminder_candidate(
    meeting_index: dict[str, ScheduledMeeting],
    state: dict[str, Any],
    now: datetime,
    *,
    is_recording: bool,
) -> ScheduledMeeting | None:
    if not is_recording:
        return None

    meeting_key = state.get("recording_meeting_key")
    if not meeting_key:
        return None

    meeting = meeting_index.get(meeting_key)
    if meeting is None or meeting.end_at is None:
        return None
    if meeting_key in set(state["prompted_stop"]):
        return None
    if now < meeting.end_at:
        return None
    return meeting


def next_meeting_label(meetings: list[ScheduledMeeting], now: datetime) -> str:
    for meeting in meetings:
        if meeting.end_at and meeting.end_at < now:
            continue
        if not meeting.end_at and meeting.start_at < now - timedelta(minutes=10):
            continue
        return f"{meeting.time_label} {meeting.title}"
    return "-"
