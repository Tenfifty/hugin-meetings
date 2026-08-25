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
# A meeting that runs over is asked about again this often, until it is stopped.
STOP_PROMPT_INTERVAL = timedelta(minutes=30)
# A recording with no known end gets its first stop prompt this long after it
# started - both a journal meeting with no end time and a recording that was
# never associated with a journal meeting at all.
OPEN_ENDED_STOP_DELAY = timedelta(minutes=30)

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


@dataclass(frozen=True)
class StopPrompt:
    """One occasion to ask whether a running recording should stop.

    ``index`` counts from the first deadline, so 0 is the meeting's scheduled
    end (or, when there is no end to key off, ``OPEN_ENDED_STOP_DELAY`` after
    the recording started) and every later index is another interval on top of
    that. ``meeting`` is None for a recording that was never associated with a
    journal meeting - those are asked about too, since a recording nobody
    scheduled is the easiest one to forget.
    """

    key_base: str
    due_at: datetime
    index: int
    interval: timedelta = STOP_PROMPT_INTERVAL
    meeting: ScheduledMeeting | None = None
    started_at: datetime | None = None

    @property
    def state_key(self) -> str:
        return f"{self.key_base}::stop{self.index}"

    @property
    def title(self) -> str:
        return self.meeting.title if self.meeting else "Unscheduled recording"

    @property
    def overdue(self) -> timedelta:
        return self.index * self.interval

    @property
    def detail(self) -> str:
        if self.meeting is not None and self.meeting.end_at is not None:
            base = f"Scheduled end: {self.meeting.end_at.strftime('%H:%M')}"
        else:
            since = self.started_at or (self.meeting.start_at if self.meeting else None)
            clock = since.strftime("%H:%M") if since else "?"
            if self.meeting is None:
                base = f"No journal meeting for it; started {clock}"
            else:
                base = f"No end time in the journal; started {clock}"
        if not self.index:
            return base
        minutes = int(self.overdue.total_seconds() // 60)
        return f"{base} - still recording, {minutes} min past due"


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


def first_stop_deadline(
    meeting: ScheduledMeeting,
    *,
    open_ended_delay: timedelta = OPEN_ENDED_STOP_DELAY,
) -> datetime:
    """When the recording for ``meeting`` first looks like it is running over."""
    if meeting.end_at is not None:
        return meeting.end_at
    return meeting.start_at + open_ended_delay


def _due_prompt(
    key_base: str,
    first_due: datetime,
    now: datetime,
    *,
    interval: timedelta,
    meeting: ScheduledMeeting | None = None,
    started_at: datetime | None = None,
) -> StopPrompt | None:
    if now < first_due:
        return None
    # Only the latest deadline is offered. Coming back to a recording that has
    # been running for hours should ask once, not once per missed interval.
    interval = max(interval, timedelta(seconds=1))
    index = int((now - first_due) // interval)
    return StopPrompt(
        key_base=key_base,
        due_at=first_due + index * interval,
        index=index,
        interval=interval,
        meeting=meeting,
        started_at=started_at,
    )


def stop_reminder_candidate(
    meeting_index: dict[str, ScheduledMeeting],
    state: dict[str, Any],
    now: datetime,
    *,
    is_recording: bool,
    recording_started_at: datetime | None = None,
    interval: timedelta = STOP_PROMPT_INTERVAL,
    open_ended_delay: timedelta = OPEN_ENDED_STOP_DELAY,
) -> StopPrompt | None:
    if not is_recording:
        return None

    meeting_key = state.get("recording_meeting_key")
    meeting = meeting_index.get(meeting_key) if meeting_key else None
    first_due = (
        first_stop_deadline(meeting, open_ended_delay=open_ended_delay)
        if meeting is not None
        else None
    )

    # An association can outlive the recording that earned it. A recording that
    # began after the meeting's deadline is not the one running over it, so the
    # meeting says nothing about when to ask - fall back to the recording.
    if first_due is not None and recording_started_at is not None:
        if recording_started_at >= first_due:
            meeting, first_due = None, None

    if meeting is not None:
        prompt = _due_prompt(
            meeting.key,
            first_due,
            now,
            interval=interval,
            meeting=meeting,
        )
    elif recording_started_at is not None:
        # Nothing in the journal to end it, so the recording's own start is the
        # only anchor there is.
        prompt = _due_prompt(
            f"recording::{recording_started_at.isoformat(timespec='seconds')}",
            recording_started_at + open_ended_delay,
            now,
            interval=interval,
            started_at=recording_started_at,
        )
    else:
        return None

    if prompt is None or prompt.state_key in set(state["prompted_stop"]):
        return None
    return prompt


def next_meeting_label(meetings: list[ScheduledMeeting], now: datetime) -> str:
    for meeting in meetings:
        if meeting.end_at and meeting.end_at < now:
            continue
        if not meeting.end_at and meeting.start_at < now - timedelta(minutes=10):
            continue
        return f"{meeting.time_label} {meeting.title}"
    return "-"
