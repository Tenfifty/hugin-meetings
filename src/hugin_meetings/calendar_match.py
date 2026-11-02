#!/usr/bin/env python3
"""Match a transcript against Google Calendar events via gws and annotate it.

Usage:
    match-calendar-event.py                              # latest transcript
    match-calendar-event.py transcript-20260409-100207.md
    match-calendar-event.py --dry-run
    match-calendar-event.py --calendar primary
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from hugin import gws
# GwsOfflineError is re-exported: context.py retries on it.
from hugin.gws import GwsError, GwsOfflineError, run_gws  # noqa: F401

from .cli_utils import resolve_transcript_md
from .pipeline import (
    CALENDAR_METADATA_END,
    CALENDAR_METADATA_START,
    RawAudioSession,
    audio_duration,
    extract_timestamp,
    metadata_block_re,
    transcript_json_for_markdown,
)
from .config import load_config

SEGMENT_TS_RE = re.compile(r"\*\*\[(\d{2}):(\d{2}):(\d{2})\]")

DEFAULT_LOOKBACK = timedelta(hours=4)
DEFAULT_LOOKAHEAD = timedelta(hours=6)
MAX_DESCRIPTION_LEN = 240
MAX_ATTENDEES = 8
MIN_OVERLAP_FRACTION = 0.5
RESOURCE_CALENDAR_DOMAIN = "resource.calendar.google.com"
NOISY_VIRTUAL_LOCATIONS = {
    "microsoft teams-möte",
    "microsoft teams meeting",
    "teams meeting",
}
RESPONSE_STATUS_PRIORITY = {
    "accepted": 3,
    "tentative": 2,
    "needsAction": 1,
    "declined": 0,
}
RESPONSE_STATUS_LABEL = {
    "accepted": "yes",
    "tentative": "maybe",
    "needsAction": "not answered",
    "declined": "no",
}


def local_tzinfo():
    return datetime.now().astimezone().tzinfo


@dataclass
class MatchWindow:
    """The stretch of wall clock a recording covers.

    Scoring only ever needed start/end/duration plus optional text for the
    title-keyword bonus, so the window can come from a finished transcript or
    straight from the raw audio — the latter lets the calendar be matched
    before transcription runs.
    """

    start: datetime
    end: datetime
    duration: timedelta
    text: str = ""


@dataclass
class Candidate:
    calendar_id: str
    calendar_name: str
    event: dict[str, Any]
    event_start: datetime
    event_end: datetime
    response_status: str
    score: float
    reasons: list[str]

    @property
    def external_attendees(self) -> list[str]:
        return external_attendees(self.event, self.calendar_id)


def session_start(ts: str) -> datetime:
    """A session timestamp as local wall-clock time."""
    return datetime.strptime(ts, "%Y%m%d-%H%M%S").replace(tzinfo=local_tzinfo())


def transcript_start_from_name(path: Path) -> datetime:
    ts = extract_timestamp(path.name)
    if not ts:
        raise GwsError(
            f"Could not infer transcript timestamp from filename: {path.name}"
        )
    return session_start(ts)


def transcript_duration(path: Path, text: str) -> timedelta:
    json_path = transcript_json_for_markdown(path)
    if json_path and json_path.exists():
        entries = json.loads(json_path.read_text())
        max_end = max((float(entry.get("end", 0.0)) for entry in entries), default=0.0)
        if max_end > 0:
            return timedelta(seconds=max_end)

    max_seconds = 0
    for hours, minutes, seconds in SEGMENT_TS_RE.findall(text):
        total = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        max_seconds = max(max_seconds, total)

    if max_seconds > 0:
        return timedelta(seconds=max_seconds)

    raise GwsError(f"Could not determine transcript duration from {path.name}")


def load_transcript(path: Path) -> MatchWindow:
    text = path.read_text()
    start = transcript_start_from_name(path)
    duration = transcript_duration(path, text)
    return MatchWindow(start, start + duration, duration, text)


def session_window(session: "RawAudioSession") -> MatchWindow:
    """The window a raw recording session covers, before any transcript exists.

    Duration is the summed wall clock of the parts of the longer channel;
    parts are recorded back to back, so their total is the meeting's length.
    """
    start = session_start(session.session_id)
    seconds = max(
        sum(audio_duration(path) for path in session.mic_parts),
        sum(audio_duration(path) for path in session.sys_parts),
    )
    if seconds <= 0:
        raise GwsError(f"No audio duration for session {session.session_id}")
    duration = timedelta(seconds=seconds)
    return MatchWindow(start, start + duration, duration)


def match_window(
    window: MatchWindow,
    *,
    calendar: str | None = None,
    include_shared: bool = False,
    lookback: timedelta = DEFAULT_LOOKBACK,
    lookahead: timedelta = DEFAULT_LOOKAHEAD,
) -> list[Candidate]:
    """Candidates for a window, best first."""
    calendars = list_calendars(calendar, include_shared)
    return collect_candidates(window, calendars, lookback, lookahead)


def serialize_candidate(candidate: Candidate) -> dict[str, Any]:
    """Store a match so it can be rendered later without asking Google again.

    The whole event is kept rather than a flattened summary: the metadata block
    and the customer matcher each want fields the other does not, and keeping
    one representation means it cannot drift from itself.
    """
    return {
        "calendar_id": candidate.calendar_id,
        "calendar_name": candidate.calendar_name,
        "event": candidate.event,
        "event_start": candidate.event_start.isoformat(),
        "event_end": candidate.event_end.isoformat(),
        "response_status": candidate.response_status,
        "score": candidate.score,
        "reasons": candidate.reasons,
    }


def deserialize_candidate(payload: dict[str, Any]) -> Candidate:
    return Candidate(
        calendar_id=payload.get("calendar_id", ""),
        calendar_name=payload.get("calendar_name", ""),
        event=payload.get("event") or {},
        event_start=datetime.fromisoformat(payload["event_start"]),
        event_end=datetime.fromisoformat(payload["event_end"]),
        response_status=payload.get("response_status", "needsAction"),
        score=float(payload.get("score", 0.0)),
        reasons=payload.get("reasons") or [],
    )


def serialize_window(window: MatchWindow) -> dict[str, Any]:
    return {
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
        "duration_seconds": window.duration.total_seconds(),
    }


def deserialize_window(payload: dict[str, Any]) -> MatchWindow:
    return MatchWindow(
        start=datetime.fromisoformat(payload["start"]),
        end=datetime.fromisoformat(payload["end"]),
        duration=timedelta(seconds=float(payload.get("duration_seconds", 0.0))),
    )


def stored_candidates(transcript_path: Path) -> tuple[list[Candidate], bool] | None:
    """The events a person already verified for this session, if any.

    Returns them with the chosen one first, and whether a person chose it.
    The list is empty when the choice was that it matched none of them.

    Re-querying Google here could annotate the transcript with a different
    event than the one confirmed before processing. The window still comes
    from the transcript, so the block describes the transcript rather than the
    raw audio it was made from.
    """
    from . import context as meeting_context

    ts = extract_timestamp(transcript_path.name)
    if not ts:
        return None
    context = meeting_context.load_context(ts)
    if context is None or not context.calendar:
        return None
    if not context.calendar.get("candidates"):
        return None
    return meeting_context.ordered_candidates(context), meeting_context.event_chosen_by_hand(context)


def is_owned_calendar(calendar: dict[str, Any]) -> bool:
    """True if the calendar is the user's own, not merely one they can manage.

    Google reports ``accessRole == "owner"`` both for the user's own calendars
    and for *another person's* personal calendar shared with "make changes and
    manage sharing". Treating the latter as owned pulls colleagues' meetings
    into the default match. The user's own personal calendar is always
    ``primary``; any other calendar whose id is a person's email address is
    someone else's. Group/resource calendars (``…@*.calendar.google.com``) the
    user has owner access to are theirs to manage, so they stay included.
    """
    if calendar.get("primary"):
        return True
    if calendar.get("accessRole") != "owner":
        return False
    calendar_id = (calendar.get("id") or "").lower()
    is_other_persons_calendar = (
        "@" in calendar_id and not calendar_id.endswith(".calendar.google.com")
    )
    return not is_other_persons_calendar


def list_calendars(
    calendar_id: str | None,
    include_shared_calendars: bool = False,
) -> list[dict[str, Any]]:
    if calendar_id:
        return [{"id": calendar_id, "summary": calendar_id, "primary": calendar_id == "primary"}]

    payload = run_gws(
        load_config(),
        "calendar",
        "calendarList",
        "list",
        "--params",
        json.dumps(
            {
                "showDeleted": False,
                "showHidden": False,
                "maxResults": 250,
                "minAccessRole": "reader" if include_shared_calendars else "owner",
            }
        ),
    )
    items = payload.get("items", [])
    calendars = [item for item in items if not item.get("deleted")]
    if include_shared_calendars:
        return calendars
    return [item for item in calendars if is_owned_calendar(item)]


def list_events(
    calendar_id: str,
    time_min: datetime,
    time_max: datetime,
) -> list[dict[str, Any]]:
    return gws.list_events(load_config(), calendar_id, time_min, time_max)


def event_time_bounds(event: dict[str, Any]) -> tuple[datetime | None, datetime | None, bool]:
    start, start_all_day = gws.parse_event_time(event.get("start", {}))
    end, end_all_day = gws.parse_event_time(event.get("end", {}))
    return start, end, start_all_day or end_all_day


def normalize_response_status(status: Any) -> str:
    """Google sends only the four canonical values; anything else is unanswered."""
    return status if status in RESPONSE_STATUS_PRIORITY else "needsAction"


def event_self_response_status(event: dict[str, Any]) -> str:
    for attendee in event.get("attendees", []):
        if attendee.get("self"):
            return normalize_response_status(attendee.get("responseStatus"))

    if event.get("organizer", {}).get("self") or event.get("creator", {}).get("self"):
        return "accepted"

    if not event.get("attendees"):
        return "accepted"

    return "needsAction"


def response_priority(status: str) -> int:
    return RESPONSE_STATUS_PRIORITY.get(status, RESPONSE_STATUS_PRIORITY["needsAction"])


def response_label(status: str) -> str:
    return RESPONSE_STATUS_LABEL.get(status, status or "not answered")


def mail_domain(email: str) -> str:
    return email.rpartition("@")[2].lower() if "@" in email else ""


def own_mail_domain(event: dict[str, Any], calendar_id: str) -> str:
    """The user's own mail domain, as the event says it: Google marks the
    user's own attendee, organizer and creator entries with ``self``."""
    people = [
        *(event.get("attendees") or []),
        event.get("organizer") or {},
        event.get("creator") or {},
    ]
    for person in people:
        if person.get("self") and mail_domain(person.get("email", "")):
            return mail_domain(person["email"])
    # A primary calendar's id is the user's address; a group calendar's is not.
    if calendar_id.endswith(".calendar.google.com"):
        return ""
    return mail_domain(calendar_id)


def external_attendees(event: dict[str, Any], calendar_id: str) -> list[str]:
    """Invitees from outside the user's own mail domain. Rooms are not people.

    Without a known own domain nobody can be called external, so none are.
    """
    own = own_mail_domain(event, calendar_id)
    if not own:
        return []
    return [
        attendee["email"]
        for attendee in event.get("attendees") or []
        if not attendee.get("resource")
        and mail_domain(attendee.get("email", "")) not in ("", own, RESOURCE_CALENDAR_DOMAIN)
    ]


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-ZåäöÅÄÖ0-9]{3,}", text.lower())
        if token not in {"och", "det", "som", "att", "med", "för", "the", "and"}
    }


def overlap_seconds(
    start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime
) -> float:
    return max(0.0, (min(end_a, end_b) - max(start_a, start_b)).total_seconds())


def score_event(
    event: dict[str, Any],
    calendar: dict[str, Any],
    window: MatchWindow,
) -> Candidate | None:
    if event.get("status") == "cancelled":
        return None

    event_start, event_end, is_all_day = event_time_bounds(event)
    if not event_start or not event_end or is_all_day:
        return None

    score = 0.0
    reasons: list[str] = []
    response_status = event_self_response_status(event)
    reasons.append(f"response {response_label(response_status)}")
    overlap = overlap_seconds(event_start, event_end, window.start, window.end)
    min_overlap = window.duration.total_seconds() * MIN_OVERLAP_FRACTION
    if overlap <= min_overlap:
        return None

    start_delta = abs((event_start - window.start).total_seconds()) / 60
    duration_delta = abs(
        ((event_end - event_start) - window.duration).total_seconds()
    ) / 60

    overlap_minutes = overlap / 60
    score += min(overlap_minutes * 2.0, 120.0)
    reasons.append(f"overlap {overlap_minutes:.0f}m")

    start_bonus = max(0.0, 45.0 - start_delta)
    if start_bonus > 0:
        score += start_bonus
        reasons.append(f"starts {start_delta:.0f}m from recording")

    duration_bonus = max(0.0, 20.0 - duration_delta / 2.0)
    if duration_bonus > 0:
        score += duration_bonus
        reasons.append(f"duration delta {duration_delta:.0f}m")

    summary = event.get("summary", "")
    summary_tokens = tokenize(summary)
    window_tokens = tokenize(window.text[:8000])
    token_hits = len(summary_tokens & window_tokens)
    if token_hits:
        token_bonus = min(token_hits * 6.0, 24.0)
        score += token_bonus
        reasons.append(f"title keywords hit {token_hits}")

    attendees = event.get("attendees", [])
    if attendees:
        score += 4.0
        reasons.append(f"{len(attendees)} attendee(s)")
    external = external_attendees(event, calendar["id"])
    if external:
        reasons.append(f"{len(external)} external")

    if event.get("hangoutLink") or event.get("conferenceData"):
        score += 4.0
        reasons.append("conference link")

    if event.get("location"):
        score += 2.0
        reasons.append("location set")

    if calendar.get("primary"):
        score += 3.0

    if score < 15.0:
        return None

    return Candidate(
        calendar_id=calendar["id"],
        calendar_name=calendar.get("summaryOverride") or calendar.get("summary") or calendar["id"],
        event=event,
        event_start=event_start,
        event_end=event_end,
        response_status=response_status,
        score=score,
        reasons=reasons,
    )


def candidate_sort_key(candidate: Candidate) -> tuple[int, bool, float]:
    # After the user's own answer, a meeting with people from outside beats one
    # with colleagues only, whatever the score. Of two overlapping meetings the
    # external one is usually the one attended, and the room booked for a
    # meeting is itself a colleagues-only event that scores just as well.
    return (
        response_priority(candidate.response_status),
        bool(candidate.external_attendees),
        candidate.score,
    )


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    best: dict[tuple[str, str, str, str], Candidate] = {}
    for candidate in candidates:
        event = candidate.event
        key = (
            event.get("iCalUID") or "",
            event.get("summary", ""),
            candidate.event_start.isoformat(),
            candidate.event_end.isoformat(),
        )
        current = best.get(key)
        if current is None or candidate_sort_key(candidate) > candidate_sort_key(current):
            best[key] = candidate
    return sorted(best.values(), key=candidate_sort_key, reverse=True)


def confidence_label(score: float) -> str:
    if score >= 110:
        return "high"
    if score >= 75:
        return "medium"
    if score >= 45:
        return "low"
    return "weak"


def format_dt(dt: datetime) -> str:
    return dt.astimezone().strftime("%Y-%m-%d %H:%M %Z")


def summarize_attendees(attendees: list[dict[str, Any]], location: str | None = None) -> str:
    if not attendees:
        return "-"

    location_text = (location or "").strip()
    names = []
    for attendee in attendees:
        name = attendee.get("displayName") or attendee.get("email") or "unknown"
        if location_text and name == location_text:
            continue
        status = attendee.get("responseStatus")
        if status and status not in {"needsAction", "accepted"}:
            name = f"{name} ({status})"
        names.append(name)
    if not names:
        return "-"
    shown = names[:MAX_ATTENDEES]
    if len(names) > MAX_ATTENDEES:
        shown.append(f"+{len(names) - MAX_ATTENDEES} more")
    return ", ".join(shown)


def shorten(text: str | None, limit: int = MAX_DESCRIPTION_LEN) -> str:
    if not text:
        return "-"
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def clean_location(text: str | None) -> str:
    if not text:
        return "-"
    compact = " ".join(text.split()).strip()
    if compact.lower() in NOISY_VIRTUAL_LOCATIONS:
        return "-"
    return compact


def clean_description(text: str | None) -> str:
    if not text:
        return "-"

    raw = text.strip()
    if not raw:
        return "-"

    lower = raw.lower()
    teams_markers = [
        "microsoft teams",
        "anslut till:",
        "mötes-id:",
        "lösenord:",
        "ring in via telefon",
        "hitta ett lokalt telefonnummer",
        "mötesalternativ",
    ]
    has_teams_boilerplate = sum(marker in lower for marker in teams_markers) >= 2
    if not has_teams_boilerplate:
        return shorten(raw)

    if re.match(r"^[_-]{10,}\s*$", raw.splitlines()[0]) or raw.lower().startswith("microsoft teams"):
        return "Microsoft Teams"

    divider_match = re.search(r"(^|\n)[_-]{10,}\n", raw)
    if divider_match:
        intro = raw[:divider_match.start()].strip()
        if intro:
            return shorten(intro)

    teams_heading = re.search(r"(^|\n)\s*Microsoft Teams", raw, re.IGNORECASE)
    if teams_heading:
        intro = raw[:teams_heading.start()].strip()
        if intro:
            return shorten(intro)

    return "Microsoft Teams"


def render_metadata(
    window: MatchWindow,
    candidates: list[Candidate],
    *,
    chosen_by_hand: bool = False,
) -> str:
    lines = [
        CALENDAR_METADATA_START,
        "## Calendar Metadata",
        f"- Transcript window: {format_dt(window.start)} to {format_dt(window.end)}",
    ]

    if candidates:
        best = candidates[0]
        lines.append(
            f"- Event time:        {format_dt(best.event_start)} to {format_dt(best.event_end)}"
        )

    lines.extend(
        [
        f"- Transcript duration: {int(window.duration.total_seconds() // 60)} minutes",
        ]
    )

    if not candidates:
        status = (
            "none of the calendar events (chosen by hand)"
            if chosen_by_hand
            else "no plausible calendar event found"
        )
        lines.append(f"- Match status: {status}")
    else:
        event = best.event
        organizer = event.get("organizer", {})
        location = clean_location(event.get("location"))
        lines.extend(
            [
                f"- Event: {event.get('summary') or '(untitled)'}",
                f"- Response: {response_label(best.response_status)}",
                f"- Organizer: {organizer.get('displayName') or organizer.get('email') or '-'}",
                f"- Attendees: {summarize_attendees(event.get('attendees', []), event.get('location'))}",
                f"- Location: {location}",
                f"- Description: {clean_description(event.get('description'))}",
            ]
        )

        alternatives = candidates[1:4]
        if alternatives:
            alt_text = "; ".join(
                f"{cand.event.get('summary') or '(untitled)'} [{format_dt(cand.event_start)}, {cand.score:.1f}]"
                for cand in alternatives
            )
            lines.append(f"- Alternatives: {alt_text}")

        if chosen_by_hand:
            lines.append("- Match status: chosen by hand")
        else:
            lines.append(
                f"- Match status: {confidence_label(best.score)} confidence (score {best.score:.1f})"
            )

    lines.append(CALENDAR_METADATA_END)
    return "\n".join(lines)


def update_transcript_markdown(path: Path, metadata_block: str) -> str:
    text = path.read_text()
    pattern = metadata_block_re(CALENDAR_METADATA_START, CALENDAR_METADATA_END)
    if pattern.search(text):
        updated = pattern.sub(metadata_block + "\n\n", text, count=1)
    else:
        lines = text.splitlines()
        insert_at = 1
        while insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1
        prefix = "\n".join(lines[:insert_at]).rstrip()
        suffix = "\n".join(lines[insert_at:]).lstrip()
        updated = f"{prefix}\n\n{metadata_block}\n\n{suffix}\n"
    path.write_text(updated)
    return updated


def collect_candidates(
    window: MatchWindow,
    calendars: list[dict[str, Any]],
    lookback: timedelta,
    lookahead: timedelta,
) -> list[Candidate]:
    time_min = window.start - lookback
    time_max = window.end + lookahead
    candidates: list[Candidate] = []
    for calendar in calendars:
        calendar_id = calendar["id"]
        try:
            events = list_events(calendar_id, time_min, time_max)
        except GwsError as exc:
            print(f"Warning: skipping calendar {calendar_id}: {exc}", file=sys.stderr)
            continue
        for event in events:
            candidate = score_event(event, calendar, window)
            if candidate is not None:
                candidates.append(candidate)
    return dedupe_candidates(candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match a transcript to a calendar event via gws and annotate the transcript"
    )
    parser.add_argument("transcript", nargs="?", help="Transcript .md file (default: latest)")
    parser.add_argument("--calendar", help="Restrict search to a single calendar ID, e.g. primary")
    parser.add_argument(
        "--include-shared-calendars",
        action="store_true",
        help="Also search readable shared/subscribed calendars (default: owned calendars only)",
    )
    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=DEFAULT_LOOKBACK.total_seconds() / 3600,
        help="How many hours before the transcript start to search (default: 4)",
    )
    parser.add_argument(
        "--lookahead-hours",
        type=float,
        default=DEFAULT_LOOKAHEAD.total_seconds() / 3600,
        help="How many hours after the transcript end to search (default: 6)",
    )
    parser.add_argument(
        "--rematch",
        action="store_true",
        help="Ask Google again instead of rendering the event stored in the session context",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the metadata block instead of modifying the transcript",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        transcript_path = resolve_transcript_md(args.transcript).resolve()
        window = load_transcript(transcript_path)
        stored = None if args.rematch else stored_candidates(transcript_path)
        chosen_by_hand = False
        if stored is not None:
            candidates, chosen_by_hand = stored
        else:
            candidates = match_window(
                window,
                calendar=args.calendar,
                include_shared=args.include_shared_calendars,
                lookback=timedelta(hours=args.lookback_hours),
                lookahead=timedelta(hours=args.lookahead_hours),
            )
        metadata_block = render_metadata(window, candidates, chosen_by_hand=chosen_by_hand)
    except GwsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(
            "gws not found on PATH. Install/configure the Google Workspace CLI first.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print(metadata_block)
        return 0

    update_transcript_markdown(transcript_path, metadata_block)
    if candidates:
        best = candidates[0]
        print(
            f"Annotated {transcript_path.name} with {best.event.get('summary') or '(untitled)'} "
            f"[{'chosen by hand' if chosen_by_hand else confidence_label(best.score) + ' confidence'}]"
        )
    else:
        found = "no event (chosen by hand)" if chosen_by_hand else "'no plausible match found'"
        print(f"Annotated {transcript_path.name} with {found}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
