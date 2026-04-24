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
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .pipeline import transcript_json_for_markdown
from .config import load_config

_cfg = load_config()
TRANSCRIPT_DIR = _cfg.transcripts_dir
METADATA_START = "<!-- calendar-metadata:start -->"
METADATA_END = "<!-- calendar-metadata:end -->"
TIMESTAMP_RE = re.compile(r"transcript-(\d{8}-\d{6})")
SEGMENT_TS_RE = re.compile(r"\*\*\[(\d{2}):(\d{2}):(\d{2})\]")
DEFAULT_GWS_CONFIG_DIR = _cfg.gws_config_dir or (Path.home() / ".config" / "gws")
DEFAULT_GWS_CREDENTIALS_FILE = DEFAULT_GWS_CONFIG_DIR / "credentials.json"
DEFAULT_GWS_BIN = _cfg.gws_bin

DEFAULT_LOOKBACK = timedelta(hours=4)
DEFAULT_LOOKAHEAD = timedelta(hours=6)
MAX_DESCRIPTION_LEN = 240
MAX_ATTENDEES = 8
MIN_OVERLAP_FRACTION = 0.5
NOISY_VIRTUAL_LOCATIONS = {
    "microsoft teams-möte",
    "microsoft teams meeting",
    "teams meeting",
}


def local_tzinfo():
    return datetime.now().astimezone().tzinfo


@dataclass
class TranscriptInfo:
    md_path: Path
    start: datetime
    end: datetime
    duration: timedelta
    text: str


@dataclass
class Candidate:
    calendar_id: str
    calendar_name: str
    event: dict[str, Any]
    event_start: datetime
    event_end: datetime
    score: float
    reasons: list[str]


class GwsError(RuntimeError):
    pass


def resolve_transcript(name: str | None) -> Path:
    if name is None:
        files = sorted(TRANSCRIPT_DIR.glob("transcript-*.md"))
        if not files:
            raise GwsError("No transcripts found.")
        return files[-1]

    path = Path(name)
    if path.exists():
        return path.resolve()
    if (TRANSCRIPT_DIR / path).exists():
        return (TRANSCRIPT_DIR / path).resolve()
    if (TRANSCRIPT_DIR / f"transcript-{path}").exists():
        return (TRANSCRIPT_DIR / f"transcript-{path}").resolve()
    raise GwsError(f"Transcript not found: {name}")


def transcript_start_from_name(path: Path) -> datetime:
    match = TIMESTAMP_RE.search(path.name)
    if not match:
        raise GwsError(
            f"Could not infer transcript timestamp from filename: {path.name}"
        )
    start = datetime.strptime(match.group(1), "%Y%m%d-%H%M%S")
    return start.replace(tzinfo=local_tzinfo())


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


def load_transcript(path: Path) -> TranscriptInfo:
    text = path.read_text()
    start = transcript_start_from_name(path)
    duration = transcript_duration(path, text)
    end = start + duration
    return TranscriptInfo(path, start, end, duration, text)


def gws_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("GOOGLE_WORKSPACE_CLI_CONFIG_DIR", str(DEFAULT_GWS_CONFIG_DIR))
    env.setdefault(
        "GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE", str(DEFAULT_GWS_CREDENTIALS_FILE)
    )
    return env


def run_gws(*args: str) -> dict[str, Any]:
    gws_bin = os.environ.get("HUGIN_GWS_BIN", DEFAULT_GWS_BIN)
    if shutil.which(gws_bin) is None:
        raise FileNotFoundError(gws_bin)

    result = subprocess.run(
        [gws_bin, *args],
        capture_output=True,
        text=True,
        env=gws_environment(),
    )

    stdout = result.stdout.strip()
    payload = None
    if stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None

    if result.returncode != 0:
        message = result.stderr.strip()
        if isinstance(payload, dict) and "error" in payload:
            message = payload["error"].get("message", message) or message
        raise GwsError(message or f"{gws_bin} {' '.join(args)} failed")

    if not isinstance(payload, dict):
        raise GwsError(f"Unexpected {gws_bin} output for {' '.join(args)}")

    if "error" in payload:
        raise GwsError(payload["error"].get("message", "gws returned an error"))

    return payload


def list_calendars(calendar_id: str | None) -> list[dict[str, Any]]:
    if calendar_id:
        return [{"id": calendar_id, "summary": calendar_id, "primary": calendar_id == "primary"}]

    payload = run_gws(
        "calendar",
        "calendarList",
        "list",
        "--params",
        json.dumps(
            {
                "showDeleted": False,
                "showHidden": False,
                "maxResults": 250,
                "minAccessRole": "reader",
            }
        ),
    )
    items = payload.get("items", [])
    return [item for item in items if not item.get("deleted")]


def list_events(
    calendar_id: str,
    time_min: datetime,
    time_max: datetime,
) -> list[dict[str, Any]]:
    payload = run_gws(
        "calendar",
        "events",
        "list",
        "--params",
        json.dumps(
            {
                "calendarId": calendar_id,
                "timeMin": time_min.isoformat(),
                "timeMax": time_max.isoformat(),
                "singleEvents": True,
                "orderBy": "startTime",
                "showDeleted": False,
                "maxResults": 250,
            }
        ),
    )
    return payload.get("items", [])


def parse_event_time(raw: dict[str, Any]) -> tuple[datetime | None, bool]:
    if "dateTime" in raw:
        value = raw["dateTime"].replace("Z", "+00:00")
        return datetime.fromisoformat(value), False
    if "date" in raw:
        value = datetime.fromisoformat(raw["date"])
        return value.replace(tzinfo=local_tzinfo()), True
    return None, False


def event_time_bounds(event: dict[str, Any]) -> tuple[datetime | None, datetime | None, bool]:
    start, start_all_day = parse_event_time(event.get("start", {}))
    end, end_all_day = parse_event_time(event.get("end", {}))
    return start, end, start_all_day or end_all_day


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
    transcript: TranscriptInfo,
) -> Candidate | None:
    if event.get("status") == "cancelled":
        return None

    event_start, event_end, is_all_day = event_time_bounds(event)
    if not event_start or not event_end or is_all_day:
        return None

    score = 0.0
    reasons: list[str] = []
    overlap = overlap_seconds(event_start, event_end, transcript.start, transcript.end)
    min_overlap = transcript.duration.total_seconds() * MIN_OVERLAP_FRACTION
    if overlap <= min_overlap:
        return None

    start_delta = abs((event_start - transcript.start).total_seconds()) / 60
    duration_delta = abs(
        ((event_end - event_start) - transcript.duration).total_seconds()
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
    transcript_tokens = tokenize(transcript.text[:8000])
    token_hits = len(summary_tokens & transcript_tokens)
    if token_hits:
        token_bonus = min(token_hits * 6.0, 24.0)
        score += token_bonus
        reasons.append(f"title keywords hit {token_hits}")

    attendees = event.get("attendees", [])
    if attendees:
        score += 4.0
        reasons.append(f"{len(attendees)} attendee(s)")

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
        score=score,
        reasons=reasons,
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
        if current is None or candidate.score > current.score:
            best[key] = candidate
    return sorted(best.values(), key=lambda item: item.score, reverse=True)


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
    transcript: TranscriptInfo,
    candidates: list[Candidate],
    searched_calendars: list[dict[str, Any]],
) -> str:
    lines = [
        METADATA_START,
        "## Calendar Metadata",
        f"- Transcript window: {format_dt(transcript.start)} to {format_dt(transcript.end)}",
    ]

    if candidates:
        best = candidates[0]
        lines.append(
            f"- Event time:        {format_dt(best.event_start)} to {format_dt(best.event_end)}"
        )

    lines.extend(
        [
        f"- Transcript duration: {int(transcript.duration.total_seconds() // 60)} minutes",
        ]
    )

    if not candidates:
        lines.extend(
            [
                "- Match status: no plausible calendar event found",
            ]
        )
    else:
        event = best.event
        organizer = event.get("organizer", {})
        location = clean_location(event.get("location"))
        lines.extend(
            [
                f"- Event: {event.get('summary') or '(untitled)'}",
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

        lines.append(
            f"- Match status: {confidence_label(best.score)} confidence (score {best.score:.1f})"
        )

    lines.append(METADATA_END)
    return "\n".join(lines)


def update_transcript_markdown(path: Path, metadata_block: str) -> str:
    text = path.read_text()
    pattern = re.compile(
        rf"{re.escape(METADATA_START)}.*?{re.escape(METADATA_END)}\n*",
        re.DOTALL,
    )
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
    transcript: TranscriptInfo,
    calendars: list[dict[str, Any]],
    lookback: timedelta,
    lookahead: timedelta,
) -> list[Candidate]:
    time_min = transcript.start - lookback
    time_max = transcript.end + lookahead
    candidates: list[Candidate] = []
    for calendar in calendars:
        calendar_id = calendar["id"]
        try:
            events = list_events(calendar_id, time_min, time_max)
        except GwsError as exc:
            print(f"Warning: skipping calendar {calendar_id}: {exc}", file=sys.stderr)
            continue
        for event in events:
            candidate = score_event(event, calendar, transcript)
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
        "--dry-run",
        action="store_true",
        help="Print the metadata block instead of modifying the transcript",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        transcript_path = resolve_transcript(args.transcript)
        transcript = load_transcript(transcript_path)
        calendars = list_calendars(args.calendar)
        candidates = collect_candidates(
            transcript,
            calendars,
            lookback=timedelta(hours=args.lookback_hours),
            lookahead=timedelta(hours=args.lookahead_hours),
        )
        metadata_block = render_metadata(transcript, candidates, calendars)
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

    update_transcript_markdown(transcript.md_path, metadata_block)
    if candidates:
        best = candidates[0]
        print(
            f"Annotated {transcript.md_path.name} with {best.event.get('summary') or '(untitled)'} "
            f"[{confidence_label(best.score)} confidence]"
        )
    else:
        print(f"Annotated {transcript.md_path.name} with 'no plausible match found'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
