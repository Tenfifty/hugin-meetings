#!/usr/bin/env python3
"""Meeting context — the first pipeline stage, before transcription.

Everything downstream used to discover what a meeting *was* only after it had
been transcribed and summarized: the calendar block was written into the
finished transcript, and the customer was guessed from the summary text, days
later. That is backwards. The calendar event and the spoken language are both
knowable the moment recording stops, and both are things the later stages want
as input rather than produce as output.

So this stage runs first and writes one ``context-{ts}.json`` per session:
what language to transcribe in, which calendar event this was, which customer
it belongs to. It is deterministic to re-run and cheap (a few seconds), and it
is meant to be *verified by a human* before the expensive stages run — hence
``verified``, which the TUI sets and the CLI never does.

The guess is always usable unverified; nothing here blocks on a person.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import langid, pipeline
from .config import load_config

CONTEXT_VERSION = 1


def context_path(ts: str) -> Path:
    return load_config().transcript_json_dir / pipeline.year_subdir(ts) / f"context-{ts}.json"


@dataclass
class MeetingContext:
    """What the later stages need to know, decided up front."""

    session_id: str
    language: dict[str, Any] = field(default_factory=dict)
    calendar: dict[str, Any] | None = None
    customer: dict[str, Any] | None = None
    note: str = ""
    verified: bool = False
    verified_at: str = ""
    applied: dict[str, Any] | None = None
    version: int = CONTEXT_VERSION

    @property
    def language_value(self) -> str:
        return self.language.get("value") or load_config().language

    @property
    def customer_label(self) -> str:
        if not self.customer:
            return "(none)"
        state = pipeline.deserialize_customer_state(self.customer)
        return state.label

    @property
    def event_title(self) -> str:
        return (self.calendar or {}).get("summary") or ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "language": self.language,
            "calendar": self.calendar,
            "customer": self.customer,
            "note": self.note,
            "verified": self.verified,
            "verified_at": self.verified_at,
            "applied": self.applied,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MeetingContext":
        return cls(
            session_id=payload.get("session_id", ""),
            language=payload.get("language") or {},
            calendar=payload.get("calendar"),
            customer=payload.get("customer"),
            note=payload.get("note") or "",
            verified=bool(payload.get("verified")),
            verified_at=payload.get("verified_at") or "",
            applied=payload.get("applied"),
            version=int(payload.get("version", CONTEXT_VERSION)),
        )


def load_context(ts: str) -> MeetingContext | None:
    path = context_path(ts)
    if not path.exists():
        return None
    try:
        return MeetingContext.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, ValueError):
        return None


def save_context(context: MeetingContext) -> Path:
    path = context_path(context.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(context.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return path


def mark_verified(context: MeetingContext) -> MeetingContext:
    """A person has looked at this and said yes. Only a UI should call it."""
    context.verified = True
    context.verified_at = datetime.now().astimezone().isoformat(timespec="seconds")
    return context


def _guess_calendar(session: pipeline.RawAudioSession) -> tuple[dict[str, Any] | None, str]:
    from . import calendar_match

    try:
        window = calendar_match.session_window(session)
        candidates, _ = calendar_match.match_window(window)
    except calendar_match.GwsError as exc:
        return None, f"calendar lookup failed: {exc}"
    except FileNotFoundError:
        return None, "gws not found on PATH"
    if not candidates:
        return None, "no plausible calendar event"
    return calendar_match.event_fields(candidates[0]), ""


def _calendar_lines(event: dict[str, Any] | None) -> dict[str, str]:
    """The event as the matcher prompt wants it: the same keys the transcript
    metadata block uses, so both stages present the same shape to the model."""
    if not event:
        return {}
    return {
        "Event": event.get("summary") or "(untitled)",
        "Event time": f"{event.get('start', '')} to {event.get('end', '')}",
        "Response": event.get("response", ""),
        "Organizer": event.get("organizer", ""),
        "Attendees": ", ".join(event.get("attendees") or []),
        "Location": event.get("location", ""),
        "Description": event.get("description", ""),
    }


def _guess_customer(event: dict[str, Any] | None, model: str) -> dict[str, Any] | None:
    if not event:
        return None
    decision = pipeline.suggest_customer_from_calendar(_calendar_lines(event), model)
    state = pipeline.state_from_decision(decision, source="calendar")
    return pipeline.serialize_customer_state(state)


def build_context(
    session: pipeline.RawAudioSession,
    *,
    model: str | None = None,
    skip_calendar: bool = False,
    skip_customer: bool = False,
) -> MeetingContext:
    """Guess everything the later stages need. Never interactive, never fatal.

    A stage that cannot answer (no calendar event, gws offline) leaves its field
    empty rather than failing the session — an unverified gap is something the
    operator can fill in, an exception is not.
    """
    context = MeetingContext(session_id=session.session_id)

    result = langid.identify_session(session.mic_parts, session.sys_parts)
    context.language = result.to_dict()

    if not skip_calendar:
        context.calendar, context.note = _guess_calendar(session)

    # No event means no domains, no title and no description — the matcher would
    # be guessing from nothing, so leave the customer for the operator instead.
    if context.calendar and not skip_customer:
        context.customer = _guess_customer(
            context.calendar, model or pipeline.DEFAULT_CUSTOMER_MODEL
        )
    return context


def ensure_context(ts: str, **kwargs: Any) -> MeetingContext:
    """Load the session's context, building and saving one if it is missing."""
    existing = load_context(ts)
    if existing is not None:
        return existing
    session = pipeline.scan_raw_audio_sessions().get(ts)
    if session is None:
        raise FileNotFoundError(f"No raw audio for session {ts}")
    context = build_context(session, **kwargs)
    save_context(context)
    return context


def main() -> int:
    parser = argparse.ArgumentParser(description="Guess language, calendar event and customer for a session")
    parser.add_argument("session", nargs="*", help="Session ids (default: the latest -n)")
    parser.add_argument("-n", type=int, default=1, help="Number of latest sessions")
    parser.add_argument("--force", action="store_true", help="Rebuild even if a context exists")
    parser.add_argument("--model", help=f"Matcher model (default: {pipeline.DEFAULT_CUSTOMER_MODEL})")
    parser.add_argument("--no-calendar", action="store_true", help="Skip the calendar lookup")
    parser.add_argument("--no-customer", action="store_true", help="Skip the customer guess")
    parser.add_argument("--json", action="store_true", help="Print the full context objects")
    args = parser.parse_args()

    sessions = pipeline.scan_raw_audio_sessions()
    wanted = args.session or sorted(sessions, reverse=True)[: args.n]

    contexts = []
    for ts in wanted:
        session = sessions.get(ts)
        if session is None:
            print(f"{ts}: no raw audio", file=sys.stderr)
            continue
        context = None if args.force else load_context(ts)
        if context is None:
            context = build_context(
                session,
                model=args.model,
                skip_calendar=args.no_calendar,
                skip_customer=args.no_customer,
            )
            save_context(context)
        contexts.append(context)
        if not args.json:
            mark = "verified" if context.verified else "unverified"
            print(
                f"{ts}  {context.language_value:2s}  {context.customer_label:24s} "
                f"{context.event_title[:40]:40s}  {mark}"
            )

    if args.json:
        print(json.dumps([context.to_dict() for context in contexts], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
