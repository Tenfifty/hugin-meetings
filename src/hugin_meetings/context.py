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
    customer_note: str = ""
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
        candidate = best_candidate(self)
        return (candidate.event.get("summary") or "") if candidate else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "language": self.language,
            "calendar": self.calendar,
            "customer": self.customer,
            "note": self.note,
            "customer_note": self.customer_note,
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
            customer_note=payload.get("customer_note") or "",
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


def verify(
    context: MeetingContext,
    *,
    language: str | None = None,
    customer: pipeline.CustomerState | None = None,
) -> MeetingContext:
    """Record a person's decision about a session, and act on it.

    This is the single write point for a verified meeting: it stamps the
    context and materializes the customer state that every later stage reads.
    Creating a customer note happens here, as the direct consequence of a
    person choosing to create one — never as a side effect further down.
    """
    if language and language != context.language.get("value"):
        context.language = {
            **context.language,
            "value": language,
            "source": "manual",
        }
    if customer is not None:
        customer.verified = True
        customer = pipeline.materialize_verified_customer_state(customer)
        context.customer = pipeline.serialize_customer_state(customer)
        pipeline.save_customer_state(context.session_id, customer)
    mark_verified(context)
    save_context(context)
    return context


def customer_state(context: MeetingContext) -> pipeline.CustomerState | None:
    if not context.customer:
        return None
    return pipeline.deserialize_customer_state(context.customer)


def record_applied(ts: str, **fields: Any) -> MeetingContext | None:
    """Note what a stage actually used, so a later change shows up as a drift.

    The transcript JSON is a bare list read positionally in several modules and
    cannot carry metadata, so the context keeps this instead.
    """
    context = load_context(ts)
    if context is None:
        return None
    context.applied = {**(context.applied or {}), **fields}
    save_context(context)
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
    # Keep the alternatives too: they are what the metadata block lists, and
    # what a person needs when the best match is the wrong meeting.
    return {
        "window": calendar_match.serialize_window(window),
        "candidates": [calendar_match.serialize_candidate(c) for c in candidates[:4]],
    }, ""


def best_candidate(context: "MeetingContext"):
    from . import calendar_match

    payload = (context.calendar or {}).get("candidates") or []
    if not payload:
        return None
    return calendar_match.deserialize_candidate(payload[0])


def calendar_block(context: "MeetingContext") -> str:
    """This session's metadata block, rendered from the stored match."""
    from . import calendar_match

    payload = context.calendar or {}
    if not payload.get("window") or not payload.get("candidates"):
        return ""
    window = calendar_match.deserialize_window(payload["window"])
    candidates = [calendar_match.deserialize_candidate(c) for c in payload["candidates"]]
    return calendar_match.render_metadata(window, candidates, [])


def calendar_fields(context: "MeetingContext") -> dict[str, str]:
    """The event as key/value pairs, for the matcher prompt and for display."""
    from . import calendar_match

    block = calendar_block(context)
    return calendar_match.metadata_fields(block) if block else {}


def _guess_customer(fields: dict[str, str], model: str) -> tuple[dict[str, Any] | None, str]:
    if not fields:
        return None, "no calendar event to match on"
    try:
        decision = pipeline.suggest_customer_from_calendar(fields, model)
    except FileNotFoundError as exc:
        # The matcher CLI is not on PATH. Common when a context guess is spawned
        # from a desktop session rather than a login shell - see llm.codex_bin.
        return None, f"matcher not found: {exc.filename or exc}"
    except Exception as exc:
        return None, f"customer guess failed: {exc}"
    state = pipeline.state_from_decision(decision, source="calendar")
    return pipeline.serialize_customer_state(state), ""


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
        context.customer, context.customer_note = _guess_customer(
            calendar_fields(context), model or pipeline.DEFAULT_CUSTOMER_MODEL
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
