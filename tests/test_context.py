from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugin_meetings import context as meeting_context


class ContextRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        patcher = patch.object(
            meeting_context,
            "context_path",
            side_effect=lambda ts: self.dir / f"context-{ts}.json",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def make(self) -> meeting_context.MeetingContext:
        return meeting_context.MeetingContext(
            session_id="20260824-140104",
            language={"value": "sv", "confidence": 0.98, "source": "langid", "probes": []},
            calendar={
                "window": {
                    "start": "2026-08-24T14:01:04+02:00",
                    "end": "2026-08-24T14:35:04+02:00",
                    "duration_seconds": 2040.0,
                },
                "candidates": [
                    {
                        "calendar_id": "primary",
                        "calendar_name": "Me",
                        "event": {
                            "summary": "Notana och AI",
                            "attendees": [{"email": "amer@notanacare.io"}],
                        },
                        "event_start": "2026-08-24T14:00:00+02:00",
                        "event_end": "2026-08-24T14:30:00+02:00",
                        "response_status": "accepted",
                        "score": 138.4,
                        "reasons": ["overlap 29m"],
                    }
                ],
            },
            customer={
                "action": "link_existing",
                "confidence": "high",
                "rationale": "Event names the customer.",
                "model": "gpt-5.6-luna",
                "verified": False,
                "customer_name": "Notana Care",
                "customer_path": None,
                "suggested_name": None,
                "source": "calendar",
            },
        )

    def test_save_and_load_round_trip(self) -> None:
        saved = meeting_context.save_context(self.make())
        loaded = meeting_context.load_context("20260824-140104")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.language_value, "sv")
        self.assertEqual(loaded.event_title, "Notana och AI")
        block = meeting_context.calendar_block(loaded)
        self.assertIn("- Event: Notana och AI", block)
        self.assertEqual(
            meeting_context.calendar_fields(loaded)["Event"], "Notana och AI"
        )
        self.assertFalse(loaded.verified)
        self.assertEqual(json.loads(saved.read_text())["version"], meeting_context.CONTEXT_VERSION)

    def test_unverified_customer_label_is_marked(self) -> None:
        """The TUI leans on this to show what still needs a human."""
        self.assertEqual(self.make().customer_label, "??Notana Care??")

    def test_mark_verified_stamps_a_time(self) -> None:
        verified = meeting_context.mark_verified(self.make())
        self.assertTrue(verified.verified)
        self.assertTrue(verified.verified_at)
        meeting_context.save_context(verified)
        self.assertTrue(meeting_context.load_context("20260824-140104").verified)

    def test_an_unprocessed_meeting_is_titled_by_its_event(self) -> None:
        """The list should name the meeting before there is a transcript to name it."""
        from hugin_meetings import pipeline

        rec = pipeline.MeetingStatus(
            timestamp="20260824-140104",
            mic_path=None,
            sys_path=None,
            mic_parts=(),
            sys_parts=(),
            transcript_json=None,
            transcript_md=None,
            summary_md=None,
            calendar_fields={},
            customer_state=None,
            anonymous_speakers=[],
            context=self.make(),
        )
        self.assertEqual(rec.title, "Notana och AI")

    def test_missing_context_is_none(self) -> None:
        self.assertIsNone(meeting_context.load_context("20260101-000000"))

    def test_corrupt_context_is_none_not_a_crash(self) -> None:
        (self.dir / "context-20260824-140104.json").write_text("{not json")
        self.assertIsNone(meeting_context.load_context("20260824-140104"))

    def test_language_falls_back_when_unset(self) -> None:
        empty = meeting_context.MeetingContext(session_id="20260824-140104")
        self.assertEqual(empty.language_value, meeting_context.load_config().language)


class CustomerGuessTests(unittest.TestCase):
    """A failed guess must not cost the session its context."""

    FIELDS = {"Event": "Notana och AI", "Attendees": "amer@notanacare.io"}

    def test_a_missing_matcher_binary_leaves_the_customer_empty(self) -> None:
        with patch.object(
            meeting_context.pipeline,
            "suggest_customer_from_calendar",
            side_effect=FileNotFoundError(2, "No such file or directory", "codex"),
        ):
            customer, note = meeting_context._guess_customer(self.FIELDS, "m")

        self.assertIsNone(customer)
        self.assertEqual(note, "matcher not found: codex")

    def test_any_other_matcher_failure_is_recorded_not_raised(self) -> None:
        with patch.object(
            meeting_context.pipeline,
            "suggest_customer_from_calendar",
            side_effect=RuntimeError("codex prompt failed"),
        ):
            customer, note = meeting_context._guess_customer(self.FIELDS, "m")

        self.assertIsNone(customer)
        self.assertEqual(note, "customer guess failed: codex prompt failed")

    def test_no_calendar_event_says_so(self) -> None:
        customer, note = meeting_context._guess_customer({}, "m")
        self.assertIsNone(customer)
        self.assertEqual(note, "no calendar event to match on")

    def test_the_note_survives_a_round_trip(self) -> None:
        ctx = meeting_context.MeetingContext(
            session_id="20260825-082229", customer_note="matcher not found: codex"
        )
        restored = meeting_context.MeetingContext.from_dict(ctx.to_dict())
        self.assertEqual(restored.customer_note, "matcher not found: codex")


if __name__ == "__main__":
    unittest.main()
