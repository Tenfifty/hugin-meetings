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

    def test_missing_context_is_none(self) -> None:
        self.assertIsNone(meeting_context.load_context("20260101-000000"))

    def test_corrupt_context_is_none_not_a_crash(self) -> None:
        (self.dir / "context-20260824-140104.json").write_text("{not json")
        self.assertIsNone(meeting_context.load_context("20260824-140104"))

    def test_language_falls_back_when_unset(self) -> None:
        empty = meeting_context.MeetingContext(session_id="20260824-140104")
        self.assertEqual(empty.language_value, meeting_context.load_config().language)


if __name__ == "__main__":
    unittest.main()
