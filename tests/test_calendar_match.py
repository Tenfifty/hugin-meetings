from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugin_meetings import calendar_match


def make_candidate(
    summary: str,
    response_status: str,
    score: float,
) -> calendar_match.Candidate:
    start = datetime(2026, 4, 24, 10, tzinfo=timezone.utc)
    end = datetime(2026, 4, 24, 11, tzinfo=timezone.utc)
    return calendar_match.Candidate(
        calendar_id="primary",
        calendar_name="Me",
        event={"summary": summary, "iCalUID": f"{summary}@example.com"},
        event_start=start,
        event_end=end,
        response_status=response_status,
        score=score,
        reasons=[],
    )


class CalendarListTests(unittest.TestCase):
    def test_default_calendar_list_keeps_only_owned_calendars(self) -> None:
        payload = {
            "items": [
                {
                    "id": "primary",
                    "summary": "Me",
                    "primary": True,
                    "accessRole": "owner",
                },
                {
                    "id": "team@example.com",
                    "summary": "Team shared",
                    "accessRole": "reader",
                },
                {
                    "id": "delegated@example.com",
                    "summary": "Delegated",
                    "accessRole": "writer",
                },
                {
                    "id": "old@example.com",
                    "summary": "Deleted",
                    "accessRole": "owner",
                    "deleted": True,
                },
            ]
        }

        with patch.object(calendar_match, "run_gws", return_value=payload) as run_gws:
            calendars = calendar_match.list_calendars(None)

        self.assertEqual([calendar["id"] for calendar in calendars], ["primary"])
        params = json.loads(run_gws.call_args.args[-1])
        self.assertEqual(params["minAccessRole"], "owner")

    def test_include_shared_calendars_preserves_readable_calendars(self) -> None:
        payload = {
            "items": [
                {"id": "primary", "summary": "Me", "primary": True, "accessRole": "owner"},
                {"id": "team@example.com", "summary": "Team shared", "accessRole": "reader"},
            ]
        }

        with patch.object(calendar_match, "run_gws", return_value=payload) as run_gws:
            calendars = calendar_match.list_calendars(
                None,
                include_shared_calendars=True,
            )

        self.assertEqual(
            [calendar["id"] for calendar in calendars],
            ["primary", "team@example.com"],
        )
        params = json.loads(run_gws.call_args.args[-1])
        self.assertEqual(params["minAccessRole"], "reader")

    def test_explicit_calendar_bypasses_default_filter(self) -> None:
        calendars = calendar_match.list_calendars("team@example.com")

        self.assertEqual(
            calendars,
            [
                {
                    "id": "team@example.com",
                    "summary": "team@example.com",
                    "primary": False,
                }
            ],
        )


class ResponseStatusTests(unittest.TestCase):
    def test_event_response_status_uses_self_attendee(self) -> None:
        event = {
            "attendees": [
                {"email": "someone@example.com", "responseStatus": "accepted"},
                {
                    "email": "me@example.com",
                    "self": True,
                    "responseStatus": "tentative",
                },
            ]
        }

        self.assertEqual(calendar_match.event_self_response_status(event), "tentative")

    def test_response_status_aliases_normalize_to_google_statuses(self) -> None:
        self.assertEqual(calendar_match.normalize_response_status("yes"), "accepted")
        self.assertEqual(calendar_match.normalize_response_status("maybe"), "tentative")
        self.assertEqual(
            calendar_match.normalize_response_status("not answered"),
            "needsAction",
        )
        self.assertEqual(calendar_match.normalize_response_status("no"), "declined")

    def test_owned_non_invite_events_count_as_accepted(self) -> None:
        self.assertEqual(
            calendar_match.event_self_response_status({"organizer": {"self": True}}),
            "accepted",
        )
        self.assertEqual(calendar_match.event_self_response_status({}), "accepted")

    def test_candidates_sort_by_response_status_before_score(self) -> None:
        candidates = calendar_match.dedupe_candidates(
            [
                make_candidate("no", "declined", 500.0),
                make_candidate("not answered", "needsAction", 400.0),
                make_candidate("maybe", "tentative", 20.0),
                make_candidate("yes", "accepted", 15.0),
            ]
        )

        self.assertEqual(
            [candidate.response_status for candidate in candidates],
            ["accepted", "tentative", "needsAction", "declined"],
        )

    def test_dedupe_prefers_better_response_status_before_score(self) -> None:
        worse_response = make_candidate("same", "declined", 500.0)
        better_response = make_candidate("same", "accepted", 15.0)

        candidates = calendar_match.dedupe_candidates([worse_response, better_response])

        self.assertEqual(candidates, [better_response])


if __name__ == "__main__":
    unittest.main()
