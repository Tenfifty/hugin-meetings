from __future__ import annotations

import json
import subprocess
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

    def test_owner_access_to_another_persons_calendar_is_not_owned(self) -> None:
        # Google reports accessRole "owner" when a colleague shares their
        # personal calendar with manage rights; that calendar is not ours and
        # must stay out of the default match. A group calendar we manage stays.
        payload = {
            "items": [
                {"id": "primary", "summary": "Me", "primary": True, "accessRole": "owner"},
                {
                    "id": "anders.bjurstrom@example.com",
                    "summary": "Anders",
                    "accessRole": "owner",
                },
                {
                    "id": "team_abc123@group.calendar.google.com",
                    "summary": "Absence",
                    "accessRole": "owner",
                },
            ]
        }

        with patch.object(calendar_match, "run_gws", return_value=payload):
            calendars = calendar_match.list_calendars(None)

        self.assertEqual(
            [calendar["id"] for calendar in calendars],
            ["primary", "team_abc123@group.calendar.google.com"],
        )

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


class ExternalAttendeeTests(unittest.TestCase):
    ME = {"email": "david.fendrich@tenfifty.io", "self": True, "responseStatus": "accepted"}
    ROOM = {"email": "c_188da@resource.calendar.google.com", "resource": True}
    # 2026-10-15: the room booked for a customer meeting scored 2 points higher
    # than the meeting itself, for having a location.
    ROOM_BOOKING = {
        "summary": "Rumsbokning, AKA",
        "organizer": {"email": "david.fendrich@tenfifty.io", "self": True},
        "attendees": [ROOM, ME, {"email": "filip.wikman@tenfifty.io"}],
    }
    CUSTOMER_MEETING = {
        "summary": "AKA - Produktutvecklingsmöte",
        "organizer": {"email": "benjamin.kevin.samuelsson@gmail.com"},
        "attendees": [
            ME,
            {"email": "filip.wikman@tenfifty.io"},
            {"email": "par.h.blixt@gmail.com"},
            {"email": "kristian@samuelsson.cc"},
        ],
    }

    def candidate(self, event, score, response_status="accepted"):
        candidate = make_candidate(event["summary"], response_status, score)
        candidate.event = event
        return candidate

    def test_colleagues_and_rooms_are_not_external(self) -> None:
        self.assertEqual(calendar_match.external_attendees(self.ROOM_BOOKING, "primary"), [])
        self.assertEqual(
            calendar_match.external_attendees(self.CUSTOMER_MEETING, "primary"),
            ["par.h.blixt@gmail.com", "kristian@samuelsson.cc"],
        )

    def test_a_primary_calendar_id_gives_the_own_domain_when_no_entry_is_marked(self) -> None:
        event = {"attendees": [{"email": "a@tenfifty.io"}, {"email": "b@aka.se"}]}
        self.assertEqual(
            calendar_match.external_attendees(event, "david.fendrich@tenfifty.io"), ["b@aka.se"]
        )
        self.assertEqual(
            calendar_match.external_attendees(event, "team@group.calendar.google.com"), []
        )

    def test_an_external_meeting_outranks_a_better_scoring_internal_one(self) -> None:
        room = self.candidate(self.ROOM_BOOKING, 140.4)
        meeting = self.candidate(self.CUSTOMER_MEETING, 138.4)

        self.assertEqual(calendar_match.dedupe_candidates([room, meeting]), [meeting, room])

    def test_the_answer_to_the_invitation_still_comes_first(self) -> None:
        room = self.candidate(self.ROOM_BOOKING, 140.4)
        declined = self.candidate(self.CUSTOMER_MEETING, 138.4, response_status="declined")

        self.assertEqual(calendar_match.dedupe_candidates([declined, room]), [room, declined])


class GwsErrorTests(unittest.TestCase):
    # What gws printed on 2026-10-15, seven seconds after the lid closed.
    OFFLINE = (
        "Authentication failed: Failed to get token: Connection failure: Hyper error: "
        "client error (Connect): dns error: failed to lookup address information"
    )

    def run_gws_failing_with(self, stderr: str) -> None:
        failed = subprocess.CompletedProcess(["gws"], 1, stdout="", stderr=stderr)
        with patch.object(calendar_match.shutil, "which", return_value="/usr/bin/gws"), patch.object(
            calendar_match.subprocess, "run", return_value=failed
        ):
            calendar_match.run_gws("calendar", "calendarList", "list")

    def test_no_network_is_told_apart_so_it_can_be_retried(self) -> None:
        with self.assertRaises(calendar_match.GwsOfflineError):
            self.run_gws_failing_with(self.OFFLINE)

    def test_a_refusal_is_not_mistaken_for_no_network(self) -> None:
        with self.assertRaises(calendar_match.GwsError) as caught:
            self.run_gws_failing_with("Authentication failed: invalid_grant")
        self.assertNotIsInstance(caught.exception, calendar_match.GwsOfflineError)


if __name__ == "__main__":
    unittest.main()
