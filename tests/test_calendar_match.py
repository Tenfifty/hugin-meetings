from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugin_meetings import calendar_match


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


if __name__ == "__main__":
    unittest.main()
