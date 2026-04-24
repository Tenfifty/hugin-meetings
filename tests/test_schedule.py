from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugin_meetings import schedule


class JournalMeetingTests(unittest.TestCase):
    def test_load_todays_journal_meetings_parses_supported_time_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "journal.md"
            journal.write_text(
                "## <2026-04-23 Thu>\n"
                "- [ ] *{09:00 - 09:30}* Yesterday\n"
                "## <2026-04-24 Fri>\n"
                "- [ ] *{09:00 - 09:30}* Morning sync\n"
                "- [ ] [10.15-11.00] Legacy format\n"
                "- [ ] *~{12:00 - 12:30}* Ignored meeting\n"
                "- [ ] *{13:00 - 18:30}* Too long\n",
                encoding="utf-8",
            )

            meetings = schedule.load_todays_journal_meetings(journal, date(2026, 4, 24))

        self.assertEqual([meeting.title for meeting in meetings], ["Morning sync", "Legacy format"])
        self.assertEqual(meetings[0].time_label, "09:00 - 09:30")
        self.assertEqual(meetings[1].start_at, datetime(2026, 4, 24, 10, 15))


class ReminderTests(unittest.TestCase):
    def test_start_and_stop_reminder_candidates_honor_prompt_state(self) -> None:
        meeting = schedule.ScheduledMeeting(
            key="m1",
            title="Planning",
            start_at=datetime(2026, 4, 24, 10, 0),
            end_at=datetime(2026, 4, 24, 10, 30),
            source_line="- [ ] *{10:00 - 10:30}* Planning",
        )
        state = schedule.default_reminder_state(date(2026, 4, 24))

        self.assertEqual(
            schedule.start_reminder_candidate(
                [meeting],
                state,
                datetime(2026, 4, 24, 10, 5),
                is_recording=False,
            ),
            meeting,
        )

        state, changed = schedule.mark_prompted(state, "start", meeting.key)
        self.assertTrue(changed)
        self.assertIsNone(
            schedule.start_reminder_candidate(
                [meeting],
                state,
                datetime(2026, 4, 24, 10, 6),
                is_recording=False,
            )
        )

        state = schedule.set_recording_meeting(state, meeting.key)
        self.assertEqual(
            schedule.stop_reminder_candidate(
                {meeting.key: meeting},
                state,
                datetime(2026, 4, 24, 10, 31),
                is_recording=True,
            ),
            meeting,
        )

    def test_associate_current_recording_requires_one_candidate(self) -> None:
        first = schedule.ScheduledMeeting(
            key="m1",
            title="One",
            start_at=datetime(2026, 4, 24, 10, 0),
            end_at=datetime(2026, 4, 24, 10, 30),
            source_line="",
        )
        second = schedule.ScheduledMeeting(
            key="m2",
            title="Two",
            start_at=datetime(2026, 4, 24, 10, 4),
            end_at=datetime(2026, 4, 24, 10, 30),
            source_line="",
        )
        state = schedule.default_reminder_state(date(2026, 4, 24))

        self.assertEqual(
            schedule.associate_current_recording(
                [first],
                state,
                datetime(2026, 4, 24, 10, 5),
                is_recording=True,
            ),
            first,
        )
        self.assertIsNone(
            schedule.associate_current_recording(
                [first, second],
                state,
                datetime(2026, 4, 24, 10, 5),
                is_recording=True,
            )
        )

    def test_next_meeting_label_skips_past_entries(self) -> None:
        past = schedule.ScheduledMeeting(
            key="past",
            title="Past",
            start_at=datetime(2026, 4, 24, 9, 0),
            end_at=datetime(2026, 4, 24, 9, 30),
            source_line="",
        )
        upcoming = schedule.ScheduledMeeting(
            key="next",
            title="Next",
            start_at=datetime(2026, 4, 24, 11, 0),
            end_at=None,
            source_line="",
        )

        self.assertEqual(
            schedule.next_meeting_label(
                [past, upcoming],
                datetime(2026, 4, 24, 10, 0),
            ),
            "11:00 Next",
        )


if __name__ == "__main__":
    unittest.main()
