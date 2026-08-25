from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
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
        prompt = schedule.stop_reminder_candidate(
            {meeting.key: meeting},
            state,
            datetime(2026, 4, 24, 10, 31),
            is_recording=True,
        )
        self.assertEqual(prompt.meeting, meeting)
        self.assertEqual(prompt.index, 0)
        self.assertEqual(prompt.detail, "Scheduled end: 10:30")

        state, changed = schedule.mark_prompted(state, "stop", prompt.state_key)
        self.assertTrue(changed)
        self.assertIsNone(
            schedule.stop_reminder_candidate(
                {meeting.key: meeting},
                state,
                datetime(2026, 4, 24, 10, 45),
                is_recording=True,
            )
        )

    def test_stop_reminder_repeats_every_half_hour_after_the_first(self) -> None:
        meeting = schedule.ScheduledMeeting(
            key="m1",
            title="Planning",
            start_at=datetime(2026, 4, 24, 10, 0),
            end_at=datetime(2026, 4, 24, 10, 30),
            source_line="",
        )
        index = {meeting.key: meeting}
        state = schedule.set_recording_meeting(
            schedule.default_reminder_state(date(2026, 4, 24)), meeting.key
        )

        # Answering "no" to the first prompt is not the end of it.
        first = schedule.stop_reminder_candidate(
            index, state, datetime(2026, 4, 24, 10, 30), is_recording=True
        )
        state, _ = schedule.mark_prompted(state, "stop", first.state_key)

        self.assertIsNone(
            schedule.stop_reminder_candidate(
                index, state, datetime(2026, 4, 24, 10, 59), is_recording=True
            )
        )
        second = schedule.stop_reminder_candidate(
            index, state, datetime(2026, 4, 24, 11, 0), is_recording=True
        )
        self.assertEqual(second.index, 1)
        self.assertEqual(second.due_at, datetime(2026, 4, 24, 11, 0))
        self.assertEqual(
            second.detail, "Scheduled end: 10:30 - still recording, 30 min past due"
        )

    def test_stop_reminder_asks_once_for_a_long_forgotten_recording(self) -> None:
        meeting = schedule.ScheduledMeeting(
            key="m1",
            title="Planning",
            start_at=datetime(2026, 4, 24, 10, 0),
            end_at=datetime(2026, 4, 24, 10, 30),
            source_line="",
        )
        index = {meeting.key: meeting}
        state = schedule.set_recording_meeting(
            schedule.default_reminder_state(date(2026, 4, 24)), meeting.key
        )

        prompt = schedule.stop_reminder_candidate(
            index, state, datetime(2026, 4, 24, 12, 40), is_recording=True
        )
        self.assertEqual(prompt.index, 4)
        self.assertEqual(prompt.overdue, timedelta(hours=2))

        # The four missed deadlines do not queue up behind it.
        state, _ = schedule.mark_prompted(state, "stop", prompt.state_key)
        self.assertIsNone(
            schedule.stop_reminder_candidate(
                index, state, datetime(2026, 4, 24, 12, 41), is_recording=True
            )
        )

    def test_stop_reminder_for_a_meeting_without_an_end_time(self) -> None:
        meeting = schedule.ScheduledMeeting(
            key="m1",
            title="Open ended",
            start_at=datetime(2026, 4, 24, 10, 0),
            end_at=None,
            source_line="",
        )
        index = {meeting.key: meeting}
        state = schedule.set_recording_meeting(
            schedule.default_reminder_state(date(2026, 4, 24)), meeting.key
        )

        self.assertIsNone(
            schedule.stop_reminder_candidate(
                index, state, datetime(2026, 4, 24, 10, 29), is_recording=True
            )
        )
        prompt = schedule.stop_reminder_candidate(
            index, state, datetime(2026, 4, 24, 10, 30), is_recording=True
        )
        self.assertEqual(prompt.index, 0)
        self.assertEqual(
            prompt.detail, "No end time in the journal; started 10:00"
        )

        state, _ = schedule.mark_prompted(state, "stop", prompt.state_key)
        again = schedule.stop_reminder_candidate(
            index, state, datetime(2026, 4, 24, 11, 0), is_recording=True
        )
        self.assertEqual(again.index, 1)

    def test_stop_reminder_for_a_recording_with_no_journal_meeting(self) -> None:
        state = schedule.default_reminder_state(date(2026, 4, 24))
        started = datetime(2026, 4, 24, 10, 0)

        self.assertIsNone(
            schedule.stop_reminder_candidate(
                {},
                state,
                datetime(2026, 4, 24, 10, 29),
                is_recording=True,
                recording_started_at=started,
            )
        )
        prompt = schedule.stop_reminder_candidate(
            {}, state, datetime(2026, 4, 24, 10, 30), is_recording=True,
            recording_started_at=started,
        )
        self.assertIsNone(prompt.meeting)
        self.assertEqual(prompt.index, 0)
        self.assertEqual(prompt.title, "Unscheduled recording")
        self.assertEqual(prompt.detail, "No journal meeting for it; started 10:00")

        state, _ = schedule.mark_prompted(state, "stop", prompt.state_key)
        self.assertIsNone(
            schedule.stop_reminder_candidate(
                {}, state, datetime(2026, 4, 24, 10, 45), is_recording=True,
                recording_started_at=started,
            )
        )
        again = schedule.stop_reminder_candidate(
            {}, state, datetime(2026, 4, 24, 11, 0), is_recording=True,
            recording_started_at=started,
        )
        self.assertEqual(again.index, 1)
        self.assertEqual(
            again.detail,
            "No journal meeting for it; started 10:00 - still recording, 30 min past due",
        )

    def test_stop_reminder_needs_an_anchor(self) -> None:
        state = schedule.default_reminder_state(date(2026, 4, 24))

        # No meeting and no known recording start: nothing to count from.
        self.assertIsNone(
            schedule.stop_reminder_candidate(
                {}, state, datetime(2026, 4, 24, 12, 0), is_recording=True
            )
        )
        # Not recording at all.
        self.assertIsNone(
            schedule.stop_reminder_candidate(
                {},
                state,
                datetime(2026, 4, 24, 12, 0),
                is_recording=False,
                recording_started_at=datetime(2026, 4, 24, 10, 0),
            )
        )

    def test_a_meeting_beats_the_recording_start_as_the_anchor(self) -> None:
        meeting = schedule.ScheduledMeeting(
            key="m1",
            title="Planning",
            start_at=datetime(2026, 4, 24, 10, 0),
            end_at=datetime(2026, 4, 24, 11, 0),
            source_line="",
        )
        state = schedule.set_recording_meeting(
            schedule.default_reminder_state(date(2026, 4, 24)), meeting.key
        )

        # 45 minutes into a recording, but the meeting runs until 11:00.
        self.assertIsNone(
            schedule.stop_reminder_candidate(
                {meeting.key: meeting},
                state,
                datetime(2026, 4, 24, 10, 45),
                is_recording=True,
                recording_started_at=datetime(2026, 4, 24, 10, 0),
            )
        )

    def test_a_stale_association_does_not_nag_a_later_recording(self) -> None:
        """Replay of 2026-08-25: a 09:00-09:45 meeting, a 10:46 recording.

        The recording_meeting_key outlived the recording it was made for, and
        every half hour the tray asked whether to stop a recording that had not
        even started when that meeting ended.
        """
        meeting = schedule.ScheduledMeeting(
            key="2026-08-25T09:00:00::Vi snackar hostplan",
            title="Vi snackar hostplan",
            start_at=datetime(2026, 8, 25, 9, 0),
            end_at=datetime(2026, 8, 25, 9, 45),
            source_line="",
        )
        index = {meeting.key: meeting}
        state = schedule.set_recording_meeting(
            schedule.default_reminder_state(date(2026, 8, 25)), meeting.key
        )
        started = datetime(2026, 8, 25, 10, 46)

        # 10:46:31 - half a minute into a recording that has nothing to do with
        # the 09:00 meeting. Before the guard this asked to stop it.
        self.assertIsNone(
            schedule.stop_reminder_candidate(
                index,
                state,
                datetime(2026, 8, 25, 10, 46, 31),
                is_recording=True,
                recording_started_at=started,
            )
        )
        self.assertIsNone(
            schedule.stop_reminder_candidate(
                index, state, datetime(2026, 8, 25, 11, 15), is_recording=True,
                recording_started_at=started,
            )
        )

        # It is still asked about on its own terms: 30 minutes after it started.
        prompt = schedule.stop_reminder_candidate(
            index, state, datetime(2026, 8, 25, 11, 16), is_recording=True,
            recording_started_at=started,
        )
        self.assertIsNone(prompt.meeting)
        self.assertEqual(prompt.detail, "No journal meeting for it; started 10:46")

    def test_a_recording_started_before_the_deadline_still_uses_the_meeting(self) -> None:
        meeting = schedule.ScheduledMeeting(
            key="m1",
            title="Planning",
            start_at=datetime(2026, 4, 24, 10, 0),
            end_at=datetime(2026, 4, 24, 10, 30),
            source_line="",
        )
        index = {meeting.key: meeting}
        state = schedule.set_recording_meeting(
            schedule.default_reminder_state(date(2026, 4, 24)), meeting.key
        )

        # Started late, but still before the meeting was due to end.
        prompt = schedule.stop_reminder_candidate(
            index, state, datetime(2026, 4, 24, 10, 31), is_recording=True,
            recording_started_at=datetime(2026, 4, 24, 10, 20),
        )
        self.assertEqual(prompt.meeting, meeting)
        self.assertEqual(prompt.detail, "Scheduled end: 10:30")

    def test_a_start_prompt_answered_far_too_late_is_not_current(self) -> None:
        """The 2026-08-25 root cause: a 09:00 prompt answered at 10:46.

        Answering it started a recording and tagged it with a meeting that had
        ended an hour earlier, which is what the stop reminders then nagged
        about every half hour.
        """
        meeting = schedule.ScheduledMeeting(
            key="2026-08-25T09:00:00::Vi snackar hostplan",
            title="Vi snackar hostplan",
            start_at=datetime(2026, 8, 25, 9, 0),
            end_at=datetime(2026, 8, 25, 9, 45),
            source_line="",
        )

        self.assertTrue(
            schedule.start_reminder_is_current(meeting, datetime(2026, 8, 25, 9, 2, 50))
        )
        self.assertFalse(
            schedule.start_reminder_is_current(meeting, datetime(2026, 8, 25, 10, 46))
        )
        # Past the 10-minute grace, even though the meeting is still running.
        self.assertFalse(
            schedule.start_reminder_is_current(meeting, datetime(2026, 8, 25, 9, 11))
        )

    def test_a_start_prompt_expires_at_the_end_of_its_window(self) -> None:
        grace = schedule.ScheduledMeeting(
            key="m1",
            title="Long enough",
            start_at=datetime(2026, 8, 25, 9, 0),
            end_at=datetime(2026, 8, 25, 10, 0),
            source_line="",
        )
        # Bounded by the 10-minute start grace.
        self.assertEqual(
            schedule.seconds_until_start_prompt_expires(grace, datetime(2026, 8, 25, 9, 0)),
            600,
        )

        short = schedule.ScheduledMeeting(
            key="m2",
            title="Shorter than the grace",
            start_at=datetime(2026, 8, 25, 9, 0),
            end_at=datetime(2026, 8, 25, 9, 5),
            source_line="",
        )
        # A meeting that ends first bounds it instead.
        self.assertEqual(
            schedule.seconds_until_start_prompt_expires(short, datetime(2026, 8, 25, 9, 0)),
            300,
        )
        # Never zero or negative - a dialog with no life left still needs a tick.
        self.assertEqual(
            schedule.seconds_until_start_prompt_expires(short, datetime(2026, 8, 25, 9, 30)),
            1,
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
