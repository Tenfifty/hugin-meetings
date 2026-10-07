from __future__ import annotations

import unittest


class YearSubdirTests(unittest.TestCase):
    def test_extracts_from_session_timestamp(self) -> None:
        from hugin_meetings.pipeline import year_subdir

        self.assertEqual(year_subdir("20260524-101112"), "2026")
        self.assertEqual(year_subdir("transcript-20260524-101112.md"), "2026")
        self.assertEqual(year_subdir("mic-20260524-101112-p01.opus"), "2026")

    def test_rejects_missing_timestamp(self) -> None:
        from hugin_meetings.pipeline import year_subdir

        with self.assertRaises(ValueError):
            year_subdir("no-timestamp-here")
