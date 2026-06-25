from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugin_meetings import slack_post


class LeadTests(unittest.TestCase):
    def test_lead_is_text_before_first_h3(self) -> None:
        body = (
            "## Meeting Summary\nShort abstract paragraph.\n\n"
            "### Purpose\nwhy\n\n### Main Points\n- a"
        )
        self.assertEqual(slack_post._lead(body), "## Meeting Summary\nShort abstract paragraph.")

    def test_lead_without_h3_falls_back_to_first_paragraph(self) -> None:
        # No `### ` subsection (e.g. bold pseudo-headings): the lead must stay
        # the first paragraph, not the whole body.
        body = (
            "**One-line gist.**\n\n"
            "**Main points**\n- a\n- b\n\n"
            "**Decisions**\n- c"
        )
        self.assertEqual(slack_post._lead(body), "**One-line gist.**")


if __name__ == "__main__":
    unittest.main()
