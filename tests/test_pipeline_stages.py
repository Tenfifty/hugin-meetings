from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugin_meetings import pipeline
from hugin_meetings.context import MeetingContext


def status(**overrides) -> pipeline.MeetingStatus:
    base = dict(
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
        context=None,
    )
    base.update(overrides)
    return pipeline.MeetingStatus(**base)


class VerificationGateTests(unittest.TestCase):
    def test_a_session_without_context_is_not_processable(self) -> None:
        rec = status()
        self.assertTrue(rec.needs_verification)
        self.assertFalse(rec.ready_for_pipeline)
        self.assertEqual(rec.short_status, ".....")

    def test_an_unverified_guess_is_still_not_processable(self) -> None:
        """A guess is not a decision — nothing expensive runs on it."""
        rec = status(context=MeetingContext(session_id="20260824-140104"))
        self.assertTrue(rec.has_context)
        self.assertFalse(rec.is_verified)
        self.assertFalse(rec.ready_for_pipeline)

    def test_verification_is_the_first_completed_step(self) -> None:
        context = MeetingContext(session_id="20260824-140104", verified=True)
        rec = status(context=context)
        self.assertTrue(rec.ready_for_pipeline)
        self.assertEqual(rec.short_status, "V....")
        self.assertEqual(rec.pipeline_steps_complete, 1)
        self.assertEqual(rec.pipeline_total_steps, 5)


class LegacyVerificationTests(unittest.TestCase):
    """Meetings from before the context stage were verified in the old place."""

    def verified_state(self) -> pipeline.CustomerState:
        return pipeline.CustomerState(
            action="link_existing",
            confidence="high",
            rationale="Confirmed in the old flow.",
            model="gpt-5.6-luna",
            verified=True,
            customer_name="Helos",
            customer_path=Path("/vault/kunder/Helos.md"),
            source="auto",
        )

    def test_a_confirmed_customer_counts_as_verified(self) -> None:
        rec = status(customer_state=self.verified_state())
        self.assertTrue(rec.is_verified)
        self.assertFalse(rec.needs_verification)

    def test_re_guessing_an_old_meeting_does_not_unsettle_it(self) -> None:
        """Building a context for an already-processed meeting is not a demotion."""
        rec = status(
            customer_state=self.verified_state(),
            context=MeetingContext(session_id="20260824-130649", verified=False),
        )
        self.assertTrue(rec.is_verified)

    def test_an_unverified_customer_state_does_not_count(self) -> None:
        state = self.verified_state()
        state.verified = False
        self.assertFalse(status(customer_state=state).is_verified)


class LinkStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.summary = Path(self.tmp.name) / "summary-20260824-140104.md"
        self.summary.write_text("## Meeting Summary\n\nBody.\n")

    def linked_state(self, verified: bool) -> pipeline.CustomerState:
        return pipeline.CustomerState(
            action="link_existing",
            confidence="high",
            rationale="Event names the customer.",
            model="gpt-5.6-luna",
            verified=verified,
            customer_name="Notana Care",
            customer_path=Path(self.tmp.name) / "Notana Care.md",
            source="calendar",
        )

    def test_summarized_but_unpublished_needs_link(self) -> None:
        rec = status(summary_md=self.summary, customer_state=self.linked_state(True))
        self.assertTrue(rec.needs_link)
        self.assertFalse(rec.pipeline_steps[-1])

    def test_a_session_with_no_customer_has_nothing_to_publish(self) -> None:
        rec = status(summary_md=self.summary, customer_state=None)
        self.assertFalse(rec.needs_link)
        self.assertTrue(rec.pipeline_steps[-1])

    def test_unverified_state_is_never_published(self) -> None:
        rec = status(summary_md=self.summary, customer_state=self.linked_state(False))
        self.assertFalse(rec.needs_link)


if __name__ == "__main__":
    unittest.main()
