from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unittest.mock import patch

from hugin_meetings import context as meeting_context
from hugin_meetings import link, pipeline
from hugin_meetings.context import MeetingContext


def status(**overrides) -> pipeline.MeetingStatus:
    base = dict(
        timestamp="20260824-140104",
        mic_parts=(),
        sys_parts=(),
        transcript_json=None,
        transcript_md=None,
        summary_md=None,
        calendar_fields={},
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
        self.assertFalse(rec.is_verified)
        self.assertFalse(rec.ready_for_pipeline)

    def test_verification_is_the_first_completed_step(self) -> None:
        context = MeetingContext(session_id="20260824-140104", verified=True)
        rec = status(context=context)
        self.assertTrue(rec.ready_for_pipeline)
        self.assertEqual(rec.short_status, "V....")
        self.assertEqual(rec.pipeline_steps_complete, 1)
        self.assertEqual(rec.pipeline_total_steps, 5)

    def test_a_session_with_parts_not_yet_archived_is_not_processable(self) -> None:
        context = MeetingContext(session_id="20260824-140104", verified=True)
        rec = status(
            context=context,
            in_progress_parts=(Path("/state/recording/mic-20260824-140104-p02.opus"),),
            recording_state="recording",
        )
        self.assertFalse(rec.ready_for_pipeline)


class CustomerFromContextTests(unittest.TestCase):
    """The context is the only place a customer lives."""

    def test_a_confirmed_customer_without_a_verified_context_does_not_count(self) -> None:
        """Verification is the context's; a verified-looking customer block does not stand in."""
        context = MeetingContext(
            session_id="20260824-130649",
            customer=pipeline.serialize_customer_state(
                pipeline.manual_customer_state(
                    customer_name="Helos", customer_path=Path("/vault/kunder/Helos.md")
                )
            ),
        )
        rec = status(context=context)
        self.assertFalse(rec.is_verified)
        self.assertEqual(rec.customer_state.customer_name, "Helos")

    def test_no_context_means_no_customer(self) -> None:
        self.assertIsNone(status().customer_state)


class LinkStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.summary = Path(self.tmp.name) / "summary-20260824-140104.md"
        self.summary.write_text("## Meeting Summary\n\nBody.\n")
        # An existing note: a missing one would be looked up by name in CUSTOMERS_DIR.
        self.note = Path(self.tmp.name) / "Notana Care.md"
        self.note.write_text("# Notana Care\n")

    def linked(self, verified: bool) -> MeetingContext:
        state = pipeline.CustomerState(
            action="link_existing",
            confidence="high",
            rationale="Event names the customer.",
            model="gpt-5.6-luna",
            verified=verified,
            customer_name="Notana Care",
            customer_path=self.note,
            source="calendar",
        )
        return MeetingContext(
            session_id="20260824-140104",
            customer=pipeline.serialize_customer_state(state),
            verified=True,
        )

    def test_summarized_but_unpublished_needs_link(self) -> None:
        rec = status(summary_md=self.summary, context=self.linked(True))
        self.assertTrue(rec.needs_link)
        self.assertFalse(rec.pipeline_steps[-1])

    def test_a_session_with_no_customer_has_nothing_to_publish(self) -> None:
        rec = status(
            summary_md=self.summary,
            context=MeetingContext(session_id="20260824-140104", verified=True),
        )
        self.assertFalse(rec.needs_link)
        self.assertTrue(rec.pipeline_steps[-1])

    def test_unverified_state_is_never_published(self) -> None:
        rec = status(summary_md=self.summary, context=self.linked(False))
        self.assertFalse(rec.needs_link)


class PublishTests(unittest.TestCase):
    """The customer is decided at verification; link_summary publishes into it at the end."""

    TS = "20260824-140104"

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.projects = base / "projects"
        for name, path in (
            ("CUSTOMERS_DIR", self.projects),
            ("TRANSCRIPT_JSON_DIR", base / "state" / "transcripts"),
            ("TRANSCRIPT_DIR", base / "transcripts"),
            ("SUMMARY_DIR", base / "summaries"),
        ):
            patcher = patch.object(pipeline, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.summary = pipeline.summary_md_path(self.TS)
        self.summary.parent.mkdir(parents=True)
        self.summary.write_text(f"{pipeline.load_config().summary_header}\n\nWe agreed on a pilot.\n")

    def verify(self, customer: pipeline.CustomerState) -> None:
        context = meeting_context.load_context(self.TS) or MeetingContext(session_id=self.TS)
        meeting_context.verify(context, customer=customer)

    def rec(self) -> pipeline.MeetingStatus:
        return status(summary_md=self.summary, context=meeting_context.load_context(self.TS))

    def test_a_new_customer_chosen_at_verification_receives_the_summary(self) -> None:
        self.verify(pipeline.manual_new_customer_state(suggested_name="Acme"))
        note = self.projects / "Acme.md"
        self.assertTrue(note.exists())
        self.assertTrue(self.rec().needs_link)

        link.link_summary(self.summary)
        link.link_summary(self.summary)  # publishing again replaces the entry

        text = note.read_text()
        self.assertEqual(text.count(f"summary-{self.TS}.md"), 1)
        self.assertIn("We agreed on a pilot.", text)
        self.assertFalse(self.rec().needs_link)

    def test_changing_the_customer_moves_the_entry(self) -> None:
        self.verify(pipeline.manual_new_customer_state(suggested_name="Acme"))
        link.link_summary(self.summary)
        self.verify(pipeline.manual_new_customer_state(suggested_name="Beta"))
        link.link_summary(self.summary)

        self.assertNotIn(f"summary-{self.TS}.md", (self.projects / "Acme.md").read_text())
        self.assertIn(f"summary-{self.TS}.md", (self.projects / "Beta.md").read_text())

    def test_an_unverified_customer_is_not_published(self) -> None:
        self.projects.mkdir()
        note = self.projects / "Acme.md"
        note.write_text("# Acme\n")
        guess = pipeline.manual_customer_state(customer_name="Acme", customer_path=note)
        guess.verified = False
        # A guess nobody has confirmed yet.
        meeting_context.save_context(
            MeetingContext(session_id=self.TS, customer=pipeline.serialize_customer_state(guess))
        )
        before = self.summary.read_text()

        link.link_summary(self.summary)

        self.assertEqual(note.read_text(), "# Acme\n")
        self.assertEqual(self.summary.read_text(), before)


if __name__ == "__main__":
    unittest.main()
