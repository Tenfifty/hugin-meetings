"""Project notes under a reorganised vault: folder names, moved notes, rewritten links."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugin_meetings import pipeline


class ProjectNoteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.vault = base / "vault"
        self.projects = self.vault / "projects"
        self.transcripts = self.vault / "meetings" / "transcripts"
        self.summaries = self.vault / "meetings" / "summaries"
        for path in (self.projects, self.transcripts, self.summaries):
            path.mkdir(parents=True)
        for patcher in (
            patch.object(pipeline, "CUSTOMERS_DIR", self.projects),
            patch.object(pipeline, "TRANSCRIPT_DIR", self.transcripts),
            patch.object(pipeline, "SUMMARY_DIR", self.summaries),
            patch.object(pipeline, "INACTIVE_DIR_NAMES", {"inaktiva"}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

        self.aka = self.projects / "AKA.md"
        self.aka.write_text("# AKA\n")
        (self.projects / "AKA").mkdir()
        (self.projects / "AKA" / "AKA-brief.md").write_text("# Brief\n")
        (self.projects / "inaktiva").mkdir()
        (self.projects / "Old material").mkdir()

    def decide(self, **payload: str) -> pipeline.CustomerDecision:
        candidates = pipeline.list_customer_notes()
        with patch.object(pipeline, "run_remote_json_prompt", return_value=payload):
            return pipeline.decide_customer("prompt", candidates, model="test-model")


class FolderGuessTests(ProjectNoteTests):
    def test_material_files_are_not_candidates(self) -> None:
        self.assertEqual([note.name for note in pipeline.list_customer_notes()], ["AKA"])

    def test_material_folder_name_links_to_its_note(self) -> None:
        decision = self.decide(action="link_existing", customer_name="aka", confidence="high")
        self.assertEqual((decision.action, decision.customer_path), ("link_existing", self.aka))

    def test_other_folder_names_are_dropped(self) -> None:
        decision = self.decide(action="link_existing", customer_name="inaktiva", confidence="high")
        self.assertEqual(decision.action, "no_match")
        self.assertIsNone(decision.customer_name)

        decision = self.decide(action="suggest_new", suggested_name="Old material", confidence="high")
        self.assertEqual(decision.action, "no_match")
        self.assertIsNone(decision.suggested_name)

    def test_suggesting_an_existing_project_links_it(self) -> None:
        decision = self.decide(action="suggest_new", suggested_name="AKA", confidence="medium")
        self.assertEqual((decision.action, decision.customer_name), ("link_existing", "AKA"))
        self.assertEqual(decision.customer_path, self.aka)

    def test_ensure_note_never_creates_one_for_a_folder(self) -> None:
        with self.assertRaises(ValueError):
            pipeline.ensure_customer_note("Old material")
        self.assertFalse((self.projects / "Old material.md").exists())


class MovedNoteTests(ProjectNoteTests):
    def test_stale_state_path_resolves_by_name(self) -> None:
        stale = self.vault / "kunder" / "AKA.md"
        state = pipeline.deserialize_customer_state(
            {"action": "link_existing", "customer_name": "AKA", "customer_path": str(stale)}
        )
        self.assertEqual(state.customer_path, self.aka)

    def test_unknown_stale_path_is_kept_visible(self) -> None:
        stale = self.vault / "kunder" / "Gone.md"
        self.assertEqual(pipeline.current_customer_path("Gone", stale), stale)

    def test_summary_header_link_with_spaces_round_trips(self) -> None:
        note = self.projects / "Notana Care.md"
        note.write_text("# Notana Care\n")
        summary = self.summaries / "2026" / "summary-20261006-103207.md"
        summary.parent.mkdir()
        metadata = pipeline.CustomerMetadata(
            status="linked", confidence="high", rationale="", transcript_path=None,
            model="summary", customer_name="Notana Care", customer_path=note,
        )
        link = pipeline.render_customer_metadata(summary, metadata)
        self.assertIn("Notana%20Care.md", link)

        summary.write_text(f"{pipeline.load_config().summary_header} {link}\n\nBody.\n")
        self.assertEqual(pipeline.parse_customer_metadata(summary).customer_path, note)


class EntryMatchingTests(ProjectNoteTests):
    def write_summary(self, ts: str) -> Path:
        summary = self.summaries / "2026" / f"summary-{ts}.md"
        summary.parent.mkdir(exist_ok=True)
        summary.write_text(f"{pipeline.load_config().summary_header}\n\n- Point.\n")
        return summary

    def test_resync_after_link_rewrite_does_not_duplicate(self) -> None:
        summary = self.write_summary("20261006-103207")
        pipeline.sync_customer_note(self.aka, summary)
        # Obsidian (or a reorganisation) rewrites the link to another form.
        rel, _ = pipeline._customer_note_links(self.aka, summary)
        self.aka.write_text(self.aka.read_text().replace(rel, "summary-20261006-103207.md"))

        pipeline.sync_customer_note(self.aka, summary)

        self.assertEqual(self.aka.read_text().count("[Full summary]("), 1)

    def test_remove_finds_entry_by_file_name(self) -> None:
        kept = self.write_summary("20261005-090000")
        removed = self.write_summary("20261006-103207")
        pipeline.sync_customer_note(self.aka, kept)
        pipeline.sync_customer_note(self.aka, removed)
        rel, _ = pipeline._customer_note_links(self.aka, removed)
        self.aka.write_text(self.aka.read_text().replace(rel, "../elsewhere/summary-20261006-103207.md"))

        pipeline.remove_customer_note_link(self.aka, removed)

        text = self.aka.read_text()
        self.assertNotIn("summary-20261006-103207", text)
        self.assertIn("summary-20261005-090000", text)


if __name__ == "__main__":
    unittest.main()
