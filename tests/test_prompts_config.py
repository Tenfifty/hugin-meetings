from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugin_meetings import pipeline
from hugin_meetings.config import MeetingsConfig, ProjectMatcherConfig

_build = MeetingsConfig.from_merged


class PromptConfigTests(unittest.TestCase):
    def test_config_loads_prompt_and_matcher_customization(self) -> None:
        cfg = _build(
            {
                "meetings": {
                    "summarize_prompt_path": "~/summary.md",
                    "summary_header": "## Custom Summary",
                    "personal_section_header": "### For Me",
                    "project_matcher": {
                        "prompt_path": "~/matcher.md",
                        "json_system_prompt": "JSON only, please.",
                        "inactive_dir_names": ["archive"],
                    },
                }
            }
        )

        self.assertEqual(cfg.summarize_prompt_path, Path("~/summary.md").expanduser())
        self.assertEqual(cfg.summary_header, "## Custom Summary")
        self.assertEqual(cfg.personal_section_header, "### For Me")
        self.assertEqual(
            cfg.project_matcher.prompt_path,
            Path("~/matcher.md").expanduser(),
        )
        self.assertEqual(cfg.project_matcher.json_system_prompt, "JSON only, please.")
        self.assertEqual(cfg.project_matcher.inactive_dir_names, ["archive"])

    def test_remote_llm_provider_is_configurable(self) -> None:
        cfg = _build(
            {
                "meetings": {
                    "llm": {
                        "provider": "claude",
                        "claude_args": ["--verbose"],
                    }
                }
            }
        )

        self.assertEqual(cfg.llm.provider, "claude")
        self.assertEqual(cfg.llm.claude_args, ["--verbose"])
        self.assertEqual(cfg.summary_model, "default")
        self.assertEqual(cfg.summary_effort, "high")
        self.assertEqual(cfg.project_matcher.model, "default")
        self.assertEqual(cfg.project_matcher.effort, "low")

    def test_local_command_provider_is_configurable(self) -> None:
        cfg = _build(
            {
                "meetings": {
                    "llm": {
                        "provider": "local",
                        "local_command": ["fake-llm", "--model", "{model}"],
                    }
                }
            }
        )

        self.assertEqual(cfg.llm.provider, "local")
        self.assertEqual(cfg.llm.local_command, ["fake-llm", "--model", "{model}"])


class ProjectMatcherPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.customers_dir = self.base / "projects"
        self.transcript_dir = self.base / "transcripts"
        self.customers_dir.mkdir()
        self.transcript_dir.mkdir()

        self.patchers = [
            patch.object(pipeline, "CUSTOMERS_DIR", self.customers_dir),
            patch.object(pipeline, "TRANSCRIPT_DIR", self.transcript_dir),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        self.addCleanup(self.tmp.cleanup)

    def test_project_matcher_prompt_template_is_customizable(self) -> None:
        (self.customers_dir / "Project Apollo.md").write_text("# Project Apollo\nImportant work.\n")

        template_path = self.base / "matcher.md"
        template_path.write_text(
            "CUSTOM TEMPLATE\n"
            "Context:\n{{candidate_context}}\n"
            "Calendar:\n{{calendar_lines}}\n"
            "Summary:\n{{summary_body}}\n"
        )

        # load_config() returns the lru-cached MeetingsConfig; patching
        # its attribute is observable to pipeline.build_customer_prompt_from.
        from hugin_meetings.config import load_config

        with patch.object(
            load_config(),
            "project_matcher",
            ProjectMatcherConfig(
                projects_dir=self.customers_dir,
                prompt_path=template_path,
            ),
        ):
            prompt, candidates = pipeline.build_customer_prompt_from(
                {}, "Discussed Project Apollo."
            )
            calendar_prompt, _ = pipeline.build_customer_prompt_from(
                {"Event": "Apollo kickoff", "Attendees": "a@apollo.example"}, ""
            )

        self.assertIn("CUSTOM TEMPLATE", prompt)
        self.assertIn("Customer: Project Apollo", prompt)
        self.assertIn("Discussed Project Apollo.", prompt)
        self.assertIn("- (no calendar metadata)", prompt)
        self.assertEqual([candidate.name for candidate in candidates], ["Project Apollo"])

        # Before transcription the same template runs in calendar mode: the
        # event is the whole evidence base, and the model is told so.
        self.assertIn("- Event: Apollo kickoff", calendar_prompt)
        self.assertIn("(not transcribed yet", calendar_prompt)

    def test_inactive_project_directory_names_are_configurable(self) -> None:
        active = self.customers_dir / "Active.md"
        archived = self.customers_dir / "archive" / "Archived.md"
        active.write_text("# Active\n")
        archived.parent.mkdir()
        archived.write_text("# Archived\n")

        with patch.object(pipeline, "INACTIVE_DIR_NAMES", {"archive"}):
            notes = pipeline.list_customer_notes()

        states = {note.name: note.is_active for note in notes}
        self.assertTrue(states["Active"])
        self.assertFalse(states["Archived"])

    def test_material_subdirectories_are_not_projects(self) -> None:
        (self.customers_dir / "AKA.md").write_text("# AKA\n")
        material = self.customers_dir / "AKA" / "AKA-brief.md"
        material.parent.mkdir()
        material.write_text("# Brief\n")
        nested = self.customers_dir / "inaktiva" / "Old" / "notes.md"
        nested.parent.mkdir(parents=True)
        nested.write_text("# Notes\n")

        names = [note.name for note in pipeline.list_customer_notes()]

        self.assertEqual(names, ["AKA"])

    def test_folder_name_guess_resolves_to_same_named_note(self) -> None:
        note = self.customers_dir / "AKA.md"
        note.write_text("# AKA\n")
        (self.customers_dir / "AKA").mkdir()

        self.assertEqual(pipeline.ensure_customer_note("AKA"), note)

    def test_folder_name_guess_without_note_creates_nothing(self) -> None:
        (self.customers_dir / "Notana").mkdir()

        with self.assertRaises(ValueError):
            pipeline.ensure_customer_note("Notana")
        self.assertFalse((self.customers_dir / "Notana.md").exists())


if __name__ == "__main__":
    unittest.main()
