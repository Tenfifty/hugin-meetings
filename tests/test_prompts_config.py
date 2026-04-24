from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugin_meetings import pipeline
from hugin_meetings.config import ProjectMatcherConfig, _build


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
        summary_path = self.base / "summary-20260424-101112.md"
        summary_path.write_text("## Meeting Summary\n\nDiscussed Project Apollo.\n")
        (self.customers_dir / "Project Apollo.md").write_text("# Project Apollo\nImportant work.\n")

        template_path = self.base / "matcher.md"
        template_path.write_text(
            "CUSTOM TEMPLATE\n"
            "Names:\n{{candidate_names}}\n"
            "Context:\n{{candidate_context}}\n"
            "Calendar:\n{{calendar_lines}}\n"
            "Summary:\n{{summary_body}}\n"
        )

        with patch.object(
            pipeline._cfg,
            "project_matcher",
            ProjectMatcherConfig(
                projects_dir=self.customers_dir,
                prompt_path=template_path,
            ),
        ):
            prompt, candidates = pipeline.build_customer_prompt(summary_path, "gpt-5.4-mini")

        self.assertIn("CUSTOM TEMPLATE", prompt)
        self.assertIn("- Project Apollo", prompt)
        self.assertIn("Discussed Project Apollo.", prompt)
        self.assertIn("- (no calendar metadata)", prompt)
        self.assertEqual([candidate.name for candidate in candidates], ["Project Apollo"])

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


if __name__ == "__main__":
    unittest.main()
