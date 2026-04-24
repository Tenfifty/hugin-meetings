from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugin_meetings.config import DEFAULT_REMOTE_MODEL, LLMConfig
from hugin_meetings import remote_llm


class Completed:
    returncode = 0
    stdout = "ok from stdout\n"
    stderr = ""


class RemoteLLMTests(unittest.TestCase):
    def test_codex_uses_default_model_with_effort_and_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = LLMConfig(provider="codex", clean_cwd=Path(tmp))

            def fake_run(cmd, **kwargs):
                out_path = Path(cmd[cmd.index("-o") + 1])
                out_path.write_text('{"ok": true}\n', encoding="utf-8")
                return Completed()

            with patch.object(remote_llm.subprocess, "run", side_effect=fake_run) as run:
                text = remote_llm.run_prompt(cfg, DEFAULT_REMOTE_MODEL, "hello", effort="low")

        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:2], ["codex", "exec"])
        self.assertNotIn("-m", cmd)
        self.assertEqual(cmd[cmd.index("-c") + 1], "model_reasoning_effort=low")
        self.assertIn("--skip-git-repo-check", cmd)
        self.assertEqual(text, '{"ok": true}')

    def test_claude_uses_clean_cwd_print_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = LLMConfig(provider="claude", clean_cwd=Path(tmp))
            with patch.object(remote_llm.subprocess, "run", return_value=Completed()) as run:
                text = remote_llm.run_prompt(cfg, DEFAULT_REMOTE_MODEL, "hello", effort="high")

        cmd = run.call_args.args[0]
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["cwd"], Path(tmp))
        self.assertNotIn("--bare", cmd)
        self.assertNotIn("--model", cmd)
        self.assertEqual(cmd[cmd.index("--effort") + 1], "high")
        self.assertIn("--print", cmd)
        self.assertIn("--no-session-persistence", cmd)
        self.assertEqual(text, "ok from stdout")

    def test_gemini_uses_clean_context_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clean_cwd = Path(tmp)
            cfg = LLMConfig(provider="gemini", clean_cwd=clean_cwd)
            with patch.object(remote_llm.subprocess, "run", return_value=Completed()) as run:
                text = remote_llm.run_prompt(cfg, DEFAULT_REMOTE_MODEL, "hello", effort="high")

            settings = json.loads((clean_cwd / ".gemini" / "settings.json").read_text())

        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:2], ["gemini", "--prompt"])
        self.assertNotIn("--model", cmd)
        self.assertNotIn("--effort", cmd)
        self.assertEqual(settings["context"]["fileName"], ".hugin-meetings-no-gemini-context.md")
        self.assertEqual(settings["context"]["discoveryMaxDirs"], 0)
        self.assertEqual(text, "ok from stdout")

    def test_local_provider_runs_configured_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = LLMConfig(
                provider="local",
                clean_cwd=Path(tmp),
                local_command=["fake-llm", "--model", "{model}", "--effort", "{effort}"],
            )
            with patch.object(remote_llm.subprocess, "run", return_value=Completed()) as run:
                text = remote_llm.run_prompt(cfg, "gemma-local", "hello", effort="low")

        cmd = run.call_args.args[0]
        kwargs = run.call_args.kwargs
        self.assertEqual(cmd, ["fake-llm", "--model", "gemma-local", "--effort", "low"])
        self.assertEqual(kwargs["input"], "hello")
        self.assertEqual(kwargs["cwd"], Path(tmp))
        self.assertEqual(text, "ok from stdout")

    def test_local_provider_requires_command(self) -> None:
        cfg = LLMConfig(provider="local")
        with self.assertRaisesRegex(RuntimeError, "local_command"):
            remote_llm.run_prompt(cfg, DEFAULT_REMOTE_MODEL, "hello")


if __name__ == "__main__":
    unittest.main()
