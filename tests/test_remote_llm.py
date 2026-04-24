from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugin_meetings.config import LLMConfig
from hugin_meetings import remote_llm


class Completed:
    returncode = 0
    stdout = "ok from stdout\n"
    stderr = ""


class RemoteLLMTests(unittest.TestCase):
    def test_codex_uses_exec_and_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = LLMConfig(provider="codex", clean_cwd=Path(tmp))

            def fake_run(cmd, **kwargs):
                out_path = Path(cmd[cmd.index("-o") + 1])
                out_path.write_text('{"ok": true}\n', encoding="utf-8")
                return Completed()

            with patch.object(remote_llm.subprocess, "run", side_effect=fake_run) as run:
                text = remote_llm.run_prompt(cfg, "gpt-5.4-mini", "hello")

        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:4], ["codex", "exec", "-m", "gpt-5.4-mini"])
        self.assertIn("--skip-git-repo-check", cmd)
        self.assertEqual(text, '{"ok": true}')

    def test_claude_uses_clean_cwd_print_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = LLMConfig(provider="claude", clean_cwd=Path(tmp))
            with patch.object(remote_llm.subprocess, "run", return_value=Completed()) as run:
                text = remote_llm.run_prompt(cfg, "sonnet", "hello")

        cmd = run.call_args.args[0]
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["cwd"], Path(tmp))
        self.assertNotIn("--bare", cmd)
        self.assertIn("--print", cmd)
        self.assertIn("--no-session-persistence", cmd)
        self.assertEqual(text, "ok from stdout")

    def test_gemini_uses_clean_context_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clean_cwd = Path(tmp)
            cfg = LLMConfig(provider="gemini", clean_cwd=clean_cwd)
            with patch.object(remote_llm.subprocess, "run", return_value=Completed()) as run:
                text = remote_llm.run_prompt(cfg, "gemini-2.5-flash", "hello")

            settings = json.loads((clean_cwd / ".gemini" / "settings.json").read_text())

        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:4], ["gemini", "--model", "gemini-2.5-flash", "--prompt"])
        self.assertEqual(settings["context"]["fileName"], ".hugin-meetings-no-gemini-context.md")
        self.assertEqual(settings["context"]["discoveryMaxDirs"], 0)
        self.assertEqual(text, "ok from stdout")


if __name__ == "__main__":
    unittest.main()
