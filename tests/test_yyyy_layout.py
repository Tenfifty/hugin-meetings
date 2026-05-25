from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hugin_meetings.migrate_yyyy import (
    Move,
    _rewrite_text,
    plan_link_rewrites,
    plan_moves,
)


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


class PlanMovesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _make(self, name: str) -> Path:
        p = self.base / name
        p.write_text("x")
        return p

    def test_moves_flat_files_to_year_subdir(self) -> None:
        self._make("transcript-20260524-101112.md")
        self._make("transcript-20251130-090000.md")
        moves = plan_moves([self.base])
        targets = {m.dst.relative_to(self.base) for m in moves}
        self.assertEqual(
            targets,
            {
                Path("2026/transcript-20260524-101112.md"),
                Path("2025/transcript-20251130-090000.md"),
            },
        )

    def test_skips_files_already_nested(self) -> None:
        nested = self.base / "2026"
        nested.mkdir()
        (nested / "transcript-20260524-101112.md").write_text("x")
        moves = plan_moves([self.base])
        self.assertEqual(moves, [])

    def test_skips_files_without_timestamp(self) -> None:
        (self.base / "README.md").write_text("docs")
        self.assertEqual(plan_moves([self.base]), [])


class VaultLinkRewriteTests(unittest.TestCase):
    def test_markdown_link_gets_year_segment(self) -> None:
        before = "[Full summary](../../audio/summaries/summary-20260524-101112.md)"
        after = _rewrite_text(before)
        self.assertEqual(
            after,
            "[Full summary](../../audio/summaries/2026/summary-20260524-101112.md)",
        )

    def test_wiki_link_gets_year_segment(self) -> None:
        before = "[[audio/summaries/summary-20260417-155318]]"
        self.assertEqual(
            _rewrite_text(before),
            "[[audio/summaries/2026/summary-20260417-155318]]",
        )

    def test_already_nested_link_is_unchanged(self) -> None:
        # Idempotent: re-running on rewritten text leaves it alone.
        already = "[Transcript](../../audio/transcripts/2026/transcript-20260524-101112.md)"
        self.assertEqual(_rewrite_text(already), already)

    def test_unrelated_links_untouched(self) -> None:
        text = "[Other](../other/path.md) and [[some/wikilink]]"
        self.assertEqual(_rewrite_text(text), text)


class PlanLinkRewritesTests(unittest.TestCase):
    def test_only_files_with_changes_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            (vault / "a.md").write_text(
                "[Transcript](audio/transcripts/transcript-20260524-101112.md)\n"
            )
            (vault / "b.md").write_text("no links here\n")
            (vault / "c.md").write_text(
                "Already migrated: [[audio/summaries/2026/summary-20260524-101112]]\n"
            )

            changes = plan_link_rewrites(vault)

        names = {c.path.name for c in changes}
        self.assertEqual(names, {"a.md"})


if __name__ == "__main__":
    unittest.main()
