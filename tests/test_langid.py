from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugin_meetings import langid


SETTINGS = {
    "model": "base",
    "compute_type": "int8",
    "languages": ("sv", "en"),
    "threshold": 0.5,
}


def probe(language: str, probability: float, channel: str = "mic") -> langid.LanguageProbe:
    return langid.LanguageProbe(channel, 0.25, language, probability, [(language, probability)])


class DecideLanguageTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.object(langid, "_settings", return_value=SETTINGS)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_unanimous_vote_wins(self) -> None:
        result = langid.decide_language(
            [probe("sv", 0.95), probe("sv", 0.89), probe("sv", 0.96)], "en"
        )
        self.assertEqual(result.value, "sv")
        self.assertEqual(result.source, "langid")
        self.assertAlmostEqual(result.confidence, 0.96)
        self.assertEqual(result.votes, {"sv": 3})
        self.assertEqual(result.note, "")

    def test_majority_wins_and_is_flagged_as_split(self) -> None:
        result = langid.decide_language(
            [probe("sv", 0.91), probe("en", 0.88), probe("en", 0.72)], "sv"
        )
        self.assertEqual(result.value, "en")
        self.assertEqual(result.note, "split vote")

    def test_tie_breaks_on_highest_probability(self) -> None:
        result = langid.decide_language([probe("sv", 0.61), probe("en", 0.93)], "sv")
        self.assertEqual(result.value, "en")

    def test_low_confidence_probes_fall_back(self) -> None:
        """The 2026-08-24 failure: a confident-sounding guess nobody should trust."""
        result = langid.decide_language(
            [probe("nl", 0.17), probe("en", 0.41), probe("nn", 0.30)], "sv"
        )
        self.assertEqual(result.value, "sv")
        self.assertEqual(result.source, "fallback")
        self.assertIn("en 0.41", result.note)

    def test_confident_probe_outside_allowlist_does_not_count(self) -> None:
        result = langid.decide_language([probe("nl", 0.97), probe("nl", 0.93)], "sv")
        self.assertEqual(result.value, "sv")
        self.assertEqual(result.source, "fallback")

    def test_no_probes_falls_back(self) -> None:
        result = langid.decide_language([], "sv")
        self.assertEqual(result.value, "sv")
        self.assertEqual(result.source, "fallback")
        self.assertEqual(result.note, "no probes")

    def test_sys_probes_join_the_same_vote(self) -> None:
        result = langid.decide_language(
            [probe("sv", 0.55), probe("en", 0.91, "sys"), probe("en", 0.88, "sys")], "sv"
        )
        self.assertEqual(result.value, "en")
        self.assertEqual(result.votes, {"sv": 1, "en": 2})


if __name__ == "__main__":
    unittest.main()
