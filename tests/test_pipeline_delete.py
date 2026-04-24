from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugin_meetings import pipeline


class DeleteMeetingEntryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

        self.raw_dir = self.base / "state" / "raw"
        self.wav_cache_dir = self.base / "state" / "cache" / "wav"
        self.transcript_json_dir = self.base / "state" / "transcripts"
        self.transcript_dir = self.base / "transcripts"
        self.summary_dir = self.base / "summaries"
        self.customers_dir = self.base / "kunder"

        for path in (
            self.raw_dir,
            self.wav_cache_dir,
            self.transcript_json_dir,
            self.transcript_dir,
            self.summary_dir,
            self.customers_dir,
        ):
            path.mkdir(parents=True)

        self.patchers = [
            patch.object(pipeline, "RAW_AUDIO_DIR", self.raw_dir),
            patch.object(pipeline, "WAV_CACHE_DIR", self.wav_cache_dir),
            patch.object(pipeline, "TRANSCRIPT_JSON_DIR", self.transcript_json_dir),
            patch.object(pipeline, "TRANSCRIPT_DIR", self.transcript_dir),
            patch.object(pipeline, "SUMMARY_DIR", self.summary_dir),
            patch.object(pipeline, "CUSTOMERS_DIR", self.customers_dir),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        self.addCleanup(self.tmp.cleanup)

    def write_file(self, path: Path, text: str = "x") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_delete_meeting_entry_removes_entry_artifacts_but_not_customer_notes(self) -> None:
        ts = "20260424-101112"
        artifacts = [
            self.raw_dir / f"mic-{ts}-p01.opus",
            self.raw_dir / f"sys-{ts}-p01.opus",
            self.wav_cache_dir / f"mic-{ts}-p01.wav",
            self.wav_cache_dir / f"sys-{ts}-p01.wav",
            self.transcript_json_dir / f"transcript-{ts}.json",
            self.transcript_json_dir / f"transcript-{ts}.customer.json",
            self.transcript_dir / f"transcript-{ts}.md",
            self.summary_dir / f"summary-{ts}.md",
        ]
        for path in artifacts:
            self.write_file(path)

        customer_note = self.write_file(
            self.customers_dir / "Example Customer.md",
            (
                "# Example Customer\n\n"
                f"## <2026-04-24 fre 10:11>\n\n"
                f"[Full summary](../summaries/summary-{ts}.md)\n"
            ),
        )
        original_customer_text = customer_note.read_text(encoding="utf-8")

        rec = pipeline.scan_recordings()[0]
        self.assertEqual(rec.timestamp, ts)

        deleted = pipeline.delete_meeting_entry(rec)

        self.assertEqual(set(deleted), set(artifacts))
        for path in artifacts:
            self.assertFalse(path.exists(), path)
        self.assertEqual(customer_note.read_text(encoding="utf-8"), original_customer_text)
        self.assertEqual(pipeline.scan_recordings(), [])


if __name__ == "__main__":
    unittest.main()
