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

        self.raw_dir = self.base / "archive" / "raw"
        self.recording_dir = self.base / "state" / "recording"
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
            patch.object(pipeline, "RECORDING_DIR", self.recording_dir),
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
        yyyy = "2026"
        # Per-session files now live under <dir>/YYYY/; wav cache stays flat.
        artifacts = [
            self.raw_dir / yyyy / f"mic-{ts}-p01.opus",
            self.raw_dir / yyyy / f"sys-{ts}-p01.opus",
            self.wav_cache_dir / f"mic-{ts}-p01.wav",
            self.wav_cache_dir / f"sys-{ts}-p01.wav",
            self.transcript_json_dir / yyyy / f"transcript-{ts}.json",
            self.transcript_json_dir / yyyy / f"transcript-{ts}.customer.json",
            self.transcript_json_dir / yyyy / f"context-{ts}.json",
            self.transcript_dir / yyyy / f"transcript-{ts}.md",
            self.summary_dir / yyyy / f"summary-{ts}.md",
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

    def test_parts_still_in_the_recording_dir_are_shown_but_never_read_or_deleted(self) -> None:
        ts = "20260424-101112"
        archived = self.write_file(self.raw_dir / "2026" / f"mic-{ts}-p01.opus")
        cut_off = self.write_file(self.recording_dir / f"mic-{ts}-p02.opus")
        only_cut_off = self.write_file(self.recording_dir / "mic-20260425-090000-p01.opus")

        recs = {rec.timestamp: rec for rec in pipeline.scan_recordings()}

        rec = recs[ts]
        self.assertEqual(rec.mic_parts, (archived,))
        self.assertEqual(rec.in_progress_parts, (cut_off,))
        # No recorder holds the session, so the part was left behind.
        self.assertEqual(rec.recording_state, "interrupted")
        self.assertEqual(recs["20260425-090000"].in_progress_parts, (only_cut_off,))
        self.assertEqual(recs["20260425-090000"].mic_parts, ())
        self.assertEqual(pipeline.delete_meeting_entry(rec), [archived])
        self.assertTrue(cut_off.exists())

    def test_a_session_held_by_a_recorder_reads_as_recording(self) -> None:
        from hugin_meetings import recording

        ts = "20260424-101112"
        session = recording.RecordingSession(audio_dir=self.raw_dir, recording_dir=self.recording_dir)
        session._hold_session_lock(ts)
        self.write_file(self.recording_dir / f"mic-{ts}-p01.opus")
        self.assertEqual(pipeline.scan_recordings()[0].recording_state, "recording")

        session._release_session_lock(ts)
        self.assertEqual(pipeline.scan_recordings()[0].recording_state, "interrupted")

if __name__ == "__main__":
    unittest.main()
