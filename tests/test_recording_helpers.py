from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugin_meetings import recording


class RecordingHelperTests(unittest.TestCase):
    def test_raw_audio_part_path_matches_pipeline_convention(self) -> None:
        path = recording.raw_audio_part_path(
            Path("/tmp/raw"),
            "mic",
            "20260424-101112",
            3,
        )

        self.assertEqual(path, Path("/tmp/raw/mic-20260424-101112-p03.opus"))

    def test_new_session_id_uses_existing_timestamp_format(self) -> None:
        self.assertEqual(
            recording.new_session_id(datetime(2026, 4, 24, 10, 11, 12)),
            "20260424-101112",
        )

    def test_build_ffmpeg_recording_command_defaults_to_pulse_opus(self) -> None:
        command = recording.build_ffmpeg_recording_command(
            source="default",
            output_path=Path("/tmp/raw/mic-20260424-101112-p01.opus"),
        )

        self.assertEqual(command[:6], ["ffmpeg", "-y", "-f", "pulse", "-i", "default"])
        self.assertIn("libopus", command)
        self.assertEqual(command[-1], "/tmp/raw/mic-20260424-101112-p01.opus")


if __name__ == "__main__":
    unittest.main()
