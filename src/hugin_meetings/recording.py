"""Reusable recording primitives for Hugin meeting frontends."""

from __future__ import annotations

import logging
import signal
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .config import load_config

RAW_AUDIO_DIR = load_config().raw_audio_dir
DEFAULT_RECORDER_LOG_PATH = Path(tempfile.gettempdir()) / "hugin-audio-recorder.log"
DEFAULT_SEGMENT_MINUTES = 65


def new_session_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d-%H%M%S")


def raw_audio_part_path(audio_dir: Path, prefix: str, session_id: str, part: int) -> Path:
    return audio_dir / f"{prefix}-{session_id}-p{part:02d}.opus"


def elapsed_label(start_time: float | None, now: float | None = None) -> str:
    if not start_time:
        return "00:00:00"
    elapsed = int((now or time.time()) - start_time)
    mins, secs = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def build_ffmpeg_recording_command(
    *,
    source: str,
    output_path: Path,
    ffmpeg_bin: str = "ffmpeg",
    input_format: str = "pulse",
) -> list[str]:
    return [
        ffmpeg_bin,
        "-y",
        "-f",
        input_format,
        "-i",
        source,
        "-ac",
        "1",
        "-c:a",
        "libopus",
        "-b:a",
        "24k",
        "-application",
        "voip",
        str(output_path),
    ]


class RecordingTrack:
    """A single ffmpeg-backed recording track."""

    def __init__(
        self,
        prefix: str,
        source: str,
        *,
        audio_dir: Path = RAW_AUDIO_DIR,
        log_path: Path = DEFAULT_RECORDER_LOG_PATH,
        input_format: str = "pulse",
        ffmpeg_bin: str = "ffmpeg",
    ):
        self.prefix = prefix
        self.source = source
        self.audio_dir = audio_dir
        self.log_path = log_path
        self.input_format = input_format
        self.ffmpeg_bin = ffmpeg_bin
        self.recording = False
        self.process = None
        self.log_file = None
        self.current_file = None
        self.session_id = None
        self.next_part = 1
        self.segment_timer = None
        self.start_time = None

    def start_segment(self) -> None:
        if not self.session_id:
            raise RuntimeError(f"No session id assigned for {self.prefix} recording")

        self.current_file = raw_audio_part_path(
            self.audio_dir, self.prefix, self.session_id, self.next_part
        )
        command = build_ffmpeg_recording_command(
            source=self.source,
            output_path=self.current_file,
            ffmpeg_bin=self.ffmpeg_bin,
            input_format=self.input_format,
        )
        logging.info(
            "Starting %s recording from %s: %s",
            self.prefix,
            self.source,
            self.current_file,
        )
        self.log_file = self.log_path.open("a", encoding="utf-8")
        self.log_file.write(
            f"\n--- ffmpeg {self.prefix} {self.session_id} p{self.next_part:02d} "
            f"{datetime.now().isoformat(timespec='seconds')} ---\n"
        )
        self.log_file.flush()
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            self._close_log_file()
            raise
        time.sleep(0.2)
        if self.process.poll() is not None:
            returncode = self.process.returncode
            self._close_log_file()
            self.process = None
            raise RuntimeError(
                f"ffmpeg exited while starting {self.prefix} recording "
                f"from {self.source} with code {returncode}"
            )
        self.next_part += 1

    def stop_segment(self) -> None:
        if self.process and self.process.poll() is None:
            logging.info("Stopping %s recording", self.prefix)
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        self._close_log_file()

    def rotate_segment(self) -> bool:
        self.stop_segment()
        if self.recording:
            self.start_segment()
        return self.recording

    def elapsed_str(self) -> str:
        return elapsed_label(self.start_time)

    def reset_session(self) -> None:
        self.current_file = None
        self.session_id = None
        self.next_part = 1
        self.start_time = None

    def _close_log_file(self) -> None:
        if self.log_file is None:
            return
        try:
            self.log_file.close()
        except Exception:
            logging.exception("Failed to close ffmpeg log file")
        self.log_file = None
