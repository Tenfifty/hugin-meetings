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
    from .pipeline import year_subdir

    return audio_dir / year_subdir(session_id) / f"{prefix}-{session_id}-p{part:02d}.opus"


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
        self.current_file.parent.mkdir(parents=True, exist_ok=True)
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


class RecordingSession:
    """Drive a mic + system-audio track pair through one recording session.

    Owns the two tracks, their shared session id, and part numbering. Frontends
    layer their own timers and UI on top; this class has no event-loop or UI
    dependencies.
    """

    def __init__(self, *, audio_dir: Path = RAW_AUDIO_DIR):
        self.mic = RecordingTrack("mic", "", audio_dir=audio_dir)
        self.system = RecordingTrack("sys", "", audio_dir=audio_dir)
        self.tracks = (self.mic, self.system)

    @property
    def recording(self) -> bool:
        return any(track.recording for track in self.tracks)

    @property
    def session_id(self) -> str | None:
        return self.mic.session_id

    def start(self, mic_source: str, system_source: str) -> None:
        session_id = new_session_id()
        start_time = time.time()
        try:
            for track, source in zip(self.tracks, (mic_source, system_source)):
                track.source = source
                track.recording = True
                track.start_time = start_time
                track.session_id = session_id
                track.next_part = 1
                track.start_segment()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        for track in self.tracks:
            track.stop_segment()
            track.recording = False
            track.reset_session()

    def rotate(self, mic_source: str | None = None, system_source: str | None = None) -> None:
        """Roll over to the next part, keeping the session id.

        Optionally switch audio sources (e.g. after a device change),
        preserving part numbering across both tracks.
        """
        next_part = max(track.next_part for track in self.tracks)
        for track in self.tracks:
            track.stop_segment()
        if mic_source is not None:
            self.mic.source = mic_source
        if system_source is not None:
            self.system.source = system_source
        started = []
        try:
            for track in self.tracks:
                track.next_part = next_part
                track.start_segment()
                started.append(track)
        except Exception:
            for track in started:
                track.stop_segment()
            raise


def main(argv: list[str] | None = None) -> int:
    """Record mic + system audio until interrupted (headless, no frontend)."""
    import argparse

    from . import audio_routes

    parser = argparse.ArgumentParser(description="Record mic + system audio for a meeting")
    parser.add_argument(
        "--segment-minutes",
        type=int,
        default=DEFAULT_SEGMENT_MINUTES,
        help="Split the recording into parts this many minutes long (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    mic_source, system_source = audio_routes.get_default_audio_routes()
    session = RecordingSession()
    session.start(mic_source, system_source)
    session_id = session.session_id

    print(f"Recording session {session_id} (Ctrl-C to stop)")
    print(f"  mic: {mic_source}")
    print(f"  sys: {system_source}")

    stopped = False

    def _stop(_signum, _frame) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    segment_seconds = args.segment_minutes * 60
    last_rotate = time.time()
    while not stopped:
        time.sleep(1)
        if time.time() - last_rotate >= segment_seconds:
            session.rotate()
            last_rotate = time.time()
        if any(t.process and t.process.poll() is not None for t in session.tracks):
            break

    session.stop()
    print(f"\nStopped. Session {session_id} saved under {RAW_AUDIO_DIR}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
