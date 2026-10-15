"""Reusable recording primitives for Hugin meeting frontends."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .config import load_config

RAW_AUDIO_DIR = load_config().raw_audio_dir
RECORDING_DIR = load_config().recording_dir
DEFAULT_RECORDER_LOG_PATH = Path(tempfile.gettempdir()) / "hugin-audio-recorder.log"
DEFAULT_SEGMENT_MINUTES = 65


def new_session_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d-%H%M%S")


def raw_audio_part_path(audio_dir: Path, prefix: str, session_id: str, part: int) -> Path:
    from .pipeline import year_subdir

    return audio_dir / year_subdir(session_id) / f"{prefix}-{session_id}-p{part:02d}.opus"


def recording_part_path(recording_dir: Path, prefix: str, session_id: str, part: int) -> Path:
    """Where a part is written while it is being recorded. Flat, same name as in the archive."""
    return recording_dir / f"{prefix}-{session_id}-p{part:02d}.opus"


def archive_part(path: Path, audio_dir: Path) -> Path:
    """Move a finished part from the recording dir to its place in the archive.

    One rename on the same filesystem, so the part appears under its final
    name whole or not at all. An archived part is never replaced
    (CONVENTIONS.md, "The archive root").
    """
    from .pipeline import year_subdir

    target = audio_dir / year_subdir(path.name) / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Archived part already exists, not replacing it: {target}")
    os.rename(path, target)
    return target


def session_lock_path(recording_dir: Path, session_id: str) -> Path:
    return recording_dir / f"{session_id}.lock"


def session_is_recording(recording_dir: Path, session_id: str) -> bool:
    """True while a recorder holds the session's lock.

    The kernel drops the lock when the recorder dies, so a session cut off by
    a crash or a power loss reads as not recording, whatever its files say.
    The parts' mtimes cannot tell: ffmpeg may not flush for minutes.
    """
    import fcntl

    try:
        fd = os.open(session_lock_path(recording_dir, session_id), os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def leftover_parts(recording_dir: Path) -> list[Path]:
    """Parts in the recording dir. Before a session starts, each is from one that was interrupted."""
    return sorted(recording_dir.glob("*.opus")) if recording_dir.exists() else []


# A session usually ends just before the lid closes; give the calendar lookup
# time for the network to come back instead of leaving the session without one.
CONTEXT_NETWORK_WAIT_SECONDS = 300


def start_context_guess(session_id: str, audio_dir: Path = RAW_AUDIO_DIR):
    """Guess language, calendar event and customer as soon as a session ends.

    The context stage is what the rest of the pipeline reads, and a person has
    to verify it before anything expensive runs — so it should be waiting for
    them, not started by them. Detached and fire-and-forget: a recorder must
    never block, and a failed guess is a gap to fill in, not a lost recording.

    Set HUGIN_NO_AUTO_CONTEXT to suppress it (tests, headless capture boxes).
    """
    import os

    from .cli_utils import resolve_sibling_bin

    if os.environ.get("HUGIN_NO_AUTO_CONTEXT"):
        return None
    # An aborted start leaves no parts behind; there is nothing to guess about.
    if not raw_audio_part_path(audio_dir, "mic", session_id, 1).exists():
        return None

    log_dir = load_config().state_dir / "logs" / "context"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / f"{session_id}.log", "ab")
    except OSError:
        logging.exception("Could not open context log for %s", session_id)
        return None

    try:
        return subprocess.Popen(
            [
                resolve_sibling_bin("hugin-meet-context"),
                session_id,
                "--wait-for-network",
                str(CONTEXT_NETWORK_WAIT_SECONDS),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError:
        logging.exception("Could not start context guess for %s", session_id)
        return None
    finally:
        log_file.close()


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
        # Its output goes to the recorder log. The banner and the progress
        # line (rewritten with \r every half second) were 99% of that log and
        # glued the recorder's own lines onto theirs; input, stream mapping
        # and warnings stay.
        "-hide_banner",
        "-nostats",
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
        recording_dir: Path = RECORDING_DIR,
        log_path: Path = DEFAULT_RECORDER_LOG_PATH,
        input_format: str = "pulse",
        ffmpeg_bin: str = "ffmpeg",
    ):
        self.prefix = prefix
        self.source = source
        self.audio_dir = audio_dir
        self.recording_dir = recording_dir
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

        # Written in the recording dir and moved into the archive by
        # stop_segment, once ffmpeg has finished the file.
        self.current_file = recording_part_path(
            self.recording_dir, self.prefix, self.session_id, self.next_part
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
                self.process.wait()
        self.process = None
        self._close_log_file()
        self._archive_current_file()

    def _archive_current_file(self) -> None:
        """The part just stopped is final: nothing writes to it again."""
        path = self.current_file
        if path is None or not path.exists():
            return
        try:
            archived = archive_part(path, self.audio_dir)
        except OSError:
            # Raising here would stop a rotation from starting the next part,
            # and the recording with it. The part stays in the recording dir,
            # where the TUI lists it as interrupted.
            logging.exception("Could not move finished part %s into the archive", path)
            return
        logging.info("Archived %s", archived)
        self.current_file = archived

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

    def __init__(self, *, audio_dir: Path = RAW_AUDIO_DIR, recording_dir: Path = RECORDING_DIR):
        self.mic = RecordingTrack("mic", "", audio_dir=audio_dir, recording_dir=recording_dir)
        self.system = RecordingTrack("sys", "", audio_dir=audio_dir, recording_dir=recording_dir)
        self.tracks = (self.mic, self.system)
        self.recording_dir = recording_dir
        self._lock_fd: int | None = None

    @property
    def recording(self) -> bool:
        return any(track.recording for track in self.tracks)

    @property
    def session_id(self) -> str | None:
        return self.mic.session_id

    @property
    def started_at(self) -> datetime | None:
        """When this session started, or None if it is not recording.

        Survives segment rotation - the tracks keep the session's start time,
        not the current part's.
        """
        if not self.mic.start_time:
            return None
        return datetime.fromtimestamp(self.mic.start_time)

    def start(self, mic_source: str, system_source: str) -> None:
        leftovers = leftover_parts(self.recording_dir)
        if leftovers:
            # Not moved or deleted: whether a part cut off by a crash belongs
            # in the archive is for a person to decide.
            logging.warning(
                "Interrupted recording parts in %s, left for you to archive or delete: %s",
                self.recording_dir,
                ", ".join(path.name for path in leftovers),
            )
        session_id = new_session_id()
        start_time = time.time()
        self._hold_session_lock(session_id)
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
        session_id = self.session_id
        for track in self.tracks:
            track.stop_segment()
            track.recording = False
            track.reset_session()
        self._release_session_lock(session_id)
        if session_id:
            start_context_guess(session_id, self.mic.audio_dir)

    def _hold_session_lock(self, session_id: str) -> None:
        """Mark the session as being recorded, for as long as this process lives."""
        import fcntl

        self.recording_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(session_lock_path(self.recording_dir, session_id), os.O_WRONLY | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
        self._lock_fd = fd

    def _release_session_lock(self, session_id: str | None) -> None:
        if self._lock_fd is None:
            return
        if session_id:
            session_lock_path(self.recording_dir, session_id).unlink(missing_ok=True)
        os.close(self._lock_fd)
        self._lock_fd = None

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
