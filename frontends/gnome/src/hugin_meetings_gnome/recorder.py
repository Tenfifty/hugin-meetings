#!/usr/bin/env python3
"""GNOME panel indicator for toggling mic and system audio recording to Opus files."""

import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('AyatanaAppIndicator3', '0.1')
from gi.repository import Gtk, AyatanaAppIndicator3, Gio, GLib

from hugin_meetings import pipeline as audio_pipeline
from hugin_meetings.config import load_config

_cfg = load_config()

def _resolve_sibling_bin(name: str) -> str:
    """Prefer the hugin-meet-* binary next to the current interpreter so the
    TUI runs in the same venv as the tray, even when gnome-terminal + a
    login shell rewrites PATH.
    """
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else name


AUDIO_TUI_BIN = _resolve_sibling_bin("hugin-meet-tui")
USER_SHELL = os.environ.get("SHELL") or "/usr/bin/zsh"

AUDIO_DIR = _cfg.raw_audio_dir
STATE_DIR = _cfg.recorder_state_dir
# Optional: path to a daily journal file. Recorder reads it to pre-populate
# today's scheduled meetings in the menu. Set journal_path in hugin.yaml
# (or meetings.journal_path) to enable; otherwise the feature is silently skipped.
JOURNAL_PATH = _cfg.journal_path
REMINDER_STATE_PATH = STATE_DIR / "recorder-reminders.json"
SEGMENT_MINUTES = 65
PENDING_REFRESH_SECONDS = 10
REMINDER_CHECK_SECONDS = 30
DEVICE_CHECK_SECONDS = 2
START_PROMPT_GRACE_SECONDS = 10 * 60
MAX_MEETING_DURATION = timedelta(hours=4)
LOG_PATH = Path("/tmp/hugin-audio-recorder.log")
TERMINAL_CANDIDATES = [
    ["gnome-terminal", "--geometry=120x40", "--", USER_SHELL, "-lic"],
    ["kgx", "-e", USER_SHELL, "-lic"],
    ["x-terminal-emulator", "-geometry", "120x40", "-e", USER_SHELL, "-lic"],
]
SECTION_HEADER_RE = re.compile(r"^\s*##\s+<(?P<date>\d{4}-\d{2}-\d{2})\b")
BRACE_TIME_RE = re.compile(
    r"\*(?P<ignored>~)?\{(?P<start>\d{1,2}[:.]\d{2})(?:\s*-\s*(?P<end>\d{1,2}[:.]\d{2}))?\}\*"
)
LEGACY_TIME_RE = re.compile(
    r"\[(?P<start>\d{1,2}[:.]\d{2})(?:\s*-\s*(?P<end>\d{1,2}[:.]\d{2}))?\]"
)
AGENDA_ITEM_RE = re.compile(r"^- \[[ xX]\]\s*(?P<body>.+)$")


logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def _log_unhandled_exception(exc_type, exc_value, exc_traceback):
    logging.exception("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = _log_unhandled_exception


def _load_pipewire_nodes():
    try:
        result = subprocess.run(
            ["pw-dump"], capture_output=True, text=True, timeout=5, check=True
        )
        return json.loads(result.stdout)
    except Exception:
        logging.exception("Failed to inspect PipeWire nodes")
        return None


def _metadata_name(value):
    if isinstance(value, dict):
        name = value.get("name")
        return name if isinstance(name, str) and name else None
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        name = parsed.get("name")
        return name if isinstance(name, str) and name else None
    return None


def _node_props(node):
    return node.get("info", {}).get("props") or node.get("props", {})


def _node_metadata(node):
    return node.get("info", {}).get("metadata") or node.get("metadata", [])


def _default_pipewire_node_name(nodes, media_class):
    default_key = f"default.audio.{media_class.removeprefix('Audio/').lower()}"
    configured_key = f"default.configured.audio.{media_class.removeprefix('Audio/').lower()}"
    candidates = {}

    for node in nodes:
        props = _node_props(node)
        if props.get("metadata.name") != "default":
            continue
        for item in _node_metadata(node):
            key = item.get("key")
            if key in {default_key, configured_key}:
                candidates[key] = _metadata_name(item.get("value"))

    for key in (default_key, configured_key):
        name = candidates.get(key)
        if name:
            return name
    return None


def _first_pipewire_node_name(nodes, media_class):
    for node in nodes:
        props = _node_props(node)
        if props.get("media.class") == media_class:
            name = props.get("node.name")
            if name:
                return name
    return None


def _resolve_default_audio_source(nodes):
    source = _default_pipewire_node_name(nodes, "Audio/Source")
    return source or _first_pipewire_node_name(nodes, "Audio/Source") or "default"


def _resolve_default_monitor_source(nodes):
    sink = _default_pipewire_node_name(nodes, "Audio/Sink")
    if sink:
        return f"{sink}.monitor"

    sink = _first_pipewire_node_name(nodes, "Audio/Sink")
    if sink:
        return f"{sink}.monitor"

    return "alsa_output.pci-0000_65_00.6.analog-stereo.monitor"


def get_default_audio_routes(log=True):
    """Get the current mic and monitor sources for ffmpeg's pulse inputs."""
    nodes = _load_pipewire_nodes()
    if nodes is None:
        logging.info("Falling back to generic PulseAudio routes")
        return "default", "alsa_output.pci-0000_65_00.6.analog-stereo.monitor"

    mic_source = _resolve_default_audio_source(nodes)
    monitor_source = _resolve_default_monitor_source(nodes)
    if log:
        logging.info(
            "Using audio routes: mic=%s sys=%s",
            mic_source,
            monitor_source,
        )
    return mic_source, monitor_source


def get_default_audio_source():
    """Get the current default microphone/source for ffmpeg's pulse input."""
    return get_default_audio_routes()[0]


def get_default_monitor_source():
    """Get the monitor source for the current default audio output sink."""
    return get_default_audio_routes()[1]


class Track:
    """A single recording track (mic or system audio)."""

    def __init__(self, prefix, pulse_source):
        self.prefix = prefix
        self.pulse_source = pulse_source
        self.recording = False
        self.process = None
        self.log_file = None
        self.current_file = None
        self.session_id = None
        self.next_part = 1
        self.segment_timer = None
        self.start_time = None

    def start_segment(self):
        if not self.session_id:
            raise RuntimeError(f"No session id assigned for {self.prefix} recording")

        self.current_file = AUDIO_DIR / f"{self.prefix}-{self.session_id}-p{self.next_part:02d}.opus"
        command = [
            "ffmpeg", "-y",
            "-f", "pulse",
            "-i", self.pulse_source,
            "-ac", "1",
            "-c:a", "libopus",
            "-b:a", "24k",
            "-application", "voip",
            str(self.current_file),
        ]
        logging.info(
            "Starting %s recording from %s: %s",
            self.prefix,
            self.pulse_source,
            self.current_file,
        )
        self.log_file = LOG_PATH.open("a", encoding="utf-8")
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
                f"from {self.pulse_source} with code {returncode}"
            )
        self.next_part += 1

    def stop_segment(self):
        if self.process and self.process.poll() is None:
            logging.info("Stopping %s recording", self.prefix)
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        self._close_log_file()

    def rotate_segment(self):
        self.stop_segment()
        if self.recording:
            self.start_segment()
        return self.recording

    def elapsed_str(self):
        if not self.start_time:
            return "00:00:00"
        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)
        hours, mins = divmod(mins, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    def reset_session(self):
        self.current_file = None
        self.session_id = None
        self.next_part = 1
        self.start_time = None

    def _close_log_file(self):
        if self.log_file is None:
            return
        try:
            self.log_file.close()
        except Exception:
            logging.exception("Failed to close ffmpeg log file")
        self.log_file = None


@dataclass(frozen=True)
class ScheduledMeeting:
    key: str
    title: str
    start_at: datetime
    end_at: datetime | None
    source_line: str

    @property
    def time_label(self):
        if not self.end_at:
            return self.start_at.strftime("%H:%M")
        return f"{self.start_at.strftime('%H:%M')} - {self.end_at.strftime('%H:%M')}"


def _parse_clock(value):
    normalized = value.replace(".", ":")
    return datetime.strptime(normalized, "%H:%M").time()


def _strip_time_markup(text):
    stripped = BRACE_TIME_RE.sub("", text)
    stripped = LEGACY_TIME_RE.sub("", stripped)
    return " ".join(stripped.split()).strip()


def load_todays_journal_meetings(journal_path, today):
    if journal_path is None or not journal_path.exists():
        return []

    lines = journal_path.read_text(encoding="utf-8").splitlines()
    in_today_section = False
    meetings = []

    for raw_line in lines:
        header_match = SECTION_HEADER_RE.match(raw_line)
        if header_match:
            in_today_section = header_match.group("date") == today.isoformat()
            continue

        if not in_today_section:
            continue

        item_match = AGENDA_ITEM_RE.match(raw_line)
        if not item_match:
            continue

        body = item_match.group("body").strip()
        time_match = BRACE_TIME_RE.search(body)
        ignored = False
        if time_match:
            ignored = bool(time_match.group("ignored"))
        else:
            time_match = LEGACY_TIME_RE.search(body)

        if not time_match or ignored:
            continue

        start_time = _parse_clock(time_match.group("start"))
        end_token = time_match.group("end")
        end_at = None
        start_at = datetime.combine(today, start_time)
        if end_token:
            end_time = _parse_clock(end_token)
            end_at = datetime.combine(today, end_time)
            if end_at <= start_at:
                continue
            if end_at - start_at > MAX_MEETING_DURATION:
                continue

        title = _strip_time_markup(body)
        if not title:
            continue

        key = f"{start_at.isoformat()}::{title}"
        meetings.append(
            ScheduledMeeting(
                key=key,
                title=title,
                start_at=start_at,
                end_at=end_at,
                source_line=raw_line,
            )
        )

    meetings.sort(key=lambda meeting: meeting.start_at)
    return meetings


class AudioRecorder:
    def __init__(self):
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        STATE_DIR.mkdir(parents=True, exist_ok=True)

        mic_source, system_source = get_default_audio_routes()
        self.mic = Track("mic", mic_source)
        self.system = Track("sys", system_source)
        self.pending_count = 0
        self.pending_refresh_at = 0.0
        self.today = date.today()
        self.scheduled_meetings = []
        self.meeting_index = {}
        self.reminder_state = self._load_reminder_state()
        self._reset_reminder_state_for_today(persist=False)
        self.journal_monitor = None

        self.indicator = AyatanaAppIndicator3.Indicator.new(
            "hugin-recorder",
            "audio-input-microphone",
            AyatanaAppIndicator3.IndicatorCategory.APPLICATION_STATUS,
        )
        self.indicator.set_status(AyatanaAppIndicator3.IndicatorStatus.ACTIVE)
        self._build_menu()
        self._reload_journal_meetings()
        self._watch_journal()
        GLib.timeout_add_seconds(1, self._update_status)
        GLib.timeout_add_seconds(REMINDER_CHECK_SECONDS, self._check_reminders)
        GLib.timeout_add_seconds(DEVICE_CHECK_SECONDS, self._check_audio_device_changes)

    @property
    def is_recording(self):
        return self.mic.recording or self.system.recording

    def _load_reminder_state(self):
        if not REMINDER_STATE_PATH.exists():
            return {
                "date": None,
                "prompted_start": [],
                "prompted_stop": [],
                "recording_meeting_key": None,
            }

        try:
            state = json.loads(REMINDER_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            logging.exception("Failed to read reminder state from %s", REMINDER_STATE_PATH)
            return {
                "date": None,
                "prompted_start": [],
                "prompted_stop": [],
                "recording_meeting_key": None,
            }

        return {
            "date": state.get("date"),
            "prompted_start": list(state.get("prompted_start", [])),
            "prompted_stop": list(state.get("prompted_stop", [])),
            "recording_meeting_key": state.get("recording_meeting_key"),
        }

    def _save_reminder_state(self):
        REMINDER_STATE_PATH.write_text(
            json.dumps(self.reminder_state, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _reset_reminder_state_for_today(self, persist=True):
        today_iso = date.today().isoformat()
        if self.reminder_state.get("date") == today_iso:
            return

        self.reminder_state = {
            "date": today_iso,
            "prompted_start": [],
            "prompted_stop": [],
            "recording_meeting_key": None,
        }
        if persist:
            self._save_reminder_state()

    def _reload_journal_meetings(self):
        try:
            self.today = date.today()
            self._reset_reminder_state_for_today()
            self.scheduled_meetings = load_todays_journal_meetings(JOURNAL_PATH, self.today)
            self.meeting_index = {meeting.key: meeting for meeting in self.scheduled_meetings}
            if (
                self.reminder_state.get("recording_meeting_key")
                and self.reminder_state["recording_meeting_key"] not in self.meeting_index
            ):
                self.reminder_state["recording_meeting_key"] = None
                self._save_reminder_state()
            logging.info("Loaded %d journal meetings for %s", len(self.scheduled_meetings), self.today)
        except Exception:
            logging.exception("Failed to reload journal meetings")

    def _watch_journal(self):
        if JOURNAL_PATH is None:
            return
        journal_file = Gio.File.new_for_path(str(JOURNAL_PATH))
        self.journal_monitor = journal_file.monitor_file(Gio.FileMonitorFlags.NONE, None)
        self.journal_monitor.connect("changed", self._on_journal_changed)

    def _on_journal_changed(self, *_args):
        self._reload_journal_meetings()

    def _recording_started_at(self):
        timestamps = [track.start_time for track in (self.mic, self.system) if track.start_time]
        if not timestamps:
            return None
        return datetime.fromtimestamp(min(timestamps))

    def _set_recording_meeting(self, meeting_key):
        self.reminder_state["recording_meeting_key"] = meeting_key
        self._save_reminder_state()

    def _clear_recording_meeting(self):
        if self.reminder_state.get("recording_meeting_key") is None:
            return
        self.reminder_state["recording_meeting_key"] = None
        self._save_reminder_state()

    def _mark_prompted(self, kind, meeting_key):
        state_key = f"prompted_{kind}"
        prompted = set(self.reminder_state[state_key])
        if meeting_key in prompted:
            return
        prompted.add(meeting_key)
        self.reminder_state[state_key] = sorted(prompted)
        self._save_reminder_state()

    def _start_recording(self, meeting_key=None):
        if self.is_recording:
            return

        mic_source, system_source = get_default_audio_routes()
        self.mic.pulse_source = mic_source
        self.system.pulse_source = system_source

        start = time.time()
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            for track in (self.mic, self.system):
                track.recording = True
                track.start_time = start
                track.session_id = session_id
                track.next_part = 1
                track.start_segment()
                track.segment_timer = GLib.timeout_add_seconds(
                    SEGMENT_MINUTES * 60, track.rotate_segment
                )
        except Exception:
            logging.exception("Failed to start recording")
            self._mark_recording_failed()
            raise

        self.toggle_item.set_label("Stop Recording")
        if meeting_key:
            self._set_recording_meeting(meeting_key)
        else:
            self._clear_recording_meeting()
            self._maybe_associate_current_recording(datetime.now())
        self._update_icon()

    def _current_audio_routes(self):
        nodes = _load_pipewire_nodes()
        if nodes is None:
            return None
        return _resolve_default_audio_source(nodes), _resolve_default_monitor_source(nodes)

    def _stop_recording(self):
        if not self.is_recording:
            return

        for track in (self.mic, self.system):
            track.stop_segment()
            track.recording = False
            if track.segment_timer:
                GLib.source_remove(track.segment_timer)
                track.segment_timer = None
            track.reset_session()
        self.toggle_item.set_label("Start Recording")
        self._clear_recording_meeting()
        self._update_icon()

    def _stop_tracks_for_rotation(self):
        for track in (self.mic, self.system):
            track.stop_segment()

    def _mark_recording_failed(self):
        for track in (self.mic, self.system):
            track.stop_segment()
            track.recording = False
            if track.segment_timer:
                GLib.source_remove(track.segment_timer)
                track.segment_timer = None
            track.reset_session()
        self.toggle_item.set_label("Start Recording")
        self._clear_recording_meeting()
        self._update_icon()

    def _rotate_recording_to_sources(self, mic_source, system_source):
        logging.info(
            "Audio device change detected; rotating recording "
            "mic=%s->%s sys=%s->%s",
            self.mic.pulse_source,
            mic_source,
            self.system.pulse_source,
            system_source,
        )
        next_part = max(self.mic.next_part, self.system.next_part)
        started_tracks = []
        self._stop_tracks_for_rotation()
        self.mic.pulse_source = mic_source
        self.system.pulse_source = system_source
        self.mic.next_part = next_part
        self.system.next_part = next_part
        try:
            for track in (self.mic, self.system):
                track.start_segment()
                started_tracks.append(track)
        except Exception:
            logging.exception("Failed to rotate recording after audio device change")
            for track in started_tracks:
                track.stop_segment()
            self._mark_recording_failed()
            self.status_item.set_label("Recorder error after device change")
            raise

    def _check_audio_device_changes(self):
        try:
            if not self.is_recording:
                return True
            current_routes = self._current_audio_routes()
            if current_routes is None:
                return True
            mic_source, system_source = current_routes
            if (
                mic_source == self.mic.pulse_source
                and system_source == self.system.pulse_source
            ):
                return True
            self._rotate_recording_to_sources(mic_source, system_source)
        except Exception:
            logging.exception("Failed during audio device change check")
        return True

    def _prompt_yes_no(self, title, text, secondary):
        dialog = Gtk.MessageDialog(
            transient_for=None,
            flags=0,
            message_type=Gtk.MessageType.QUESTION,
            buttons=Gtk.ButtonsType.NONE,
            text=text,
        )
        dialog.set_title(title)
        dialog.format_secondary_text(secondary)
        dialog.set_keep_above(True)
        dialog.add_buttons(
            "_No", Gtk.ResponseType.NO,
            "_Yes", Gtk.ResponseType.YES,
        )
        dialog.set_default_response(Gtk.ResponseType.YES)
        response = dialog.run()
        dialog.destroy()
        return response == Gtk.ResponseType.YES

    def _maybe_associate_current_recording(self, now):
        if not self.is_recording or self.reminder_state.get("recording_meeting_key"):
            return

        candidates = [
            meeting
            for meeting in self.scheduled_meetings
            if 0 <= (now - meeting.start_at).total_seconds() <= START_PROMPT_GRACE_SECONDS
            and (meeting.end_at is None or now <= meeting.end_at)
        ]
        if len(candidates) == 1:
            logging.info("Associating current recording with %s", candidates[0].title)
            self._set_recording_meeting(candidates[0].key)

    def _check_start_reminders(self, now):
        if self.is_recording:
            return

        prompted = set(self.reminder_state["prompted_start"])
        for meeting in self.scheduled_meetings:
            if meeting.key in prompted:
                continue
            age_seconds = (now - meeting.start_at).total_seconds()
            if age_seconds < 0 or age_seconds > START_PROMPT_GRACE_SECONDS:
                continue

            logging.info("Prompting to start recording for %s", meeting.title)
            should_record = self._prompt_yes_no(
                "Start recording?",
                f'Record "{meeting.title}" now?',
                f"Scheduled time: {meeting.time_label}",
            )
            self._mark_prompted("start", meeting.key)
            if should_record and not self.is_recording:
                self._start_recording(meeting.key)
            return

    def _check_stop_reminders(self, now):
        if not self.is_recording:
            return

        self._maybe_associate_current_recording(now)
        meeting_key = self.reminder_state.get("recording_meeting_key")
        if not meeting_key:
            return

        meeting = self.meeting_index.get(meeting_key)
        if meeting is None or meeting.end_at is None:
            return
        if meeting_key in set(self.reminder_state["prompted_stop"]):
            return
        if now < meeting.end_at:
            return

        logging.info("Prompting to stop recording for %s", meeting.title)
        should_stop = self._prompt_yes_no(
            "Stop recording?",
            f'Stop recording for "{meeting.title}"?',
            f"Scheduled end: {meeting.end_at.strftime('%H:%M')}",
        )
        self._mark_prompted("stop", meeting.key)
        if should_stop and self.is_recording:
            self._stop_recording()

    def _check_reminders(self):
        try:
            self._reset_reminder_state_for_today()
            now = datetime.now()
            self._check_start_reminders(now)
            self._check_stop_reminders(now)
        except Exception:
            logging.exception("Failed during reminder check")
        return True

    def _build_menu(self):
        menu = Gtk.Menu()

        self.toggle_item = Gtk.MenuItem(label="Start Recording")
        self.toggle_item.connect("activate", self._on_toggle)
        menu.append(self.toggle_item)

        self.status_item = Gtk.MenuItem(label="Idle")
        self.status_item.set_sensitive(False)
        menu.append(self.status_item)

        self.pending_item = Gtk.MenuItem(label="Pending pipeline: 0")
        self.pending_item.set_sensitive(False)
        menu.append(self.pending_item)

        self.next_meeting_item = Gtk.MenuItem(label="Next meeting: -")
        self.next_meeting_item.set_sensitive(False)
        menu.append(self.next_meeting_item)

        open_tui_item = Gtk.MenuItem(label="Open Audio TUI")
        open_tui_item.connect("activate", self._on_open_tui)
        menu.append(open_tui_item)

        sep = Gtk.SeparatorMenuItem()
        menu.append(sep)

        quit_item = Gtk.MenuItem(label="Quit")
        quit_item.connect("activate", self._on_quit)
        menu.append(quit_item)

        menu.show_all()
        self.indicator.set_menu(menu)

    def _on_toggle(self, _):
        try:
            if self.is_recording:
                self._stop_recording()
            else:
                self._start_recording()
        except Exception:
            logging.exception("Failed to toggle recording")
            self.status_item.set_label("Error while toggling recording")

    def _update_icon(self):
        if self.is_recording:
            self.indicator.set_icon_full("media-record", "Recording")
        elif self.pending_count > 0:
            self.indicator.set_icon_full("dialog-warning", "Pending audio processing")
        else:
            self.indicator.set_icon_full("audio-input-microphone", "Idle")

    def _refresh_pending_count(self):
        now = time.time()
        if now < self.pending_refresh_at:
            return

        try:
            recordings = audio_pipeline.scan_recordings()
            self.pending_count = sum(1 for rec in recordings if rec.needs_pipeline)
        except Exception:
            # The recorder should keep working even if pipeline state inspection fails.
            logging.exception("Failed to refresh pending pipeline count")

        self.pending_refresh_at = now + PENDING_REFRESH_SECONDS

    def _update_status(self):
        try:
            self._refresh_pending_count()

            parts = []
            if self.mic.recording:
                parts.append(f"Mic {self.mic.elapsed_str()}")
            if self.system.recording:
                parts.append(f"Sys {self.system.elapsed_str()}")
            self.status_item.set_label(" | ".join(parts) if parts else "Idle")
            self.pending_item.set_label(f"Pending pipeline: {self.pending_count}")
            self.next_meeting_item.set_label(f"Next meeting: {self._next_meeting_label()}")
            self._update_icon()
        except Exception:
            logging.exception("Failed during status update")
            self.status_item.set_label("Recorder error")
        return True  # keep timer alive

    def _next_meeting_label(self):
        now = datetime.now()
        for meeting in self.scheduled_meetings:
            if meeting.end_at and meeting.end_at < now:
                continue
            if not meeting.end_at and meeting.start_at < now - timedelta(minutes=10):
                continue
            return f"{meeting.time_label} {meeting.title}"
        return "-"

    def _on_open_tui(self, _):
        command = (
            "printf '\\033[8;40;120t'; "
            f"{shlex.quote(AUDIO_TUI_BIN)}; "
            f"exec {shlex.quote(USER_SHELL)} -i"
        )
        env = os.environ.copy()
        for var in (
            "GDK_PIXBUF_MODULEDIR",
            "GDK_PIXBUF_MODULE_FILE",
            "GIO_MODULE_DIR",
            "GSETTINGS_SCHEMA_DIR",
            "GTK_EXE_PREFIX",
            "GTK_IM_MODULE_FILE",
            "GTK_PATH",
            "LOCPATH",
        ):
            env.pop(var, None)

        for terminal_cmd in TERMINAL_CANDIDATES:
            try:
                subprocess.Popen(
                    [*terminal_cmd, command],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                logging.info("Launched audio-tui via %s", terminal_cmd[0])
                self.status_item.set_label("Opened Audio TUI")
                return
            except FileNotFoundError:
                continue
            except Exception:
                logging.exception("Failed to launch audio-tui via %s", terminal_cmd[0])
                self.status_item.set_label("Failed to open Audio TUI")
                return

        logging.error("No terminal emulator found for audio-tui launch")
        self.status_item.set_label("No terminal found for Audio TUI")

    def _on_quit(self, _):
        self._stop_recording()
        Gtk.main_quit()


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    logging.info("Starting Hugin audio recorder")
    AudioRecorder()
    Gtk.main()


if __name__ == "__main__":
    main()
