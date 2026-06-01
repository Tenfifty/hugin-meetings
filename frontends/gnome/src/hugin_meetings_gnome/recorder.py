#!/usr/bin/env python3
"""GNOME panel indicator for toggling meeting audio recording."""

import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import date, datetime

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('AyatanaAppIndicator3', '0.1')
from gi.repository import Gtk, AyatanaAppIndicator3, Gio, GLib

from hugin_meetings import audio_routes
from hugin_meetings import pipeline as audio_pipeline
from hugin_meetings import recording
from hugin_meetings import schedule as meeting_schedule
from hugin_meetings.cli_utils import resolve_sibling_bin
from hugin_meetings.config import load_config

_cfg = load_config()

AUDIO_TUI_BIN = resolve_sibling_bin("hugin-meet-tui")
USER_SHELL = os.environ.get("SHELL") or "/usr/bin/zsh"

AUDIO_DIR = _cfg.raw_audio_dir
STATE_DIR = _cfg.recorder_state_dir
# Optional: path to a daily journal file. Recorder reads it to pre-populate
# today's scheduled meetings in the menu. Set journal_path in hugin.yaml
# (or meetings.journal_path) to enable; otherwise the feature is silently skipped.
JOURNAL_PATH = _cfg.journal_path
REMINDER_STATE_PATH = STATE_DIR / "recorder-reminders.json"
SEGMENT_MINUTES = recording.DEFAULT_SEGMENT_MINUTES
PENDING_REFRESH_SECONDS = 10
REMINDER_CHECK_SECONDS = 30
DEVICE_CHECK_SECONDS = 2
LOG_PATH = recording.DEFAULT_RECORDER_LOG_PATH
TERMINAL_CANDIDATES = [
    ["gnome-terminal", "--geometry=120x40", "--", USER_SHELL, "-lic"],
    ["kgx", "-e", USER_SHELL, "-lic"],
    ["x-terminal-emulator", "-geometry", "120x40", "-e", USER_SHELL, "-lic"],
]


logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def _log_unhandled_exception(exc_type, exc_value, exc_traceback):
    logging.exception("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = _log_unhandled_exception


class AudioRecorder:
    def __init__(self):
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        STATE_DIR.mkdir(parents=True, exist_ok=True)

        self.session = recording.RecordingSession(audio_dir=AUDIO_DIR)
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
        return self.session.recording

    def _load_reminder_state(self):
        return meeting_schedule.load_reminder_state(REMINDER_STATE_PATH)

    def _save_reminder_state(self):
        meeting_schedule.save_reminder_state(REMINDER_STATE_PATH, self.reminder_state)

    def _reset_reminder_state_for_today(self, persist=True):
        self.reminder_state, changed = meeting_schedule.reset_reminder_state_for_today(
            self.reminder_state
        )
        if changed and persist:
            self._save_reminder_state()

    def _reload_journal_meetings(self):
        try:
            self.today = date.today()
            self._reset_reminder_state_for_today()
            self.scheduled_meetings = meeting_schedule.load_todays_journal_meetings(
                JOURNAL_PATH, self.today
            )
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

    def _set_recording_meeting(self, meeting_key):
        self.reminder_state = meeting_schedule.set_recording_meeting(
            self.reminder_state, meeting_key
        )
        self._save_reminder_state()

    def _clear_recording_meeting(self):
        if self.reminder_state.get("recording_meeting_key") is None:
            return
        self.reminder_state = meeting_schedule.set_recording_meeting(
            self.reminder_state, None
        )
        self._save_reminder_state()

    def _mark_prompted(self, kind, meeting_key):
        self.reminder_state, changed = meeting_schedule.mark_prompted(
            self.reminder_state, kind, meeting_key
        )
        if changed:
            self._save_reminder_state()

    def _start_recording(self, meeting_key=None):
        if self.is_recording:
            return

        mic_source, system_source = audio_routes.get_default_audio_routes()
        try:
            self.session.start(mic_source, system_source)
        except Exception:
            logging.exception("Failed to start recording")
            self._teardown_recording()
            raise

        for track in self.session.tracks:
            track.segment_timer = GLib.timeout_add_seconds(
                SEGMENT_MINUTES * 60, track.rotate_segment
            )

        self.toggle_item.set_label("Stop Recording")
        if meeting_key:
            self._set_recording_meeting(meeting_key)
        else:
            self._clear_recording_meeting()
            self._maybe_associate_current_recording(datetime.now())
        self._update_icon()

    def _current_audio_routes(self):
        nodes = audio_routes.load_pipewire_nodes()
        if nodes is None:
            return None
        return (
            audio_routes.resolve_default_audio_source(nodes),
            audio_routes.resolve_default_monitor_source(nodes),
        )

    def _stop_recording(self):
        if not self.is_recording:
            return
        self._teardown_recording()

    def _teardown_recording(self):
        for track in self.session.tracks:
            if track.segment_timer:
                GLib.source_remove(track.segment_timer)
                track.segment_timer = None
        self.session.stop()
        self.toggle_item.set_label("Start Recording")
        self._clear_recording_meeting()
        self._update_icon()

    def _rotate_recording_to_sources(self, mic_source, system_source):
        logging.info(
            "Audio device change detected; rotating recording "
            "mic=%s->%s sys=%s->%s",
            self.session.mic.source,
            mic_source,
            self.session.system.source,
            system_source,
        )
        try:
            self.session.rotate(mic_source, system_source)
        except Exception:
            logging.exception("Failed to rotate recording after audio device change")
            self._teardown_recording()
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
                mic_source == self.session.mic.source
                and system_source == self.session.system.source
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
        meeting = meeting_schedule.associate_current_recording(
            self.scheduled_meetings,
            self.reminder_state,
            now,
            is_recording=self.is_recording,
        )
        if meeting:
            logging.info("Associating current recording with %s", meeting.title)
            self._set_recording_meeting(meeting.key)

    def _check_start_reminders(self, now):
        meeting = meeting_schedule.start_reminder_candidate(
            self.scheduled_meetings,
            self.reminder_state,
            now,
            is_recording=self.is_recording,
        )
        if not meeting:
            return

        logging.info("Prompting to start recording for %s", meeting.title)
        should_record = self._prompt_yes_no(
            "Start recording?",
            f'Record "{meeting.title}" now?',
            f"Scheduled time: {meeting.time_label}",
        )
        self._mark_prompted("start", meeting.key)
        if should_record and not self.is_recording:
            self._start_recording(meeting.key)

    def _check_stop_reminders(self, now):
        self._maybe_associate_current_recording(now)
        meeting = meeting_schedule.stop_reminder_candidate(
            self.meeting_index,
            self.reminder_state,
            now,
            is_recording=self.is_recording,
        )
        if not meeting:
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
            if self.session.mic.recording:
                parts.append(f"Mic {self.session.mic.elapsed_str()}")
            if self.session.system.recording:
                parts.append(f"Sys {self.session.system.elapsed_str()}")
            self.status_item.set_label(" | ".join(parts) if parts else "Idle")
            self.pending_item.set_label(f"Pending pipeline: {self.pending_count}")
            self.next_meeting_item.set_label(f"Next meeting: {self._next_meeting_label()}")
            self._update_icon()
        except Exception:
            logging.exception("Failed during status update")
            self.status_item.set_label("Recorder error")
        return True  # keep timer alive

    def _next_meeting_label(self):
        return meeting_schedule.next_meeting_label(self.scheduled_meetings, datetime.now())

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
