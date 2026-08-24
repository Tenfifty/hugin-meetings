#!/usr/bin/env python3
"""Interactive TUI for the Hugin audio pipeline."""

from __future__ import annotations

import argparse
import contextlib
import curses
import importlib.util
import io
import os
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

from . import pipeline as audio_pipeline
from .cli_utils import resolve_sibling_bin
from .config import load_config

STATE_DIR = load_config().state_dir
LOG_DIR = STATE_DIR / "logs" / "audio-tui"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AudioTui:
    def __init__(self, summary_model: str, customer_model: str):
        self.summary_model = summary_model
        self.customer_model = customer_model
        self.recordings = audio_pipeline.scan_recordings()
        self.selected = 0
        self.message = ""
        self.log_lines: list[str] = []
        self.enroll_module = None
        self.last_command_log: Path | None = None

    def refresh(self) -> None:
        self.recordings = audio_pipeline.scan_recordings()
        if self.recordings:
            self.selected = min(self.selected, len(self.recordings) - 1)
        else:
            self.selected = 0

    def append_log(self, message: str) -> None:
        for line in message.splitlines():
            line = line.rstrip().expandtabs()
            if line:
                self.log_lines.append(line)
        self.log_lines = self.log_lines[-200:]

    def set_message(self, message: str) -> None:
        lines = [line.strip() for line in message.splitlines() if line.strip()]
        self.message = " | ".join(lines)

    def selected_recording(self) -> audio_pipeline.MeetingStatus | None:
        if not self.recordings:
            return None
        return self.recordings[self.selected]

    def run(self, stdscr) -> None:
        curses.curs_set(0)
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        curses.init_pair(5, curses.COLOR_CYAN, -1)

        while True:
            self.draw_main(stdscr)
            key = stdscr.getch()

            if key in (ord("q"), ord("Q")):
                return
            if key == curses.KEY_UP and self.selected > 0:
                self.selected -= 1
            elif key == curses.KEY_DOWN and self.selected < len(self.recordings) - 1:
                self.selected += 1
            elif key in (ord("r"), ord("R")):
                self.refresh()
                self.set_message("Refreshed meeting list.")
            elif key in (ord("p"), ord("P")):
                self.process_pending(stdscr)
            elif key in (ord("l"), ord("L")):
                rec = self.selected_recording()
                if rec:
                    self.customer_link_flow(stdscr, rec)
                    self.refresh()
            elif key in (ord("v"), ord("V")):
                rec = self.selected_recording()
                if rec:
                    self.verify_customer(stdscr, rec)
                    self.refresh()
            elif key in (ord("x"), ord("X")):
                rec = self.selected_recording()
                if rec:
                    self.remove_customer(stdscr, rec)
                    self.refresh()
            elif key in (ord("d"), ord("D")):
                rec = self.selected_recording()
                if rec:
                    self.delete_entry_flow(stdscr, rec)
                    self.refresh()
            elif key in (ord("\n"), curses.KEY_ENTER, 10, 13):
                if self.recordings:
                    self.open_recording(stdscr, self.recordings[self.selected])

    def draw_main(self, stdscr) -> None:
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        unverified = sum(1 for rec in self.recordings if rec.needs_verification)
        ready = sum(1 for rec in self.recordings if rec.ready_for_pipeline)
        header = (
            f"Hugin Audio TUI  |  meetings: {len(self.recordings)}  "
            f"needs verification: {unverified}  ready to process: {ready}"
        )
        stdscr.addstr(0, 0, header[: w - 1], curses.A_BOLD)

        info = "Enter: open  p: process verified  v: verify  l: verify screen  x: remove customer  d: delete entry  r: refresh  q: quit"
        stdscr.addstr(1, 0, info[: w - 1], curses.A_DIM)

        table_top = 3
        table_height = max(5, h - 10)
        visible = self.recordings[:table_height]
        if self.selected >= table_height and self.recordings:
            start = self.selected - table_height + 1
            visible = self.recordings[start : start + table_height]
        else:
            start = 0

        for idx, rec in enumerate(visible):
            y = table_top + idx
            actual_index = start + idx
            selector = ">" if actual_index == self.selected else " "
            enroll = f" anon:{len(rec.anonymous_speakers)}" if rec.anonymous_speakers else ""
            customer_label = self.customer_label(rec)
            part_label = f"p{rec.raw_part_count}" if rec.raw_part_count else "--"
            line = (
                f"{selector} {rec.timestamp}  {part_label:>3}  {rec.short_status}  "
                f"{rec.pipeline_steps_complete}/{rec.pipeline_total_steps}  "
                f"{self.language_label(rec):<4}"
                f"{customer_label[:22].ljust(22)}  {rec.title}{enroll}"
            )
            attr = curses.color_pair(1) | curses.A_BOLD if actual_index == self.selected else curses.A_NORMAL
            stdscr.addstr(y, 0, line[: w - 1], attr)

        log_top = table_top + table_height + 1
        if log_top < h - 2:
            stdscr.addstr(log_top, 0, "Recent activity", curses.color_pair(5) | curses.A_BOLD)
            log_lines = self.log_lines[-max(1, h - log_top - 3) :]
            for offset, line in enumerate(log_lines, start=1):
                if log_top + offset >= h - 1:
                    break
                stdscr.addstr(log_top + offset, 0, line[: w - 1])

        if self.message:
            stdscr.addstr(h - 1, 0, self.message[: w - 1], curses.color_pair(3))

        stdscr.refresh()

    def pending_commands(self, rec: audio_pipeline.MeetingStatus) -> list[tuple[str, list[str]]]:
        commands: list[tuple[str, list[str]]] = []
        if not rec.is_verified:
            raise RuntimeError(
                f"{rec.timestamp} is not verified — check language and customer first (v)"
            )
        if not rec.has_transcript:
            if not rec.mic_parts:
                raise RuntimeError(f"Missing mic audio for {rec.timestamp}")
            part_desc = f"{rec.raw_part_count} part(s)" if rec.raw_part_count > 1 else rec.mic_parts[0].name
            commands.append(
                (
                    f"Transcribing {rec.timestamp} ({part_desc})",
                    [resolve_sibling_bin("hugin-meet-transcribe"), rec.timestamp],
                )
            )
            rec = self.find_recording(rec.timestamp) or rec

        transcript_arg = f"transcript-{rec.timestamp}.md"
        if not rec.has_calendar_metadata:
            commands.append(
                (
                    f"Matching calendar for {transcript_arg}",
                    [resolve_sibling_bin("hugin-meet-match-calendar"), transcript_arg],
                )
            )
        if not rec.has_summary:
            commands.append(
                (
                    f"Summarizing {transcript_arg}",
                    [resolve_sibling_bin("hugin-meet-summarize"), "--model", self.summary_model, transcript_arg],
                )
            )
        # Publishing into the customer note is its own stage now that
        # verification happens before there is anything to publish.
        commands.append(
            (
                f"Linking {rec.timestamp} into its customer note",
                [resolve_sibling_bin("hugin-meet-link"), rec.timestamp],
            )
        )
        return commands

    def language_label(self, rec: audio_pipeline.MeetingStatus) -> str:
        """The language a session will be transcribed in.

        Verification covers language as well as customer, so it has to be
        visible where verification happens. A trailing ``?`` means the same
        thing the ``??`` around a customer means — nobody has confirmed it yet
        — but a two-letter code does not need the column width to say so.

        Why the probe landed where it did (and whether it abstained at all) is
        on the verification screen, which is where you decide.
        """
        context = rec.context
        if context is None:
            return "-"
        value = context.language_value
        return value if rec.is_verified else f"{value}?"

    def customer_label(self, rec: audio_pipeline.MeetingStatus) -> str:
        """A confirmed customer wins over a guess, wherever each of them lives.

        The context is where a guess exists before verification; the customer
        state is where a decision lands. Re-guessing an old meeting must not
        make its settled customer look unsettled.
        """
        from . import context as meeting_context

        guess = (
            meeting_context.customer_state(rec.context)
            if rec.context is not None
            else None
        )
        candidates = [rec.customer_state, guess]
        for state in candidates:
            if state is not None and state.verified:
                return state.label
        for state in candidates:
            if state is not None:
                return state.label
        return "-"

    def run_command(self, stdscr, title: str, cmd: list[str]) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = self._command_slug(cmd)
        log_path = LOG_DIR / f"{stamp}-{slug}.log"
        self.last_command_log = log_path
        self.append_log(f"$ {' '.join(cmd)}")
        self.set_message(title)
        self.draw_progress(stdscr, title, [])

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        command_log: list[str] = []
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"$ {' '.join(cmd)}\n")
            log_file.write(f"Title: {title}\n\n")
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                cleaned = line.rstrip()
                if cleaned:
                    command_log.append(cleaned)
                    self.append_log(cleaned)
                    self.draw_progress(stdscr, title, command_log[-20:])

        code = process.wait()
        if code != 0:
            raise RuntimeError(f"{title} failed with exit code {code} (log: {log_path})")

    def _command_slug(self, cmd: list[str]) -> str:
        for part in reversed(cmd):
            if part.startswith("--"):
                continue
            name = Path(part).name
            if name and not name.startswith("transcript-"):
                return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:40] or "command"
        return "command"

    def draw_progress(self, stdscr, title: str, lines: list[str]) -> None:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, title[: w - 1], curses.A_BOLD)
        stdscr.addstr(1, 0, "Pipeline running...", curses.color_pair(5))
        for idx, line in enumerate(lines[-(h - 4) :], start=3):
            if idx >= h - 1:
                break
            stdscr.addstr(idx, 0, line[: w - 1])
        stdscr.refresh()

    def process_pending(self, stdscr) -> None:
        pending = [rec for rec in reversed(self.recordings) if rec.ready_for_pipeline]
        unverified = sum(1 for rec in self.recordings if rec.needs_verification)
        if not pending:
            if unverified:
                self.set_message(f"{unverified} recording(s) need verification first (v).")
            else:
                self.set_message("No pending recordings.")
            return

        total = len(pending)
        for idx, rec in enumerate(pending, start=1):
            self.set_message(f"Processing {idx}/{total}: {rec.timestamp}")
            try:
                for title, cmd in self.pending_commands(rec):
                    self.run_command(stdscr, f"[{idx}/{total}] {title}", cmd)
                self.refresh()
                self.append_log(f"Finished pipeline for {rec.timestamp}")
            except Exception as exc:
                self.append_log(f"ERROR: {exc}")
                self.set_message(str(exc))
                break
            finally:
                self.refresh()
        else:
            self.set_message(f"Processed {total} pending recording(s).")

    def find_recording(self, timestamp: str) -> audio_pipeline.MeetingStatus | None:
        for rec in self.recordings:
            if rec.timestamp == timestamp:
                return rec
        return None

    def open_recording(self, stdscr, rec: audio_pipeline.MeetingStatus) -> None:
        while True:
            rec = self.find_recording(rec.timestamp) or rec
            self.draw_recording(stdscr, rec)
            key = stdscr.getch()
            if key in (ord("b"), ord("B"), ord("q"), ord("Q"), 27):
                return
            if key in (ord("g"), ord("G")):
                if rec.needs_pipeline:
                    try:
                        for title, cmd in self.pending_commands(rec):
                            self.run_command(stdscr, title, cmd)
                        self.refresh()
                        rec = self.find_recording(rec.timestamp) or rec
                    except Exception as exc:
                        self.set_message(str(exc))
                    finally:
                        self.refresh()
                else:
                    self.set_message("Pipeline already complete for this meeting.")
            elif key in (ord("e"), ord("E")):
                self.run_enrollment(stdscr, rec)
                self.refresh()
            elif key in (ord("l"), ord("L")):
                self.customer_link_flow(stdscr, rec)
                self.refresh()
            elif key in (ord("v"), ord("V")):
                self.verify_customer(stdscr, rec)
                self.refresh()
            elif key in (ord("x"), ord("X")):
                self.remove_customer(stdscr, rec)
                self.refresh()
            elif key in (ord("s"), ord("S")):
                self.slack_post_flow(stdscr, rec)
                self.refresh()
            elif key in (ord("d"), ord("D")):
                if self.delete_entry_flow(stdscr, rec):
                    self.refresh()
                    return

    def draw_recording(self, stdscr, rec: audio_pipeline.MeetingStatus) -> None:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, f"Meeting {rec.timestamp}", curses.A_BOLD)
        stdscr.addstr(1, 0, f"Title: {rec.title}"[: w - 1])
        stdscr.addstr(
            2,
            0,
            (
                f"Parts: mic={len(rec.mic_parts)}, sys={len(rec.sys_parts)}  "
                if rec.raw_part_count
                else ""
            )[: w - 1],
        )
        stdscr.addstr(
            3,
            0,
            (
                f"Pipeline: transcript={'yes' if rec.has_transcript else 'no'}, "
                f"calendar={'yes' if rec.has_calendar_metadata else 'no'}, "
                f"summary={'yes' if rec.has_summary else 'no'}"
            )[: w - 1],
        )

        if rec.customer_state:
            customer_line = f"Customer: {rec.customer_state.label} [{rec.customer_state.confidence}]"
            customer_attr = curses.color_pair(2) if rec.customer_state.verified else curses.color_pair(3)
        else:
            customer_line = "Customer: none"
            customer_attr = curses.color_pair(3)
        language_note = ""
        if rec.context is not None:
            language = rec.context.language or {}
            votes = language.get("votes") or {}
            if votes:
                language_note = "  [" + ", ".join(f"{k} x{v}" for k, v in votes.items()) + "]"
            elif language.get("note"):
                language_note = f"  [{language['note']}]"
        stdscr.addstr(
            4,
            0,
            (f"Language: {self.language_label(rec)}{language_note}")[: w - 1],
            curses.color_pair(2) if rec.is_verified else curses.color_pair(3),
        )
        stdscr.addstr(5, 0, customer_line[: w - 1], customer_attr)

        enroll_line = (
            "Anonymous speakers: " + ", ".join(rec.anonymous_speakers)
            if rec.anonymous_speakers
            else "Anonymous speakers: none"
        )
        stdscr.addstr(6, 0, enroll_line[: w - 1])

        excerpt = ""
        if rec.summary_md and rec.summary_md.exists():
            excerpt = audio_pipeline.summary_excerpt(rec.summary_md, max_len=500)
        elif rec.transcript_md and rec.transcript_md.exists():
            excerpt = rec.title

        stdscr.addstr(8, 0, "e: enroll  l: verify screen  v: verify  x: remove customer  d: delete entry  g: run pipeline  s: post to Slack  b: back", curses.A_DIM)
        stdscr.addstr(10, 0, "Summary excerpt", curses.color_pair(5) | curses.A_BOLD)
        for idx, line in enumerate(textwrap.wrap(excerpt, width=max(20, w - 2))[: max(1, h - 13)], start=11):
            if idx >= h - 1:
                break
            stdscr.addstr(idx, 0, line[: w - 1])
        stdscr.refresh()

    def run_enrollment(self, stdscr, rec: audio_pipeline.MeetingStatus) -> None:
        if not rec.transcript_json:
            self.set_message("No transcript JSON available yet.")
            return

        if self.enroll_module is None:
            from . import enroll as enroll_module
            self.enroll_module = enroll_module

        assignments = self.enroll_module.interactive_enroll(stdscr, rec.transcript_json)
        if not assignments:
            self.set_message("No speaker assignments made.")
            return

        self.draw_progress(stdscr, "Enrolling speakers...", ["Extracting speaker embeddings..."])
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                self.enroll_module.do_enrollment(rec.transcript_json, assignments)
        except SystemExit as exc:
            self.append_log(buffer.getvalue().strip())
            self.set_message(f"Enrollment failed: {exc}")
            return

        output = buffer.getvalue().strip()
        if output:
            for line in output.splitlines():
                self.append_log(line)
        self.set_message(f"Enrolled {len(assignments)} speaker mapping(s).")

    def customer_link_flow(self, stdscr, rec: audio_pipeline.MeetingStatus) -> None:
        """Verify a session before it is processed: language, and customer.

        This screen used to run at the end, on a summary. It now runs at the
        start, on the context — same keys, but the decision is an input to the
        pipeline instead of a correction after it.
        """
        while True:
            rec = self.find_recording(rec.timestamp) or rec
            self.draw_customer_screen(stdscr, rec)
            key = stdscr.getch()
            if key in (ord("b"), ord("B"), ord("q"), ord("Q"), 27):
                return
            if key in (ord("r"), ord("R"), ord("g"), ord("G")):
                try:
                    self.run_command(
                        stdscr,
                        f"Guessing context for {rec.timestamp}",
                        [resolve_sibling_bin("hugin-meet-context"), rec.timestamp, "--force"],
                    )
                    self.set_message("Context guess updated.")
                    self.refresh()
                except Exception as exc:
                    self.set_message(f"Context guess failed: {exc}")
            elif key in (ord("v"), ord("V"), ord("a"), ord("A")):
                self.verify_customer(stdscr, rec)
                self.refresh()
            elif key in (ord("m"), ord("M")):
                chosen = self.pick_customer(stdscr)
                if chosen is None:
                    continue
                self.apply_customer(
                    rec,
                    audio_pipeline.manual_customer_state(
                        customer_name=chosen.name,
                        customer_path=chosen.path,
                    ),
                )
                self.refresh()
            elif key in (ord("n"), ord("N")):
                name = self.prompt_text(stdscr, "New customer/org name: ")
                if not name:
                    continue
                self.apply_customer(
                    rec, audio_pipeline.manual_new_customer_state(suggested_name=name)
                )
                self.refresh()
            elif key in (ord("l"), ord("L")):
                self.set_language(stdscr, rec)
                self.refresh()
            elif key in (ord("c"), ord("C")):
                self.remove_customer(stdscr, rec)
                self.refresh()

    def session_context(self, rec: audio_pipeline.MeetingStatus):
        if rec.context is None:
            self.set_message("No context yet. Press g to guess.")
            return None
        return rec.context

    def apply_customer(self, rec: audio_pipeline.MeetingStatus, state) -> None:
        """Record a person's customer choice, and publish it if there is a summary.

        Choosing is verifying — this is the point at which a new customer note
        gets created, as the direct result of the choice.
        """
        from . import context as meeting_context

        context = self.session_context(rec)
        if context is None:
            return
        meeting_context.verify(context, customer=state)
        stored = meeting_context.customer_state(context)
        self.append_log(f"Verified customer for {rec.timestamp}: {stored.label}")
        self.set_message(f"Verified customer: {stored.label}")
        self.publish_if_summarized(rec)

    def publish_if_summarized(self, rec: audio_pipeline.MeetingStatus) -> None:
        """A customer changed after summarizing still has to reach the note."""
        if not rec.summary_md or not rec.summary_md.exists():
            return
        from . import link

        try:
            self.append_log(link.link_summary(rec.summary_md))
        except Exception as exc:
            self.append_log(f"ERROR linking {rec.timestamp}: {exc}")

    def set_language(self, stdscr, rec: audio_pipeline.MeetingStatus) -> None:
        from . import context as meeting_context

        context = self.session_context(rec)
        if context is None:
            return
        language = self.prompt_text(stdscr, "Language: ", context.language_value)
        if not language:
            return
        meeting_context.verify(context, language=language.strip())
        self.append_log(f"Language for {rec.timestamp}: {language.strip()}")
        self.set_message(f"Language set to {language.strip()}.")
        if rec.has_transcript:
            self.set_message(
                f"Language set to {language.strip()} — transcript already exists, re-transcribe to apply."
            )

    def draw_customer_screen(self, stdscr, rec: audio_pipeline.MeetingStatus) -> None:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, f"Verify {rec.timestamp}", curses.A_BOLD)
        stdscr.addstr(1, 0, f"Meeting: {rec.title}"[: w - 1])

        context = rec.context
        y = 3
        if context is None:
            stdscr.addstr(y, 0, "No context yet. Press g to guess.", curses.color_pair(3))
            stdscr.addstr(h - 1, 0, "g: guess  b: back"[: w - 1], curses.A_DIM)
            stdscr.refresh()
            return

        from . import context as meeting_context

        language = context.language or {}
        votes = language.get("votes") or {}
        vote_text = ", ".join(f"{lang} x{count}" for lang, count in votes.items()) or "-"
        state = meeting_context.customer_state(context)
        event = meeting_context.calendar_fields(context)

        blocks = [
            (
                "Verification",
                [
                    f"Status: {'VERIFIED' if context.verified else 'NOT VERIFIED'}"
                    + (f" ({context.verified_at})" if context.verified_at else ""),
                    f"Applied so far: {context.applied or '-'}",
                ],
            ),
            (
                "Language",
                [
                    f"Language: {context.language_value}",
                    f"Source: {language.get('source', '-')}  confidence: {language.get('confidence', '-')}",
                    f"Windows: {vote_text}",
                    f"Note: {language.get('note') or '-'}",
                ],
            ),
            (
                "Calendar",
                [
                    f"Event: {event.get('Event', '-')}",
                    f"Time: {event.get('Event time', '-')}",
                    f"Attendees: {event.get('Attendees', '-')}",
                    f"Match: {event.get('Match status', context.note or '-')}",
                ],
            ),
            (
                "Customer",
                [
                    f"Customer: {state.label if state else '(none)'}",
                    f"Action: {state.action if state else '-'}  confidence: {state.confidence if state else '-'}",
                    f"Model/source: {state.model if state else '-'} / {state.source if state else '-'}",
                    f"Basis: {(state.rationale if state else '') or '-'}",
                ],
            ),
        ]

        for title, body in blocks:
            if y >= h - 3:
                break
            stdscr.addstr(y, 0, title, curses.color_pair(5) | curses.A_BOLD)
            y += 1
            for line in body:
                for wrapped in textwrap.wrap(line, width=max(20, w - 2)) or [""]:
                    if y >= h - 3:
                        break
                    stdscr.addstr(y, 0, wrapped[: w - 1])
                    y += 1
            y += 1

        footer = "v: accept guess  m: pick existing  n: free text  l: language  g: re-guess  c: remove  b: back"
        stdscr.addstr(h - 1, 0, footer[: w - 1], curses.A_DIM)
        stdscr.refresh()

    def verify_customer(self, stdscr, rec: audio_pipeline.MeetingStatus) -> None:
        """Accept the guess as it stands."""
        from . import context as meeting_context

        context = self.session_context(rec)
        if context is None:
            return
        state = meeting_context.customer_state(context)
        if state is None:
            self.set_message("No customer guessed. Pick one (m) or type a name (n).")
            return
        self.draw_progress(stdscr, "Verifying...", [state.label])
        self.apply_customer(rec, state)

    def remove_customer(self, stdscr, rec: audio_pipeline.MeetingStatus) -> None:
        """Drop the customer entirely — a verified session with nowhere to publish.

        This is the third outcome of verification: the meeting is confirmed and
        may be processed, but it does not belong in any customer note.
        """
        from . import context as meeting_context

        if rec.summary_md and rec.summary_md.exists():
            audio_pipeline.remove_customer_link(rec.summary_md)
        else:
            audio_pipeline.clear_customer_state(rec.timestamp)
        context = rec.context
        if context is not None:
            context.customer = None
            meeting_context.mark_verified(context)
            meeting_context.save_context(context)
        self.append_log(f"Removed customer for {rec.timestamp}")
        self.set_message("Removed customer link; session stays verified.")

    def slack_post_flow(self, stdscr, rec: audio_pipeline.MeetingStatus) -> None:
        if not rec.summary_md or not rec.summary_md.exists():
            self.set_message("No summary yet.")
            return
        state = rec.customer_state
        if not state or not state.customer_path:
            self.set_message("No project linked. Link a project first (l).")
            return
        cmd = [resolve_sibling_bin("hugin-meet-slack-post"), str(rec.summary_md)]
        try:
            self.run_command(stdscr, f"Posting {rec.timestamp} to Slack...", cmd)
            self.set_message("Posted to Slack.")
        except Exception as exc:
            self.set_message(str(exc))

    def delete_entry_flow(self, stdscr, rec: audio_pipeline.MeetingStatus) -> bool:
        paths = audio_pipeline.meeting_artifact_paths(rec)
        if not paths:
            self.set_message("No files found for this entry.")
            return False

        self.draw_delete_confirmation(stdscr, rec, paths)
        confirmation = self.prompt_text(stdscr, "Type DELETE to delete this entry: ")
        if confirmation != "DELETE":
            self.set_message("Delete cancelled.")
            return False

        try:
            deleted = audio_pipeline.delete_meeting_entry(rec)
        except Exception as exc:
            self.append_log(f"ERROR deleting {rec.timestamp}: {exc}")
            self.set_message(f"Delete failed: {exc}")
            return False

        self.append_log(f"Deleted entry {rec.timestamp}: {len(deleted)} file(s)")
        self.set_message(f"Deleted {rec.timestamp} ({len(deleted)} file(s)). Customer notes were left unchanged.")
        return True

    def draw_delete_confirmation(
        self,
        stdscr,
        rec: audio_pipeline.MeetingStatus,
        paths: list[Path],
    ) -> None:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, f"Delete meeting {rec.timestamp}", curses.color_pair(4) | curses.A_BOLD)
        stdscr.addstr(1, 0, "This removes raw audio, transcripts, summaries, cached WAVs, and customer-state JSON."[: w - 1])
        stdscr.addstr(2, 0, "Project/customer notes are left unchanged."[: w - 1])
        stdscr.addstr(4, 0, "Files to delete", curses.color_pair(5) | curses.A_BOLD)
        for offset, path in enumerate(paths[: max(1, h - 8)], start=5):
            if offset >= h - 2:
                break
            stdscr.addstr(offset, 0, str(path)[: w - 1])
        if len(paths) > max(1, h - 8):
            stdscr.addstr(h - 2, 0, f"... and {len(paths) - max(1, h - 8)} more"[: w - 1])
        stdscr.refresh()

    def prompt_text(self, stdscr, prompt: str, default: str = "") -> str | None:
        h, w = stdscr.getmaxyx()
        curses.curs_set(1)
        curses.echo()
        stdscr.addstr(h - 1, 0, " " * (w - 1))
        stdscr.addstr(h - 1, 0, prompt[: w - 1], curses.A_BOLD)
        stdscr.refresh()
        try:
            value = stdscr.getstr(h - 1, min(len(prompt), w - 1), max(1, w - len(prompt) - 1))
        except (KeyboardInterrupt, EOFError):
            value = b""
        finally:
            curses.noecho()
            curses.curs_set(0)
        text = value.decode("utf-8").strip()
        if not text and default:
            return default
        return text or None

    def pick_customer(self, stdscr) -> audio_pipeline.CustomerNote | None:
        query = self.prompt_text(stdscr, "Customer search: ")
        if query is None:
            return None

        notes = audio_pipeline.list_customer_notes()
        matches = [
            note
            for note in notes
            if query.lower() in note.name.lower()
        ]
        if not matches:
            self.set_message(f"No customers match '{query}'.")
            return None

        index = 0
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            stdscr.addstr(0, 0, f"Select customer for '{query}'", curses.A_BOLD)
            stdscr.addstr(1, 0, "Enter: choose  q: cancel", curses.A_DIM)
            visible = matches[: h - 4]
            if index >= len(visible):
                visible = matches[index - (h - 5) : index + 1]
            start_index = matches.index(visible[0]) if visible else 0
            for offset, note in enumerate(visible):
                y = 3 + offset
                actual = start_index + offset
                marker = ">" if actual == index else " "
                active_marker = "A" if note.is_active else "I"
                text = f"{marker} [{active_marker}] {note.name}"
                attr = curses.color_pair(1) | curses.A_BOLD if actual == index else curses.A_NORMAL
                stdscr.addstr(y, 0, text[: w - 1], attr)
            stdscr.refresh()

            key = stdscr.getch()
            if key in (ord("q"), ord("Q"), 27):
                return None
            if key == curses.KEY_UP and index > 0:
                index -= 1
            elif key == curses.KEY_DOWN and index < len(matches) - 1:
                index += 1
            elif key in (ord("\n"), curses.KEY_ENTER, 10, 13):
                return matches[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hugin audio pipeline TUI")
    parser.add_argument(
        "--summary-model",
        default=audio_pipeline.summarize_tool.DEFAULT_MODEL,
        help="Model for summary generation when the pipeline needs summarization.",
    )
    parser.add_argument(
        "--customer-model",
        default=audio_pipeline.DEFAULT_CUSTOMER_MODEL,
        help="Model for customer-link suggestion.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = AudioTui(summary_model=args.summary_model, customer_model=args.customer_model)
    curses.wrapper(app.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
