#!/usr/bin/env python3
"""Enroll speaker voices for recognition from diarized transcripts.

Usage:
    hugin-meet-enroll                               # interactive, uses latest transcript
    hugin-meet-enroll transcript-20260408-223310.json
    hugin-meet-enroll --list
    hugin-meet-enroll --remove Alice                # forget a person entirely
    hugin-meet-enroll --remove-session Alice 20260408-223310 [--channel sys] [--span 120-185]
    hugin-meet-enroll --always "David Fendrich"     # candidate in every meeting
    hugin-meet-enroll --rebuild                     # re-embed the whole cache

Names are only ever added here, by a person. Each enrollment or correction is
an event written once into the archive (``speakers.py``); ``speakers/`` is a
cache folded from them, and every command here brings it up to date after
writing. Transcription never writes events. It also only *applies* an enrolled
name when that name is among the attendees of the session's verified calendar
event (see ``transcribe.speaker_allowlist``), so an enrollment under a name the
invitation does not list is warned about below: it would be dead weight at
best and a misattribution at worst.
"""

import argparse
import curses
import json
import sys
from pathlib import Path

import numpy as np
from . import speakers
from .speakers import (
    MIN_READY_SEGMENTS,
    EnrollmentError,
    PyannoteEmbedder,
    cosine_similarities,
    slugify,
)
from .pipeline import (
    TRANSCRIPT_JSON_DIR,
    is_backchannel,
    transcript_json_for_markdown,
)
from .config import load_config

TRANSCRIPT_DIR = load_config().transcripts_dir
SPEAKERS_DIR = load_config().speakers_dir
ENROLLMENTS_DIR = load_config().enrollments_dir

# Segment filtering
MIN_SEGMENT_DURATION = 2.5    # seconds
MAX_SEGMENT_DURATION = 30.0   # seconds (very long = likely misattributed)
MIN_WORDS = 4                 # reject backchannels

# Embedding quality
OUTLIER_THRESHOLD = 0.3       # reject segment if cosine to current centroid < this


# --- Segment filtering ---

def enrollable(entry: dict) -> bool:
    """Long enough, short enough and wordy enough to say who is speaking."""
    duration = entry["end"] - entry["start"]
    if duration < MIN_SEGMENT_DURATION or duration > MAX_SEGMENT_DURATION:
        return False
    text = entry["text"].strip()
    if len(text.split()) < MIN_WORDS:
        return False
    return not is_backchannel(text)


def filter_segments(entries: list[dict], channel: str, speaker_label: str) -> list[dict]:
    """Filter segments for enrollment quality."""
    return [
        e for e in entries
        if e["speaker"] == speaker_label and e["channel"] == channel and enrollable(e)
    ]


def accepted_mask(embeddings: np.ndarray, existing_centroid: np.ndarray | None) -> np.ndarray:
    """Which new embeddings an enrollment keeps.

    Against an existing profile: those within ``OUTLIER_THRESHOLD`` of its
    centroid. For a first enrollment: all but the bottom 10% by cosine to
    their own mean, as likely noise.
    """
    if existing_centroid is not None:
        return cosine_similarities(embeddings, existing_centroid) >= OUTLIER_THRESHOLD
    sims = cosine_similarities(embeddings, embeddings.mean(axis=0))
    return sims >= np.percentile(sims, 10)


# --- Transcript resolution ---

def latest_transcript() -> Path | None:
    files = sorted(TRANSCRIPT_JSON_DIR.rglob("transcript-*.json"))
    return files[-1] if files else None


def resolve_transcript(name: str | None) -> Path:
    if name is None:
        path = latest_transcript()
        if not path:
            print("No transcripts found.", file=sys.stderr)
            sys.exit(1)
        return path
    path = Path(name)
    if path.exists():
        return path
    md_json = transcript_json_for_markdown(TRANSCRIPT_DIR / path.name)
    if md_json and md_json.exists():
        return md_json
    if (TRANSCRIPT_DIR / path).exists():
        md_path = TRANSCRIPT_DIR / path
        md_json = transcript_json_for_markdown(md_path)
        if md_json and md_json.exists():
            return md_json
    if (TRANSCRIPT_DIR / f"transcript-{path}").exists():
        md_path = TRANSCRIPT_DIR / f"transcript-{path}"
        md_json = transcript_json_for_markdown(md_path)
        if md_json and md_json.exists():
            return md_json
    if (TRANSCRIPT_JSON_DIR / path.name).exists():
        return TRANSCRIPT_JSON_DIR / path.name
    if (TRANSCRIPT_JSON_DIR / f"transcript-{path.name}").exists():
        return TRANSCRIPT_JSON_DIR / f"transcript-{path.name}"
    # Final fallback: YYYY/-nested lookup via rglob.
    matches = list(TRANSCRIPT_JSON_DIR.rglob(path.name)) or list(
        TRANSCRIPT_JSON_DIR.rglob(f"transcript-{path.name}")
    )
    if matches:
        return matches[0]
    print(f"Transcript not found: {name}", file=sys.stderr)
    sys.exit(1)


def get_speaker_previews(entries: list[dict]) -> dict[str, list[str]]:
    """Group transcript text by channel:speaker composite key."""
    previews = {}
    for e in entries:
        key = f"{e['channel']}:{e['speaker']}"
        if key not in previews:
            previews[key] = []
        previews[key].append(e["text"].strip())
    return previews


# --- Interactive TUI ---

def interactive_enroll(stdscr, transcript_path: Path):
    curses.curs_set(0)
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_GREEN, -1)

    entries = json.loads(transcript_path.read_text())
    previews = get_speaker_previews(entries)
    speaker_labels = sorted(previews.keys())

    if not speaker_labels:
        stdscr.addstr(0, 0, "No speakers found in transcript.")
        stdscr.getch()
        return []

    selected = 0
    results = []

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        title = f" {transcript_path.name} "
        stdscr.addstr(0, 0, title[:w], curses.A_BOLD)

        enrolled_names = {r[0]: r[1] for r in results}
        list_width = 30

        for i, spk in enumerate(speaker_labels):
            y = i + 2
            if y >= h - 2:
                break

            seg_count = len(previews[spk])
            label = f" {spk} ({seg_count} segs)"

            if spk in enrolled_names:
                label += f" -> {enrolled_names[spk]}"

            label = label[:list_width].ljust(list_width)

            if i == selected:
                stdscr.addstr(y, 0, label, curses.color_pair(1) | curses.A_BOLD)
            else:
                stdscr.addstr(y, 0, label)

        # Preview
        preview_x = list_width + 2
        preview_w = w - preview_x - 1
        if preview_w > 10:
            spk = speaker_labels[selected]
            stdscr.addstr(1, preview_x, f"--- {spk} ---"[:preview_w], curses.color_pair(2))

            lines = previews[spk]
            y = 2
            for line in lines:
                if y >= h - 2:
                    break
                while line and y < h - 2:
                    stdscr.addstr(y, preview_x, line[:preview_w])
                    line = line[preview_w:]
                    y += 1

        footer = " ↑↓:select  Enter:name speaker  q:done "
        stdscr.addstr(h - 1, 0, footer[:w], curses.A_DIM)

        stdscr.refresh()
        key = stdscr.getch()

        if key == curses.KEY_UP and selected > 0:
            selected -= 1
        elif key == curses.KEY_DOWN and selected < len(speaker_labels) - 1:
            selected += 1
        elif key in (ord('\n'), curses.KEY_ENTER, 10, 13):
            spk = speaker_labels[selected]
            name = prompt_name(stdscr, spk, h)
            if name:
                results.append((spk, name))
        elif key in (ord('q'), ord('Q'), 27):
            break

    return results


def prompt_name(stdscr, speaker_label: str, h: int) -> str | None:
    curses.curs_set(1)
    stdscr.addstr(h - 1, 0, " " * (stdscr.getmaxyx()[1] - 1))
    prompt = f"Name for {speaker_label}: "
    stdscr.addstr(h - 1, 0, prompt, curses.A_BOLD)
    stdscr.refresh()

    curses.echo()
    try:
        name = stdscr.getstr(h - 1, len(prompt), 40).decode("utf-8").strip()
    except (KeyboardInterrupt, EOFError):
        name = ""
    curses.noecho()
    curses.curs_set(0)
    return name if name else None


# --- Enrollment logic ---

def require_event_store() -> None:
    """Refuse to write events while ``speakers/`` still holds pre-event profiles.

    The first cache update would replace them with what the events say, and
    until they are migrated the events say nothing.
    """
    if speakers.is_legacy_store(SPEAKERS_DIR):
        raise EnrollmentError(
            f"{SPEAKERS_DIR} still holds profiles from before enrollment events. "
            "Migrate them first (see README, 'Enrolled speakers')."
        )


def update_cache(embedder: speakers.Embedder, force: bool = False) -> dict[str, speakers.Person]:
    return speakers.update_cache(SPEAKERS_DIR, ENROLLMENTS_DIR, embedder, force=force)


def do_enrollment(transcript_path: Path, assignments: list[tuple[str, str]]):
    """Write one enrollment event per assignment, then update the cache from the events.

    Raises ``EnrollmentError`` when the session audio is missing or the model
    cannot be loaded; nothing is written for that assignment or any after it.
    """
    entries = json.loads(transcript_path.read_text())
    ts = transcript_path.stem.removeprefix("transcript-")
    require_event_store()
    warn_names_not_invited(ts, [name for _, name in assignments])

    with PyannoteEmbedder() as embedder:
        # The outlier check below compares against the current centroids.
        update_cache(embedder)
        for composite_label, name in assignments:
            # composite_label is "channel:SPEAKER_XX"
            channel, speaker_label = composite_label.split(":", 1)
            print(f"\nProcessing {composite_label} -> {name}")

            candidates = filter_segments(entries, channel, speaker_label)
            if not candidates:
                print(f"  No suitable segments for {composite_label} (all too short/backchannel)")
                continue
            print(f"  {len(candidates)} candidate segments")

            spans = [(e["start"], e["end"]) for e in candidates]
            embeddings = embedder.embed(ts, channel, spans)
            existing = speakers.load_cached_centroid(SPEAKERS_DIR, slugify(name))
            mask = accepted_mask(embeddings, existing)
            accepted = [span for span, keep in zip(spans, mask) if keep]
            if len(accepted) < len(spans):
                print(f"  Rejected {len(spans) - len(accepted)}/{len(spans)} outlier embeddings")
            if not accepted:
                print("  All embeddings were outliers — skipping")
                continue

            path = speakers.write_event(ENROLLMENTS_DIR, {
                "type": "enroll",
                "name": name,
                "slug": slugify(name),
                "session": ts,
                "channel": channel,
                "segments": [{"start": a, "end": b} for a, b in accepted],
                "embedding_model": embedder.model_id,
                "diarization_label": composite_label,
            })
            print(f"  Wrote {path.name}")
            people = update_cache(embedder)
            total = len(people[slugify(name)].segments)
            status = "READY" if total >= MIN_READY_SEGMENTS else f"need {MIN_READY_SEGMENTS - total} more"
            print(f"  {name}: {total} segments ({status})")

    # Update transcript files with the new names
    if assignments:
        update_transcript(transcript_path, assignments)


def warn_names_not_invited(session_id: str, names: list[str]) -> list[str]:
    """Print a warning for each name absent from the session's calendar attendees.

    Returns the names warned about. Silent when the session has no context or
    the event has no attendee list — there is nothing to check against then.
    """
    from . import context as meeting_context
    from .transcribe import attendee_matches_speaker, event_attendee_names

    context = meeting_context.load_context(session_id)
    candidate = meeting_context.best_candidate(context) if context else None
    attendees = event_attendee_names(candidate.event) if candidate else []
    if not attendees:
        return []
    missing = [
        n for n in names
        if not any(attendee_matches_speaker(a, n) for a in attendees)
    ]
    for n in missing:
        print(
            f"  WARNING: '{n}' is not among the calendar attendees "
            f"({', '.join(attendees)}). Enrolled anyway — check that this is the right person."
        )
    return missing


def update_transcript(transcript_path: Path, assignments: list[tuple[str, str]]):
    """Replace anonymous speaker labels with real names in transcript files."""
    # Build rename map: (channel, SPEAKER_XX) -> name
    rename = {}
    for composite_label, name in assignments:
        channel, speaker_label = composite_label.split(":", 1)
        rename[(channel, speaker_label)] = name

    # Update JSON
    entries = json.loads(transcript_path.read_text())
    changed = 0
    for e in entries:
        key = (e["channel"], e["speaker"])
        if key in rename:
            e["speaker"] = rename[key]
            changed += 1

    transcript_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))

    # Update markdown
    md_path = transcript_path.with_suffix(".md")
    if md_path.exists():
        ts = transcript_path.stem.removeprefix("transcript-")
        md_path.write_text(_format_transcript(entries, ts))

    print(f"\nUpdated transcript: renamed {changed} segment(s) in {transcript_path.name}")


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_transcript(entries: list[dict], ts: str) -> str:
    """Format entries as readable markdown (mirrors transcribe.py)."""
    lines = [f"# Transcript {ts}", ""]
    prev_speaker = None
    prev_channel = None
    for e in entries:
        tag = f"{e['channel']}:{e['speaker']}"
        if tag != f"{prev_channel}:{prev_speaker}":
            start_fmt = _fmt_time(e["start"])
            lines.append(f"\n**[{start_fmt}] {tag}**\n")
            prev_speaker = e["speaker"]
            prev_channel = e["channel"]
        lines.append(e["text"])
    return "\n".join(lines) + "\n"


# --- CLI ---

def list_speakers():
    status, reason = speakers.cache_status(SPEAKERS_DIR, ENROLLMENTS_DIR)
    if status == "legacy":
        print(f"({reason}; not yet migrated to enrollment events)")
    elif status == "stale":
        print(f"(cache out of date: {reason}; run hugin-meet-enroll --update-cache)")
    names = [
        d for d in sorted(SPEAKERS_DIR.iterdir()) if d.is_dir() and (d / "meta.json").exists()
    ] if SPEAKERS_DIR.exists() else []
    if not names:
        print("No enrolled speakers.")
        return
    for d in names:
        meta = json.loads((d / "meta.json").read_text())
        display = meta.get("display_name", d.name)
        total = meta.get("total_segments", 0)
        ready = meta.get("ready", False)
        status = "ready" if ready else f"need {max(0, MIN_READY_SEGMENTS - total)} more"
        sources = meta.get("sources", [])
        source_str = ", ".join(
            f"{s['channel']}-{s['session']}" if "session" in s else s.get("file", "?")
            for s in sources[-3:]
        )
        always = "  [always a candidate]" if meta.get("always_candidate") else ""
        print(f"  {display:20s}  {total:3d} segments  ({status})  from: {source_str}{always}")


def _current_people() -> dict[str, speakers.Person]:
    return speakers.fold(speakers.load_events(ENROLLMENTS_DIR))


def _write_and_update(event: dict) -> None:
    path = speakers.write_event(ENROLLMENTS_DIR, event)
    print(f"Wrote {path}")
    with PyannoteEmbedder() as embedder:
        update_cache(embedder)


def remove_speaker(name: str):
    require_event_store()
    person = speakers.resolve_person(_current_people(), name)
    _write_and_update({"type": "remove_person", "name": person.name, "slug": person.slug})
    print(f"Removed '{person.name}'")


def parse_span(text: str) -> dict:
    """'120-185.5' -> {'start': 120.0, 'end': 185.5}, seconds in the session timeline."""
    try:
        start, end = (float(x) for x in text.split("-", 1))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected START-END in seconds, got {text!r}")
    return {"start": start, "end": end}


def remove_session(name: str, session: str, channel: str | None, spans: list[dict] | None):
    """Drop a person's segments from one session, or only those overlapping ``spans``."""
    require_event_store()
    people = _current_people()
    person = speakers.resolve_person(people, name)
    event = {
        "type": "remove_segments", "name": person.name, "slug": person.slug,
        "session": session, "channel": channel,
    }
    if spans:
        event["segments"] = spans
    # A permanent event that changes nothing is a typo; refuse it.
    after = speakers.fold([*speakers.load_events(ENROLLMENTS_DIR), {"schema": speakers.SCHEMA, **event}])
    removed = len(person.segments) - len(after[person.slug].segments if person.slug in after else ())
    if removed == 0:
        raise EnrollmentError(f"{person.name} has no segments there; nothing to remove")
    _write_and_update(event)
    print(f"Removed {removed} segment(s) of {session} from '{person.name}'")


def set_always_candidate(name: str, value: bool):
    """Mark a speaker as a match candidate in every meeting, invited or not.

    Meant for the person whose machine records: they are never on their own
    attendee list in a useful way, and a meeting with no attendee list at all
    would otherwise have no candidates.
    """
    require_event_store()
    person = speakers.resolve_person(_current_people(), name)
    if person.always_candidate != value:
        _write_and_update({
            "type": "always_candidate", "name": person.name, "slug": person.slug, "value": value,
        })
    state = "always a candidate" if value else "candidate only when invited"
    print(f"{person.name}: {state}")


def main():
    parser = argparse.ArgumentParser(description="Enroll speaker voices for recognition")
    parser.add_argument("transcript", nargs="?", help="Transcript JSON file (default: latest)")
    parser.add_argument("--list", action="store_true", help="List enrolled speakers")
    parser.add_argument("--remove", metavar="NAME", help="Forget an enrolled speaker entirely")
    parser.add_argument(
        "--remove-session", nargs=2, metavar=("NAME", "SESSION"),
        help="Drop NAME's segments from SESSION (all of them, or those overlapping --span)",
    )
    parser.add_argument("--channel", choices=speakers.CHANNELS, help="With --remove-session: only this channel")
    parser.add_argument(
        "--span", type=parse_span, action="append", metavar="START-END",
        help="With --remove-session: seconds in the session timeline; repeatable",
    )
    parser.add_argument(
        "--always", metavar="NAME",
        help="Make NAME a match candidate in every meeting, whether invited or not",
    )
    parser.add_argument(
        "--not-always", metavar="NAME",
        help="Undo --always: NAME is a candidate only when the invitation lists them",
    )
    parser.add_argument(
        "--update-cache", action="store_true",
        help="Bring speakers/ up to date with the enrollment events",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild speakers/ from the enrollment events, re-embedding every segment",
    )
    args = parser.parse_args()

    try:
        if args.list:
            list_speakers()
        elif args.remove:
            remove_speaker(args.remove)
        elif args.remove_session:
            remove_session(*args.remove_session, args.channel, args.span)
        elif args.always:
            set_always_candidate(args.always, True)
        elif args.not_always:
            set_always_candidate(args.not_always, False)
        elif args.update_cache or args.rebuild:
            with PyannoteEmbedder() as embedder:
                people = update_cache(embedder, force=args.rebuild)
            print(f"{SPEAKERS_DIR}: {len(people)} speaker(s) from {ENROLLMENTS_DIR}")
        else:
            transcript_path = resolve_transcript(args.transcript)
            print(f"Using transcript: {transcript_path.name}")
            assignments = curses.wrapper(interactive_enroll, transcript_path)
            if assignments:
                print(f"\nEnrolling {len(assignments)} speaker(s)...")
                do_enrollment(transcript_path, assignments)
            else:
                print("No speakers to enroll.")
    except EnrollmentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
