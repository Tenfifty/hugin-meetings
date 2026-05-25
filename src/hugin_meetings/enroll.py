#!/usr/bin/env python3
"""Enroll speaker voices for recognition from diarized transcripts.

Usage:
    enroll-speaker.py                               # interactive, uses latest transcript
    enroll-speaker.py transcript-20260408-223310.json
    enroll-speaker.py --list
    enroll-speaker.py --remove David
"""

import argparse
import curses
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from .cli_utils import get_hf_token
from .pipeline import (
    TRANSCRIPT_JSON_DIR,
    is_backchannel,
    raw_audio_session,
    transcript_json_for_markdown,
)
from .config import load_config

AUDIO_DIR = load_config().raw_audio_dir
TRANSCRIPT_DIR = load_config().transcripts_dir
SPEAKERS_DIR = load_config().speakers_dir

# Segment filtering
MIN_SEGMENT_DURATION = 2.5    # seconds
MAX_SEGMENT_DURATION = 30.0   # seconds (very long = likely misattributed)
MIN_WORDS = 4                 # reject backchannels

# Embedding quality
MIN_READY_SEGMENTS = 20       # don't use for matching until we have this many
OUTLIER_THRESHOLD = 0.3       # reject segment if cosine to current centroid < this
AMBIGUITY_GAP = 0.1           # reject if gap between best and second-best match < this
MATCH_THRESHOLD = 0.5         # minimum cosine similarity for a match


# --- Speaker storage ---

def speaker_dir(name: str) -> Path:
    return SPEAKERS_DIR / name.lower().replace(" ", "_")


def load_speaker_meta(name: str) -> dict | None:
    meta_path = speaker_dir(name) / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return None


def save_speaker(name: str, meta: dict, embeddings: np.ndarray, centroid: np.ndarray):
    d = speaker_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    np.save(d / "embeddings.npy", embeddings)
    np.save(d / "centroid.npy", centroid)


def load_speaker_centroid(name: str) -> np.ndarray | None:
    path = speaker_dir(name) / "centroid.npy"
    if path.exists():
        return np.load(path)
    return None


def load_speaker_embeddings(name: str) -> np.ndarray | None:
    path = speaker_dir(name) / "embeddings.npy"
    if path.exists():
        return np.load(path)
    return None


def list_enrolled() -> list[str]:
    if not SPEAKERS_DIR.exists():
        return []
    return [
        d.name for d in sorted(SPEAKERS_DIR.iterdir())
        if d.is_dir() and (d / "meta.json").exists()
    ]


def load_all_centroids() -> dict[str, np.ndarray]:
    """Load all enrolled speaker centroids for matching."""
    result = {}
    for name in list_enrolled():
        meta = load_speaker_meta(name)
        if meta and meta.get("ready", False):
            centroid = load_speaker_centroid(name)
            if centroid is not None:
                display_name = meta.get("display_name", name)
                result[display_name] = centroid
    return result


# --- Segment filtering ---

def filter_segments(entries: list[dict], channel: str, speaker_label: str) -> list[dict]:
    """Filter segments for enrollment quality."""
    candidates = []
    for e in entries:
        if e["speaker"] != speaker_label or e["channel"] != channel:
            continue
        duration = e["end"] - e["start"]
        if duration < MIN_SEGMENT_DURATION or duration > MAX_SEGMENT_DURATION:
            continue
        text = e["text"].strip()
        if len(text.split()) < MIN_WORDS:
            continue
        if is_backchannel(text):
            continue
        candidates.append(e)
    return candidates


# --- Embedding extraction ---

def load_embedding_model(hf_token: str, device: str):
    """Load the embedding model from the diarization pipeline."""
    import torch
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=hf_token
    ).to(torch.device(device))
    return pipeline._embedding


def extract_segment_embeddings(
    audio_path: Path, segments: list[dict], emb_model, device: str
) -> np.ndarray:
    """Extract one embedding per segment by slicing audio and running through model."""
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(str(audio_path))
    # Resample to 16kHz mono if needed
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    embeddings = []
    for seg in segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = waveform[:, start_sample:end_sample]

        if chunk.shape[1] < emb_model.min_num_samples:
            continue

        # Model expects (batch, channel, samples)
        chunk = chunk.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = emb_model(chunk)
        if isinstance(emb, np.ndarray):
            embeddings.append(emb.squeeze())
        else:
            embeddings.append(emb.squeeze().cpu().numpy())

    return np.array(embeddings) if embeddings else np.empty((0, 256))


def filter_by_centroid(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Keep only embeddings that are close enough to the centroid."""
    if len(embeddings) == 0:
        return embeddings
    sims = cosine_similarities(embeddings, centroid)
    mask = sims >= OUTLIER_THRESHOLD
    return embeddings[mask]


def cosine_similarities(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Cosine similarity between each embedding and a centroid."""
    norms_e = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    norm_c = np.linalg.norm(centroid) + 1e-8
    return (embeddings @ centroid) / (norms_e.squeeze() * norm_c)


# --- Transcript resolution ---

def latest_transcript() -> Path | None:
    files = sorted(TRANSCRIPT_JSON_DIR.glob("transcript-*.json"))
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


def session_audio_path(session_id: str, channel: str, tmp_dir: Path) -> Path | None:
    session = raw_audio_session(session_id)
    if session is None:
        return None

    parts = session.mic_parts if channel == "mic" else session.sys_parts
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    output_path = tmp_dir / f"{channel}-{session_id}.opus"
    list_path = tmp_dir / f"{channel}-{session_id}.txt"
    lines = []
    for path in parts:
        escaped = str(path).replace("'", r"'\''")
        lines.append(f"file '{escaped}'\n")
    list_path.write_text("".join(lines), encoding="utf-8")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output_path


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

def do_enrollment(transcript_path: Path, assignments: list[tuple[str, str]]):
    """Extract embeddings for assigned speakers and store them."""
    import torch

    entries = json.loads(transcript_path.read_text())
    ts = transcript_path.stem.removeprefix("transcript-")

    hf_token = get_hf_token()
    if not hf_token:
        print("No HuggingFace token found. Run: huggingface-cli login", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading embedding model...")
    emb_model = load_embedding_model(hf_token, device)

    with tempfile.TemporaryDirectory(prefix=f"enroll-{ts}-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for composite_label, name in assignments:
            # composite_label is "channel:SPEAKER_XX"
            channel, speaker_label = composite_label.split(":", 1)
            print(f"\nProcessing {composite_label} -> {name}")

            # Filter segments
            candidates = filter_segments(entries, channel, speaker_label)
            if not candidates:
                print(f"  No suitable segments for {composite_label} (all too short/backchannel)")
                continue

            audio_path = session_audio_path(ts, channel, tmp_dir)

            if audio_path is None or not audio_path.exists():
                print(f"  Audio file not found for {channel} session {ts}")
                continue

            print(f"  {len(candidates)} candidate segments from {audio_path.name}")

            # Extract embeddings
            print("  Extracting embeddings...")
            new_embeddings = extract_segment_embeddings(
                audio_path, candidates, emb_model, device
            )
            if len(new_embeddings) == 0:
                print("  No embeddings extracted")
                continue

            print(f"  Got {len(new_embeddings)} embeddings")

            # Load existing data if any
            existing_embeddings = load_speaker_embeddings(name)
            existing_meta = load_speaker_meta(name)

            if existing_embeddings is not None and len(existing_embeddings) > 0:
                # Filter new embeddings: only keep those close to existing centroid
                centroid = np.load(speaker_dir(name) / "centroid.npy")
                sims = cosine_similarities(new_embeddings, centroid)
                close_mask = sims >= OUTLIER_THRESHOLD
                accepted = new_embeddings[close_mask]
                rejected = len(new_embeddings) - len(accepted)
                if rejected > 0:
                    print(f"  Rejected {rejected}/{len(new_embeddings)} outlier embeddings")
                if len(accepted) == 0:
                    print("  All embeddings were outliers — skipping")
                    continue
                all_embeddings = np.vstack([existing_embeddings, accepted])
                new_count = len(accepted)
            else:
                # First enrollment: compute initial centroid, remove obvious outliers
                initial_centroid = new_embeddings.mean(axis=0)
                sims = cosine_similarities(new_embeddings, initial_centroid)
                # Remove bottom 10% as likely noise
                threshold = np.percentile(sims, 10)
                mask = sims >= threshold
                all_embeddings = new_embeddings[mask]
                new_count = len(all_embeddings)
                if len(all_embeddings) == 0:
                    all_embeddings = new_embeddings
                    new_count = len(all_embeddings)

            centroid = all_embeddings.mean(axis=0)
            total = len(all_embeddings)
            ready = total >= MIN_READY_SEGMENTS

            meta = {
                "display_name": name,
                "total_segments": total,
                "ready": ready,
                "sources": (existing_meta or {}).get("sources", []) + [
                    {"file": audio_path.name, "label": composite_label, "added": new_count}
                ],
            }

            save_speaker(name, meta, all_embeddings, centroid)
            status = "READY" if ready else f"need {MIN_READY_SEGMENTS - total} more"
            print(f"  Saved: {total} embeddings ({status})")

    # Update transcript files with the new names
    if assignments:
        update_transcript(transcript_path, assignments)


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
    names = list_enrolled()
    if not names:
        print("No enrolled speakers.")
        return
    for name in names:
        meta = load_speaker_meta(name)
        if not meta:
            continue
        display = meta.get("display_name", name)
        total = meta.get("total_segments", 0)
        ready = meta.get("ready", False)
        status = "ready" if ready else f"need {max(0, MIN_READY_SEGMENTS - total)} more"
        sources = meta.get("sources", [])
        source_str = ", ".join(s["file"] for s in sources[-3:])  # show last 3
        print(f"  {display:20s}  {total:3d} segments  ({status})  from: {source_str}")


def remove_speaker(name: str):
    import shutil
    d = speaker_dir(name)
    if not d.exists():
        # Try case-insensitive
        for n in list_enrolled():
            meta = load_speaker_meta(n)
            if meta and meta.get("display_name", "").lower() == name.lower():
                d = speaker_dir(n)
                break
    if not d.exists():
        print(f"Speaker '{name}' not found.", file=sys.stderr)
        sys.exit(1)
    shutil.rmtree(d)
    print(f"Removed '{name}'")


def main():
    parser = argparse.ArgumentParser(description="Enroll speaker voices for recognition")
    parser.add_argument("transcript", nargs="?", help="Transcript JSON file (default: latest)")
    parser.add_argument("--list", action="store_true", help="List enrolled speakers")
    parser.add_argument("--remove", metavar="NAME", help="Remove an enrolled speaker")
    args = parser.parse_args()

    if args.list:
        list_speakers()
    elif args.remove:
        remove_speaker(args.remove)
    else:
        transcript_path = resolve_transcript(args.transcript)
        print(f"Using transcript: {transcript_path.name}")
        assignments = curses.wrapper(interactive_enroll, transcript_path)
        if assignments:
            print(f"\nEnrolling {len(assignments)} speaker(s)...")
            do_enrollment(transcript_path, assignments)
        else:
            print("No speakers to enroll.")


if __name__ == "__main__":
    main()
