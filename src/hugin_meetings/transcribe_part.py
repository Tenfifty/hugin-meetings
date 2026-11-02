#!/usr/bin/env python3
"""Transcribe and diarize one raw audio part in its own process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe a single recording part")
    parser.add_argument("--mic", required=True, help="Mic part opus file")
    parser.add_argument("--sys", help="Matching system-audio part opus file")
    parser.add_argument("--part-index", required=True, type=int, help="Part index (1-based)")
    parser.add_argument("--json-out", required=True, help="Where to write merged part entries")
    parser.add_argument("--language", required=True, help="Language to transcribe in, from the session context")
    parser.add_argument("--use-part-suffix", action="store_true", help="Suffix anonymous speakers with the part id")
    parser.add_argument("--no-diarize", action="store_true", help="Skip diarization")
    parser.add_argument(
        "--allowed-speakers", required=True,
        help="JSON list of enrolled names this part may be labelled with "
        "(from the session's calendar event); '[]' allows nobody.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from . import transcribe

    entries = transcribe.process_part(
        Path(args.mic),
        Path(args.sys) if args.sys else None,
        part_index=args.part_index,
        use_part_suffix=args.use_part_suffix,
        language=args.language,
        allowed_speakers=json.loads(args.allowed_speakers),
        do_diarize=not args.no_diarize,
    )
    Path(args.json_out).write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
