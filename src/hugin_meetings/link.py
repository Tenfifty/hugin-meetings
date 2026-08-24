#!/usr/bin/env python3
"""Publish a finished summary into its customer note.

Verification used to do this: pressing ``v`` in the TUI both approved the
customer guess and wrote the summary into that customer's note. Now that
verification happens before the meeting is transcribed, there is no summary to
publish at that point, so publishing becomes its own stage at the other end of
the pipeline.

Keeping it separate from ``summarize`` means the summarizer stays about
summarizing, and that re-publishing after changing your mind about the customer
is one cheap command rather than a re-run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import pipeline
from .cli_utils import resolve_transcript_md
from .config import load_config

SUMMARY_DIR = load_config().summaries_dir


def summary_path_for(ts: str) -> Path:
    return SUMMARY_DIR / pipeline.year_subdir(ts) / f"summary-{ts}.md"


def link_summary(summary_path: Path) -> str:
    """Write the customer link into the summary and the summary into the note.

    Returns a one-line report. Publishing an unverified state is refused: the
    customer note is a curated document, and only a person's decision belongs
    in it.
    """
    ts = pipeline.extract_timestamp(summary_path.name)
    if not ts:
        raise ValueError(f"Could not determine timestamp from {summary_path.name}")
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary at {summary_path}")

    state = pipeline.load_customer_state(ts, summary_path)
    if state is None:
        return f"{ts}: no customer state — verify the session context first"
    if not state.verified:
        return f"{ts}: customer state is unverified ({state.label}), not publishing"
    if state.action != "link_existing" or not state.customer_path:
        return f"{ts}: no customer to publish to ({state.action})"

    pipeline.write_customer_metadata(
        summary_path, pipeline.metadata_from_state(summary_path, state)
    )
    return f"{ts}: linked into {state.customer_path.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish summaries into their customer notes")
    parser.add_argument("target", nargs="?", help="Summary or transcript file, or session id (default: latest)")
    parser.add_argument("--all", action="store_true", help="Publish every summary still waiting for it")
    args = parser.parse_args()

    if args.all:
        pending = [rec for rec in pipeline.scan_recordings() if rec.needs_link]
        if not pending:
            print("Nothing to link.")
            return 0
        for rec in pending:
            print(link_summary(rec.summary_md))
        return 0

    if args.target and pipeline.extract_timestamp(args.target) == args.target:
        summary = summary_path_for(args.target)
    else:
        transcript = resolve_transcript_md(load_config().transcripts_dir, args.target)
        summary = summary_path_for(pipeline.extract_timestamp(transcript.name))

    try:
        print(link_summary(summary))
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
