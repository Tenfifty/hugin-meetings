#!/usr/bin/env python3
"""Summarize a meeting transcript with the configured LLM provider (codex / claude / agy).

Usage:
    summarize.py                                    # latest transcript
    summarize.py transcript-20260409-100207.md
    summarize.py --all                              # all unsummarized
"""

import argparse
from pathlib import Path

from hugin.llm import run_prompt
from hugin.prompts import resolve_prompt

from .cli_utils import resolve_transcript_md
from .config import load_config
from .pipeline import extract_timestamp, summary_md_path

TRANSCRIPT_DIR = load_config().transcripts_dir
SUMMARY_DIR = load_config().summaries_dir
DEFAULT_MODEL = load_config().summary_model

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt() -> str:
    cfg = load_config()
    path = resolve_prompt(
        base="summary",
        language=cfg.language,
        explicit=cfg.summarize_prompt_path,
        package_dir=_PROMPTS_DIR,
    )
    text = path.read_text(encoding="utf-8")
    return text.replace("{{user_name}}", cfg.user_name or "mig")


def summarize_remote(model_id: str, transcript_text: str) -> str:
    prompt = f"{_load_prompt()}\n\nTranscript:\n\n{transcript_text}"

    print(f"  Running {load_config().llm.provider} with model: {model_id}")
    # A provider failure used to print a line and write an empty summary,
    # which link could then publish. It now stops the stage instead.
    summary = run_prompt(load_config().llm, model_id, prompt, effort=load_config().summary_effort)
    if not summary.strip():
        raise RuntimeError(f"{load_config().llm.provider} returned an empty summary")
    return summary


def process_transcript(model_key: str, md_path: Path):
    out_path = summary_md_path(extract_timestamp(md_path.name))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing = out_path.read_text().strip()
        if existing:
            print(f"  Already summarized: {out_path.name}")
            return
        print(f"  Replacing empty summary: {out_path.name}")

    transcript_text = md_path.read_text()

    print(f"  Summarizing: {md_path.name}")
    summary = summarize_remote(model_key, transcript_text)

    out_path.write_text(summary + "\n")
    print(f"  Wrote {out_path}")


def find_unsummarized() -> list[Path]:
    summarized = {
        p.stem.removeprefix("summary-")
        for p in SUMMARY_DIR.rglob("summary-*.md")
        if p.read_text().strip()
    }
    transcripts = sorted(TRANSCRIPT_DIR.rglob("transcript-*.md"))
    return [t for t in transcripts if t.stem.removeprefix("transcript-") not in summarized]


def main():
    parser = argparse.ArgumentParser(description="Summarize meeting transcripts")
    parser.add_argument("transcript", nargs="?", help="Transcript .md file (default: latest)")
    parser.add_argument("--all", action="store_true", help="Summarize all unsummarized transcripts")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model for the configured provider (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    if args.all:
        unsummarized = find_unsummarized()
        if not unsummarized:
            print("Nothing to summarize.")
            return
        print(f"Found {len(unsummarized)} unsummarized transcript(s)")
        for md_path in unsummarized:
            process_transcript(args.model, md_path)
    else:
        md_path = resolve_transcript_md(args.transcript)
        process_transcript(args.model, md_path)


if __name__ == "__main__":
    main()
