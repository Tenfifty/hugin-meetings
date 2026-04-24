#!/usr/bin/env python3
"""Summarize a meeting transcript using a local LLM (Gemma 4 via llama.cpp).

Usage:
    summarize.py                                    # latest transcript
    summarize.py transcript-20260409-100207.md
    summarize.py --all                              # all unsummarized
"""

import argparse
import sys
from pathlib import Path

from .config import load_config
from .remote_llm import run_prompt
_cfg = load_config()

TRANSCRIPT_DIR = _cfg.transcripts_dir
SUMMARY_DIR = _cfg.summaries_dir
MODELS_DIR = _cfg.models_dir
LOCAL_MODELS = {
    "small": MODELS_DIR / "gemma-4-E4B-it-Q4_K_M.gguf",
    "large": MODELS_DIR / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
}
DEFAULT_MODEL = _cfg.summary_model

_DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "summary_default.md"


def _load_prompt() -> str:
    path = _cfg.summarize_prompt_path or _DEFAULT_PROMPT_PATH
    return path.read_text(encoding="utf-8")


SYSTEM_PROMPT = _load_prompt()


def load_local_model(model_key: str):
    from llama_cpp import Llama

    model_path = LOCAL_MODELS.get(model_key)
    if not model_path or not model_path.exists():
        print(f"Model not found: {model_key} ({model_path})", file=sys.stderr)
        print(f"Available: {', '.join(k for k, v in LOCAL_MODELS.items() if v.exists())}")
        sys.exit(1)

    threshold_bytes = int(_cfg.summarize_hybrid_threshold_gb * 1024**3)
    is_hybrid = model_path.stat().st_size > threshold_bytes
    if _cfg.summarize_n_gpu_layers is not None:
        n_gpu_layers = _cfg.summarize_n_gpu_layers
    else:
        n_gpu_layers = _cfg.summarize_hybrid_n_gpu_layers if is_hybrid else -1

    print(f"Loading model: {model_path.name} (n_gpu_layers={n_gpu_layers})")
    return Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=8192,
        flash_attn=not is_hybrid,
        verbose=False,
    )


def clean_summary_text(text: str) -> str:
    import re

    text = re.sub(r"(<unused\d+>)+", "", text)
    text = re.sub(r"<\|channel\|>[-*_\w]*thought[-*_\w]*\n?", "", text)
    text = re.sub(r"<channel\|>[-*_\w]*thought[-*_\w]*\n?", "", text)
    text = re.sub(r"<\|?channel\|?>", "", text)
    text = re.sub(r"^[-*_\w]*thought[-*_\w]*\n+", "", text)

    for marker in (
        _cfg.summary_header,
        "## Mötessammanfattning",
        "Här är en sammanfattning av mötet:",
        "**Sammanfattning",
        "Sammanfattning av mötet:",
    ):
        idx = text.find(marker)
        if idx > 0:
            text = text[idx:]
            break

    return text.strip()


def summarize_local(model, transcript_text: str) -> str:
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript_text},
        ],
        max_tokens=2048,
        temperature=0.3,
    )
    return clean_summary_text(response["choices"][0]["message"]["content"])


def summarize_remote(model_id: str, transcript_text: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nTranscript:\n\n{transcript_text}"

    print(f"  Running {_cfg.llm.provider} with model: {model_id}")
    try:
        return run_prompt(_cfg.llm, model_id, prompt, effort=_cfg.summary_effort)
    except RuntimeError as exc:
        print(f"  {_cfg.llm.provider} failed: {str(exc)[:500]}", file=sys.stderr)
        return ""


def resolve_transcript(name: str | None) -> Path:
    if name is None:
        files = sorted(TRANSCRIPT_DIR.glob("transcript-*.md"))
        if not files:
            print("No transcripts found.", file=sys.stderr)
            sys.exit(1)
        return files[-1]
    path = Path(name)
    if path.exists():
        return path
    if (TRANSCRIPT_DIR / path).exists():
        return TRANSCRIPT_DIR / path
    if (TRANSCRIPT_DIR / f"transcript-{path}").exists():
        return TRANSCRIPT_DIR / f"transcript-{path}"
    print(f"Transcript not found: {name}", file=sys.stderr)
    sys.exit(1)


def process_transcript(model_key: str, model, md_path: Path):
    ts = md_path.stem.removeprefix("transcript-")
    out_path = SUMMARY_DIR / f"summary-{ts}.md"

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing = out_path.read_text().strip()
        if existing:
            print(f"  Already summarized: {out_path.name}")
            return
        print(f"  Replacing empty summary: {out_path.name}")

    transcript_text = md_path.read_text()

    if model_key in LOCAL_MODELS:
        est_tokens = len(transcript_text) / 3.5
        if est_tokens > 5500:
            print(f"  Warning: transcript is ~{int(est_tokens)} tokens, may be truncated")

    print(f"  Summarizing: {md_path.name}")
    if model_key in LOCAL_MODELS:
        summary = summarize_local(model, transcript_text)
    else:
        summary = summarize_remote(model_key, transcript_text)

    out_path.write_text(summary + "\n")
    print(f"  Wrote {out_path}")


def find_unsummarized() -> list[Path]:
    summarized = {
        p.stem.removeprefix("summary-")
        for p in SUMMARY_DIR.glob("summary-*.md")
        if p.read_text().strip()
    }
    transcripts = sorted(TRANSCRIPT_DIR.glob("transcript-*.md"))
    return [t for t in transcripts if t.stem.removeprefix("transcript-") not in summarized]


def main():
    parser = argparse.ArgumentParser(description="Summarize meeting transcripts")
    parser.add_argument("transcript", nargs="?", help="Transcript .md file (default: latest)")
    parser.add_argument("--all", action="store_true", help="Summarize all unsummarized transcripts")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"Model to use (default: {DEFAULT_MODEL}; use small/large for local models, "
            "otherwise the configured remote provider is used)"
        ),
    )
    args = parser.parse_args()

    model = None
    if args.model in LOCAL_MODELS:
        model = load_local_model(args.model)

    if args.all:
        unsummarized = find_unsummarized()
        if not unsummarized:
            print("Nothing to summarize.")
            return
        print(f"Found {len(unsummarized)} unsummarized transcript(s)")
        for md_path in unsummarized:
            process_transcript(args.model, model, md_path)
    else:
        md_path = resolve_transcript(args.transcript)
        process_transcript(args.model, model, md_path)


if __name__ == "__main__":
    main()
