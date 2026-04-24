#!/usr/bin/env python3
"""Summarize a meeting transcript using a local LLM (Gemma 4 via llama.cpp).

Usage:
    summarize.py                                    # latest transcript
    summarize.py transcript-20260409-100207.md
    summarize.py --all                              # all unsummarized
"""

import argparse
import sys
import tempfile
from pathlib import Path

from .config import load_config
_cfg = load_config()

TRANSCRIPT_DIR = _cfg.transcripts_dir
SUMMARY_DIR = _cfg.summaries_dir
MODELS_DIR = _cfg.models_dir
LOCAL_MODELS = {
    "small": MODELS_DIR / "gemma-4-E4B-it-Q4_K_M.gguf",
    "large": MODELS_DIR / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
}
# codex exec models
CODEX_MODELS = {"gpt-5.4", "gpt-5.4-mini"}
DEFAULT_MODEL = "gpt-5.4"
CODEX_CLEAN_CWD = Path(tempfile.gettempdir()) / "codex-clean"

_DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "summary_default.md"


def _load_prompt() -> str:
    override = _cfg.raw.get("meetings", {}).get("summarize_prompt_path")
    path = Path(override).expanduser() if override else _DEFAULT_PROMPT_PATH
    return path.read_text(encoding="utf-8")


SYSTEM_PROMPT = _load_prompt()


def load_local_model(model_key: str):
    from llama_cpp import Llama

    model_path = LOCAL_MODELS.get(model_key)
    if not model_path or not model_path.exists():
        print(f"Model not found: {model_key} ({model_path})", file=sys.stderr)
        print(f"Available: {', '.join(k for k, v in LOCAL_MODELS.items() if v.exists())}")
        sys.exit(1)

    # 26B MoE needs hybrid CPU/GPU (won't fit in 8GB VRAM)
    is_hybrid = model_path.stat().st_size > 10 * 1024**3

    print(f"Loading model: {model_path.name}")
    return Llama(
        model_path=str(model_path),
        n_gpu_layers=-1 if not is_hybrid else 10,
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


def summarize_codex(codex_model: str, transcript_text: str) -> str:
    import subprocess

    prompt = f"{SYSTEM_PROMPT}\n\nTranscript:\n\n{transcript_text}"

    model_id = codex_model

    CODEX_CLEAN_CWD.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as out:
        out_path = out.name

    print(f"  Running codex exec with model: {model_id}")
    result = subprocess.run(
        [
            "codex", "exec",
            "-m", model_id,
            "-C", str(CODEX_CLEAN_CWD),
            "-c", "model_reasoning_effort=medium",
            "--skip-git-repo-check",
            "--ephemeral",
            "-o", out_path,
            "-",
        ],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        print(f"  codex exec failed: {result.stderr[:500]}", file=sys.stderr)
        return ""

    text = Path(out_path).read_text()
    Path(out_path).unlink(missing_ok=True)
    return text.strip()


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

    if model_key not in CODEX_MODELS:
        est_tokens = len(transcript_text) / 3.5
        if est_tokens > 5500:
            print(f"  Warning: transcript is ~{int(est_tokens)} tokens, may be truncated")

    print(f"  Summarizing: {md_path.name}")
    if model_key in CODEX_MODELS:
        summary = summarize_codex(model_key, transcript_text)
    else:
        summary = summarize_local(model, transcript_text)

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
    all_models = list(LOCAL_MODELS.keys()) + sorted(CODEX_MODELS)
    parser = argparse.ArgumentParser(description="Summarize meeting transcripts")
    parser.add_argument("transcript", nargs="?", help="Transcript .md file (default: latest)")
    parser.add_argument("--all", action="store_true", help="Summarize all unsummarized transcripts")
    parser.add_argument("--model", choices=all_models, default=DEFAULT_MODEL,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    model = None
    if args.model not in CODEX_MODELS:
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
