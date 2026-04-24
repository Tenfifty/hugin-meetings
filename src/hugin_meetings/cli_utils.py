"""Small helpers shared by command-line and desktop entry points."""

from __future__ import annotations

import sys
from pathlib import Path


def resolve_sibling_bin(name: str) -> str:
    """Prefer a helper binary installed next to the current interpreter."""
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else name


def get_hf_token() -> str | None:
    """Return the current HuggingFace token, if any."""
    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


def resolve_transcript_md(transcripts_dir: Path, name: str | None) -> Path:
    """Resolve a transcript .md file by filename, timestamp, or latest."""
    if name is None:
        files = sorted(transcripts_dir.glob("transcript-*.md"))
        if not files:
            print("No transcripts found.", file=sys.stderr)
            sys.exit(1)
        return files[-1]
    path = Path(name)
    if path.exists():
        return path
    for candidate in (transcripts_dir / path, transcripts_dir / f"transcript-{path}"):
        if candidate.exists():
            return candidate
    print(f"Transcript not found: {name}", file=sys.stderr)
    sys.exit(1)
