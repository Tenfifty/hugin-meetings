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


def _resolve_session_file(name: str | None, kind: str, directory: Path, path_for) -> Path:
    """A session file by path, by any name holding its timestamp, or the latest."""
    from .pipeline import extract_timestamp

    if name is None:
        files = sorted(directory.rglob(f"{kind}-*.md"))
        if not files:
            print(f"No {kind}s found.", file=sys.stderr)
            sys.exit(1)
        return files[-1]
    path = Path(name)
    if path.exists():
        return path
    ts = extract_timestamp(path.name)
    if ts and path_for(ts).exists():
        return path_for(ts)
    print(f"{kind.capitalize()} not found: {name}", file=sys.stderr)
    sys.exit(1)


def resolve_transcript_md(name: str | None) -> Path:
    """Resolve a transcript .md file by path, filename, timestamp, or latest."""
    from . import pipeline

    return _resolve_session_file(name, "transcript", pipeline.TRANSCRIPT_DIR, pipeline.transcript_md_path)


def resolve_summary_md(name: str | None) -> Path:
    """Resolve a summary .md file by path, filename, timestamp, or latest."""
    from . import pipeline

    return _resolve_session_file(name, "summary", pipeline.SUMMARY_DIR, pipeline.summary_md_path)
