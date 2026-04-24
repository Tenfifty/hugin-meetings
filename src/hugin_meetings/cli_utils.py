"""Small helpers shared by command-line and desktop entry points."""

from __future__ import annotations

import sys
from pathlib import Path


def resolve_sibling_bin(name: str) -> str:
    """Prefer a helper binary installed next to the current interpreter."""
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else name
