"""``hugin-meet-migrate-yyyy`` — move timestamped files into YYYY/ subdirs.

Idempotent two-stage migration:

1. **Files** — for each of ``transcripts_dir`` / ``summaries_dir`` /
   ``raw_audio_dir`` / ``transcript_json_dir``, move every timestamped
   file that lives directly in the dir into a ``YYYY/`` subdir
   (year extracted from the filename's ``YYYYMMDD-HHMMSS`` timestamp).
2. **Vault links** — rewrite Markdown ``[text](audio/X/file)`` and
   Obsidian ``[[audio/X/file]]`` references so they include the
   matching ``YYYY/`` component.

Default mode is dry-run. Pass ``--apply`` to commit. Vault files are
backed up to a timestamped directory under the meetings ``state_dir``
before being rewritten.

Stop the recorder daemon before running this — otherwise an in-progress
recording can race with the file moves. The migration is re-runnable:
files already in YYYY/ and links already containing a year segment are
left alone.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import load_config
from .pipeline import TS_RE, year_subdir


# Matches the path part of a link to transcripts/summaries — handles
# `[Transcript](../../audio/transcripts/transcript-20260524-123456.md)`,
# `[[audio/summaries/summary-20260524-123456]]`, and bare-path lists.
# We only match when there is NO year directory between the kind and
# the filename, so the substitution is idempotent.
VAULT_LINK_RE = re.compile(
    r"(audio/(?:transcripts|summaries)/)"
    r"((?:transcript|summary)-(\d{4})\d{4}-\d{6}(?:\.md)?)"
)


@dataclass
class Move:
    src: Path
    dst: Path


@dataclass
class LinkChange:
    path: Path
    old_text: str
    new_text: str

    @property
    def n_substitutions(self) -> int:
        return len(VAULT_LINK_RE.findall(self.old_text))


def _is_already_nested(path: Path, root: Path) -> bool:
    """True if ``path`` sits under ``root/YYYY/<file>``."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    parts = rel.parts
    return len(parts) >= 2 and len(parts[0]) == 4 and parts[0].isdigit()


def plan_moves(dirs: list[Path]) -> list[Move]:
    moves: list[Move] = []
    for d in dirs:
        if not d.exists():
            continue
        for child in sorted(d.iterdir()):
            if not child.is_file():
                continue
            ts = TS_RE.search(child.name)
            if not ts:
                continue
            yyyy = year_subdir(child.name)
            target = d / yyyy / child.name
            if child == target:
                continue
            moves.append(Move(src=child, dst=target))
    return moves


def _rewrite_text(text: str) -> str:
    return VAULT_LINK_RE.sub(r"\1\3/\2", text)


def plan_link_rewrites(vault: Path) -> list[LinkChange]:
    changes: list[LinkChange] = []
    if not vault.exists():
        return changes
    for path in sorted(vault.rglob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "audio/transcripts/" not in text and "audio/summaries/" not in text:
            continue
        new = _rewrite_text(text)
        if new != text:
            changes.append(LinkChange(path=path, old_text=text, new_text=new))
    return changes


def _print_plan(moves: list[Move], link_changes: list[LinkChange]) -> None:
    if moves:
        print(f"\n{len(moves)} file(s) to move:")
        by_dir: dict[Path, list[Move]] = {}
        for m in moves:
            by_dir.setdefault(m.src.parent, []).append(m)
        for d, mlist in sorted(by_dir.items()):
            print(f"  {d}  ({len(mlist)} files)")
            for m in mlist[:3]:
                print(f"    -> {m.dst.relative_to(d)}")
            if len(mlist) > 3:
                print(f"    ... and {len(mlist) - 3} more")
    else:
        print("\nNo files to move.")

    total_subs = sum(c.n_substitutions for c in link_changes)
    if link_changes:
        print(f"\n{len(link_changes)} vault file(s) with {total_subs} link substitution(s):")
        for c in link_changes:
            print(f"  {c.path}  ({c.n_substitutions} substitution(s))")
    else:
        print("\nNo vault links to rewrite.")


def _apply_moves(moves: list[Move]) -> int:
    n = 0
    for m in moves:
        if m.dst.exists():
            print(f"  skip (target exists): {m.dst}", file=sys.stderr)
            continue
        m.dst.parent.mkdir(parents=True, exist_ok=True)
        m.src.rename(m.dst)
        n += 1
    return n


def _apply_link_rewrites(link_changes: list[LinkChange], backup_root: Path, vault: Path) -> int:
    if not link_changes:
        return 0
    backup_root.mkdir(parents=True, exist_ok=True)
    print(f"\nVault file backups -> {backup_root}")
    n = 0
    for c in link_changes:
        rel = c.path.relative_to(vault) if c.path.is_relative_to(vault) else Path(c.path.name)
        backup = backup_root / rel
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(c.path, backup)
        c.path.write_text(c.new_text, encoding="utf-8")
        n += 1
    return n


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hugin-meet-migrate-yyyy",
        description="Move per-session files into YYYY/ subdirs and rewrite vault links.",
    )
    p.add_argument("--apply", action="store_true", help="Commit changes (default: dry-run)")
    p.add_argument("--no-files", action="store_true", help="Skip the file-move stage")
    p.add_argument("--no-links", action="store_true", help="Skip the vault-link stage")
    p.add_argument("--vault", type=Path, default=None, help="Override vault_path from config")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config()
    vault = args.vault or cfg.vault_path

    dirs = [cfg.transcripts_dir, cfg.summaries_dir, cfg.raw_audio_dir, cfg.transcript_json_dir]
    print("Migration target directories:")
    for d in dirs:
        print(f"  {d}")
    print(f"Vault: {vault or '(not configured — skipping links)'}")

    moves = [] if args.no_files else plan_moves(dirs)
    link_changes = [] if args.no_links or not vault else plan_link_rewrites(vault)

    _print_plan(moves, link_changes)

    if not args.apply:
        print("\nDry run — nothing written. Re-run with --apply to commit.")
        return 0

    moved = _apply_moves(moves)
    backup_root = cfg.state_dir / f"migrate-yyyy-backup-{dt.datetime.now():%Y%m%d-%H%M%S}"
    rewritten = _apply_link_rewrites(link_changes, backup_root, vault) if vault else 0
    print(f"\nDone. Moved {moved} file(s), rewrote {rewritten} vault file(s).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
