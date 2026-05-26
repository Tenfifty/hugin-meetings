"""``hugin-meet-install-gnome-tray`` — install the tray .desktop files.

Writes ``hugin-recorder.desktop`` to both ``~/.local/share/applications/``
(so the tray shows up in the GNOME activities menu) and
``~/.config/autostart/`` (so it launches on login). The ``Exec=`` line
points at the launcher script inside this package's repo checkout —
absolute path is substituted in at install time so neither file
depends on PATH lookups.

Idempotent: existing files are left alone unless ``--force``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

DESKTOP_FILENAME = "hugin-recorder.desktop"


def _launcher_path() -> Path:
    """Resolve the launcher.sh that ships with this package.

    Package layout: ``frontends/gnome/src/hugin_meetings_gnome/install.py``
    → launcher at ``frontends/gnome/desktop/launcher.sh``.
    """
    pkg_dir = Path(__file__).resolve().parent
    return pkg_dir.parent.parent / "desktop" / "launcher.sh"


def _render(launcher: Path, *, autostart: bool, venv: Path | None = None) -> str:
    exec_line = f"Exec={launcher}"
    if venv is not None:
        exec_line = f"Exec=env HUGIN_MEETINGS_VENV={venv} {launcher}"
    lines = [
        "[Desktop Entry]",
        "Name=Hugin Recorder",
        "Comment=Audio capture indicator for mic and system audio",
        exec_line,
        "Icon=audio-input-microphone",
        "Type=Application",
    ]
    if autostart:
        lines.append("X-GNOME-Autostart-enabled=true")
    else:
        # App-menu-only categorisation and startup behaviour.
        lines.extend(["Categories=AudioVideo;Utility;", "StartupNotify=false"])
    return "\n".join(lines) + "\n"


@dataclass
class Target:
    path: Path
    autostart: bool


def _targets(no_autostart: bool, app_menu_dir: Path, autostart_dir: Path) -> list[Target]:
    targets = [Target(app_menu_dir / DESKTOP_FILENAME, autostart=False)]
    if not no_autostart:
        targets.append(Target(autostart_dir / DESKTOP_FILENAME, autostart=True))
    return targets


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hugin-meet-install-gnome-tray",
        description="Install the Hugin Recorder tray .desktop files.",
    )
    p.add_argument(
        "--launcher",
        type=Path,
        default=None,
        help="Path to launcher.sh (default: bundled with this package)",
    )
    p.add_argument(
        "--venv",
        type=Path,
        default=None,
        help="Set HUGIN_MEETINGS_VENV in the generated .desktop files",
    )
    p.add_argument(
        "--no-autostart",
        action="store_true",
        help="Skip the autostart copy under ~/.config/autostart/",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    p.add_argument("--dry-run", action="store_true", help="Show actions, write nothing")
    p.add_argument(
        "--app-menu-dir",
        type=Path,
        default=Path.home() / ".local" / "share" / "applications",
        help=argparse.SUPPRESS,  # tests only
    )
    p.add_argument(
        "--autostart-dir",
        type=Path,
        default=Path.home() / ".config" / "autostart",
        help=argparse.SUPPRESS,  # tests only
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    launcher = (args.launcher or _launcher_path()).resolve()
    if not launcher.exists():
        print(f"Launcher not found: {launcher}", file=sys.stderr)
        return 2
    if not os.access(launcher, os.X_OK):
        print(f"Launcher not executable: {launcher}", file=sys.stderr)
        return 2
    venv = args.venv.expanduser().resolve() if args.venv else None

    print(f"Launcher: {launcher}")
    if venv is not None:
        print(f"Virtualenv: {venv}")
    targets = _targets(args.no_autostart, args.app_menu_dir, args.autostart_dir)

    wrote = 0
    for target in targets:
        marker = "+"
        suffix = ""
        if target.path.exists() and not args.force:
            marker = "."
            suffix = " (exists; --force to overwrite)"
            print(f"  {marker} {target.path}{suffix}")
            continue
        if args.dry_run:
            print(f"  {marker} {target.path} (dry-run)")
            continue

        target.path.parent.mkdir(parents=True, exist_ok=True)
        target.path.write_text(
            _render(launcher, autostart=target.autostart, venv=venv),
            encoding="utf-8",
        )
        print(f"  {marker} {target.path}")
        wrote += 1

    if args.dry_run:
        print("Dry run — nothing written.")
    elif wrote == 0:
        print("Nothing to do. Pass --force to overwrite existing files.")
    else:
        print(f"Wrote {wrote} file(s). Log out and back in to pick up the autostart entry.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
