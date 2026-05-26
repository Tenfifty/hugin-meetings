from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugin_meetings_gnome.install import _launcher_path, _render, main


def _make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class RenderTests(unittest.TestCase):
    def test_render_substitutes_launcher_absolute_path(self) -> None:
        out = _render(Path("/some/where/launcher.sh"), autostart=False)
        self.assertIn("Exec=/some/where/launcher.sh", out)
        self.assertNotIn("@LAUNCHER@", out)

    def test_render_can_pin_virtualenv(self) -> None:
        out = _render(
            Path("/some/where/launcher.sh"),
            autostart=False,
            venv=Path("/opt/hugin-venv"),
        )
        self.assertIn(
            "Exec=env HUGIN_MEETINGS_VENV=/opt/hugin-venv /some/where/launcher.sh",
            out,
        )

    def test_autostart_adds_xdg_autostart_marker(self) -> None:
        autostart = _render(Path("/x/launcher.sh"), autostart=True)
        menu = _render(Path("/x/launcher.sh"), autostart=False)
        self.assertIn("X-GNOME-Autostart-enabled=true", autostart)
        self.assertNotIn("X-GNOME-Autostart-enabled", menu)

    def test_both_variants_include_required_keys(self) -> None:
        for autostart in (True, False):
            out = _render(Path("/x/launcher.sh"), autostart=autostart)
            for key in ("Type=Application", "Name=Hugin Recorder", "Icon="):
                self.assertIn(key, out)


class LauncherPathTests(unittest.TestCase):
    def test_resolves_relative_to_package(self) -> None:
        # The bundled launcher lives at <repo>/frontends/gnome/desktop/launcher.sh
        launcher = _launcher_path()
        self.assertTrue(launcher.name == "launcher.sh")
        self.assertTrue(launcher.exists(), f"missing: {launcher}")
        self.assertTrue(os.access(launcher, os.X_OK))


class MainCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.app_menu = self.base / "applications"
        self.autostart = self.base / "autostart"
        self.launcher = _make_executable(self.base / "launcher.sh")
        self.addCleanup(self.tmp.cleanup)

    def _run(self, *extra: str) -> int:
        return main([
            "--launcher", str(self.launcher),
            "--app-menu-dir", str(self.app_menu),
            "--autostart-dir", str(self.autostart),
            *extra,
        ])

    def test_writes_both_desktop_files_with_absolute_launcher(self) -> None:
        rc = self._run()
        self.assertEqual(rc, 0)
        menu = (self.app_menu / "hugin-recorder.desktop").read_text()
        autostart = (self.autostart / "hugin-recorder.desktop").read_text()
        self.assertIn(f"Exec={self.launcher}", menu)
        self.assertIn(f"Exec={self.launcher}", autostart)
        self.assertIn("X-GNOME-Autostart-enabled=true", autostart)
        self.assertNotIn("X-GNOME-Autostart-enabled", menu)

    def test_venv_flag_writes_env_into_desktop_file(self) -> None:
        venv = self.base / "venv"
        rc = self._run("--venv", str(venv), "--no-autostart")
        self.assertEqual(rc, 0)
        menu = (self.app_menu / "hugin-recorder.desktop").read_text()
        self.assertIn(f"Exec=env HUGIN_MEETINGS_VENV={venv} {self.launcher}", menu)

    def test_no_autostart_flag_skips_autostart_file(self) -> None:
        self._run("--no-autostart")
        self.assertTrue((self.app_menu / "hugin-recorder.desktop").exists())
        self.assertFalse((self.autostart / "hugin-recorder.desktop").exists())

    def test_dry_run_writes_nothing(self) -> None:
        rc = self._run("--dry-run")
        self.assertEqual(rc, 0)
        self.assertFalse(self.app_menu.exists())
        self.assertFalse(self.autostart.exists())

    def test_existing_files_preserved_without_force(self) -> None:
        self.app_menu.mkdir(parents=True)
        target = self.app_menu / "hugin-recorder.desktop"
        target.write_text("user edits")
        self._run("--no-autostart")
        self.assertEqual(target.read_text(), "user edits")

    def test_force_overwrites(self) -> None:
        self.app_menu.mkdir(parents=True)
        target = self.app_menu / "hugin-recorder.desktop"
        target.write_text("user edits")
        self._run("--no-autostart", "--force")
        self.assertIn(f"Exec={self.launcher}", target.read_text())

    def test_missing_launcher_errors(self) -> None:
        rc = main([
            "--launcher", str(self.base / "nope.sh"),
            "--app-menu-dir", str(self.app_menu),
            "--autostart-dir", str(self.autostart),
        ])
        self.assertEqual(rc, 2)

    def test_non_executable_launcher_errors(self) -> None:
        plain = self.base / "plain.sh"
        plain.write_text("not executable")
        rc = main([
            "--launcher", str(plain),
            "--app-menu-dir", str(self.app_menu),
            "--autostart-dir", str(self.autostart),
        ])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
