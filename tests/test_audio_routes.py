from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from hugin_meetings import audio_routes


class PulseMonitorFallbackTests(unittest.TestCase):
    def test_default_pulse_monitor_uses_current_default_sink(self) -> None:
        result = subprocess.CompletedProcess(
            ["pactl", "get-default-sink"],
            0,
            stdout="alsa_output.usb-speakers\n",
            stderr="",
        )
        with patch.object(audio_routes.subprocess, "run", return_value=result):
            self.assertEqual(
                audio_routes.default_pulse_monitor_source(),
                "alsa_output.usb-speakers.monitor",
            )

    def test_default_pulse_monitor_has_generic_fallback(self) -> None:
        with patch.object(audio_routes.subprocess, "run", side_effect=FileNotFoundError):
            self.assertEqual(
                audio_routes.default_pulse_monitor_source(),
                audio_routes.DEFAULT_MONITOR_SOURCE,
            )

    def test_audio_routes_use_pactl_when_pipewire_dump_is_unavailable(self) -> None:
        with (
            patch.object(audio_routes, "load_pipewire_nodes", return_value=None),
            patch.object(
                audio_routes,
                "default_pulse_monitor_source",
                return_value="sink.monitor",
            ),
        ):
            self.assertEqual(
                audio_routes.get_default_audio_routes(log=False),
                (audio_routes.DEFAULT_PULSE_SOURCE, "sink.monitor"),
            )


if __name__ == "__main__":
    unittest.main()
