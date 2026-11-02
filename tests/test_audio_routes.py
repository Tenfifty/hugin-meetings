from __future__ import annotations

import unittest
from unittest.mock import patch

from hugin_meetings import audio_routes


class PipeWireUnavailableTests(unittest.TestCase):
    def test_no_current_routes_without_pipewire(self) -> None:
        with patch.object(audio_routes, "load_pipewire_nodes", return_value=None):
            self.assertIsNone(audio_routes.current_audio_routes())

    def test_default_routes_fall_back_to_generic_sources(self) -> None:
        with patch.object(audio_routes, "load_pipewire_nodes", return_value=None):
            self.assertEqual(
                audio_routes.get_default_audio_routes(),
                (audio_routes.DEFAULT_PULSE_SOURCE, audio_routes.DEFAULT_MONITOR_SOURCE),
            )


if __name__ == "__main__":
    unittest.main()
