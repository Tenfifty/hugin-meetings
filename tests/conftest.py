"""Point the whole suite at a throwaway config before anything imports hugin_meetings.

Most modules read ``load_config()`` at import time into module constants
(``pipeline.CUSTOMERS_DIR``, ``transcribe.SPEAKERS_DIR``, ...). Without this, a
test that forgets to patch one of them reads or writes the real vault, the
real ``~/.hugin_audio`` or the real archive. Here every configured directory
lives in a temporary sandbox instead, so such a test touches nothing that
matters. Tests should still patch what they use; this is the backstop.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

_SANDBOX = Path(tempfile.mkdtemp(prefix="hugin-meetings-tests-"))
atexit.register(shutil.rmtree, _SANDBOX, ignore_errors=True)

_config_dir = _SANDBOX / "config"
_config_dir.mkdir()
(_config_dir / "hugin.yaml").write_text(
    f"archive_root: {_SANDBOX / 'archive'}\nvault_path: {_SANDBOX / 'vault'}\n"
)
(_config_dir / "meetings.yaml").write_text(
    "meetings:\n"
    f"  transcripts_dir: {_SANDBOX / 'vault' / 'transcripts'}\n"
    f"  summaries_dir: {_SANDBOX / 'vault' / 'summaries'}\n"
    f"  state_dir: {_SANDBOX / 'state'}\n"
    "  project_matcher:\n"
    f"    projects_dir: {_SANDBOX / 'vault' / 'projects'}\n"
)
os.environ["HUGIN_CONFIG_DIR"] = str(_config_dir)
