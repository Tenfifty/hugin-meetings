from __future__ import annotations

import pytest

from hugin_meetings import recording


def make_track(tmp_path, name="mic-20260424-101112-p01.opus"):
    track = recording.RecordingTrack(
        "mic", "", audio_dir=tmp_path / "archive" / "raw", recording_dir=tmp_path / "recording"
    )
    track.current_file = tmp_path / "recording" / name
    track.current_file.parent.mkdir(parents=True)
    track.current_file.write_bytes(b"opus")
    return track


def test_a_part_is_written_outside_the_archive() -> None:
    path = recording.recording_part_path(recording.RECORDING_DIR, "sys", "20260424-101112", 3)
    assert path == recording.RECORDING_DIR / "sys-20260424-101112-p03.opus"


def test_a_stopped_part_moves_into_the_archive(tmp_path) -> None:
    track = make_track(tmp_path)

    track.stop_segment()

    archived = tmp_path / "archive" / "raw" / "2026" / "mic-20260424-101112-p01.opus"
    assert archived.read_bytes() == b"opus"
    assert list((tmp_path / "recording").iterdir()) == []
    assert track.current_file == archived


def test_an_archived_part_is_never_replaced(tmp_path, caplog) -> None:
    track = make_track(tmp_path)
    existing = tmp_path / "archive" / "raw" / "2026" / "mic-20260424-101112-p01.opus"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"first")

    with pytest.raises(FileExistsError):
        recording.archive_part(track.current_file, track.audio_dir)
    # stop_segment must not raise, or a rotation would stop the recording.
    track.stop_segment()

    assert existing.read_bytes() == b"first"
    assert track.current_file.exists()
    assert "Could not move finished part" in caplog.text


def test_leftover_parts_are_what_the_recording_dir_holds(tmp_path) -> None:
    assert recording.leftover_parts(tmp_path / "missing") == []
    track = make_track(tmp_path)
    (tmp_path / "recording" / "20260424-101112.lock").touch()
    assert recording.leftover_parts(tmp_path / "recording") == [track.current_file]
