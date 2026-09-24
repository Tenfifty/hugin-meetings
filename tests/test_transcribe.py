from pathlib import Path

from hugin_meetings import transcribe


def test_diarize_skips_empty_transcription() -> None:
    result = {"segments": [], "language": "sv"}

    class UnexpectedDiarizer:
        def __call__(self, _path: str):
            raise AssertionError("diarizer must not run without transcript segments")

    assert transcribe.diarize(
        Path("silent.opus"),
        result,
        "cpu",
        "nemo",
        UnexpectedDiarizer(),
        None,
    ) is result


def test_nemo_all_silence_drops_hallucinated_segments(monkeypatch) -> None:
    result = {
        "segments": [{"start": 0.0, "end": 5.0, "text": " Undertexter från Amara.org-gemenskapen"}],
        "language": "sv",
    }
    monkeypatch.setattr(transcribe, "_ensure_pcm_wav", lambda path: path)
    monkeypatch.setattr(transcribe, "audio_duration", lambda path: 6.1)
    monkeypatch.setattr(transcribe, "_cuda_memory_stats", lambda: "")

    class SilentDiarizer:
        def __call__(self, _path: str):
            raise ValueError("All files present in manifest contains silence, aborting next steps")

    out = transcribe.diarize(Path("sys-p03.opus"), result, "cpu", "nemo", SilentDiarizer(), None)
    assert out["segments"] == []


def test_nemo_other_value_errors_propagate(monkeypatch) -> None:
    import pytest

    monkeypatch.setattr(transcribe, "_ensure_pcm_wav", lambda path: path)
    monkeypatch.setattr(transcribe, "audio_duration", lambda path: 6.1)
    monkeypatch.setattr(transcribe, "_cuda_memory_stats", lambda: "")

    class BrokenDiarizer:
        def __call__(self, _path: str):
            raise ValueError("something else")

    with pytest.raises(ValueError, match="something else"):
        transcribe.diarize(
            Path("sys-p03.opus"), {"segments": [{"text": "hej"}]}, "cpu", "nemo", BrokenDiarizer(), None,
        )
