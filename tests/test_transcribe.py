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
