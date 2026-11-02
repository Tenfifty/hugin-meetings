from hugin_meetings import transcribe


def _seg(start, text, speaker="speaker_0"):
    return {"start": start, "end": start + 2.0, "text": f" {text} ", "speaker": speaker}


def test_merge_drops_mic_echoes_of_sys_and_keeps_the_rest():
    """The text dedup is the backstop behind echo cancellation (see echo.py)."""
    sys_result = {"segments": [_seg(10.0, "vi borde titta på budgeten"), _seg(40.0, "okej")]}
    mic_result = {"segments": [
        _seg(10.4, "vi borde titta på budgeten"),  # the speaker heard through the mic
        _seg(12.0, "absolut, jag skickar den"),     # the local person answering
        _seg(30.0, "vi borde titta på budgeten"),  # said again locally, 20 s later
    ]}

    merged = transcribe.merge_channels(mic_result, sys_result)

    assert [(e["channel"], e["start"]) for e in merged] == [
        ("sys", 10.0), ("mic", 12.0), ("mic", 30.0), ("sys", 40.0),
    ]
    assert merged[0]["text"] == "vi borde titta på budgeten"


def test_merge_without_sys_keeps_the_mic_as_is():
    mic_result = {"segments": [_seg(0.0, "hej")]}
    assert [e["channel"] for e in transcribe.merge_channels(mic_result, None)] == ["mic"]
