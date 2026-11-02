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


def test_subtitle_artifacts_are_dropped_only_as_whole_segments():
    # Every variant seen in the transcripts up to 2026-11-02, and the real
    # speech that mentions the same words.
    artifacts = [
        "Textning.nu",
        " Textning.nu",
        "Undertexter från Amara.org-gemenskapen",
        "Tack för att du tittade på den här videon!",
        "Tack för att ni tittade på den här videon.",
    ]
    speech = [
        "Men vill man ha en testtjänst från dem så prenumererar man på det.",
        "Textning ska vara längre och man kan också av pedagogiska skäl lägga en liten progressbar på videorna när de kör.",
        "Den kunde haft en sån här som undertexter har, en vit skugga.",
        "Kolla på Textning.nu sen.",
    ]
    result = {"language": "sv", "segments": [{"text": t} for t in artifacts + speech]}
    kept = transcribe.drop_subtitle_artifacts(result)
    assert [s["text"] for s in kept["segments"]] == speech
    assert kept["language"] == "sv"


def test_subtitle_markup_goes_and_is_cleaned_out_of_speech():
    # KB-Whisper's kind, as found in the transcripts: entities without their
    # semicolon inside speech, loops, and <i> tags.
    segments = [
        {"start": 0.0, "end": 1.0, "text": " &amp;"},
        {"start": 1.0, "end": 2.0, "text": "<i><i><i><i><i>"},
        {"start": 2.0, "end": 3.0, "text": "&lt&gt&gt&lt&gt&lt&gt&lt"},
        {"start": 3.0, "end": 4.0, "text": " Så om ni gör mycket M&amp,A och det är stora pengar."},
        {"start": 4.0, "end": 5.0, "text": "Nja, så &lt&gt&gt&lt&gt&lt&gt&lt&gt&lt&gt&lt&lt&gt"},
        {"start": 5.0, "end": 6.0, "text": "<i>Det här är sant.</i>"},
        {"start": 6.0, "end": 7.0, "text": " ... ... ..."},
        {"start": 7.0, "end": 8.0, "text": " Smith &amp;amp, ConMed och Arthrex."},
    ]
    kept = transcribe.drop_subtitle_artifacts({"segments": segments})["segments"]
    assert [s["text"] for s in kept] == [
        " Så om ni gör mycket M&,A och det är stora pengar.",
        "Nja, så ",
        "Det här är sant.",
        # Not markup: left for whoever decides about empty segments.
        " ... ... ...",
        " Smith &, ConMed och Arthrex.",
    ]
    assert [s["start"] for s in kept] == [3.0, 4.0, 5.0, 6.0, 7.0]
