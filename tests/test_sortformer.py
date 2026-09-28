from pathlib import Path

import numpy as np

from hugin_meetings import sortformer as sf
from hugin_meetings import transcribe


def _voice(seed: int, dim: int = 16) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim)
    return v / np.linalg.norm(v)


def _pieces_for(segments, voice_of):
    """Embedding pieces for ``segments``, each embedded as its speaker's voice plus noise."""
    pieces = sf.embedding_pieces(segments)
    rng = np.random.default_rng(99)
    emb = np.stack([voice_of(p) + 0.05 * rng.normal(size=16) for p in pieces])
    return pieces, emb


def test_pieces_drop_overlap_and_fold_short_tail():
    segments = [(0.0, 7.0, "a"), (6.0, 8.0, "b")]
    pieces = sf.embedding_pieces(segments)
    # a: 0-6 alone -> two 3 s pieces; b: 7-8 alone is too short for a piece.
    assert pieces == [(0.0, 3.0, "a"), (3.0, 6.0, "a")]
    assert sf.embedding_pieces([(0.0, 7.0, "a")]) == [(0.0, 3.0, "a"), (3.0, 7.0, "a")]


def test_two_people_sharing_one_slot_are_split():
    # Sortformer ran out of slots: one label, David for 60 s then Casper for 60 s.
    segments = [(0.0, 120.0, "speaker_0")]
    david, casper = _voice(1), _voice(2)
    pieces, emb = _pieces_for(segments, lambda p: david if p[0] < 60 else casper)
    out = sf.refine_speakers(segments, pieces, emb)
    assert [(a, b) for a, b, _ in out] == [(0.0, 60.0), (60.0, 120.0)]
    assert out[0][2] != out[1][2]


def test_one_voice_on_two_slots_is_merged():
    # Filip came back after a silence in a new slot.
    segments = [(0.0, 60.0, "speaker_1"), (60.0, 90.0, "speaker_0"), (90.0, 150.0, "speaker_2")]
    filip, david = _voice(3), _voice(4)
    pieces, emb = _pieces_for(segments, lambda p: david if p[2] == "speaker_0" else filip)
    out = sf.refine_speakers(segments, pieces, emb)
    labels = [label for *_, label in out]
    assert labels[0] == labels[2] != labels[1]
    assert len(set(labels)) == 2


def test_similar_voices_talking_at_once_are_not_merged():
    # Cosine ~0.8: close enough to merge on voice alone, but they overlap 30 s.
    base, other = _voice(5), _voice(6)
    a = base
    b = 0.8 * base + 0.6 * (other - (other @ base) * base) / np.linalg.norm(other - (other @ base) * base)
    segments = [(0.0, 60.0, "speaker_0"), (30.0, 90.0, "speaker_1")]
    pieces, emb = _pieces_for(segments, lambda p: a if p[2] == "speaker_0" else b)
    out = sf.refine_speakers(segments, pieces, emb)
    assert len({label for *_, label in out}) == 2


def test_same_voice_on_two_overlapping_slots_is_merged():
    # Sortformer can light two slots for one voice at once; >= SURE_SAME_COS wins.
    voice = _voice(7)
    segments = [(0.0, 60.0, "speaker_0"), (40.0, 100.0, "speaker_1")]
    pieces, emb = _pieces_for(segments, lambda p: voice)
    out = sf.refine_speakers(segments, pieces, emb)
    assert {label for *_, label in out} == {"speaker_0"}


def test_sortformer_no_speech_drops_hallucinated_segments(monkeypatch):
    result = {
        "segments": [{"start": 0.0, "end": 5.0, "text": " Undertexter från Amara.org-gemenskapen"}],
        "language": "sv",
    }
    monkeypatch.setattr(transcribe, "_ensure_pcm_wav", lambda path: path)
    monkeypatch.setattr(transcribe, "audio_duration", lambda path: 6.1)
    monkeypatch.setattr(transcribe, "_cuda_memory_stats", lambda: "")
    out = transcribe.diarize(Path("sys-p03.opus"), result, "cpu", "sortformer", lambda _path: [], None)
    assert out["segments"] == []


def test_sortformer_labels_reach_the_words(monkeypatch):
    result = {"segments": [
        {"start": 0.0, "end": 2.0, "text": "hej", "words": [{"word": "hej", "start": 0.0, "end": 2.0}]},
        {"start": 3.0, "end": 5.0, "text": "hallå", "words": [{"word": "hallå", "start": 3.0, "end": 5.0}]},
    ]}
    monkeypatch.setattr(transcribe, "_ensure_pcm_wav", lambda path: path)
    monkeypatch.setattr(transcribe, "audio_duration", lambda path: 6.0)
    monkeypatch.setattr(transcribe, "_cuda_memory_stats", lambda: "")
    diarizer = lambda _path: [(0.0, 2.5, "speaker_0"), (2.5, 6.0, "speaker_1")]
    out = transcribe.diarize(Path("mic-p01.opus"), result, "cpu", "sortformer", diarizer, None)
    assert [seg["speaker"] for seg in out["segments"]] == ["speaker_0", "speaker_1"]
