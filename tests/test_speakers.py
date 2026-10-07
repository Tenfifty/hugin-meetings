"""Enrollment events are the truth; speakers/ is a cache folded from them."""

import hashlib
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from hugin_meetings import speakers, transcribe
from hugin_meetings.enroll import accepted_mask

T0 = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)


class FakeEmbedder:
    """A fixed pseudo-random vector per span, and a record of what was embedded."""

    def __init__(self, model_id="fake/m1"):
        self.model_id = model_id
        self.calls: list[tuple[str, str, float, float]] = []

    def embed(self, session, channel, spans):
        rows = []
        for start, end in spans:
            self.calls.append((session, channel, start, end))
            key = f"{self.model_id}|{session}|{channel}|{start}|{end}".encode()
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
            rows.append(np.random.default_rng(seed).normal(size=8))
        return np.array(rows)


def all_audio(session, channel):
    return True


def enroll(name, session, channel, spans, label="SPEAKER_0"):
    return {
        "type": "enroll", "name": name, "slug": speakers.slugify(name),
        "session": session, "channel": channel,
        "segments": [{"start": a, "end": b} for a, b in spans],
        "embedding_model": "fake/m1", "diarization_label": f"{channel}:{label}",
    }


def remove_person(name):
    return {"type": "remove_person", "name": name, "slug": speakers.slugify(name)}


def remove_segments(name, session, channel=None, spans=None):
    event = {"type": "remove_segments", "name": name, "slug": speakers.slugify(name),
             "session": session, "channel": channel}
    if spans is not None:
        event["segments"] = [{"start": a, "end": b} for a, b in spans]
    return event


def always(name, value=True):
    return {"type": "always_candidate", "name": name, "slug": speakers.slugify(name), "value": value}


def write_all(events_dir, events, order=None):
    """Write ``events`` with timestamps T0, T0+1s, ... ; ``order`` is the order of writing."""
    for i in order or range(len(events)):
        speakers.write_event(events_dir, events[i], now=T0 + timedelta(seconds=i))


def build(tmp_path, events, embedder=None, order=None, name="speakers"):
    events_dir = tmp_path / f"events-{name}"
    write_all(events_dir, events, order)
    speakers_dir = tmp_path / name
    people = speakers.update_cache(speakers_dir, events_dir, embedder or FakeEmbedder(), has_audio=all_audio)
    return people, speakers_dir, events_dir


def spans(n, offset=0.0):
    return [(offset + 10.0 * i, offset + 10.0 * i + 4.0) for i in range(n)]


def test_an_event_is_never_replaced(tmp_path):
    first = speakers.write_event(tmp_path, enroll("Alice", "20260401-100000", "mic", spans(1)), now=T0)
    original = first.read_bytes()

    with pytest.raises(FileExistsError):
        speakers.write_event(tmp_path, enroll("Alice", "20260402-100000", "sys", spans(2)), now=T0)

    assert first.name == "20261007T120000.000000Z-alice.json"
    assert first.read_bytes() == original
    assert [p.name for p in tmp_path.iterdir()] == [first.name]  # no temp file left behind
    stored = json.loads(original)
    assert stored["schema"] == speakers.SCHEMA and stored["created"] == "20261007T120000.000000Z"


@pytest.mark.parametrize("bad", [
    {"channel": "both"},
    {"segments": []},
    {"segments": [{"start": 5.0, "end": 2.0}]},
    {"slug": "bob"},
    {"session": "yesterday"},
])
def test_malformed_events_are_refused(tmp_path, bad):
    with pytest.raises(speakers.EnrollmentError):
        speakers.write_event(tmp_path, {**enroll("Alice", "20260401-100000", "mic", spans(1)), **bad})
    assert list(tmp_path.iterdir()) == []


def test_fold_applies_removals_in_order():
    s1, s2 = "20260401-100000", "20260402-100000"
    events = [
        enroll("Alice", s1, "mic", spans(5)),
        enroll("Alice", s2, "sys", spans(3)),
        enroll("Bob", s1, "sys", spans(4)),
        always("Bob"),
        remove_segments("Alice", s1, spans=[(19.0, 31.0)]),   # overlaps 20-24 and 30-34
        remove_segments("Alice", s2, channel="sys"),          # the whole session
        remove_person("Bob"),
        enroll("Bob", s2, "mic", spans(2)),                   # starts afresh
        remove_segments("Carol", s1),                         # nobody: no effect
    ]
    people = speakers.fold(events)

    assert sorted(people) == ["alice", "bob"]
    assert [(s.session, s.start) for s in people["alice"].sorted_segments()] == [
        (s1, 0.0), (s1, 10.0), (s1, 40.0),
    ]
    assert [(s.session, s.channel) for s in people["bob"].sorted_segments()] == [(s2, "mic")] * 2
    assert people["bob"].always_candidate is False  # the removal took the flag with it


def test_a_person_with_every_segment_removed_disappears(tmp_path):
    s1 = "20260401-100000"
    people, speakers_dir, _ = build(tmp_path, [
        enroll("Alice", s1, "mic", spans(3)),
        enroll("Bob", s1, "sys", spans(3)),
        remove_segments("Alice", s1),
    ])
    assert list(people) == ["bob"]
    assert sorted(d.name for d in speakers_dir.iterdir()) == ["bob", "cache.json"]


def test_the_cache_depends_on_the_event_set_not_the_order_of_writing(tmp_path):
    s1, s2 = "20260401-100000", "20260402-100000"
    events = [
        enroll("Alice", s2, "sys", spans(12, offset=100)),
        enroll("Alice", s1, "mic", spans(12)),
        enroll("Bob", s1, "sys", spans(25)),
        always("Alice"),
    ]
    _, a, _ = build(tmp_path, events, order=[0, 1, 2, 3], name="a")
    _, b, _ = build(tmp_path, events, order=[3, 2, 1, 0], name="b")
    # Alice's two enrollments swapped in time: the same segment set, so the same profile.
    swapped = [events[1], events[0], events[2], events[3]]
    _, c, _ = build(tmp_path, swapped, name="c")

    for other in (b, c):
        for slug in ("alice", "bob"):
            assert np.array_equal(np.load(a / slug / "centroid.npy"), np.load(other / slug / "centroid.npy"))
            assert json.loads((a / slug / "meta.json").read_text()) == json.loads((other / slug / "meta.json").read_text())

    meta = json.loads((a / "alice" / "meta.json").read_text())
    assert meta["display_name"] == "Alice" and meta["total_segments"] == 24
    assert meta["ready"] is True and meta["always_candidate"] is True


def test_ready_is_computed_from_the_segment_count(tmp_path):
    _, speakers_dir, _ = build(tmp_path, [
        enroll("Alice", "20260401-100000", "mic", spans(speakers.MIN_READY_SEGMENTS)),
        enroll("Bob", "20260401-100000", "sys", spans(speakers.MIN_READY_SEGMENTS - 1)),
    ])
    assert json.loads((speakers_dir / "alice" / "meta.json").read_text())["ready"] is True
    assert json.loads((speakers_dir / "bob" / "meta.json").read_text())["ready"] is False


def test_the_cache_is_updated_per_person_and_rebuilt_on_a_model_change(tmp_path):
    s1, s2 = "20260401-100000", "20260402-100000"
    embedder = FakeEmbedder("fake/m1")
    _, speakers_dir, events_dir = build(tmp_path, [
        enroll("Alice", s1, "mic", spans(3)),
        enroll("Bob", s1, "sys", spans(2)),
    ], embedder)
    assert len(embedder.calls) == 5
    assert speakers.cache_status(speakers_dir, events_dir, "fake/m1") == ("current", "")

    # Nothing changed: nothing is embedded.
    speakers.update_cache(speakers_dir, events_dir, embedder, has_audio=all_audio)
    assert len(embedder.calls) == 5

    # A new event: stale, and only its segments are embedded.
    speakers.write_event(events_dir, enroll("Alice", s2, "sys", spans(2)), now=T0 + timedelta(hours=1))
    assert speakers.cache_status(speakers_dir, events_dir, "fake/m1")[0] == "stale"
    speakers.update_cache(speakers_dir, events_dir, embedder, has_audio=all_audio)
    assert embedder.calls[5:] == [(s2, "sys", 0.0, 4.0), (s2, "sys", 10.0, 14.0)]

    # A new model: stale, and everything is embedded again.
    status, reason = speakers.cache_status(speakers_dir, events_dir, "fake/m2")
    assert status == "stale" and "fake/m1 -> fake/m2" in reason
    m2 = FakeEmbedder("fake/m2")
    speakers.update_cache(speakers_dir, events_dir, m2, has_audio=all_audio)
    assert len(m2.calls) == 7
    assert speakers.cache_status(speakers_dir, events_dir, "fake/m2") == ("current", "")
    assert json.loads((speakers_dir / "alice" / "meta.json").read_text())["embedding_model"] == "fake/m2"

    # Wiping the cache is safe: the same events give the same profile.
    before = np.load(speakers_dir / "alice" / "centroid.npy")
    for d in speakers_dir.iterdir():
        if d.is_dir():
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
    speakers.update_cache(speakers_dir, events_dir, FakeEmbedder("fake/m2"), has_audio=all_audio)
    assert np.array_equal(before, np.load(speakers_dir / "alice" / "centroid.npy"))


def test_an_event_without_audio_aborts_the_update(tmp_path):
    events_dir = tmp_path / "events"
    write_all(events_dir, [
        enroll("Alice", "20260401-100000", "mic", spans(2)),
        enroll("Bob", "20260402-100000", "sys", spans(2)),
    ])

    def only_alice(session, channel):
        return session == "20260401-100000"

    with pytest.raises(speakers.EnrollmentError, match="Bob.*sys of session 20260402-100000"):
        speakers.update_cache(tmp_path / "speakers", events_dir, FakeEmbedder(), has_audio=only_alice)
    assert not (tmp_path / "speakers" / speakers.CACHE_FILE).exists()


def test_a_short_embedder_answer_aborts_the_update(tmp_path):
    class Dropping(FakeEmbedder):
        def embed(self, session, channel, spans):
            return super().embed(session, channel, spans)[1:]

    events_dir = tmp_path / "events"
    write_all(events_dir, [enroll("Alice", "20260401-100000", "mic", spans(2))])
    with pytest.raises(speakers.EnrollmentError, match="1 rows for 2 segments"):
        speakers.update_cache(tmp_path / "speakers", events_dir, Dropping(), has_audio=all_audio)


def test_a_pre_event_store_is_neither_overwritten_nor_built_over(tmp_path):
    legacy = tmp_path / "speakers" / "alice"
    legacy.mkdir(parents=True)
    (legacy / "meta.json").write_text(json.dumps({"display_name": "Alice", "ready": True}))
    np.save(legacy / "centroid.npy", np.ones(8))

    assert speakers.cache_status(tmp_path / "speakers", tmp_path / "events")[0] == "legacy"
    with pytest.raises(speakers.LegacySpeakerStore):
        speakers.update_cache(tmp_path / "speakers", tmp_path / "events", FakeEmbedder(), has_audio=all_audio)
    assert (legacy / "centroid.npy").exists()


def test_transcribe_reads_the_cache(tmp_path, monkeypatch):
    _, speakers_dir, _ = build(tmp_path, [
        enroll("David Fendrich", "20260401-100000", "mic", spans(20)),
        always("David Fendrich"),
        enroll("Alice Andersson", "20260401-100000", "sys", spans(20)),
        enroll("Bob Berg", "20260401-100000", "sys", spans(5, offset=1000)),  # not ready
    ])
    monkeypatch.setattr(transcribe, "SPEAKERS_DIR", speakers_dir)

    assert transcribe.speaker_allowlist(None) == ["David Fendrich"]
    david = np.load(speakers_dir / "david_fendrich" / "centroid.npy")
    alice = np.load(speakers_dir / "alice_andersson" / "centroid.npy")
    assert transcribe.match_speakers({"SPEAKER_0": david, "SPEAKER_1": alice}) == {
        "SPEAKER_0": "David Fendrich", "SPEAKER_1": "Alice Andersson",
    }


def test_accepted_mask_against_an_existing_profile_and_a_first_one():
    rng = np.random.default_rng(0)
    centroid = np.ones(8)
    near = centroid + 0.1 * rng.normal(size=(5, 8))
    far = -centroid + 0.1 * rng.normal(size=(2, 8))
    mask = accepted_mask(np.vstack([near, far]), centroid)
    assert mask.tolist() == [True] * 5 + [False] * 2

    first = accepted_mask(rng.normal(size=(20, 8)) + 3, None)
    assert first.sum() == 18  # the bottom 10% go
