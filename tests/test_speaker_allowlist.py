"""The calendar decides who may be named; the voice decides which of them."""

import json

import numpy as np
import pytest

from hugin_meetings import transcribe
from hugin_meetings.context import MeetingContext


def _enroll(speakers_dir, name, centroid, *, ready=True, always=False):
    d = speakers_dir / name.lower().replace(" ", "_")
    d.mkdir(parents=True)
    meta = {"display_name": name, "ready": ready, "total_segments": 30}
    if always:
        meta["always_candidate"] = True
    (d / "meta.json").write_text(json.dumps(meta))
    np.save(d / "centroid.npy", np.array(centroid, dtype=float))


def _context(attendees=None, organizer=None):
    event = {"summary": "x"}
    if attendees is not None:
        event["attendees"] = attendees
    if organizer is not None:
        event["organizer"] = organizer
    payload = {
        "calendar_id": "primary",
        "calendar_name": "p",
        "event": event,
        "event_start": "2026-04-23T15:00:00+02:00",
        "event_end": "2026-04-23T15:30:00+02:00",
        "response_status": "accepted",
        "score": 1.0,
        "reasons": [],
    }
    return MeetingContext(session_id="20260423-150149", calendar={"candidates": [payload]})


@pytest.fixture
def speakers(tmp_path, monkeypatch):
    monkeypatch.setattr(transcribe, "SPEAKERS_DIR", tmp_path)
    _enroll(tmp_path, "David Fendrich", [1, 0, 0], always=True)
    _enroll(tmp_path, "Charlotte Eriksson", [0, 1, 0])
    _enroll(tmp_path, "Kristina Bjurström", [0, 0.8, 0.6])
    _enroll(tmp_path, "Johannes Öhlin", [0, 0, 1])
    _enroll(tmp_path, "Not Ready", [1, 1, 1], ready=False)
    return tmp_path


def test_name_tokens_fold_accents_and_emails():
    assert transcribe.name_tokens("Pär Blixt") == {"par", "blixt"}
    assert transcribe.name_tokens("charlotte.eriksson@generategroup.se") == {"charlotte", "eriksson"}
    assert transcribe.name_tokens("Öhlin, Johannes") == {"ohlin", "johannes"}


def test_attendee_matching_rules():
    m = transcribe.attendee_matches_speaker
    assert m("Charlotte Eriksson", "Charlotte Eriksson")
    assert m("charlotte.eriksson@generategroup.se", "Charlotte Eriksson")
    assert m("johannes.ohlin@tenfifty.io", "Johannes Öhlin")
    # a bare first name (email local part) is enough
    assert m("amer", "Amer Mohammed")
    # but a full name must match in full
    assert not m("Charlotte Svensson", "Charlotte Eriksson")
    assert not m("Anders Bjurström", "Kristina Bjurström")
    assert not m("", "Charlotte Eriksson")


def test_allowlist_is_invited_plus_always(speakers):
    ctx = _context(
        attendees=[
            {"displayName": "Kristina Bjurström", "email": "k@x.se"},
            {"email": "david.fendrich@tenfifty.io", "self": True},
        ]
    )
    assert transcribe.speaker_allowlist(ctx) == ["David Fendrich", "Kristina Bjurström"]


def test_allowlist_counts_the_organizer(speakers):
    ctx = _context(attendees=[{"email": "x@y.se"}], organizer={"displayName": "Charlotte Eriksson"})
    assert "Charlotte Eriksson" in transcribe.speaker_allowlist(ctx)


def test_allowlist_without_attendees_is_only_always(speakers):
    assert transcribe.speaker_allowlist(_context(attendees=None)) == ["David Fendrich"]
    assert transcribe.speaker_allowlist(_context(attendees=[])) == ["David Fendrich"]
    assert transcribe.speaker_allowlist(None) == ["David Fendrich"]


def test_allowlist_skips_not_ready(speakers):
    ctx = _context(attendees=[{"displayName": "Not Ready"}])
    assert transcribe.speaker_allowlist(ctx) == ["David Fendrich"]


def test_match_respects_allowlist(speakers):
    # A voice near the "Charlotte" direction, as an uninvited woman's would be.
    emb = {"SPEAKER_0": [0, 1.0, 0.05]}
    assert transcribe.match_speakers(emb) == {"SPEAKER_0": "Charlotte Eriksson"}
    assert transcribe.match_speakers(emb, ["David Fendrich"]) == {}
    assert transcribe.match_speakers(emb, []) == {}
    assert transcribe.match_speakers(emb, ["David Fendrich", "Charlotte Eriksson"]) == {
        "SPEAKER_0": "Charlotte Eriksson"
    }


def test_ambiguity_is_judged_among_allowed_only(speakers):
    # Equidistant between Charlotte and Kristina: ambiguous when both are
    # candidates, decided when only one is invited.
    emb = {"SPEAKER_0": [0, 0.9, 0.35]}
    both = transcribe.match_speakers(emb, ["Charlotte Eriksson", "Kristina Bjurström"])
    assert both == {}
    only = transcribe.match_speakers(emb, ["Kristina Bjurström"])
    assert only == {"SPEAKER_0": "Kristina Bjurström"}
