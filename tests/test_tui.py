from __future__ import annotations

from hugin_meetings.tui import AudioTui


def make_tui() -> AudioTui:
    tui = AudioTui.__new__(AudioTui)
    tui.log_lines = []
    tui.message = ""
    return tui


def test_append_log_splits_multiline_messages_into_terminal_lines() -> None:
    tui = make_tui()

    tui.append_log("first line\nsecond line\nthird line")

    assert tui.log_lines == ["first line", "second line", "third line"]


def test_set_message_collapses_multiline_errors_to_one_terminal_line() -> None:
    tui = make_tui()

    tui.set_message("command failed\nerror detail")

    assert tui.message == "command failed | error detail"


def make_status(context=None):
    from hugin_meetings import pipeline

    return pipeline.MeetingStatus(
        timestamp="20260824-140104",
        mic_parts=(),
        sys_parts=(),
        transcript_json=None,
        transcript_md=None,
        summary_md=None,
        calendar_fields={},
        anonymous_speakers=[],
        context=context,
    )


def test_language_column_shows_what_will_be_transcribed() -> None:
    from hugin_meetings.context import MeetingContext

    context = MeetingContext(
        session_id="20260824-140104",
        language={"value": "sv", "source": "langid", "confidence": 0.98},
        verified=True,
    )

    assert make_tui().language_label(make_status(context=context)) == "sv"


def test_an_unverified_language_is_marked_unverified() -> None:
    """Same meaning as ?? on a customer, compact enough for a two-letter code."""
    from hugin_meetings.context import MeetingContext

    context = MeetingContext(
        session_id="20260824-140104",
        language={"value": "sv", "source": "langid", "confidence": 0.98},
    )

    assert make_tui().language_label(make_status(context=context)) == "sv?"


def test_a_migrated_legacy_context_shows_no_language() -> None:
    """Verified before langid existed: nothing was probed, so nothing is claimed."""
    from hugin_meetings.context import MeetingContext

    context = MeetingContext(session_id="20260601-100000", verified=True)

    assert make_tui().language_label(make_status(context=context)) == "-"


def test_the_customer_column_shows_the_context_customer() -> None:
    from hugin_meetings.context import MeetingContext

    context = MeetingContext(
        session_id="20260824-140104",
        customer={"action": "link_existing", "customer_name": "Helos", "verified": False},
    )

    assert make_tui().customer_label(make_status(context=context)) == "??Helos??"
    assert make_tui().customer_label(make_status()) == "-"


def test_vote_text_lists_the_counted_windows() -> None:
    assert AudioTui.vote_text({"votes": {"sv": 3, "en": 1}}) == "sv x3, en x1"
    assert AudioTui.vote_text({}) == ""


def test_no_context_means_no_language_to_show() -> None:
    assert make_tui().language_label(make_status()) == "-"


def test_enter_opens_verification_while_a_decision_is_pending() -> None:
    from hugin_meetings.context import MeetingContext

    pending = make_status(context=MeetingContext(session_id="20260824-140104"))

    assert AudioTui.enter_opens_verification(pending) is True


def test_enter_opens_the_meeting_once_it_is_settled() -> None:
    from hugin_meetings.context import MeetingContext

    settled = make_status(
        context=MeetingContext(session_id="20260824-140104", verified=True)
    )

    assert AudioTui.enter_opens_verification(settled) is False


def test_reguessing_is_refused_once_a_decision_exists() -> None:
    """--force rebuilds from scratch, which would drop the operator's choice."""
    from hugin_meetings.context import MeetingContext

    settled = make_status(
        context=MeetingContext(session_id="20260824-140104", verified=True)
    )

    assert AudioTui.reguess_would_discard_decision(settled) is True


def test_reguessing_is_allowed_while_it_is_still_only_a_guess() -> None:
    from hugin_meetings.context import MeetingContext

    pending = make_status(
        context=MeetingContext(session_id="20260824-140104", note="gws not found on PATH")
    )

    assert AudioTui.reguess_would_discard_decision(pending) is False


def test_reguessing_is_allowed_when_there_is_no_context_at_all() -> None:
    assert AudioTui.reguess_would_discard_decision(make_status()) is False
