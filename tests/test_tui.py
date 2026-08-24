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


def make_status(context=None, customer_state=None):
    from hugin_meetings import pipeline

    return pipeline.MeetingStatus(
        timestamp="20260824-140104",
        mic_path=None,
        sys_path=None,
        mic_parts=(),
        sys_parts=(),
        transcript_json=None,
        transcript_md=None,
        summary_md=None,
        calendar_fields={},
        customer_state=customer_state,
        anonymous_speakers=[],
        context=context,
    )


def test_language_column_shows_what_will_be_transcribed() -> None:
    from hugin_meetings.context import MeetingContext

    context = MeetingContext(
        session_id="20260824-140104",
        language={"value": "sv", "source": "langid", "confidence": 0.98},
    )

    assert make_tui().language_label(make_status(context=context)) == "sv"


def test_a_language_nobody_decided_is_flagged() -> None:
    """The probe abstained and the configured default stood in — worth a look."""
    from hugin_meetings.context import MeetingContext

    context = MeetingContext(
        session_id="20260824-140104",
        language={"value": "sv", "source": "fallback", "note": "no probe cleared"},
    )

    assert make_tui().language_label(make_status(context=context)) == "sv?"


def test_no_context_means_no_language_to_show() -> None:
    assert make_tui().language_label(make_status()) == "-"
