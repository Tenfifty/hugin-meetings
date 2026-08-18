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
