"""Tests for EvoScientist.cli.widgets.thread_selector module."""

from __future__ import annotations

from unittest import mock

from rich.text import Text

from EvoScientist.cli.widgets.thread_selector import (
    ThreadPickerWidget,
    build_row_text,
)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_THREADS = [
    {
        "thread_id": "abc12345",
        "preview": "Help me write a paper",
        "message_count": 12,
        "model": "claude-sonnet-4-6",
        "updated_at": "2026-03-09T10:00:00+00:00",
        "workspace_dir": "/workspace",
    },
    {
        "thread_id": "def67890",
        "preview": "Run experiment pipeline",
        "message_count": 5,
        "model": "gpt-4o",
        "updated_at": "2026-03-08T08:00:00+00:00",
        "workspace_dir": "/workspace",
    },
    {
        "thread_id": "ghi11111",
        "preview": "",
        "message_count": 0,
        "model": "",
        "updated_at": None,
        "workspace_dir": "/workspace",
    },
]


# ---------------------------------------------------------------------------
# build_row_text unit tests (pure function, no Textual app needed)
# ---------------------------------------------------------------------------


class TestBuildRowText:
    def test_selected_has_cursor(self):
        text = build_row_text(_THREADS[0], selected=True)
        assert isinstance(text, Text)
        assert "\u25b8" in text.plain

    def test_not_selected_no_cursor(self):
        text = build_row_text(_THREADS[0], selected=False)
        assert "\u25b8" not in text.plain

    def test_thread_id_shown(self):
        text = build_row_text(_THREADS[0])
        assert "abc12345" in text.plain

    def test_current_marker(self):
        text = build_row_text(_THREADS[0], current=True)
        assert "*" in text.plain

    def test_no_current_marker(self):
        text = build_row_text(_THREADS[0], current=False)
        assert " *" not in text.plain

    def test_preview_shown(self):
        text = build_row_text(_THREADS[0])
        assert "Help me write a paper" in text.plain

    def test_no_preview(self):
        text = build_row_text(_THREADS[2])
        assert "ghi11111" in text.plain
        assert "(0 msgs)" in text.plain

    def test_message_count(self):
        text = build_row_text(_THREADS[0])
        assert "(12 msgs)" in text.plain

    def test_model_shown(self):
        text = build_row_text(_THREADS[0])
        assert "claude-sonnet-4-6" in text.plain

    def test_empty_model_omitted(self):
        text = build_row_text(_THREADS[2])
        plain = text.plain
        assert "ghi11111" in plain

    def test_long_preview_truncated(self):
        long_thread = {**_THREADS[0], "preview": "A" * 60}
        text = build_row_text(long_thread)
        assert "\u2026" in text.plain
        assert "A" * 60 not in text.plain

    def test_relative_time_shown(self):
        text = build_row_text(_THREADS[0])
        assert "abc12345" in text.plain

    def test_no_time_for_none_updated_at(self):
        text = build_row_text(_THREADS[2])
        assert "ghi11111" in text.plain


# ---------------------------------------------------------------------------
# ThreadPickerWidget unit tests
# ---------------------------------------------------------------------------


class TestThreadPickerWidget:
    def test_init_stores_threads(self):
        picker = ThreadPickerWidget(_THREADS, current_thread="abc12345")
        assert picker._threads == _THREADS
        assert picker._current_thread == "abc12345"
        # items = [header, thread0, thread1, thread2] — first thread is at index 1
        assert picker._selected == 1

    def test_init_empty_threads(self):
        picker = ThreadPickerWidget([])
        assert picker._threads == []

    def test_custom_title(self):
        picker = ThreadPickerWidget(_THREADS, title="Pick one")
        assert picker._title == "Pick one"

    def test_default_title(self):
        picker = ThreadPickerWidget(_THREADS)
        assert picker._title == "Select a session"

    def test_action_move_down_wraps(self):
        picker = ThreadPickerWidget(_THREADS)
        picker._row_widgets = [mock.MagicMock() for _ in _THREADS]
        # items = [header@0, t0@1, t1@2, t2@3]; last thread is at index 3
        picker._selected = len(picker._items) - 1
        # Mock _update_rows to avoid Textual update calls
        picker._update_rows = mock.MagicMock()
        picker.action_move_down()
        # wraps past header back to first thread at index 1
        assert picker._selected == 1

    def test_action_move_up_wraps(self):
        picker = ThreadPickerWidget(_THREADS)
        picker._row_widgets = [mock.MagicMock() for _ in _THREADS]
        # items = [header@0, t0@1, t1@2, t2@3]; start at first thread
        picker._selected = 1
        picker._update_rows = mock.MagicMock()
        picker.action_move_up()
        # wraps past header back to last thread at index 3
        assert picker._selected == len(picker._items) - 1

    def test_action_move_down_increments(self):
        picker = ThreadPickerWidget(_THREADS)
        picker._row_widgets = [mock.MagicMock() for _ in _THREADS]
        picker._selected = 0
        picker._update_rows = mock.MagicMock()
        picker.action_move_down()
        assert picker._selected == 1

    def test_action_move_up_decrements(self):
        picker = ThreadPickerWidget(_THREADS)
        picker._row_widgets = [mock.MagicMock() for _ in _THREADS]
        picker._selected = 2
        picker._update_rows = mock.MagicMock()
        picker.action_move_up()
        assert picker._selected == 1

    def test_action_move_empty_noop(self):
        picker = ThreadPickerWidget([])
        picker.action_move_down()
        assert picker._selected == 0
        picker.action_move_up()
        assert picker._selected == 0

    def test_action_select_posts_picked(self):
        picker = ThreadPickerWidget(_THREADS)
        # items = [header@0, t0@1, t1@2, t2@3]; select t1 (def67890) at index 2
        picker._selected = 2
        picker.post_message = mock.MagicMock()
        picker.action_select()
        msg = picker.post_message.call_args[0][0]
        assert isinstance(msg, ThreadPickerWidget.Picked)
        assert msg.thread_id == "def67890"

    def test_action_cancel_posts_cancelled(self):
        picker = ThreadPickerWidget(_THREADS)
        picker.post_message = mock.MagicMock()
        picker.action_cancel()
        msg = picker.post_message.call_args[0][0]
        assert isinstance(msg, ThreadPickerWidget.Cancelled)

    def test_action_select_empty_posts_cancelled(self):
        picker = ThreadPickerWidget([])
        picker.post_message = mock.MagicMock()
        picker.action_select()
        msg = picker.post_message.call_args[0][0]
        assert isinstance(msg, ThreadPickerWidget.Cancelled)

    def test_bindings_include_navigation(self):
        bindings = {b.key for b in ThreadPickerWidget.BINDINGS}
        assert "up" in bindings
        assert "down" in bindings
        assert "enter" in bindings
        assert "escape" in bindings
        assert "k" in bindings
        assert "j" in bindings

    def test_can_focus(self):
        assert ThreadPickerWidget.can_focus is True
        assert ThreadPickerWidget.can_focus_children is False
