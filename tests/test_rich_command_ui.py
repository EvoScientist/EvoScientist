"""Tests for the Rich CLI CommandUI adapter."""

from unittest.mock import MagicMock

import pytest
from rich.console import Console
from rich.table import Table

from tests.conftest import run_async as _run


def _make_ui(**kwargs):
    """Build a RichCLICommandUI backed by a MagicMock console."""
    from EvoScientist.cli.rich_command_ui import RichCLICommandUI

    console = MagicMock(spec=Console)
    ui = RichCLICommandUI(console, **kwargs)
    return ui, console


class TestBasicIO:
    """Core CommandUI methods used by /model path."""

    def test_supports_interactive_true(self):
        ui, _ = _make_ui()
        assert ui.supports_interactive is True

    def test_append_system_forwards_style(self):
        ui, console = _make_ui()
        ui.append_system("hello", style="green")
        console.print.assert_called_once_with("hello", style="green")

    def test_append_system_default_style(self):
        ui, console = _make_ui()
        ui.append_system("info")
        console.print.assert_called_once_with("info", style="dim")

    def test_mount_renderable_preserves_type(self):
        ui, console = _make_ui()
        table = Table(title="demo")
        ui.mount_renderable(table)
        console.print.assert_called_once_with(table)

    def test_flush_is_async_noop(self):
        ui, console = _make_ui()
        _run(ui.flush())
        # flush should not print anything
        console.print.assert_not_called()


class TestWaitForModelPick:
    """CLI model picker fallback: print table + return None."""

    def test_returns_none(self):
        ui, _ = _make_ui()
        entries = [
            ("claude-sonnet-4-6", "anthropic/claude-sonnet", "anthropic"),
            ("gpt-4o", "openai/gpt-4o", "openai"),
        ]
        result = _run(
            ui.wait_for_model_pick(
                entries,
                current_model="claude-sonnet-4-6",
                current_provider="anthropic",
            )
        )
        assert result is None

    def test_prints_table_with_current_model_marker(self):
        ui, console = _make_ui()
        entries = [
            ("claude-sonnet-4-6", "anthropic/claude-sonnet", "anthropic"),
            ("gpt-4o", "openai/gpt-4o", "openai"),
        ]
        _run(
            ui.wait_for_model_pick(
                entries,
                current_model="claude-sonnet-4-6",
                current_provider="anthropic",
            )
        )
        # First call renders the Table (Rich renderable), second prints usage.
        assert console.print.call_count == 2
        first_arg = console.print.call_args_list[0].args[0]
        assert isinstance(first_arg, Table)

        usage_arg = console.print.call_args_list[1].args[0]
        assert "Usage: /model" in usage_arg
        assert "--save" in usage_arg

    def test_no_current_model_no_marker(self):
        ui, console = _make_ui()
        entries = [("claude-sonnet-4-6", "anthropic/claude-sonnet", "anthropic")]
        _run(
            ui.wait_for_model_pick(
                entries,
                current_model=None,
                current_provider=None,
            )
        )
        # Just asserts the coroutine runs without marker-branch issues.
        assert console.print.call_count == 2

    def test_empty_entries_still_prints_header_and_usage(self):
        ui, console = _make_ui()
        result = _run(
            ui.wait_for_model_pick(
                [],
                current_model=None,
                current_provider=None,
            )
        )
        assert result is None
        # Header table + usage hint should still be printed even with
        # no entries.
        assert console.print.call_count == 2


class TestUpdateStatusHook:
    """update_status_after_model_change is a deliberate no-op on CLI."""

    def test_no_op(self):
        ui, console = _make_ui()
        ui.update_status_after_model_change("claude-opus-4-6", "anthropic")
        console.print.assert_not_called()


class TestPhaseAMigrated:
    """Phase A migration: quit, clear, thread-pick fallback, and compact status.

    These replaced the original ``NotImplementedError`` stubs once the
    corresponding commands were migrated through the shared CommandManager
    dispatch block in ``interactive.py``.
    """

    def test_request_quit_fires_callback(self):
        called = []
        ui, _ = _make_ui(on_request_quit=lambda: called.append("q"))
        ui.request_quit()
        assert called == ["q"]

    def test_request_quit_without_callback_is_noop(self):
        ui, console = _make_ui()
        ui.request_quit()
        console.clear.assert_not_called()
        console.print.assert_not_called()

    def test_force_quit_fires_callback(self):
        called = []
        ui, _ = _make_ui(on_force_quit=lambda: called.append("fq"))
        ui.force_quit()
        assert called == ["fq"]

    def test_clear_chat_fires_callback(self):
        called = []
        ui, console = _make_ui(on_clear_chat=lambda: called.append("cls"))
        ui.clear_chat()
        assert called == ["cls"]
        # Callback owns clearing — adapter should not also clear
        console.clear.assert_not_called()

    def test_clear_chat_default_falls_back_to_console_clear(self):
        ui, console = _make_ui()
        ui.clear_chat()
        console.clear.assert_called_once()

    def test_update_status_after_compact_fires_callback(self):
        received = []
        ui, _ = _make_ui(on_status_after_compact=received.append)
        ui.update_status_after_compact(1234)
        assert received == [1234]

    def test_update_status_after_compact_without_callback_is_noop(self):
        ui, console = _make_ui()
        # Does not raise; does not print.
        ui.update_status_after_compact(500)
        console.print.assert_not_called()

    def test_wait_for_thread_pick_returns_none(self):
        ui, _ = _make_ui()
        threads = [
            {
                "thread_id": "abc123",
                "preview": "hello",
                "message_count": 3,
                "model": "claude",
                "updated_at": "2026-04-22T10:00:00Z",
            },
        ]
        result = _run(ui.wait_for_thread_pick(threads, "abc123", "Select thread"))
        assert result is None

    def test_wait_for_thread_pick_prints_table_with_current_marker(self):
        ui, console = _make_ui()
        threads = [
            {
                "thread_id": "abc123",
                "preview": "hello",
                "message_count": 3,
                "model": "claude",
                "updated_at": None,
            },
            {
                "thread_id": "def456",
                "preview": "",
                "message_count": 0,
                "model": None,
                "updated_at": None,
            },
        ]
        _run(ui.wait_for_thread_pick(threads, "abc123", "Select thread"))
        # Table + usage hint
        assert console.print.call_count == 2
        table_arg = console.print.call_args_list[0].args[0]
        assert isinstance(table_arg, Table)
        hint_arg = console.print.call_args_list[1].args[0]
        assert "/resume" in hint_arg
        assert "/delete" in hint_arg


class TestCompactIndicator:
    """start/stop_compacting_indicator duck-typed by CompactCommand."""

    def test_indicator_pair_wraps_console_status(self):
        ui, console = _make_ui()
        # Simulate a context manager returned by console.status()
        status_cm = MagicMock()
        console.status.return_value = status_cm
        ui.start_compacting_indicator()
        console.status.assert_called_once()
        status_cm.__enter__.assert_called_once()
        ui.stop_compacting_indicator()
        status_cm.__exit__.assert_called_once_with(None, None, None)

    def test_stop_without_start_is_noop(self):
        ui, console = _make_ui()
        # Should not raise even if start was never called.
        ui.stop_compacting_indicator()
        console.status.assert_not_called()


class TestUnmigratedStubs:
    """Phase B/C stubs still raise NotImplementedError until migrated."""

    def test_wait_for_skill_browse(self):
        ui, _ = _make_ui()
        with pytest.raises(NotImplementedError, match="/evoskills"):
            _run(ui.wait_for_skill_browse([], set(), ""))

    def test_wait_for_mcp_browse(self):
        ui, _ = _make_ui()
        with pytest.raises(NotImplementedError, match="/install-mcp"):
            _run(ui.wait_for_mcp_browse([], set(), ""))

    def test_start_new_session(self):
        ui, _ = _make_ui()
        with pytest.raises(NotImplementedError, match="/new"):
            ui.start_new_session()

    def test_handle_session_resume(self):
        ui, _ = _make_ui()
        with pytest.raises(NotImplementedError, match="/resume"):
            _run(ui.handle_session_resume("tid"))
