"""Tests for the Rich CLI CommandUI adapter."""

from unittest.mock import MagicMock

from rich.console import Console
from rich.table import Table

from tests.conftest import run_async as _run


def _make_ui():
    """Build a RichCLICommandUI backed by a MagicMock console."""
    from EvoScientist.cli.rich_command_ui import RichCLICommandUI

    console = MagicMock(spec=Console)
    ui = RichCLICommandUI(console)
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


class TestLifecycleSignals:
    """Lifecycle commands set signal flags instead of directly mutating state."""

    def test_request_quit_sets_flag(self):
        ui, _ = _make_ui()
        assert ui._quit_requested is False
        ui.request_quit()
        assert ui._quit_requested is True

    def test_force_quit_sets_flag(self):
        ui, _ = _make_ui()
        ui.force_quit()
        assert ui._quit_requested is True

    def test_start_new_session_sets_flag(self):
        ui, _ = _make_ui()
        assert ui._new_session_requested is False
        ui.start_new_session()
        assert ui._new_session_requested is True

    def test_handle_session_resume_stores_request(self):
        ui, _ = _make_ui()
        assert ui._resume_request is None
        _run(ui.handle_session_resume("thread-123", "/some/workspace"))
        assert ui._resume_request == ("thread-123", "/some/workspace")

    def test_handle_session_resume_no_workspace(self):
        ui, _ = _make_ui()
        _run(ui.handle_session_resume("thread-456"))
        assert ui._resume_request == ("thread-456", None)

    def test_clear_chat_calls_console_clear(self):
        ui, console = _make_ui()
        ui.clear_chat()
        console.clear.assert_called_once()

    def test_reset_signals_clears_all(self):
        ui, _ = _make_ui()
        ui._quit_requested = True
        ui._new_session_requested = True
        ui._resume_request = ("tid", "/ws")
        ui._compact_tokens = 42
        ui.reset_signals()
        assert ui._quit_requested is False
        assert ui._new_session_requested is False
        assert ui._resume_request is None
        assert ui._compact_tokens is None


class TestCompactIndicator:
    """Compacting indicator uses Rich status spinner."""

    def test_update_status_after_compact_stores_tokens(self):
        ui, _ = _make_ui()
        assert ui._compact_tokens is None
        ui.update_status_after_compact(12345)
        assert ui._compact_tokens == 12345

    def test_start_stop_compacting_indicator(self):
        ui, console = _make_ui()
        # Mock the status context manager
        status_ctx = MagicMock()
        console.status.return_value = status_ctx
        ui.start_compacting_indicator()
        console.status.assert_called_once()
        status_ctx.__enter__.assert_called_once()

        ui.stop_compacting_indicator()
        status_ctx.__exit__.assert_called_once_with(None, None, None)
        assert ui._status_ctx is None

    def test_stop_compacting_indicator_noop_when_not_started(self):
        ui, _ = _make_ui()
        # Should not raise
        ui.stop_compacting_indicator()
