"""Tests for the Rich CLI CommandUI adapter."""

from unittest.mock import MagicMock

import pytest
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


class TestUnmigratedMethodsStub:
    """Protocol methods not yet wired for CLI must raise NotImplementedError.

    These stubs are signposts for the A1 migration (see
    cli-commandmanager-migration.md). Each one is replaced by a real
    implementation when its corresponding command is migrated.
    """

    def test_wait_for_thread_pick(self):
        ui, _ = _make_ui()
        with pytest.raises(NotImplementedError, match="/threads"):
            _run(ui.wait_for_thread_pick([], "tid", "title"))

    def test_wait_for_skill_browse(self):
        ui, _ = _make_ui()
        with pytest.raises(NotImplementedError, match="/skills"):
            _run(ui.wait_for_skill_browse([], set(), ""))

    def test_wait_for_mcp_browse(self):
        ui, _ = _make_ui()
        with pytest.raises(NotImplementedError, match="/mcp"):
            _run(ui.wait_for_mcp_browse([], set(), ""))

    def test_clear_chat(self):
        ui, _ = _make_ui()
        with pytest.raises(NotImplementedError, match="/clear"):
            ui.clear_chat()

    def test_request_quit(self):
        ui, _ = _make_ui()
        with pytest.raises(NotImplementedError, match="/exit"):
            ui.request_quit()

    def test_force_quit(self):
        ui, _ = _make_ui()
        with pytest.raises(NotImplementedError, match="/exit"):
            ui.force_quit()

    def test_start_new_session(self):
        ui, _ = _make_ui()
        with pytest.raises(NotImplementedError, match="/new"):
            ui.start_new_session()

    def test_handle_session_resume(self):
        ui, _ = _make_ui()
        with pytest.raises(NotImplementedError, match="/resume"):
            _run(ui.handle_session_resume("tid"))
