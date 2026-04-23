"""Tests for the Rich CLI CommandUI adapter."""

from unittest.mock import MagicMock

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


class TestWaitForThreadPick:
    """Phase B questionary picker — ported from the pre-migration
    ``_cmd_resume`` implementation in ``interactive.py``."""

    def _fake_prompt(self, selected):
        """Build a stub matching ``questionary.select(...)`` return."""
        prompt = MagicMock()
        prompt.ask.return_value = selected
        prompt.application.layout.find_all_windows.return_value = []
        return prompt

    def _threads(self):
        return [
            {
                "thread_id": "abc123",
                "preview": "hello",
                "message_count": 3,
                "model": "claude",
                "updated_at": None,
                "workspace_dir": "/w",
            },
            {
                "thread_id": "def456",
                "preview": "x" * 60,  # triggers truncation path
                "message_count": 0,
                "model": None,
                "updated_at": None,
                "workspace_dir": "/w",
            },
        ]

    def test_returns_selected_thread_id(self, monkeypatch):
        import EvoScientist.cli.rich_command_ui as mod

        ui, _ = _make_ui()
        prompt = self._fake_prompt("abc123")
        called = {}

        def fake_select(title, choices, style):
            called["title"] = title
            called["choices"] = choices
            return prompt

        monkeypatch.setattr("questionary.select", fake_select)
        result = _run(ui.wait_for_thread_pick(self._threads(), "abc123", "pick:"))
        assert result == "abc123"
        assert called["title"] == "pick:"
        # _build_items prepends a workspace header — choices has headers +
        # thread Choice entries.
        assert len(called["choices"]) >= 2
        # Table import removed; this test no longer depends on console output.
        assert mod.RichCLICommandUI is not None  # sanity

    def test_cancel_returns_none(self, monkeypatch):
        ui, _ = _make_ui()
        prompt = self._fake_prompt(None)
        monkeypatch.setattr("questionary.select", lambda *a, **k: prompt)
        result = _run(ui.wait_for_thread_pick(self._threads(), "abc123", "pick:"))
        assert result is None

    def test_current_thread_marker_in_label(self, monkeypatch):
        ui, _ = _make_ui()
        prompt = self._fake_prompt(None)
        captured_choices: list = []

        def fake_select(title, choices, style):
            captured_choices.extend(choices)
            return prompt

        monkeypatch.setattr("questionary.select", fake_select)
        _run(ui.wait_for_thread_pick(self._threads(), "abc123", "pick:"))
        # At least one Choice title contains "abc123 *" (current marker)
        choice_titles = [getattr(c, "title", "") for c in captured_choices]
        assert any("abc123 *" in t for t in choice_titles)


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


class TestPhaseBMigrated:
    """Session lifecycle callbacks (start/resume) filled in Phase B."""

    def test_start_new_session_fires_callback(self):
        called = []
        ui, _ = _make_ui(on_start_new_session=lambda: called.append("new"))
        ui.start_new_session()
        assert called == ["new"]

    def test_start_new_session_without_callback_is_noop(self):
        ui, console = _make_ui()
        ui.start_new_session()
        console.print.assert_not_called()

    def test_handle_session_resume_awaits_callback(self):
        from unittest.mock import AsyncMock

        cb = AsyncMock()
        ui, _ = _make_ui(on_handle_session_resume=cb)
        _run(ui.handle_session_resume("tid-x", "/workspace"))
        cb.assert_awaited_once_with("tid-x", "/workspace")

    def test_handle_session_resume_without_callback_is_noop(self):
        ui, _ = _make_ui()
        # Should not raise
        _run(ui.handle_session_resume("tid-x"))

    def test_handle_session_resume_workspace_defaults_none(self):
        from unittest.mock import AsyncMock

        cb = AsyncMock()
        ui, _ = _make_ui(on_handle_session_resume=cb)
        _run(ui.handle_session_resume("tid-x"))
        cb.assert_awaited_once_with("tid-x", None)


class TestPhaseCMigrated:
    """Skill/MCP browse pickers delegate to questionary helpers via
    ``asyncio.to_thread`` since questionary blocks the event loop."""

    def test_skill_browse_delegates_to_picker(self, monkeypatch):
        from unittest.mock import MagicMock

        picker = MagicMock(return_value=["skill-a", "skill-b"])
        monkeypatch.setattr(
            "EvoScientist.cli.skills_cmd._pick_skills_interactive",
            picker,
        )
        ui, _ = _make_ui()
        result = _run(ui.wait_for_skill_browse([{"name": "a"}], {"installed"}, "core"))
        assert result == ["skill-a", "skill-b"]
        picker.assert_called_once_with([{"name": "a"}], {"installed"}, "core")

    def test_skill_browse_cancel_returns_none(self, monkeypatch):
        from unittest.mock import MagicMock

        monkeypatch.setattr(
            "EvoScientist.cli.skills_cmd._pick_skills_interactive",
            MagicMock(return_value=None),
        )
        ui, _ = _make_ui()
        result = _run(ui.wait_for_skill_browse([], set(), ""))
        assert result is None

    def test_mcp_browse_delegates_to_picker(self, monkeypatch):
        from unittest.mock import MagicMock

        sentinel_entries = [MagicMock(name="entry1"), MagicMock(name="entry2")]
        picker = MagicMock(return_value=sentinel_entries)
        monkeypatch.setattr(
            "EvoScientist.cli.mcp_install_cmd._browse_and_select",
            picker,
        )
        ui, _ = _make_ui()
        result = _run(ui.wait_for_mcp_browse([MagicMock()], {"configured"}, ""))
        assert result is sentinel_entries
        picker.assert_called_once()

    def test_mcp_browse_cancel_returns_none(self, monkeypatch):
        from unittest.mock import MagicMock

        monkeypatch.setattr(
            "EvoScientist.cli.mcp_install_cmd._browse_and_select",
            MagicMock(return_value=None),
        )
        ui, _ = _make_ui()
        result = _run(ui.wait_for_mcp_browse([], set(), ""))
        assert result is None
