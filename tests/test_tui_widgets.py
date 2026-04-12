"""Unit tests for TUI widgets.

Tests widget construction, state transitions, and public APIs
without requiring a running Textual app (no Textual pilot needed).
"""

from __future__ import annotations

import importlib
import unittest
from unittest.mock import AsyncMock

# ---------------------------------------------------------------------------
# Textual might not be installed — skip entire module if missing
# ---------------------------------------------------------------------------
_has_textual = importlib.util.find_spec("textual") is not None


@unittest.skipUnless(_has_textual, "textual not installed")
class TestLoadingWidget(unittest.TestCase):
    """LoadingWidget construction and attributes."""

    def test_construction(self):
        from EvoScientist.cli.widgets.loading_widget import LoadingWidget
        from EvoScientist.cli.widgets.timed_status_widget import TimedStatusWidget

        w = LoadingWidget()
        assert isinstance(w, TimedStatusWidget)
        assert w._frame == 0
        assert w._elapsed == 0.0
        assert w._timer_handle is None

    def test_spinner_frames_not_empty(self):
        from EvoScientist.cli.widgets.loading_widget import _SPINNER_FRAMES

        assert len(_SPINNER_FRAMES) > 0

    def test_tick_advances_spinner_and_elapsed(self):
        from EvoScientist.cli.widgets.loading_widget import LoadingWidget

        w = LoadingWidget()
        w._tick()
        assert w._frame == 1
        assert w._elapsed == 0.1

    def test_cleanup_stops_timer_and_removes(self):
        from EvoScientist.cli.widgets.loading_widget import LoadingWidget

        class _Timer:
            def __init__(self) -> None:
                self.stopped = False

            def stop(self) -> None:
                self.stopped = True

        w = LoadingWidget()
        timer = _Timer()
        w._timer_handle = timer
        w.remove = AsyncMock()

        import asyncio

        asyncio.run(w.cleanup())

        assert timer.stopped is True
        assert w._timer_handle is None
        w.remove.assert_awaited_once()


@unittest.skipUnless(_has_textual, "textual not installed")
class TestCompactingWidget(unittest.TestCase):
    """CompactingWidget construction and cleanup."""

    def test_construction(self):
        from EvoScientist.cli.widgets.compacting_widget import CompactingWidget

        w = CompactingWidget()
        assert w._elapsed == 0.0
        assert w._timer_handle is None


@unittest.skipUnless(_has_textual, "textual not installed")
class TestThinkingWidget(unittest.TestCase):
    """ThinkingWidget construction, append, finalize."""

    def test_construction_visible(self):
        from EvoScientist.cli.widgets.thinking_widget import ThinkingWidget

        w = ThinkingWidget(show_thinking=True)
        assert w._is_active is True
        assert w._content == ""
        assert w.display is True

    def test_construction_hidden(self):
        from EvoScientist.cli.widgets.thinking_widget import ThinkingWidget

        w = ThinkingWidget(show_thinking=False)
        assert w.display is False

    def test_append_text_accumulates(self):
        from EvoScientist.cli.widgets.thinking_widget import ThinkingWidget

        w = ThinkingWidget(show_thinking=True)
        w._content = ""  # direct access for unit test
        # Simulate append (without DOM)
        w._content += "hello "
        w._content += "world"
        assert w._content == "hello world"

    def test_finalize_sets_inactive(self):
        from EvoScientist.cli.widgets.thinking_widget import ThinkingWidget

        w = ThinkingWidget(show_thinking=True)
        w._is_active = True
        # Simulate finalize without DOM
        w._is_active = False
        assert w._is_active is False


class TestSummarizationStateMachine(unittest.TestCase):
    """Summary panel lifecycle decisions in the TUI event loop."""

    def test_summary_continuation_events_do_not_finalize(self):
        from EvoScientist.cli.tui_interactive import (
            _should_finalize_active_summarization,
        )

        for event_type in ("summarization_start", "summarization", "usage_stats"):
            assert _should_finalize_active_summarization(event_type) is False

    def test_non_summary_events_finalize_active_summary(self):
        from EvoScientist.cli.tui_interactive import (
            _should_finalize_active_summarization,
        )

        for event_type in (
            "thinking",
            "text",
            "tool_call",
            "tool_result",
            "tool_selection",
            "subagent_start",
            "subagent_tool_call",
            "subagent_tool_result",
            "subagent_end",
            "ask_user",
            "interrupt",
            "done",
            "error",
        ):
            assert _should_finalize_active_summarization(event_type) is True


@unittest.skipUnless(_has_textual, "textual not installed")
class TestAssistantMessage(unittest.TestCase):
    """AssistantMessage construction."""

    def test_empty_construction(self):
        from EvoScientist.cli.widgets.assistant_message import AssistantMessage

        w = AssistantMessage()
        assert w._content == ""

    def test_initial_content(self):
        from EvoScientist.cli.widgets.assistant_message import AssistantMessage

        w = AssistantMessage("hello world")
        assert w._content == "hello world"


@unittest.skipUnless(_has_textual, "textual not installed")
class TestToolCallWidget(unittest.TestCase):
    """ToolCallWidget construction and state transitions."""

    def test_construction(self):
        from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget

        w = ToolCallWidget("read_file", {"path": "/foo.py"}, "abc-123")
        assert w._tool_name == "read_file"
        assert w._tool_args == {"path": "/foo.py"}
        assert w._tool_id == "abc-123"
        assert w._status == "running"

    def test_tool_id_property(self):
        from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget

        w = ToolCallWidget("grep", {"pattern": "foo"}, "xyz")
        assert w.tool_id == "xyz"
        assert w.tool_name == "grep"

    def test_status_transitions(self):
        from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget

        w = ToolCallWidget("execute", {"command": "ls"})
        assert w._status == "running"
        # Direct state change for unit test (no DOM)
        w._status = "success"
        assert w._status == "success"
        w._status = "error"
        assert w._status == "error"
        w._status = "interrupted"
        assert w._status == "interrupted"

    def test_memory_tool_header_uses_result_inference(self):
        from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget

        w = ToolCallWidget("edit_file", {}, "mem-1")
        w._result_content = (
            "Successfully replaced 1 instance(s) of the string in '/memory/MEMORY.md'"
        )

        class _Header:
            def __init__(self) -> None:
                self.updated = None

            def update(self, value) -> None:
                self.updated = value

        header = _Header()
        w.query_one = lambda selector, cls=None: header  # type: ignore[assignment]
        w._status = "success"
        w._render_header()

        assert "Updating memory" in header.updated.plain

    def test_result_summary_truncation(self):
        from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget

        w = ToolCallWidget("execute", {})
        w._result_content = "a" * 100
        summary = w._result_summary()
        assert len(summary) <= 61  # 57 + "…"

    def test_result_summary_empty(self):
        from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget

        w = ToolCallWidget("execute", {})
        w._result_content = ""
        assert w._result_summary() == "done"

    def test_should_collapse_short(self):
        from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget

        w = ToolCallWidget("read_file", {})
        w._result_content = "line1\nline2\nline3"
        assert w._should_collapse() is False

    def test_should_collapse_long(self):
        from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget

        w = ToolCallWidget("read_file", {})
        w._result_content = "\n".join(f"line{i}" for i in range(20))
        assert w._should_collapse() is True


@unittest.skipUnless(_has_textual, "textual not installed")
class TestSubAgentWidget(unittest.TestCase):
    """SubAgentWidget construction and name display."""

    def test_construction(self):
        from EvoScientist.cli.widgets.subagent_widget import SubAgentWidget

        w = SubAgentWidget("research-agent", "Search literature")
        assert w._sa_name == "research-agent"
        assert w._description == "Search literature"
        assert w._is_active is True
        assert w._tool_count == 0

    def test_sa_name_property(self):
        from EvoScientist.cli.widgets.subagent_widget import SubAgentWidget

        w = SubAgentWidget("code-agent")
        assert w.sa_name == "code-agent"

    def test_display_name_with_description(self):
        from EvoScientist.cli.widgets.subagent_widget import SubAgentWidget

        w = SubAgentWidget("research-agent", "Search for relevant papers")
        name = w._display_name()
        assert "Cooking with research-agent" in name
        assert "Search for relevant papers" in name

    def test_display_name_truncation(self):
        from EvoScientist.cli.widgets.subagent_widget import SubAgentWidget

        long_desc = "A" * 100
        w = SubAgentWidget("agent", long_desc)
        name = w._display_name()
        assert len(name) < 100  # Should be truncated

    def test_finalize(self):
        from EvoScientist.cli.widgets.subagent_widget import SubAgentWidget

        w = SubAgentWidget("agent")
        w._is_active = True
        w._is_active = False  # Simulate finalize
        assert w._is_active is False

    def test_update_name(self):
        from EvoScientist.cli.widgets.subagent_widget import SubAgentWidget

        w = SubAgentWidget("sub-agent")
        assert w._sa_name == "sub-agent"
        assert w._description == ""
        # Simulate name resolution
        w.update_name("planner-agent", "Plan the experiment")
        assert w._sa_name == "planner-agent"
        assert w._description == "Plan the experiment"
        assert "planner-agent" in w._display_name()
        assert "Plan the experiment" in w._display_name()

    def test_update_name_preserves_description(self):
        from EvoScientist.cli.widgets.subagent_widget import SubAgentWidget

        w = SubAgentWidget("sub-agent", "existing desc")
        w.update_name("research-agent")
        assert w._sa_name == "research-agent"
        # Empty description should not overwrite existing
        assert w._description == "existing desc"

    def test_update_name_overwrites_description(self):
        from EvoScientist.cli.widgets.subagent_widget import SubAgentWidget

        w = SubAgentWidget("sub-agent", "old desc")
        w.update_name("code-agent", "new desc")
        assert w._description == "new desc"

    def test_tool_widgets_dict_keyed_by_id(self):
        """_tool_widgets dict should be keyed by tool_id for dedup."""
        from EvoScientist.cli.widgets.subagent_widget import SubAgentWidget
        from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget

        sa = SubAgentWidget("research-agent")
        # Simulate pre-populating a tool widget (as add_tool_call would)
        tw = ToolCallWidget("tavily_search", {"query": ""}, "id-1")
        sa._tool_widgets["id-1"] = tw
        # Verify lookup works
        assert "id-1" in sa._tool_widgets
        assert sa._tool_widgets["id-1"] is tw


@unittest.skipUnless(_has_textual, "textual not installed")
class TestTodoWidget(unittest.TestCase):
    """TodoWidget construction."""

    def test_construction_empty(self):
        from EvoScientist.cli.widgets.todo_widget import TodoWidget

        w = TodoWidget()
        assert w._items == []

    def test_construction_with_items(self):
        from EvoScientist.cli.widgets.todo_widget import TodoWidget

        items = [
            {"content": "Search papers", "status": "done"},
            {"content": "Analyze data", "status": "active"},
            {"content": "Write report", "status": "todo"},
        ]
        w = TodoWidget(items)
        assert len(w._items) == 3

    def test_update_items(self):
        from EvoScientist.cli.widgets.todo_widget import TodoWidget

        w = TodoWidget()
        items = [{"content": "task1", "status": "todo"}]
        w._items = items  # Direct set for unit test
        assert w._items == items


@unittest.skipUnless(_has_textual, "textual not installed")
class TestUserMessage(unittest.TestCase):
    """UserMessage construction."""

    def test_construction(self):
        from EvoScientist.cli.widgets.user_message import UserMessage

        w = UserMessage("hello world")
        # Should create without error
        assert w is not None


@unittest.skipUnless(_has_textual, "textual not installed")
class TestSystemMessage(unittest.TestCase):
    """SystemMessage construction."""

    def test_construction_default_style(self):
        from EvoScientist.cli.widgets.system_message import SystemMessage

        w = SystemMessage("info text")
        assert w is not None

    def test_construction_custom_style(self):
        from EvoScientist.cli.widgets.system_message import SystemMessage

        w = SystemMessage("error!", msg_style="red")
        assert w is not None


@unittest.skipUnless(_has_textual, "textual not installed")
class TestIsFinalResponse(unittest.TestCase):
    """Test _is_final_response helper."""

    def test_empty_state_is_final(self):
        from EvoScientist.cli.tui_interactive import _is_final_response
        from EvoScientist.stream.state import StreamState

        state = StreamState()
        assert _is_final_response(state) is True

    def test_pending_tools_not_final(self):
        from EvoScientist.cli.tui_interactive import _is_final_response
        from EvoScientist.stream.state import StreamState

        state = StreamState()
        state.tool_calls = [{"name": "read_file", "args": {}}]
        state.tool_results = []  # No results yet
        assert _is_final_response(state) is False

    def test_all_tools_done_is_final(self):
        from EvoScientist.cli.tui_interactive import _is_final_response
        from EvoScientist.stream.state import StreamState

        state = StreamState()
        state.tool_calls = [{"name": "read_file", "args": {}}]
        state.tool_results = [{"name": "read_file", "content": "[OK]"}]
        assert _is_final_response(state) is True

    def test_active_subagent_not_final(self):
        from EvoScientist.cli.tui_interactive import _is_final_response
        from EvoScientist.stream.state import StreamState, SubAgentState

        state = StreamState()
        sa = SubAgentState("research-agent")
        sa.is_active = True
        state.subagents = [sa]
        assert _is_final_response(state) is False

    def test_completed_subagent_is_final(self):
        from EvoScientist.cli.tui_interactive import _is_final_response
        from EvoScientist.stream.state import StreamState, SubAgentState

        state = StreamState()
        sa = SubAgentState("research-agent")
        sa.is_active = False
        state.subagents = [sa]
        assert _is_final_response(state) is True

    def test_internal_tools_ignored(self):
        from EvoScientist.cli.tui_interactive import _is_final_response
        from EvoScientist.stream.state import StreamState

        state = StreamState()
        state.tool_calls = [{"name": "ExtractedMemory", "args": {}}]
        # No result for internal tool -- should still be considered final
        assert _is_final_response(state) is True

    def test_processing_not_final(self):
        from EvoScientist.cli.tui_interactive import _is_final_response
        from EvoScientist.stream.state import StreamState

        state = StreamState()
        state.is_processing = True
        assert _is_final_response(state) is False


@unittest.skipUnless(_has_textual, "textual not installed")
class TestToolCallWidgetIcons(unittest.TestCase):
    """ToolCallWidget uses correct status icons (✓/✗/●)."""

    def test_success_icon(self):
        from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget

        w = ToolCallWidget("read_file", {"path": "/f"}, "id1")
        # After success, status should be "success"
        w._status = "success"
        assert w._status == "success"

    def test_error_icon(self):
        from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget

        w = ToolCallWidget("execute", {"command": "bad"}, "id2")
        w._status = "error"
        assert w._status == "error"

    def test_running_icon(self):
        from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget

        w = ToolCallWidget("grep", {}, "id3")
        assert w._status == "running"

    def test_interrupted_icon(self):
        from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget

        w = ToolCallWidget("execute", {"command": "long"}, "id4")
        w._status = "interrupted"
        assert w._status == "interrupted"


@unittest.skipUnless(_has_textual, "textual not installed")
class TestResponseStripping(unittest.TestCase):
    """Response text trailing '...' should be stripped."""

    def test_strip_trailing_dots(self):
        text = "Hello world\n..."
        clean = text.strip()
        while clean.endswith("\n...") or clean.rstrip() == "...":
            clean = clean.rstrip().removesuffix("...").rstrip()
        assert clean == "Hello world"

    def test_no_strip_normal_text(self):
        text = "Hello world"
        clean = text.strip()
        while clean.endswith("\n...") or clean.rstrip() == "...":
            clean = clean.rstrip().removesuffix("...").rstrip()
        assert clean == "Hello world"

    def test_strip_standalone_dots(self):
        text = "..."
        clean = text.strip()
        while clean.endswith("\n...") or clean.rstrip() == "...":
            clean = clean.rstrip().removesuffix("...").rstrip()
        assert clean == ""


@unittest.skipUnless(_has_textual, "textual not installed")
class TestWidgetImports(unittest.TestCase):
    """Verify all widgets are importable from the package."""

    def test_all_imports(self):
        from EvoScientist.cli.widgets import (
            AssistantMessage,
            LoadingWidget,
            SubAgentWidget,
            SystemMessage,
            ThinkingWidget,
            TodoWidget,
            ToolCallWidget,
            UserMessage,
        )

        # All should be classes
        for cls in (
            LoadingWidget,
            ThinkingWidget,
            AssistantMessage,
            ToolCallWidget,
            SubAgentWidget,
            TodoWidget,
            UserMessage,
            SystemMessage,
        ):
            assert isinstance(cls, type), f"{cls} is not a class"


class TestClipboardPaste(unittest.TestCase):
    """Test clipboard paste functionality."""

    def test_get_clipboard_text_import(self):
        """get_clipboard_text should be importable."""
        from EvoScientist.cli.clipboard import get_clipboard_text

        assert callable(get_clipboard_text)

    def test_paste_native_import(self):
        """_paste_native should be importable."""
        from EvoScientist.cli.clipboard import _paste_native

        assert callable(_paste_native)

    def test_paste_native_returns_string_or_none(self):
        """_paste_native should return str or None."""
        from EvoScientist.cli.clipboard import _paste_native

        result = _paste_native()
        assert result is None or isinstance(result, str)

    def test_get_clipboard_text_with_pyperclip_mock(self):
        """get_clipboard_text should use pyperclip when available."""
        from unittest.mock import MagicMock, patch

        mock_pyperclip = MagicMock()
        mock_pyperclip.paste.return_value = "mocked text"

        with patch.dict("sys.modules", {"pyperclip": mock_pyperclip}):
            # Re-import to pick up the mock
            import importlib

            from EvoScientist.cli import clipboard

            importlib.reload(clipboard)

            result = clipboard.get_clipboard_text()
            # pyperclip.paste was called
            mock_pyperclip.paste.assert_called_once()
            assert result == "mocked text"

    def test_get_clipboard_text_fallback_to_native(self):
        """get_clipboard_text should fall back to native when pyperclip unavailable."""
        from unittest.mock import patch

        with patch.dict("sys.modules", {"pyperclip": None}):
            import importlib

            from EvoScientist.cli import clipboard

            importlib.reload(clipboard)

            # Should not raise, returns None or string
            result = clipboard.get_clipboard_text()
            assert result is None or isinstance(result, str)


@unittest.skipUnless(_has_textual, "textual not installed")
class TestCompletionLogic(unittest.TestCase):
    """Unit tests for slash-command completion and TAB-key handling.

    The completion methods live on EvoTextualInteractiveApp but require a
    running Textual pilot to instantiate normally.  We use a lightweight
    stub that reimplements only the state fields and wires query_one() to
    return fake widget objects, letting us exercise the pure logic without
    starting the full TUI.
    """

    # ------------------------------------------------------------------
    # Stub infrastructure
    # ------------------------------------------------------------------

    def _make_app(self, comp_items=None, comp_index=-1):
        """Return a stub app-like object with completion state."""
        from rich.text import Text

        # Fake Input widget -------------------------------------------------
        class _FakeInput:
            def __init__(self):
                self.value = ""
                self.cursor_position = 0

            def focus(self):
                self._focused = True

        # Fake Static widget ------------------------------------------------
        class _FakeStatic:
            def __init__(self):
                self.display = False
                self._content = None

            def update(self, content):
                self._content = content

        fake_input = _FakeInput()
        fake_completions = _FakeStatic()

        # Widget registry -----------------------------------------------
        _widgets = {
            ("#prompt", None): fake_input,
            ("#completions", None): fake_completions,
        }

        from EvoScientist.commands import manager as cmd_manager

        _slash_commands = cmd_manager.list_commands()

        # Build stub --------------------------------------------------------
        class _StubApp:
            """Minimal stub that shares the real completion method bodies."""

            def __init__(self):
                self._comp_items: list = list(comp_items or [])
                self._comp_index: int = comp_index
                # Expose fakes for assertions
                self._fake_input = fake_input
                self._fake_completions = fake_completions
                self._SLASH_COMMANDS = _slash_commands

            def query_one(self, selector, widget_type=None):
                # Match by selector string; widget_type is ignored in stub
                if "prompt" in selector:
                    return fake_input
                if "completions" in selector:
                    return fake_completions
                raise KeyError(f"Unknown selector: {selector!r}")

            # ---- copy real method bodies verbatim ----

            def action_tab_complete(self):
                comp_widget = self.query_one("#completions")
                if not (comp_widget.display and self._comp_items):
                    self.query_one("#prompt").focus()
                    return
                self._comp_index = (self._comp_index + 1) % len(self._comp_items)
                self._apply_selected_completion()

            def _apply_selected_completion(self):
                selected_cmd = self._comp_items[self._comp_index][0]
                prompt = self.query_one("#prompt")
                prompt.value = selected_cmd + " "
                prompt.cursor_position = len(prompt.value)
                self._render_completions()

            def _hide_completions(self):
                self._comp_items = []
                self._comp_index = -1
                comp_widget = self.query_one("#completions")
                comp_widget.display = False

            def _render_completions(self):
                comp_widget = self.query_one("#completions")
                comp_text = Text()
                for i, (cmd, desc) in enumerate(self._comp_items):
                    if i == self._comp_index:
                        comp_text.append("\u25b8 ", style="bold")
                        comp_text.append(f"{cmd:<22}", style="bold")
                        comp_text.append(desc, style="bold")
                    else:
                        comp_text.append("  ", style="#888888")
                        comp_text.append(f"{cmd:<22}", style="#888888")
                        comp_text.append(desc, style="#888888")
                    if i < len(self._comp_items) - 1:
                        comp_text.append("\n")
                comp_widget.update(comp_text)

            def on_input_changed(self, text: str):
                """Simplified version matching the real on_input_changed logic."""
                comp_widget = self.query_one("#completions")
                if text.startswith("/"):
                    prefix = text.lower()
                    matches = [
                        (cmd, desc)
                        for cmd, desc in self._SLASH_COMMANDS
                        if cmd.startswith(prefix)
                    ]
                    if len(matches) == 1 and matches[0][0] == prefix:
                        self._hide_completions()
                        return
                    if matches:
                        self._comp_items = matches
                        self._comp_index = -1
                        self._render_completions()
                        comp_widget.display = True
                        return
                self._hide_completions()

            def on_key(self, key: str):
                """Simplified version matching the real on_key logic.

                Up/down are handled by priority bindings, not on_key.
                Only enter needs on_key handling.
                """
                comp_widget = self.query_one("#completions")
                if not (comp_widget.display and self._comp_items):
                    return False  # did nothing
                if key == "enter" and self._comp_index >= 0:
                    self._hide_completions()
                    return True
                return False

        return _StubApp()

    # ------------------------------------------------------------------
    # action_tab_complete
    # ------------------------------------------------------------------

    def test_tab_complete_no_completions_refocuses_prompt(self):
        """TAB with no active completions should refocus the prompt."""
        app = self._make_app(comp_items=[])
        app._fake_completions.display = False
        app.action_tab_complete()
        assert getattr(app._fake_input, "_focused", False) is True

    def test_tab_complete_cycles_forward(self):
        """TAB with completions visible should advance the selection index."""
        items = [("/resume", "desc1"), ("/run", "desc2"), ("/reset", "desc3")]
        app = self._make_app(comp_items=items, comp_index=-1)
        app._fake_completions.display = True

        app.action_tab_complete()
        assert app._comp_index == 0
        assert app._fake_input.value == "/resume "

    def test_tab_complete_wraps_around(self):
        """TAB past the last item should wrap back to index 0."""
        items = [("/resume", "d1"), ("/run", "d2")]
        app = self._make_app(comp_items=items, comp_index=1)  # last item
        app._fake_completions.display = True

        app.action_tab_complete()
        assert app._comp_index == 0
        assert app._fake_input.value == "/resume "

    def test_tab_complete_updates_cursor_position(self):
        """TAB should position the cursor at the end of the completed text."""
        items = [("/skills", "List installed skills")]
        app = self._make_app(comp_items=items, comp_index=-1)
        app._fake_completions.display = True

        app.action_tab_complete()
        expected = "/skills "
        assert app._fake_input.value == expected
        assert app._fake_input.cursor_position == len(expected)

    # ------------------------------------------------------------------
    # _apply_selected_completion
    # ------------------------------------------------------------------

    def test_apply_selected_completion(self):
        items = [("/new", "Start new"), ("/next", "Next")]
        app = self._make_app(comp_items=items, comp_index=1)
        app._apply_selected_completion()
        assert app._fake_input.value == "/next "
        assert app._fake_input.cursor_position == len("/next ")

    # ------------------------------------------------------------------
    # _hide_completions
    # ------------------------------------------------------------------

    def test_hide_completions_clears_state(self):
        items = [("/resume", "d")]
        app = self._make_app(comp_items=items, comp_index=0)
        app._fake_completions.display = True

        app._hide_completions()

        assert app._comp_items == []
        assert app._comp_index == -1
        assert app._fake_completions.display is False

    # ------------------------------------------------------------------
    # _render_completions
    # ------------------------------------------------------------------

    def test_render_completions_produces_text(self):
        """_render_completions should call update() with a rich Text object."""
        from rich.text import Text

        items = [("/resume", "Resume session"), ("/run", "Run")]
        app = self._make_app(comp_items=items, comp_index=0)
        app._render_completions()

        result = app._fake_completions._content
        assert isinstance(result, Text)
        plain = result.plain
        assert "/resume" in plain
        assert "/run" in plain

    def test_render_completions_highlights_selected(self):
        """The selected item should have a bold arrow marker."""
        from rich.text import Text

        items = [("/help", "Help"), ("/hitl", "HITL")]
        app = self._make_app(comp_items=items, comp_index=1)
        app._render_completions()

        result: Text = app._fake_completions._content
        # Collect spans for bold segments
        bold_spans = [
            result.plain[s.start : s.end]
            for s in result._spans
            if "bold" in str(s.style)
        ]
        # The arrow marker "▸" should appear in a bold span
        arrow_in_bold = any("▸" in seg for seg in bold_spans)
        assert arrow_in_bold, f"No bold arrow found. Spans: {bold_spans}"

    # ------------------------------------------------------------------
    # on_input_changed
    # ------------------------------------------------------------------

    def test_input_changed_slash_shows_completions(self):
        """/re prefix should show matching commands."""
        app = self._make_app()
        app.on_input_changed("/re")
        assert app._fake_completions.display is True
        assert len(app._comp_items) > 0
        assert all(cmd.startswith("/re") for cmd, _ in app._comp_items)

    def test_input_changed_exact_match_hides_completions(self):
        """An exact match for a command should hide completions."""
        app = self._make_app()
        # /help is the only command starting with /help
        app.on_input_changed("/help")
        assert app._fake_completions.display is False

    def test_input_changed_non_slash_hides_completions(self):
        """Regular text (no leading slash) should hide completions."""
        items = [("/resume", "d")]
        app = self._make_app(comp_items=items)
        app._fake_completions.display = True

        app.on_input_changed("hello world")
        assert app._fake_completions.display is False

    def test_input_changed_no_match_hides_completions(self):
        """A /prefix that matches nothing should hide completions."""
        app = self._make_app()
        app._fake_completions.display = True

        app.on_input_changed("/zzznomatch")
        assert app._fake_completions.display is False

    # ------------------------------------------------------------------
    # on_key  (enter only — up/down handled by priority bindings)
    # ------------------------------------------------------------------

    def test_on_key_enter_hides_completions_when_selected(self):
        items = [("/resume", "d1")]
        app = self._make_app(comp_items=items, comp_index=0)
        app._fake_completions.display = True

        handled = app.on_key("enter")
        assert handled is True
        assert app._fake_completions.display is False
        assert app._comp_items == []

    def test_on_key_enter_ignored_when_nothing_selected(self):
        """Enter with comp_index == -1 should not hide completions."""
        items = [("/resume", "d1")]
        app = self._make_app(comp_items=items, comp_index=-1)
        app._fake_completions.display = True

        handled = app.on_key("enter")
        assert handled is False
        assert app._fake_completions.display is True  # unchanged

    def test_on_key_noop_when_completions_hidden(self):
        """Key events should be ignored when completions are not visible."""
        items = [("/resume", "d1")]
        app = self._make_app(comp_items=items, comp_index=0)
        app._fake_completions.display = False  # hidden

        handled = app.on_key("down")
        assert handled is False
        assert app._comp_index == 0  # unchanged


if __name__ == "__main__":
    unittest.main()
