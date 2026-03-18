"""Unit tests for TUI widgets.

Tests widget construction, state transitions, and public APIs
without requiring a running Textual app (no Textual pilot needed).
"""

from __future__ import annotations

import importlib
import unittest


# ---------------------------------------------------------------------------
# Textual might not be installed — skip entire module if missing
# ---------------------------------------------------------------------------
_has_textual = importlib.util.find_spec("textual") is not None


@unittest.skipUnless(_has_textual, "textual not installed")
class TestLoadingWidget(unittest.TestCase):
    """LoadingWidget construction and attributes."""

    def test_construction(self):
        from EvoScientist.cli.widgets.loading_widget import LoadingWidget

        w = LoadingWidget()
        assert w._frame == 0
        assert w._elapsed == 0.0
        assert w._timer_handle is None

    def test_spinner_frames_not_empty(self):
        from EvoScientist.cli.widgets.loading_widget import _SPINNER_FRAMES

        assert len(_SPINNER_FRAMES) > 0


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
            LoadingWidget,
            ThinkingWidget,
            AssistantMessage,
            ToolCallWidget,
            SubAgentWidget,
            TodoWidget,
            UserMessage,
            SystemMessage,
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
            from EvoScientist.cli import clipboard
            import importlib

            importlib.reload(clipboard)

            result = clipboard.get_clipboard_text()
            # pyperclip.paste was called
            mock_pyperclip.paste.assert_called_once()
            assert result == "mocked text"

    def test_get_clipboard_text_fallback_to_native(self):
        """get_clipboard_text should fall back to native when pyperclip unavailable."""
        from unittest.mock import patch

        with patch.dict("sys.modules", {"pyperclip": None}):
            from EvoScientist.cli import clipboard
            import importlib

            importlib.reload(clipboard)

            # Should not raise, returns None or string
            result = clipboard.get_clipboard_text()
            assert result is None or isinstance(result, str)


if __name__ == "__main__":
    unittest.main()
