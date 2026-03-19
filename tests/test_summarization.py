"""Tests for the summarization event pipeline and display widgets."""

from EvoScientist.stream.emitter import StreamEventEmitter
from EvoScientist.stream.events import _extract_summarization_text
from EvoScientist.stream.state import StreamState

# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


class TestSummarizationEmitter:
    """StreamEventEmitter.summarization()."""

    def test_event_type(self):
        ev = StreamEventEmitter.summarization("hello")
        assert ev.type == "summarization"

    def test_event_data(self):
        ev = StreamEventEmitter.summarization("ctx compressed")
        assert ev.data["type"] == "summarization"
        assert ev.data["content"] == "ctx compressed"

    def test_empty_content(self):
        ev = StreamEventEmitter.summarization("")
        assert ev.data["content"] == ""


# ---------------------------------------------------------------------------
# StreamState
# ---------------------------------------------------------------------------


class TestSummarizationState:
    """StreamState handling of summarization events."""

    def test_initial_state(self):
        state = StreamState()
        assert state.summarization_text == ""

    def test_handle_summarization(self):
        state = StreamState()
        etype = state.handle_event({"type": "summarization", "content": "summary"})
        assert etype == "summarization"
        assert state.summarization_text == "summary"

    def test_accumulates_chunks(self):
        """Summarization chunks are accumulated (streaming)."""
        state = StreamState()
        state.handle_event({"type": "summarization", "content": "first"})
        state.handle_event({"type": "summarization", "content": "second"})
        assert state.summarization_text == "firstsecond"

    def test_get_display_args_includes_field(self):
        state = StreamState()
        state.handle_event({"type": "summarization", "content": "ctx"})
        args = state.get_display_args()
        assert "summarization_text" in args
        assert args["summarization_text"] == "ctx"

    def test_does_not_affect_thinking(self):
        state = StreamState()
        state.handle_event({"type": "thinking", "content": "think"})
        state.handle_event({"type": "summarization", "content": "sum"})
        assert state.thinking_text == "think"
        assert state.summarization_text == "sum"

    def test_does_not_affect_response(self):
        state = StreamState()
        state.handle_event({"type": "text", "content": "hello"})
        state.handle_event({"type": "summarization", "content": "sum"})
        assert state.response_text == "hello"
        assert state.summarization_text == "sum"


# ---------------------------------------------------------------------------
# Rich CLI display
# ---------------------------------------------------------------------------


def _render_group(group) -> str:
    """Render a Rich Group to plain text for assertion checks."""
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=True)
    console.print(group)
    return buf.getvalue()


class TestSummarizationRichDisplay:
    """create_streaming_display() with summarization_text."""

    def test_no_panel_when_empty(self):
        from EvoScientist.stream.display import create_streaming_display

        group = create_streaming_display(summarization_text="")
        rendered = _render_group(group)
        assert "Context Summarized" not in rendered

    def test_panel_rendered(self):
        from EvoScientist.stream.display import create_streaming_display

        group = create_streaming_display(
            summarization_text="The conversation was about ML.",
            response_text="ok",
        )
        rendered = _render_group(group)
        assert "Context Summarized" in rendered

    def test_long_text_truncated(self):
        from EvoScientist.stream.display import create_streaming_display

        long_text = "x" * 500
        group = create_streaming_display(
            summarization_text=long_text,
            response_text="ok",
        )
        rendered = _render_group(group)
        assert "..." in rendered


# ---------------------------------------------------------------------------
# TUI SummarizationWidget
# ---------------------------------------------------------------------------


class TestSummarizationWidget:
    """SummarizationWidget (Textual TUI)."""

    def test_init_collapsed(self):
        from EvoScientist.cli.widgets.summarization_widget import SummarizationWidget

        w = SummarizationWidget()
        assert w._collapsed is True
        assert w._content == ""

    def test_set_content(self):
        from EvoScientist.cli.widgets.summarization_widget import SummarizationWidget

        w = SummarizationWidget()
        w._content = "test content"
        assert w._content == "test content"

    def test_char_count_label_small(self):
        from EvoScientist.cli.widgets.summarization_widget import SummarizationWidget

        w = SummarizationWidget()
        w._content = "hello"
        assert w._char_count_label() == "5 chars"

    def test_char_count_label_large(self):
        from EvoScientist.cli.widgets.summarization_widget import SummarizationWidget

        w = SummarizationWidget()
        w._content = "x" * 2500
        assert w._char_count_label() == "2.5k chars"

    def test_append_text(self):
        from EvoScientist.cli.widgets.summarization_widget import SummarizationWidget

        w = SummarizationWidget()
        w.append_text("hello ")
        w.append_text("world")
        assert w._content == "hello world"
        assert w._is_active is True

    def test_finalize(self):
        from EvoScientist.cli.widgets.summarization_widget import SummarizationWidget

        w = SummarizationWidget()
        w.append_text("some summary")
        w.finalize()
        assert w._is_active is False
        assert w._collapsed is True

    def test_toggle_collapsed(self):
        from EvoScientist.cli.widgets.summarization_widget import SummarizationWidget

        w = SummarizationWidget()
        assert w._collapsed is True
        w._collapsed = not w._collapsed
        assert w._collapsed is False
        w._collapsed = not w._collapsed
        assert w._collapsed is True


# ---------------------------------------------------------------------------
# _extract_summarization_text helper
# ---------------------------------------------------------------------------


class TestExtractSummarizationText:
    """Content extraction from summarization chunks."""

    def test_string_content(self):
        class Msg:
            content = "hello world"

        assert _extract_summarization_text(Msg()) == "hello world"

    def test_content_blocks(self):
        class Msg:
            def __init__(self):
                self.content = [
                    {"type": "text", "text": "part1"},
                    {"type": "text", "text": "part2"},
                ]

        assert _extract_summarization_text(Msg()) == "part1part2"

    def test_content_blocks_with_index(self):
        """Content blocks may include 'index' field — should still extract text."""

        class Msg:
            def __init__(self):
                self.content = [{"type": "text", "text": " vs", "index": 1}]

        assert _extract_summarization_text(Msg()) == " vs"

    def test_empty_list(self):
        class Msg:
            def __init__(self):
                self.content = []

        assert _extract_summarization_text(Msg()) == ""

    def test_no_content_attr(self):
        class Msg:
            pass

        assert _extract_summarization_text(Msg()) == ""

    def test_mixed_block_types(self):
        class Msg:
            def __init__(self):
                self.content = [
                    {"type": "text", "text": "hello"},
                    {"type": "image", "url": "..."},
                ]

        assert _extract_summarization_text(Msg()) == "hello"

    def test_string_blocks_in_list(self):
        class Msg:
            def __init__(self):
                self.content = ["hello", "world"]

        assert _extract_summarization_text(Msg()) == "helloworld"
