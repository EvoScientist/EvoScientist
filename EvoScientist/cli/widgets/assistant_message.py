"""Assistant message widget with incremental Markdown rendering."""

from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import Markdown

from ...stream.display import _fix_markdown_heading_spacing
from .timestamp_mixin import TimestampClickMixin


class AssistantMessage(TimestampClickMixin, Vertical):
    """Displays the assistant's final Markdown response.

    Mount once, then call :meth:`append_content` for each text chunk.
    When streaming finishes, call :meth:`stop_stream`.

    Each ``append_content`` call re-renders only *this* widget's Markdown —
    not the entire chat history — which is the core improvement over the
    old "rebuild Rich Group every 100 ms" approach.
    """

    DEFAULT_CSS = """
    AssistantMessage {
        height: auto;
        margin: 1 0 0 0;
    }
    AssistantMessage Markdown {
        margin: 0;
        padding: 0;
    }
    """

    def __init__(self, initial_content: str = "") -> None:
        super().__init__()
        self._content = initial_content
        self._flush_pending = False

    def compose(self):
        yield Markdown("")

    def on_mount(self) -> None:
        """Render ``initial_content`` once the widget enters the DOM."""
        if self._content:
            self.query_one(Markdown).update(
                _fix_markdown_heading_spacing(self._content)
            )

    async def append_content(self, text: str) -> None:
        """Append text and schedule a debounced Markdown re-render."""
        self._content += text
        if not self._flush_pending:
            self._flush_pending = True
            self.set_timer(0.1, self._flush_markdown)

    def _flush_markdown(self) -> None:
        """Flush accumulated content to the Markdown widget on a display copy."""
        self._flush_pending = False
        self.query_one(Markdown).update(_fix_markdown_heading_spacing(self._content))

    async def stop_stream(self) -> None:
        """Finalize the stream — ensure final content is rendered."""
        self._flush_pending = False
        if self._content:
            self.query_one(Markdown).update(
                _fix_markdown_heading_spacing(self._content)
            )
