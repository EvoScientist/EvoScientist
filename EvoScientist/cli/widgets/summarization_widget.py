"""Summarization panel widget for context compression display.

Renders a Rich Panel showing that LangGraph's summarization middleware
has compressed older conversation history. Yellow/amber border to
distinguish from the blue thinking panel. Default collapsed; click to
expand/collapse.

Supports streaming: text arrives incrementally via ``append_text()``
while the widget shows a live "Summarizing..." indicator, then switches
to a collapsed preview once ``finalize()`` is called.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.text import Text
from textual.events import Click

from .timed_status_widget import TimedStatusWidget

_MAX_COLLAPSED_CHARS = 80
_MAX_EXPANDED_CHARS = 3000


class SummarizationWidget(TimedStatusWidget):
    """Collapsible panel showing context summarization.

    Streams text via ``append_text()`` (shows live spinner while active).
    Defaults to collapsed after streaming ends; click to expand/collapse.

    Usage::

        w = SummarizationWidget()
        await container.mount(w)
        w.append_text("The conversation ")
        w.append_text("covered ...")
        w.finalize()  # stop spinner, collapse
    """

    DEFAULT_CSS = """
    SummarizationWidget {
        height: auto;
        margin: 0 0 1 0;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._content = ""
        self._collapsed = True
        self._is_active = True  # still receiving chunks

    def _should_tick(self) -> bool:
        return self._is_active

    def _char_count_label(self) -> str:
        n = len(self._content)
        if n >= 1000:
            return f"{n / 1000:.1f}k chars"
        return f"{n:,} chars"

    def _refresh_display(self) -> None:
        secs = self.elapsed_seconds
        if not self._content:
            if self._is_active:
                self.update(
                    Panel(
                        Text("Summarizing...", style="dim italic"),
                        title=f"Context Summarizing... ({secs}s)",
                        border_style="#f59e0b",
                        padding=(0, 1),
                    )
                )
            else:
                self.update("")
            return

        if self._is_active:
            # While streaming: show latest content tail (like thinking widget)
            title = f"Context Summarizing... ({secs}s)"
            tail = self._content.rstrip()
            if len(tail) > 200:
                tail = tail[-200:]
            body = Text(tail, style="dim italic")
        elif self._collapsed:
            title = f"Context Summarized ({self._char_count_label()})"
            first_line = self._content.strip().split("\n")[0].strip()
            if len(first_line) > _MAX_COLLAPSED_CHARS:
                first_line = first_line[: _MAX_COLLAPSED_CHARS - 3] + "\u2026"
            preview = Text(first_line, style="dim italic")
            preview.append("  [click to expand]", style="dim italic")
            body = preview
        else:
            title = f"Context Summarized ({self._char_count_label()})"
            display = self._content.rstrip()
            if len(display) > _MAX_EXPANDED_CHARS:
                half = _MAX_EXPANDED_CHARS // 2
                display = (
                    display[:half] + "\n\n... (truncated) ...\n\n" + display[-half:]
                )
            body = (
                Text(display, style="dim italic")
                if display
                else Text("(empty)", style="dim")
            )

        self.update(Panel(body, title=title, border_style="#f59e0b", padding=(0, 1)))

    def append_text(self, text: str) -> None:
        """Append a chunk of summarization text (streaming)."""
        self._content += text
        self._refresh_display()

    def finalize(self) -> None:
        """Mark streaming as complete — switch to collapsed preview."""
        self._is_active = False
        self._collapsed = True
        self._stop_timer()
        self._refresh_display()

    def set_content(self, text: str) -> None:
        """Set the full summarization text at once (non-streaming fallback)."""
        self._content = text
        self._is_active = False
        self._refresh_display()

    def on_click(self, event: Click) -> None:
        """Toggle collapsed/expanded state."""
        if self._content and not self._is_active:
            self._collapsed = not self._collapsed
            self._refresh_display()
