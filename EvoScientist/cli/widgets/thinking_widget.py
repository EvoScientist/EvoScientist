"""Thinking panel widget for extended thinking display.

Renders a Rich Panel matching the Rich CLI's style:
blue border, "Thinking" title with spinner, dim content.
Collapses to a one-line summary after finalize; click to expand/collapse.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.text import Text
from textual.events import Click
from textual.widgets import Static

_SPINNER_FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
_MAX_DISPLAY_CHARS = 1000
_MAX_EXPANDED_CHARS = 5000


class ThinkingWidget(Static):
    """Collapsible panel showing the model's extended thinking.

    Uses Rich ``Panel`` for rendering, matching the Rich CLI output exactly.
    While streaming, shows tail-truncated content with spinner.
    After finalize, collapses to a one-line summary; click to expand/collapse.

    Usage::

        w = ThinkingWidget(show_thinking=True)
        await container.mount(w)
        w.append_text("reasoning chunk...")
        w.finalize()  # stop spinner, collapse
    """

    DEFAULT_CSS = """
    ThinkingWidget {
        height: auto;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, *, show_thinking: bool = True) -> None:
        super().__init__("")
        self._content = ""
        self._is_active = True
        self._show = show_thinking
        self._collapsed = False
        self._frame = 0
        self._timer_handle = None
        if not show_thinking:
            self.display = False

    def on_mount(self) -> None:
        self._timer_handle = self.set_interval(0.1, self._tick)
        self._refresh_display()

    def _tick(self) -> None:
        if self._is_active:
            self._frame = (self._frame + 1) % len(_SPINNER_FRAMES)
            self._refresh_display()

    def _char_count_label(self) -> str:
        n = len(self._content)
        if n >= 1000:
            return f"{n / 1000:.1f}k chars"
        return f"{n:,} chars"

    def _refresh_display(self) -> None:
        if self._is_active:
            # Streaming: show tail-truncated content with spinner
            char = _SPINNER_FRAMES[self._frame]
            title = f"Thinking {char}"
            display = self._content.rstrip()
            if len(display) > _MAX_DISPLAY_CHARS:
                display = "..." + display[-_MAX_DISPLAY_CHARS:]
            body = Text(display, style="dim") if display else Text("...", style="dim")
        elif self._collapsed:
            # Finalized + collapsed: first line preview + click hint
            title = f"Thinking ({self._char_count_label()})"
            first_line = self._content.strip().split("\n")[0].strip()
            if len(first_line) > 80:
                first_line = first_line[:77] + "\u2026"
            preview = Text(first_line, style="dim")
            preview.append("  [click to expand]", style="dim italic")
            body = preview
        else:
            # Finalized + expanded: show with middle elision
            title = f"Thinking ({self._char_count_label()})"
            display = self._content.rstrip()
            if len(display) > _MAX_EXPANDED_CHARS:
                half = _MAX_EXPANDED_CHARS // 2
                display = (
                    display[:half] + "\n\n... (truncated) ...\n\n" + display[-half:]
                )
            body = (
                Text(display, style="dim") if display else Text("(empty)", style="dim")
            )

        self.update(Panel(body, title=title, border_style="blue", padding=(0, 1)))

    def append_text(self, chunk: str) -> None:
        """Append a chunk of thinking text."""
        self._content += chunk
        self._refresh_display()

    def finalize(self) -> None:
        """Mark thinking as complete — stop spinner, collapse."""
        self._is_active = False
        self._collapsed = True
        if self._timer_handle is not None:
            self._timer_handle.stop()
            self._timer_handle = None
        self._refresh_display()

    def on_click(self, event: Click) -> None:
        """Toggle collapsed/expanded state after finalize."""
        if self._is_active:
            return
        self._collapsed = not self._collapsed
        self._refresh_display()
