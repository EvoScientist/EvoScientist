"""Collapsible widget for manual /compact summary results."""

from __future__ import annotations

from rich.panel import Panel
from rich.text import Text
from textual.events import Click
from textual.widgets import Static

_MAX_COLLAPSED_CHARS = 80
_MAX_EXPANDED_CHARS = 3000


class CompactSummaryWidget(Static):
    """Collapsible panel showing the generated manual compact summary."""

    DEFAULT_CSS = """
    CompactSummaryWidget {
        height: auto;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, summary_text: str) -> None:
        super().__init__("")
        self._content = (summary_text or "").strip()
        self._collapsed = True
        self._refresh_display()

    def _char_count_label(self) -> str:
        n = len(self._content)
        if n >= 1000:
            return f"{n / 1000:.1f}k chars"
        return f"{n:,} chars"

    def _refresh_display(self) -> None:
        if not self._content:
            self.update(
                Panel(
                    Text("(empty summary)", style="dim"),
                    title="Context Compacted",
                    border_style="#f59e0b",
                    padding=(0, 1),
                )
            )
            return

        if self._collapsed:
            title = f"Context Compacted ({self._char_count_label()})"
            first_line = self._content.strip().split("\n")[0].strip()
            if len(first_line) > _MAX_COLLAPSED_CHARS:
                first_line = first_line[: _MAX_COLLAPSED_CHARS - 3] + "..."
            preview = Text(first_line, style="dim italic")
            preview.append("  [click to expand]", style="dim italic")
            body = preview
        else:
            title = f"Context Compacted ({self._char_count_label()})"
            display = self._content.rstrip()
            if len(display) > _MAX_EXPANDED_CHARS:
                half = _MAX_EXPANDED_CHARS // 2
                display = (
                    display[:half] + "\n\n... (truncated) ...\n\n" + display[-half:]
                )
            body = Text(display, style="dim italic")

        self.update(Panel(body, title=title, border_style="#f59e0b", padding=(0, 1)))

    def on_click(self, event: Click) -> None:
        """Toggle collapsed/expanded state."""
        if self._content:
            self._collapsed = not self._collapsed
            self._refresh_display()
