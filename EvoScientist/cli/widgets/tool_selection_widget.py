"""Tool selection panel widget for LLMToolSelectorMiddleware display.

Renders a Rich Panel showing which tools were selected for the current
query. Dim border to indicate it's informational, not interactive.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.text import Text
from textual.widgets import Static


class ToolSelectionWidget(Static):
    """Static panel showing selected tools.

    Usage::

        w = ToolSelectionWidget(["ls", "read_file", "execute"])
        await container.mount(w)
    """

    DEFAULT_CSS = """
    ToolSelectionWidget {
        height: auto;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, tools: list[str]) -> None:
        super().__init__("")
        tools_str = ", ".join(tools)
        self.update(
            Panel(
                Text(tools_str, style="cyan"),
                title=f"Adaptive Selected Tools ({len(tools)})",
                border_style="#2d7d46",
                padding=(0, 1),
            )
        )
