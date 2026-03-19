"""Token usage statistics widget."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class UsageWidget(Static):
    """Displays token usage stats, right-aligned with styled numbers."""

    DEFAULT_CSS = """
    UsageWidget {
        height: auto;
        text-align: right;
    }
    """

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        stats = Text(justify="right")
        stats.append("[", style="dim italic")
        stats.append("Usage: ", style="dim italic")
        stats.append(f"{input_tokens:,}", style="cyan italic")
        stats.append(" in · ", style="dim italic")
        stats.append(f"{output_tokens:,}", style="green italic")
        stats.append(" out", style="dim italic")
        stats.append("]", style="dim italic")
        super().__init__(stats)
