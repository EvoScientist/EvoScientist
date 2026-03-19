"""User message widget."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class UserMessage(Static):
    """Displays user input with a blue prompt marker."""

    DEFAULT_CSS = """
    UserMessage {
        height: auto;
        margin: 1 0 0 0;
    }
    """

    def __init__(self, content: str) -> None:
        renderable = Text.assemble(
            ("> ", "bold #38bdf8"),
            (content, "#e5e7eb"),
        )
        super().__init__(renderable)
