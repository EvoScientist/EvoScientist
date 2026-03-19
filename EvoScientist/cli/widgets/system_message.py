"""System message widget."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class SystemMessage(Static):
    """Displays a system/status message (replaces ``_append_system``)."""

    DEFAULT_CSS = """
    SystemMessage {
        height: auto;
    }
    """

    def __init__(self, content: str, *, msg_style: str = "dim") -> None:
        super().__init__(Text(content, style=msg_style))
