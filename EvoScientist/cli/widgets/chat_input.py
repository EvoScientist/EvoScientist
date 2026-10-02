"""Multi-line chat input widget with Enter-to-submit and modifier+Enter newline."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import ClassVar

from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class ChatTextArea(TextArea):
    """TextArea that submits on Enter and inserts newline on modifier+Enter.

    Emits :class:`ChatTextArea.Submitted` when the user presses Enter
    with non-empty text.  Modifier+Enter (Option+Enter on macOS,
    Ctrl+J everywhere) inserts a literal newline instead.

    An optional *before_submit* callback can be set to intercept Enter
    before submission.  If it returns ``True`` the submit is suppressed
    (the callback handled the event itself).
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding(
            "shift+enter,ctrl+j,alt+enter,ctrl+enter",
            "insert_newline",
            "New Line",
            show=False,
            priority=True,
        ),
    ]

    DEFAULT_CSS = """
    ChatTextArea {
        height: auto;
        min-height: 1;
        max-height: 8;
        border: none;
        padding: 0;
        background: transparent;
    }
    ChatTextArea:focus {
        border: none;
    }
    ChatTextArea .text-area--cursor-line {
        background: transparent;
    }
    """

    class Submitted(Message):
        """Posted when the user presses Enter to submit."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def __init__(
        self,
        *,
        placeholder: str = "",
        id: str | None = None,
        before_submit: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(
            id=id,
            language=None,
            show_line_numbers=False,
            soft_wrap=True,
        )
        self._placeholder = placeholder
        self.before_submit: Callable[[], bool] | None = before_submit

    @property
    def value(self) -> str:
        """Get the current text content."""
        return self.text

    @value.setter
    def value(self, new_value: str) -> None:
        """Set the text content (clears and replaces)."""
        self.clear()
        if new_value:
            self.insert(new_value)

    def action_insert_newline(self) -> None:
        """Insert a literal newline character."""
        self.insert("\n")

    async def _on_key(self, event: events.Key) -> None:
        """Handle Enter as submit.

        If *before_submit* is set and returns ``True``, the submit is
        suppressed (the callback already handled the Enter press, e.g.
        to apply a completion selection).
        """
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            # Let the host intercept Enter (e.g. for completion selection)
            if self.before_submit and self.before_submit():
                return
            value = self.text.strip()
            if value:
                self.post_message(self.Submitted(value))
            return
        await super()._on_key(event)

    @staticmethod
    def newline_shortcut_label() -> str:
        """Return the platform-native label for the newline shortcut."""
        return "Option+Enter" if sys.platform == "darwin" else "Ctrl+J"
