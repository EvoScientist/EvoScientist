"""Inline thread picker widget for /resume and /delete in TUI.

Keyboard-driven widget mounted directly into the chat container (like
ApprovalWidget).  Posts ``ThreadPickerWidget.Picked`` when user selects
a thread, or ``ThreadPickerWidget.Cancelled`` on Esc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_row_text(
    thread: dict,
    *,
    selected: bool = False,
    current: bool = False,
) -> Text:
    """Build a Rich Text object for a single thread row.

    Pure function — no Textual app context required, safe for unit tests.
    """
    from ...sessions import _format_relative_time

    tid = thread["thread_id"]
    preview = thread.get("preview", "") or ""
    msgs = thread.get("message_count", 0)
    model = thread.get("model", "") or ""
    when = _format_relative_time(thread.get("updated_at"))

    line = Text()
    cursor = "\u25b8 " if selected else "  "
    line.append(cursor, style="bold cyan" if selected else "dim")
    line.append(f"{tid}", style="bold" if selected else "")
    if current:
        line.append(" *", style="bold green")
    line.append("  ")
    if preview:
        display_preview = preview[:40] + "\u2026" if len(preview) > 40 else preview
        line.append(display_preview, style="")
        line.append("  ", style="")
    line.append(f"({msgs} msgs)", style="dim")
    if model:
        line.append(f"  {model}", style="dim italic")
    if when:
        line.append(f"  {when}", style="dim")
    return line


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


class ThreadPickerWidget(Widget):
    """Inline thread picker — mounts in chat, keyboard-driven.

    Posts ``Picked(thread_id)`` on Enter, ``Cancelled()`` on Esc.
    Follows the same pattern as ``ApprovalWidget``.
    """

    can_focus = True
    can_focus_children = False

    DEFAULT_CSS = """
    ThreadPickerWidget {
        height: auto;
        max-height: 22;
        margin: 1 0;
        padding: 0 1;
        background: $surface;
        border: solid $primary;
    }
    ThreadPickerWidget .picker-title {
        height: 1;
        text-style: bold;
        color: $primary;
    }
    ThreadPickerWidget .picker-rows {
        height: auto;
        max-height: 16;
        overflow-y: auto;
    }
    ThreadPickerWidget .picker-row {
        height: 1;
        padding: 0 1;
    }
    ThreadPickerWidget .picker-row-selected {
        background: $primary;
        text-style: bold;
    }
    ThreadPickerWidget .picker-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("k", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("j", "move_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    class Picked(Message):
        """Posted when user picks a thread."""

        def __init__(self, thread_id: str) -> None:
            super().__init__()
            self.thread_id = thread_id

    class Cancelled(Message):
        """Posted when user cancels selection."""

    def __init__(
        self,
        threads: list[dict],
        *,
        current_thread: str | None = None,
        title: str = "Select a session",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._threads = threads
        self._current_thread = current_thread
        self._title = title
        self._selected = 0
        self._row_widgets: list[Static] = []

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="picker-title")
        with Container(classes="picker-rows"):
            for _ in self._threads:
                widget = Static("", classes="picker-row")
                self._row_widgets.append(widget)
                yield widget
        yield Static(
            "\u2191/\u2193 navigate \u00b7 Enter select \u00b7 Esc cancel",
            classes="picker-help",
        )

    def on_mount(self) -> None:
        self._update_rows()
        self.call_later(self.focus)

    def _update_rows(self) -> None:
        for i, (thread, widget) in enumerate(
            zip(self._threads, self._row_widgets, strict=False)
        ):
            is_current = thread["thread_id"] == self._current_thread
            text = build_row_text(
                thread, selected=(i == self._selected), current=is_current
            )
            widget.update(text)
            widget.remove_class("picker-row-selected")
            if i == self._selected:
                widget.add_class("picker-row-selected")
                widget.scroll_visible()

    def action_move_up(self) -> None:
        if not self._threads:
            return
        self._selected = (self._selected - 1) % len(self._threads)
        self._update_rows()

    def action_move_down(self) -> None:
        if not self._threads:
            return
        self._selected = (self._selected + 1) % len(self._threads)
        self._update_rows()

    def action_select(self) -> None:
        if not self._threads:
            self.post_message(self.Cancelled())
            return
        self.post_message(self.Picked(self._threads[self._selected]["thread_id"]))

    def action_cancel(self) -> None:
        self.post_message(self.Cancelled())

    def on_blur(self, event: events.Blur) -> None:
        """Re-focus to keep focus trapped until decision is made."""
        self.call_after_refresh(self.focus)
