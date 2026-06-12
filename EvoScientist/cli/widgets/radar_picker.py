"""Inline radar history picker widget for /radar history in TUI.

Keyboard-driven widget mounted into the chat container. Shows a list of
past radar reports by date. Posts ``RadarPickerWidget.Picked`` on Enter,
or ``RadarPickerWidget.Cancelled`` on Esc.
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


def _build_row_text(entry: dict, *, selected: bool = False) -> Text:
    """Build a single radar entry row."""
    line = Text()
    cursor = "\u25b8 " if selected else "  "
    line.append(cursor, style="bold cyan" if selected else "dim")
    line.append(entry["date"], style="bold" if selected else "")
    line.append(f"  ({entry['paper_count']} papers)", style="dim")
    summary = entry.get("summary", "")
    if summary:
        display = summary[:50] + "\u2026" if len(summary) > 50 else summary
        line.append(f"  {display}", style="dim italic")
    return line


class RadarPickerWidget(Widget):
    """Inline radar history picker — mounts in chat, keyboard-driven.

    Posts ``Picked(date_str)`` on Enter, ``Cancelled()`` on Esc.
    """

    can_focus = True
    can_focus_children = False

    DEFAULT_CSS = """
    RadarPickerWidget {
        height: auto;
        max-height: 20;
        margin: 1 0;
        padding: 0 1;
        background: $surface;
        border: solid $primary;
    }
    RadarPickerWidget .picker-title {
        height: 1;
        text-style: bold;
        color: $primary;
    }
    RadarPickerWidget .picker-rows {
        height: auto;
        max-height: 16;
        overflow-y: auto;
    }
    RadarPickerWidget .picker-row {
        height: 1;
        padding: 0 1;
    }
    RadarPickerWidget .picker-row-selected {
        background: $primary;
        text-style: bold;
    }
    RadarPickerWidget .picker-help {
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
        def __init__(self, date_str: str) -> None:
            super().__init__()
            self.date_str = date_str

    class Cancelled(Message):
        """Posted when user cancels selection."""

    def __init__(
        self,
        entries: list[dict],
        *,
        title: str = ">>> Select radar report <<<",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._entries = entries
        self._title = title
        self._selected = 0 if entries else -1
        self._row_widgets: list[Static] = []

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="picker-title")
        with Container(classes="picker-rows"):
            for _ in self._entries:
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
        for i, (entry, widget) in enumerate(
            zip(self._entries, self._row_widgets, strict=False)
        ):
            is_selected = i == self._selected
            widget.remove_class("picker-row-selected")
            widget.update(_build_row_text(entry, selected=is_selected))
            if is_selected:
                widget.add_class("picker-row-selected")
                widget.scroll_visible()

    def action_move_up(self) -> None:
        if self._entries and self._selected > 0:
            self._selected -= 1
            self._update_rows()

    def action_move_down(self) -> None:
        if self._entries and self._selected < len(self._entries) - 1:
            self._selected += 1
            self._update_rows()

    def action_select(self) -> None:
        if not self._entries or self._selected < 0:
            self.post_message(self.Cancelled())
            return
        self.post_message(self.Picked(self._entries[self._selected]["date"]))

    def action_cancel(self) -> None:
        self.post_message(self.Cancelled())

    def on_blur(self, event: events.Blur) -> None:
        self.call_after_refresh(self.focus)
