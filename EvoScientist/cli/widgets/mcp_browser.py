"""Inline MCP server browser widget for /install-mcp in TUI.

Two-phase keyboard-driven widget (mirrors SkillBrowserWidget):
  Phase 1 — tag picker (arrow keys + Enter to select, or Esc for all)
  Phase 2 — server checkbox (arrow keys to navigate, Space to toggle, Enter to confirm)

Posts ``MCPBrowserWidget.Confirmed`` with selected MCPServerEntry objects,
or ``MCPBrowserWidget.Cancelled`` on Esc.
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

    from ...mcp.registry import MCPServerEntry


class MCPBrowserWidget(Widget):
    """Inline MCP server browser — mounts in chat, keyboard-driven.

    Phase 1: Tag picker (select a tag filter or "All").
    Phase 2: Server checkbox (toggle servers, confirm to install).
    """

    can_focus = True
    can_focus_children = False

    DEFAULT_CSS = """
    MCPBrowserWidget {
        height: auto;
        max-height: 30;
        margin: 1 0;
        padding: 0 1;
        background: $surface;
        border: solid $primary;
    }
    MCPBrowserWidget .browser-title {
        height: 1;
        text-style: bold;
        color: $primary;
    }
    MCPBrowserWidget .browser-rows {
        height: auto;
        max-height: 20;
        overflow-y: auto;
    }
    MCPBrowserWidget .browser-row {
        height: 1;
        padding: 0 1;
    }
    MCPBrowserWidget .browser-row-selected {
        background: $primary;
        text-style: bold;
    }
    MCPBrowserWidget .browser-help {
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
        Binding("enter", "confirm", "Confirm", show=False),
        Binding("space", "toggle", "Toggle", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    class Confirmed(Message):
        """Posted when user confirms server selection."""

        def __init__(self, entries: list[MCPServerEntry]) -> None:
            super().__init__()
            self.entries = entries

    class Cancelled(Message):
        """Posted when user cancels."""

    def __init__(
        self,
        servers: list[MCPServerEntry],
        installed_names: set[str],
        *,
        pre_filter_tag: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._servers = servers
        self._installed_names = installed_names
        self._pre_filter_tag = pre_filter_tag.lower()
        self._selected = 0
        self._row_widgets: list[Static] = []
        self._title_widget: Static | None = None
        self._help_widget: Static | None = None

        # Phase 1: tag picker
        # Phase 2: server checkbox
        self._phase: int = 1
        self._tag_items: list[tuple[str, int]] = []
        self._server_items: list[MCPServerEntry] = []
        self._checked: set[int] = set()

        # Build tag list
        from collections import Counter

        tag_counter: Counter[str] = Counter()
        for s in self._servers:
            for t in s.tags:
                tag_counter[t.lower()] += 1
        sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
        self._tag_items = [("all", len(self._servers))] + sorted_tags

        # If pre-filtered, skip to phase 2
        if self._pre_filter_tag:
            self._server_items = [
                s
                for s in self._servers
                if self._pre_filter_tag in [t.lower() for t in s.tags]
            ]
            if self._server_items:
                self._phase = 2
            else:
                self._pre_filter_tag = ""

    def compose(self) -> ComposeResult:
        self._title_widget = Static("", classes="browser-title")
        yield self._title_widget
        with Container(classes="browser-rows"):
            max_rows = max(len(self._tag_items), len(self._servers))
            for _ in range(max_rows):
                widget = Static("", classes="browser-row")
                self._row_widgets.append(widget)
                yield widget
        self._help_widget = Static("", classes="browser-help")
        yield self._help_widget

    def on_mount(self) -> None:
        self.call_after_refresh(self._update_display)
        self.call_later(self.focus)

    def _update_display(self) -> None:
        if self._phase == 1:
            self._render_tag_picker()
        else:
            self._render_server_checkbox()

    def _render_tag_picker(self) -> None:
        if self._title_widget:
            self._title_widget.update("Filter by tag:")
        if self._help_widget:
            self._help_widget.update("\u2191/\u2193 navigate \u00b7 Enter select \u00b7 Esc cancel")

        for i, widget in enumerate(self._row_widgets):
            if i < len(self._tag_items):
                tag, count = self._tag_items[i]
                is_selected = i == self._selected
                text = Text()
                cursor = "\u25b8 " if is_selected else "  "
                text.append(cursor, style="bold cyan" if is_selected else "dim")
                label = f"{tag} ({count})"
                text.append(label, style="bold" if is_selected else "")
                widget.update(text)
                widget.display = True
                widget.remove_class("browser-row-selected")
                if is_selected:
                    widget.add_class("browser-row-selected")
                    widget.scroll_visible()
            else:
                widget.update("")
                widget.display = False

    def _row_content_width(self) -> int:
        try:
            w = self.size.width
            if w > 0:
                return w - 6
        except Exception:
            pass
        try:
            return self.app.size.width - 10
        except Exception:
            return 100

    def _truncate(self, desc: str, name: str, *, suffix: str = "") -> str:
        overhead = 2 + 2 + len(name) + 3 + len(suffix)
        max_len = max(20, self._row_content_width() - overhead)
        if len(desc) <= max_len:
            return desc
        return desc[: max_len - 1] + "\u2026"

    def _render_server_checkbox(self) -> None:
        n_checked = len(
            [
                i
                for i in self._checked
                if self._server_items[i].name not in self._installed_names
            ]
        )
        if self._title_widget:
            self._title_widget.update(
                f"Select MCP servers to install ({n_checked} selected):"
            )
        if self._help_widget:
            self._help_widget.update(
                "\u2191/\u2193 navigate \u00b7 Space toggle \u00b7 Enter install \u00b7 Esc cancel"
            )

        for i, widget in enumerate(self._row_widgets):
            if i < len(self._server_items):
                entry = self._server_items[i]
                is_selected = i == self._selected
                is_installed = entry.name in self._installed_names
                is_checked = i in self._checked

                text = Text()
                cursor = "\u25b8 " if is_selected else "  "
                text.append(cursor, style="bold cyan" if is_selected else "dim")

                desc = entry.description or entry.label

                if is_installed:
                    suffix = "  (configured)"
                    desc = self._truncate(desc, entry.name, suffix=suffix)
                    text.append("\u2713 ", style="green")
                    text.append(entry.name, style="green dim")
                    text.append(f" \u2014 {desc}", style="dim")
                    text.append(suffix, style="dim italic")
                elif is_checked:
                    desc = self._truncate(desc, entry.name)
                    text.append("\u25cf ", style="green bold")
                    text.append(entry.name, style="bold")
                    text.append(f" \u2014 {desc}", style="")
                else:
                    desc = self._truncate(desc, entry.name)
                    text.append("\u25cb ", style="dim")
                    text.append(entry.name, style="bold" if is_selected else "")
                    text.append(f" \u2014 {desc}", style="dim")

                widget.update(text)
                widget.display = True
                widget.remove_class("browser-row-selected")
                if is_selected:
                    widget.add_class("browser-row-selected")
                    widget.scroll_visible()
            else:
                widget.update("")
                widget.display = False

    def _current_items_count(self) -> int:
        if self._phase == 1:
            return len(self._tag_items)
        return len(self._server_items)

    def action_move_up(self) -> None:
        n = self._current_items_count()
        if not n:
            return
        self._selected = (self._selected - 1) % n
        self._update_display()

    def action_move_down(self) -> None:
        n = self._current_items_count()
        if not n:
            return
        self._selected = (self._selected + 1) % n
        self._update_display()

    def action_toggle(self) -> None:
        if self._phase != 2:
            return
        if not self._server_items:
            return
        entry = self._server_items[self._selected]
        if entry.name in self._installed_names:
            return
        if self._selected in self._checked:
            self._checked.discard(self._selected)
        else:
            self._checked.add(self._selected)
        self._update_display()

    def action_confirm(self) -> None:
        if self._phase == 1:
            if not self._tag_items:
                return
            tag, _ = self._tag_items[self._selected]
            if tag == "all":
                self._server_items = list(self._servers)
            else:
                self._server_items = [
                    s
                    for s in self._servers
                    if tag in [t.lower() for t in s.tags]
                ]
            self._phase = 2
            self._selected = 0
            self._checked = set()
            self._update_display()
        else:
            entries = [
                self._server_items[i]
                for i in sorted(self._checked)
                if self._server_items[i].name not in self._installed_names
            ]
            self.post_message(self.Confirmed(entries))

    def action_cancel(self) -> None:
        if self._phase == 2 and not self._pre_filter_tag:
            self._phase = 1
            self._selected = 0
            self._checked = set()
            self._update_display()
        else:
            self.post_message(self.Cancelled())

    def on_blur(self, event: events.Blur) -> None:
        self.call_after_refresh(self.focus)
