"""Inline skill browser widget for /install-skills in TUI.

Two-phase keyboard-driven widget:
  Phase 1 — tag picker (arrow keys + Enter to select, or Esc for all)
  Phase 2 — skill checkbox (arrow keys to navigate, Space to toggle, Enter to confirm)

Posts ``SkillBrowserWidget.Confirmed`` with selected install sources,
or ``SkillBrowserWidget.Cancelled`` on Esc.
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


class SkillBrowserWidget(Widget):
    """Inline skill browser — mounts in chat, keyboard-driven.

    Phase 1: Tag picker (select a tag filter or "All").
    Phase 2: Skill checkbox (toggle skills, confirm to install).
    """

    can_focus = True
    can_focus_children = False

    DEFAULT_CSS = """
    SkillBrowserWidget {
        height: auto;
        max-height: 30;
        margin: 1 0;
        padding: 0 1;
        background: $surface;
        border: solid $primary;
    }
    SkillBrowserWidget .browser-title {
        height: 1;
        text-style: bold;
        color: $primary;
    }
    SkillBrowserWidget .browser-rows {
        height: auto;
        max-height: 20;
        overflow-y: auto;
    }
    SkillBrowserWidget .browser-row {
        height: 1;
        padding: 0 1;
    }
    SkillBrowserWidget .browser-row-selected {
        background: $primary;
        text-style: bold;
    }
    SkillBrowserWidget .browser-help {
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
        """Posted when user confirms skill selection."""

        def __init__(self, install_sources: list[str]) -> None:
            super().__init__()
            self.install_sources = install_sources

    class Cancelled(Message):
        """Posted when user cancels."""

    def __init__(
        self,
        index: list[dict],
        installed_names: set[str],
        *,
        pre_filter_tag: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._index = index
        self._installed_names = installed_names
        self._pre_filter_tag = pre_filter_tag.lower()
        self._selected = 0
        self._row_widgets: list[Static] = []
        self._title_widget: Static | None = None
        self._help_widget: Static | None = None

        # Phase 1: tag picker
        # Phase 2: skill checkbox
        self._phase: int = 1
        self._tag_items: list[tuple[str, int]] = []  # (tag, count)
        self._skill_items: list[dict] = []  # filtered skills
        self._checked: set[int] = set()  # indices of checked skills

        # Build tag list (sorted by count desc, then alphabetically)
        from collections import Counter

        tag_counter: Counter[str] = Counter()
        for s in self._index:
            for t in s.get("tags", []):
                tag_counter[t.lower()] += 1
        sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
        self._tag_items = [("all", len(self._index))] + sorted_tags

        # If pre-filtered, skip to phase 2
        if self._pre_filter_tag:
            self._skill_items = [
                s
                for s in self._index
                if self._pre_filter_tag in [t.lower() for t in s.get("tags", [])]
            ]
            if self._skill_items:
                self._phase = 2
            else:
                # No matches — show tag picker anyway
                self._pre_filter_tag = ""

    def compose(self) -> ComposeResult:
        self._title_widget = Static("", classes="browser-title")
        yield self._title_widget
        with Container(classes="browser-rows"):
            # Pre-allocate enough rows for the larger of tag list or skill list
            max_rows = max(len(self._tag_items), len(self._index))
            for _ in range(max_rows):
                widget = Static("", classes="browser-row")
                self._row_widgets.append(widget)
                yield widget
        self._help_widget = Static("", classes="browser-help")
        yield self._help_widget

    def on_mount(self) -> None:
        # Defer rendering until after layout so self.size is populated
        self.call_after_refresh(self._update_display)
        self.call_later(self.focus)

    def _update_display(self) -> None:
        if self._phase == 1:
            self._render_tag_picker()
        else:
            self._render_skill_checkbox()

    def _render_tag_picker(self) -> None:
        if self._title_widget:
            self._title_widget.update("Filter by tag:")
        if self._help_widget:
            self._help_widget.update("↑/↓ navigate · Enter select · Esc cancel")

        for i, widget in enumerate(self._row_widgets):
            if i < len(self._tag_items):
                tag, count = self._tag_items[i]
                is_selected = i == self._selected
                text = Text()
                cursor = "▸ " if is_selected else "  "
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
        """Get the usable character width for a row's text content.

        Accounts for widget border, widget padding, and row padding.
        Falls back to terminal width if the widget hasn't been laid out yet.
        """
        try:
            w = self.size.width
            if w > 0:
                # border (2) + widget padding-left/right (2) + row padding-left/right (2)
                return w - 6
        except Exception:
            pass
        # Fallback: use terminal width minus reasonable chrome
        try:
            return self.app.size.width - 10
        except Exception:
            return 100

    def _truncate(self, desc: str, name: str, *, suffix: str = "") -> str:
        """Truncate a description to fit the row, adding ellipsis if needed."""
        # cursor(2) + indicator(2) + name + " — "(3) + suffix
        overhead = 2 + 2 + len(name) + 3 + len(suffix)
        max_len = max(20, self._row_content_width() - overhead)
        if len(desc) <= max_len:
            return desc
        return desc[: max_len - 1] + "…"

    def _render_skill_checkbox(self) -> None:
        n_checked = len(
            [
                i
                for i in self._checked
                if self._skill_items[i]["name"] not in self._installed_names
            ]
        )
        if self._title_widget:
            self._title_widget.update(
                f"Select skills to install ({n_checked} selected):"
            )
        if self._help_widget:
            self._help_widget.update(
                "↑/↓ navigate · Space toggle · Enter install · Esc cancel"
            )

        for i, widget in enumerate(self._row_widgets):
            if i < len(self._skill_items):
                skill = self._skill_items[i]
                is_selected = i == self._selected
                is_installed = skill["name"] in self._installed_names
                is_checked = i in self._checked

                text = Text()
                cursor = "▸ " if is_selected else "  "
                text.append(cursor, style="bold cyan" if is_selected else "dim")

                if is_installed:
                    suffix = "  (installed)"
                    desc = self._truncate(
                        desc=skill["description"],
                        name=skill["name"],
                        suffix=suffix,
                    )
                    text.append("✓ ", style="green")
                    text.append(skill["name"], style="green dim")
                    text.append(f" — {desc}", style="dim")
                    text.append(suffix, style="dim italic")
                elif is_checked:
                    desc = self._truncate(skill["description"], skill["name"])
                    text.append("● ", style="green bold")
                    text.append(skill["name"], style="bold")
                    text.append(f" — {desc}", style="")
                else:
                    desc = self._truncate(skill["description"], skill["name"])
                    text.append("○ ", style="dim")
                    text.append(skill["name"], style="bold" if is_selected else "")
                    text.append(f" — {desc}", style="dim")

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
        return len(self._skill_items)

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
        """Toggle skill selection (phase 2 only)."""
        if self._phase != 2:
            return
        if not self._skill_items:
            return
        skill = self._skill_items[self._selected]
        if skill["name"] in self._installed_names:
            return  # Can't toggle installed skills
        if self._selected in self._checked:
            self._checked.discard(self._selected)
        else:
            self._checked.add(self._selected)
        # Move down after toggle for quick multi-select
        n = len(self._skill_items)
        if n > 1:
            self._selected = (self._selected + 1) % n
        self._update_display()

    def action_confirm(self) -> None:
        if self._phase == 1:
            # Transition to phase 2
            if not self._tag_items:
                return
            tag, _ = self._tag_items[self._selected]
            if tag == "all":
                self._skill_items = list(self._index)
            else:
                self._skill_items = [
                    s
                    for s in self._index
                    if tag in [t.lower() for t in s.get("tags", [])]
                ]
            self._phase = 2
            self._selected = 0
            self._checked = set()
            self._update_display()
        else:
            # Confirm selection
            sources = [
                self._skill_items[i]["install_source"]
                for i in sorted(self._checked)
                if self._skill_items[i]["name"] not in self._installed_names
            ]
            self.post_message(self.Confirmed(sources))

    def action_cancel(self) -> None:
        if self._phase == 2 and not self._pre_filter_tag:
            # Go back to tag picker
            self._phase = 1
            self._selected = 0
            self._checked = set()
            self._update_display()
        else:
            self.post_message(self.Cancelled())

    def on_blur(self, event: events.Blur) -> None:
        """Re-focus to keep focus trapped until decision is made."""
        self.call_after_refresh(self.focus)
