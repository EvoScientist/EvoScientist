"""Shared engine for the TUI's keyboard-driven picker/browser widgets.

Every inline picker (model picker, thread picker, skill/MCP browsers)
follows the same pattern: a flat item list where some rows are
selectable, a wrapping highlight cursor, Enter/Esc terminal messages,
and focus trapped inside the widget until a decision is made.  This
module owns that machinery so the widgets only provide their data
model and row rendering.

Subclassing contract: Textual dispatches same-named handlers at EVERY
level of the MRO, so subclasses must NOT define ``on_mount``/``on_blur``
— they implement the ``_refresh_view()`` hook (and override
``_should_refocus_on_blur()`` if focus may legitimately leave, e.g. a
child ``Input``).  Message classes (``Picked``/``Confirmed``/
``Cancelled``) stay defined in each widget: their handler names
(``on_<widget>_<message>``) derive from the defining class.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.widget import Widget
from textual.widgets import Static

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult

# ---------------------------------------------------------------------------
# Pure selection helpers
# ---------------------------------------------------------------------------


def first_selectable_index(
    items: list[Any], is_selectable: Callable[[Any], bool]
) -> int:
    """Index of the first selectable item, or 0 when none qualifies."""
    for i, item in enumerate(items):
        if is_selectable(item):
            return i
    return 0


def move_selection(
    items: list[Any],
    current: int,
    direction: int,
    is_selectable: Callable[[Any], bool],
) -> int:
    """Next selectable index from *current*, wrapping around the list.

    Non-selectable rows (headers, separators) are skipped; when no
    selectable row exists the *current* index is returned unchanged.
    """
    if not items:
        return current
    i = (current + direction) % len(items)
    steps = 0
    while not is_selectable(items[i]) and steps < len(items):
        i = (i + direction) % len(items)
        steps += 1
    return i if is_selectable(items[i]) else current


# ---------------------------------------------------------------------------
# Widget base
# ---------------------------------------------------------------------------


class PickerWidgetBase(Widget):
    """Focus-trapped inline picker: mount-focus, blur-refocus, row
    highlight bookkeeping and description truncation."""

    can_focus = True
    can_focus_children = False

    def _refresh_view(self) -> None:
        """Render the current state into the row widgets."""
        raise NotImplementedError

    def _should_refocus_on_blur(self) -> bool:
        """Whether blur should snap focus back (focus trap)."""
        return True

    def on_mount(self) -> None:
        # Deferred so self.size is populated for width-aware rendering.
        self.call_after_refresh(self._refresh_view)
        self.call_later(self.focus)

    def on_blur(self, event: events.Blur) -> None:
        if self._should_refocus_on_blur():
            self.call_after_refresh(self.focus)

    @staticmethod
    def apply_row_highlight(
        widget: Static, selected: bool, css_class: str = "picker-row-selected"
    ) -> None:
        """Toggle the selected-row CSS class and keep the row in view."""
        widget.remove_class(css_class)
        if selected:
            widget.add_class(css_class)
            widget.scroll_visible()

    def _row_content_width(self) -> int:
        """Usable character width for a row's text content (accounts for
        widget border/padding; falls back to terminal width pre-layout)."""
        try:
            w = self.size.width
            if w > 0:
                # border (2) + widget padding (2) + row padding (2)
                return w - 6
        except Exception:
            pass
        try:
            return self.app.size.width - 10
        except Exception:
            return 100

    def _truncate(self, desc: str, name: str, *, suffix: str = "") -> str:
        """Truncate a description to fit the row, adding ellipsis."""
        # cursor(2) + indicator(2) + name + " — "(3) + suffix
        overhead = 2 + 2 + len(name) + 3 + len(suffix)
        max_len = max(20, self._row_content_width() - overhead)
        if len(desc) <= max_len:
            return desc
        return desc[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# Two-phase tag-filter → checkbox browser
# ---------------------------------------------------------------------------


class TagCheckboxBrowserBase(PickerWidgetBase):
    """Two-phase multi-select browser shared by the skill and MCP browsers.

    Phase 1 — tag picker (Enter selects a tag filter, "all" included).
    Phase 2 — checkbox list (Space toggles, Enter confirms, Esc goes back
    to phase 1 unless the widget was constructed pre-filtered).

    Subclasses provide the data adapters (``_item_name`` / ``_item_tags``
    / ``_item_desc``), the phase-2 texts, and ``_post_confirmed()``.
    """

    DEFAULT_CSS = """
    TagCheckboxBrowserBase {
        height: auto;
        max-height: 30;
        margin: 1 0;
        padding: 0 1;
        background: $surface;
        border: solid $primary;
    }
    TagCheckboxBrowserBase .browser-title {
        height: 1;
        text-style: bold;
        color: $primary;
    }
    TagCheckboxBrowserBase .browser-rows {
        height: auto;
        max-height: 20;
        overflow-y: auto;
    }
    TagCheckboxBrowserBase .browser-row {
        height: 1;
        padding: 0 1;
    }
    TagCheckboxBrowserBase .browser-row-selected {
        background: $primary;
        text-style: bold;
    }
    TagCheckboxBrowserBase .browser-help {
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

    # -- subclass adapters --------------------------------------------

    _INSTALLED_SUFFIX: ClassVar[str] = "  (installed)"
    _PHASE2_TITLE: ClassVar[str] = "Select items to install"
    _PHASE2_CONFIRM_LABEL: ClassVar[str] = "install"

    def _item_name(self, item: Any) -> str:
        raise NotImplementedError

    def _item_tags(self, item: Any) -> list[str]:
        raise NotImplementedError

    def _item_desc(self, item: Any) -> str:
        raise NotImplementedError

    def _post_confirmed(self, items: list[Any]) -> None:
        """Post the widget-specific ``Confirmed`` message."""
        raise NotImplementedError

    def _post_cancelled(self) -> None:
        """Post the widget-specific ``Cancelled`` message."""
        raise NotImplementedError

    # -- state ---------------------------------------------------------

    def __init__(
        self,
        items: list[Any],
        installed_names: set[str],
        *,
        pre_filter_tag: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._all_items = items
        self._installed_names = installed_names
        self._pre_filter_tag = pre_filter_tag.lower()
        self._selected = 0
        self._row_widgets: list[Static] = []
        self._title_widget: Static | None = None
        self._help_widget: Static | None = None

        self._phase: int = 1
        self._filtered_items: list[Any] = []
        self._checked: set[int] = set()

        # Build tag list (sorted by count desc, then alphabetically)
        from collections import Counter

        tag_counter: Counter[str] = Counter()
        for item in self._all_items:
            for t in self._item_tags(item):
                tag_counter[t.lower()] += 1
        sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
        self._tag_items: list[tuple[str, int]] = [
            ("all", len(self._all_items)),
            *sorted_tags,
        ]

        # If pre-filtered, skip to phase 2
        if self._pre_filter_tag:
            self._filtered_items = self._items_with_tag(self._pre_filter_tag)
            if self._filtered_items:
                self._phase = 2
            else:
                self._pre_filter_tag = ""

    def _items_with_tag(self, tag: str) -> list[Any]:
        if tag == "all":
            return list(self._all_items)
        return [
            item
            for item in self._all_items
            if tag in [t.lower() for t in self._item_tags(item)]
        ]

    # -- layout ---------------------------------------------------------

    def compose(self) -> ComposeResult:
        self._title_widget = Static("", classes="browser-title")
        yield self._title_widget
        with Container(classes="browser-rows"):
            max_rows = max(len(self._tag_items), len(self._all_items))
            for _ in range(max_rows):
                widget = Static("", classes="browser-row")
                self._row_widgets.append(widget)
                yield widget
        self._help_widget = Static("", classes="browser-help")
        yield self._help_widget

    # -- rendering ------------------------------------------------------

    def _refresh_view(self) -> None:
        if self._phase == 1:
            self._render_tag_picker()
        else:
            self._render_checkbox_list()

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
                text.append(f"{tag} ({count})", style="bold" if is_selected else "")
                widget.update(text)
                widget.display = True
                self.apply_row_highlight(widget, is_selected, "browser-row-selected")
            else:
                widget.update("")
                widget.display = False

    def _render_checkbox_list(self) -> None:
        n_checked = len(
            [
                i
                for i in self._checked
                if self._item_name(self._filtered_items[i]) not in self._installed_names
            ]
        )
        if self._title_widget:
            self._title_widget.update(f"{self._PHASE2_TITLE} ({n_checked} selected):")
        if self._help_widget:
            self._help_widget.update(
                "↑/↓ navigate · Space toggle · "
                f"Enter {self._PHASE2_CONFIRM_LABEL} · Esc cancel"
            )

        for i, widget in enumerate(self._row_widgets):
            if i < len(self._filtered_items):
                item = self._filtered_items[i]
                name = self._item_name(item)
                is_selected = i == self._selected
                is_installed = name in self._installed_names
                is_checked = i in self._checked

                text = Text()
                cursor = "▸ " if is_selected else "  "
                text.append(cursor, style="bold cyan" if is_selected else "dim")

                if is_installed:
                    suffix = self._INSTALLED_SUFFIX
                    desc = self._truncate(self._item_desc(item), name, suffix=suffix)
                    text.append("✓ ", style="green")
                    text.append(name, style="green dim")
                    text.append(f" — {desc}", style="dim")
                    text.append(suffix, style="dim italic")
                elif is_checked:
                    desc = self._truncate(self._item_desc(item), name)
                    text.append("● ", style="green bold")
                    text.append(name, style="bold")
                    text.append(f" — {desc}", style="")
                else:
                    desc = self._truncate(self._item_desc(item), name)
                    text.append("○ ", style="dim")
                    text.append(name, style="bold" if is_selected else "")
                    text.append(f" — {desc}", style="dim")

                widget.update(text)
                widget.display = True
                self.apply_row_highlight(widget, is_selected, "browser-row-selected")
            else:
                widget.update("")
                widget.display = False

    # -- actions ----------------------------------------------------------

    def _current_items_count(self) -> int:
        if self._phase == 1:
            return len(self._tag_items)
        return len(self._filtered_items)

    def action_move_up(self) -> None:
        n = self._current_items_count()
        if not n:
            return
        self._selected = (self._selected - 1) % n
        self._refresh_view()

    def action_move_down(self) -> None:
        n = self._current_items_count()
        if not n:
            return
        self._selected = (self._selected + 1) % n
        self._refresh_view()

    def action_toggle(self) -> None:
        """Toggle checkbox selection (phase 2 only)."""
        if self._phase != 2 or not self._filtered_items:
            return
        if self._item_name(self._filtered_items[self._selected]) in (
            self._installed_names
        ):
            return  # Can't toggle already-installed items
        if self._selected in self._checked:
            self._checked.discard(self._selected)
        else:
            self._checked.add(self._selected)
        self._refresh_view()

    def action_confirm(self) -> None:
        if self._phase == 1:
            if not self._tag_items:
                return
            tag, _ = self._tag_items[self._selected]
            self._filtered_items = self._items_with_tag(tag)
            self._phase = 2
            self._selected = 0
            self._checked = set()
            self._refresh_view()
        else:
            items = [
                self._filtered_items[i]
                for i in sorted(self._checked)
                if self._item_name(self._filtered_items[i]) not in self._installed_names
            ]
            self._post_confirmed(items)

    def action_cancel(self) -> None:
        if self._phase == 2 and not self._pre_filter_tag:
            # Go back to tag picker
            self._phase = 1
            self._selected = 0
            self._checked = set()
            self._refresh_view()
        else:
            self._post_cancelled()
