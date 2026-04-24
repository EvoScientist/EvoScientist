"""Inline model picker widget for /model command in TUI.

Keyboard-driven widget mounted directly into the chat container.
Models are grouped by provider with a search/filter input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from rich.text import Text
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult


# Sentinel ``model_id`` used for the "Custom Ollama model..." pseudo-row.
# Selecting this row switches the widget into free-text input mode instead
# of posting ``Picked`` — the user types a model name, Enter confirms.
_CUSTOM_OLLAMA_ID = "__custom_ollama__"


def _build_items(
    entries: list[tuple[str, str, str]],
    current_model: str | None = None,
    current_provider: str | None = None,
    filter_text: str = "",
) -> list[dict]:
    """Build the flat item list rendered by ModelPickerWidget.

    Returns a list of::

        {"type": "header", "label": str}
        {"type": "model",  "name": str, "model_id": str, "provider": str, "current": bool}
    """
    # Apply filter. The Custom Ollama sentinel is the user's escape hatch
    # when no local models match; it must remain visible regardless of filter.
    if filter_text:
        ft = filter_text.lower()
        entries = [
            (n, mid, p)
            for n, mid, p in entries
            if mid == _CUSTOM_OLLAMA_ID or ft in n.lower() or ft in p.lower()
        ]

    # Group by provider preserving order. Deduplicate the Custom Ollama
    # sentinel defensively — if callers somehow pass two sentinel rows
    # (state reuse, stale merges), collapse them into one to avoid
    # rendering duplicate "Custom Ollama model..." rows in the picker.
    groups: dict[str, list[tuple[str, str, str]]] = {}
    seen_sentinel = False
    for name, model_id, provider in entries:
        if model_id == _CUSTOM_OLLAMA_ID:
            if seen_sentinel:
                continue
            seen_sentinel = True
        if provider not in groups:
            groups[provider] = []
        groups[provider].append((name, model_id, provider))

    items: list[dict] = []
    for provider, models in groups.items():
        items.append({"type": "header", "label": provider})
        for name, model_id, prov in models:
            is_current = name == current_model and prov == current_provider
            items.append(
                {
                    "type": "model",
                    "name": name,
                    "model_id": model_id,
                    "provider": prov,
                    "current": is_current,
                }
            )
    return items


class ModelPickerWidget(Widget):
    """Inline model picker -- mounts in chat, keyboard-driven.

    Posts ``Picked(name, provider)`` on Enter, ``Cancelled()`` on Esc.
    Type to filter models.
    """

    can_focus = True
    # Required so the Custom Ollama ``Input`` child can hold focus when the
    # user is typing a model name.
    can_focus_children = True

    DEFAULT_CSS = """
    ModelPickerWidget {
        height: auto;
        max-height: 30;
        margin: 1 0;
        padding: 0 1;
        background: $surface;
        border: solid $primary;
    }
    ModelPickerWidget .picker-custom-input {
        height: 3;
        margin: 1 0 0 0;
    }
    ModelPickerWidget .picker-title {
        height: 1;
        text-style: bold;
        color: $primary;
    }
    ModelPickerWidget .picker-filter {
        height: 1;
        padding: 0 1;
        color: $text;
    }
    ModelPickerWidget .picker-rows {
        height: auto;
        max-height: 22;
        overflow-y: auto;
    }
    ModelPickerWidget .picker-header {
        height: 1;
        padding: 0 1;
        margin-top: 1;
    }
    ModelPickerWidget .picker-row {
        height: 1;
        padding: 0 1;
    }
    ModelPickerWidget .picker-row-selected {
        background: $primary;
        text-style: bold;
    }
    ModelPickerWidget .picker-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("backspace", "backspace", "Backspace", show=False),
    ]

    class Picked(Message):
        def __init__(self, name: str, provider: str) -> None:
            super().__init__()
            self.name = name
            self.provider = provider

    class Cancelled(Message):
        """Posted when user cancels selection."""

    def __init__(
        self,
        entries: list[tuple[str, str, str]],
        *,
        current_model: str | None = None,
        current_provider: str | None = None,
        title: str = ">>> Select model <<<",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._entries = entries
        self._current_model = current_model
        self._current_provider = current_provider
        self._title = title
        self._filter_text = ""
        self._items = _build_items(
            entries,
            current_model=current_model,
            current_provider=current_provider,
        )
        self._selected = self._first_model_index()
        self._row_widgets: list[Static] = []
        self._filter_widget: Static | None = None
        # "list" = arrow-key selection over models; "input" = free-text entry
        # for Custom Ollama model name. Transitions: selecting the sentinel
        # row enters input mode; Esc or Up arrow inside input mode returns to
        # list mode without closing the picker.
        self._mode: Literal["list", "input"] = "list"
        self._custom_input: Input | None = None

    def _first_model_index(self) -> int:
        for i, item in enumerate(self._items):
            if item["type"] == "model":
                return i
        return 0

    def _move(self, direction: int) -> None:
        if not self._items:
            return
        i = (self._selected + direction) % len(self._items)
        steps = 0
        while self._items[i]["type"] != "model" and steps < len(self._items):
            i = (i + direction) % len(self._items)
            steps += 1
        if self._items[i]["type"] == "model":
            self._selected = i
            self._update_rows()

    def _rebuild(self) -> None:
        """Rebuild items from filter and re-render."""
        self._items = _build_items(
            self._entries,
            current_model=self._current_model,
            current_provider=self._current_provider,
            filter_text=self._filter_text,
        )
        self._selected = self._first_model_index()
        # Re-mount rows
        rows_container = self.query_one(".picker-rows", Container)
        for w in list(rows_container.children):
            w.remove()
        self._row_widgets.clear()
        for item in self._items:
            css = "picker-header" if item["type"] == "header" else "picker-row"
            widget = Static("", classes=css)
            self._row_widgets.append(widget)
            rows_container.mount(widget)
        self._update_rows()
        self._update_filter()

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="picker-title")
        self._filter_widget = Static("", classes="picker-filter")
        yield self._filter_widget
        with Container(classes="picker-rows"):
            for item in self._items:
                css = "picker-header" if item["type"] == "header" else "picker-row"
                widget = Static("", classes=css)
                self._row_widgets.append(widget)
                yield widget
        # Hidden until the user selects "Custom Ollama model..." \u2014 then shown
        # and focused for free-text entry of an Ollama model name.
        self._custom_input = Input(
            placeholder="Type Ollama model name (e.g. llama3.3)...",
            classes="picker-custom-input",
        )
        self._custom_input.display = False
        yield self._custom_input
        yield Static(
            "\u2191/\u2193 navigate \u00b7 Enter select \u00b7 Type to filter \u00b7 Esc cancel",
            classes="picker-help",
        )

    def on_mount(self) -> None:
        self._update_rows()
        self._update_filter()
        self.call_later(self.focus)

    def _update_filter(self) -> None:
        if self._filter_widget is not None:
            if self._filter_text:
                t = Text()
                t.append("  Filter: ", style="dim")
                t.append(self._filter_text, style="bold")
                t.append("\u2588", style="blink")
                self._filter_widget.update(t)
            else:
                self._filter_widget.update(
                    Text("  Type to filter...", style="dim italic")
                )

    def _update_rows(self) -> None:
        for i, (item, widget) in enumerate(
            zip(self._items, self._row_widgets, strict=False)
        ):
            widget.remove_class("picker-row-selected")
            if item["type"] == "header":
                t = Text()
                t.append("\u2500\u2500 ", style="bold cyan")
                t.append(item["label"], style="bold cyan")
                widget.update(t)
            else:
                is_selected = i == self._selected
                t = Text()
                cursor = "\u25b8 " if is_selected else "  "
                t.append(cursor, style="bold cyan" if is_selected else "dim")
                t.append(item["name"], style="bold" if is_selected else "")
                if item["current"]:
                    t.append(" *", style="bold green")
                t.append(f"  ({item['provider']})", style="dim italic")
                widget.update(t)
                if is_selected:
                    widget.add_class("picker-row-selected")
                    widget.scroll_visible()

    def on_key(self, event: events.Key) -> None:
        # In input mode, the Input child owns printable keys + backspace.
        if self._mode == "input":
            return
        # Let bindings handle special keys
        if event.key in ("up", "down", "enter", "escape", "backspace"):
            return
        # Printable characters -> filter
        if event.character and event.character.isprintable():
            self._filter_text += event.character
            self._rebuild()
            event.prevent_default()

    def action_backspace(self) -> None:
        if self._mode == "input":
            # Input widget handles its own backspace.
            return
        if self._filter_text:
            self._filter_text = self._filter_text[:-1]
            self._rebuild()

    def action_move_up(self) -> None:
        if self._mode == "input":
            # Up from the Input field escapes back to list selection.
            self._exit_input_mode()
            return
        self._move(-1)

    def action_move_down(self) -> None:
        if self._mode == "input":
            # Down in input mode is ambiguous; absorb rather than toggle.
            return
        self._move(1)

    def action_select(self) -> None:
        if self._mode == "input":
            self._submit_custom_input()
            return
        if not self._items or self._selected >= len(self._items):
            self.post_message(self.Cancelled())
            return
        item = self._items[self._selected]
        if item["type"] != "model":
            self.post_message(self.Cancelled())
            return
        if item["provider"] == "ollama" and item["model_id"] == _CUSTOM_OLLAMA_ID:
            self._enter_input_mode()
            return
        self.post_message(self.Picked(item["name"], item["provider"]))

    def action_cancel(self) -> None:
        if self._mode == "input":
            # Esc returns to list selection; does NOT close the picker.
            self._exit_input_mode()
            return
        self.post_message(self.Cancelled())

    def on_blur(self, event: events.Blur) -> None:
        # When the Input child has focus we must NOT steal it back.
        if self._mode == "input":
            return
        self.call_after_refresh(self.focus)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Safety net: Enter fired inside the Input widget rather than
        bubbling to ``action_select``. Route to the same submit path."""
        if event.input is self._custom_input:
            event.stop()
            self._submit_custom_input()

    def _enter_input_mode(self) -> None:
        """Show the Custom Ollama Input and move focus into it."""
        self._mode = "input"
        if self._custom_input is not None:
            self._custom_input.display = True
            # Carry any filter text over as a nice touch — user may have
            # started typing a model name thinking it would filter.
            self._custom_input.value = self._filter_text
            self._custom_input.focus()

    def _exit_input_mode(self) -> None:
        """Hide the Input and return focus to the list."""
        self._mode = "list"
        if self._custom_input is not None:
            self._custom_input.display = False
            self._custom_input.value = ""
        self.focus()

    def _submit_custom_input(self) -> None:
        """Confirm the typed Ollama model name. Empty input is a no-op —
        user can Esc out or keep typing."""
        typed = (self._custom_input.value if self._custom_input else "").strip()
        if not typed:
            return
        self.post_message(self.Picked(typed, "ollama"))
