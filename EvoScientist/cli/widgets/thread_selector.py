"""Inline thread picker widget for /resume and /delete in TUI.

Keyboard-driven widget mounted directly into the chat container (like
ApprovalWidget).  Posts ``ThreadPickerWidget.Picked`` when user selects
a thread, or ``ThreadPickerWidget.Cancelled`` on Esc.

Threads are grouped into a two-level hierarchy:

  L1 header  — common ancestor path shared by 2+ workspaces, or the
               workspace path itself for standalone workspaces.
  L2 subheader — relative sub-path shown only when a group contains
                 multiple workspaces.  Run-mode dirs are marked with 🔁.
  thread row — indented under their sub-path (or directly under L1 for
               standalone groups).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from prompt_toolkit.styles import Style as PtStyle  # type: ignore[import-untyped]
from rich.text import Text
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult


# Style for questionary pickers used by the Rich CLI ``/resume`` and
# ``/delete`` interactive selectors. Matches the slash-completion menu's
# visual language: gray (#888888) for non-selected, bold for selected,
# no background changes.
PICKER_STYLE = PtStyle.from_dict(
    {
        "questionmark": "#888888",
        "question": "",
        "pointer": "bold",
        "highlighted": "bold",
        "text": "#888888",
        "answer": "bold",
    }
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _normalize_path(path: str) -> str:
    """Strip trailing slash and replace home directory with ~."""
    import os

    if not path:
        return ""
    path = path.rstrip("/")
    home = os.path.expanduser("~")
    if path.startswith(home):
        path = "~" + path[len(home) :]
    return path


def _common_prefix_depth(p1: str, p2: str) -> int:
    """Return the number of leading path components shared by *p1* and *p2*."""
    depth = 0
    for a, b in zip(p1.split("/"), p2.split("/"), strict=False):
        if a == b:
            depth += 1
        else:
            break
    return depth


def _is_run_path(rel: str) -> bool:
    """Return True if *rel* (relative to group ancestor) is a run-mode dir."""
    return "runs" in rel.split("/")


def _group_by_ancestor(norm_paths: list[str]) -> dict[str, list[str]]:
    """Group normalized paths by their deepest common ancestor.

    Two paths are placed in the same group when they share a common prefix
    of at least 2 components (e.g. ``~/Projects``).  Paths with no such
    shared prefix become standalone single-item groups keyed by their own
    full path.

    The returned dict is ordered by first appearance in *norm_paths*.

    Complexity: O(n log n) — paths are sorted lexicographically so the
    maximum common prefix depth for each path is found by comparing only
    its immediate neighbours in sorted order (not all pairs).
    """
    sorted_paths = sorted(norm_paths)
    n = len(sorted_paths)

    # Single pass over sorted list: max common prefix is always with a neighbour.
    sorted_best: dict[str, int] = {}
    for i, p in enumerate(sorted_paths):
        best = 1  # at minimum depth 1 (~)
        if i > 0:
            best = max(best, _common_prefix_depth(p, sorted_paths[i - 1]))
        if i < n - 1:
            best = max(best, _common_prefix_depth(p, sorted_paths[i + 1]))
        sorted_best[p] = best

    path_to_ancestor: dict[str, str] = {
        p: ("/".join(p.split("/")[:best]) if best >= 2 else p)
        for p, best in sorted_best.items()
    }

    groups: dict[str, list[str]] = {}
    for p in norm_paths:
        anc = path_to_ancestor[p]
        if anc not in groups:
            groups[anc] = []
        groups[anc].append(p)
    return groups


# ---------------------------------------------------------------------------
# Item builders
# ---------------------------------------------------------------------------


def _build_items(threads: list[dict]) -> list[dict]:
    """Build the flat item list rendered by ThreadPickerWidget.

    Returns a list whose elements are one of::

        {"type": "header",    "label": str}
        {"type": "subheader", "label": str}
        {"type": "thread",    "thread": dict, "indented": bool}

    *indented* is True for thread rows that sit under a L2 subheader.
    """
    if not threads:
        return []

    # Map normalized path -> list[thread dicts], preserving first-seen order
    raw_to_threads: dict[str, list[dict]] = {}
    seen_order: list[str] = []
    for t in threads:
        raw = t.get("workspace_dir", "") or ""
        norm = _normalize_path(raw) or raw
        if norm not in raw_to_threads:
            raw_to_threads[norm] = []
            seen_order.append(norm)
        raw_to_threads[norm].append(t)

    groups = _group_by_ancestor(seen_order)

    items: list[dict] = []
    for ancestor, norm_paths in groups.items():
        multi = len(norm_paths) > 1

        # L1 header — the common ancestor (or the sole workspace path)
        items.append({"type": "header", "label": ancestor or "(no workspace)"})

        for norm_path in norm_paths:
            if multi:
                # L2 subheader — relative path from ancestor
                rel = norm_path[len(ancestor) :].lstrip("/")
                if not rel:
                    # norm_path IS the ancestor (standalone group of 1 that shares
                    # an ancestor with others); show just the last path component
                    rel = norm_path.split("/")[-1] or norm_path
                icon = "🔁" if _is_run_path(rel) else "📁"
                items.append({"type": "subheader", "label": f"{icon} {rel}"})

            for t in raw_to_threads[norm_path]:
                items.append({"type": "thread", "thread": t, "indented": multi})

    return items


def build_header_text(label: str) -> Text:
    """L1 header: ``── 📂 <label>``."""
    t = Text()
    t.append("\u2500\u2500 \U0001f4c2 ", style="bold cyan")
    t.append(label, style="bold cyan")
    return t


def build_subheader_text(label: str) -> Text:
    """L2 subheader: ``  <icon> <rel-path>`` (indented, dim)."""
    t = Text()
    t.append("  ", style="")
    t.append(label, style="dim")
    return t


def build_row_text(
    thread: dict,
    *,
    selected: bool = False,
    current: bool = False,
    indented: bool = False,
) -> Text:
    """Thread row.  *indented* adds extra leading space for L2-grouped rows."""
    from ...sessions import _format_relative_time

    tid = thread["thread_id"]
    preview = thread.get("preview", "") or ""
    msgs = thread.get("message_count", 0)
    model = thread.get("model", "") or ""
    when = _format_relative_time(thread.get("updated_at"))

    line = Text()
    # Extra indent when nested under a subheader
    if indented:
        line.append("  ", style="")
    cursor = "\u25b8 " if selected else "  "
    line.append(cursor, style="bold cyan" if selected else "dim")
    line.append(tid, style="bold" if selected else "")
    if current:
        line.append(" *", style="bold green")
    line.append("  ")
    if preview:
        display_preview = preview[:40] + "\u2026" if len(preview) > 40 else preview
        line.append(display_preview)
        line.append("  ")
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
    Threads are displayed in a two-level workspace hierarchy.
    """

    can_focus = True
    can_focus_children = False

    DEFAULT_CSS = """
    ThreadPickerWidget {
        height: auto;
        max-height: 26;
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
        max-height: 20;
        overflow-y: auto;
    }
    ThreadPickerWidget .picker-header {
        height: 1;
        padding: 0 1;
        margin-top: 1;
    }
    ThreadPickerWidget .picker-subheader {
        height: 1;
        padding: 0 1;
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
        self._items = _build_items(threads)
        self._selected = self._first_thread_index()
        self._row_widgets: list[Static] = []

    def _first_thread_index(self) -> int:
        for i, item in enumerate(self._items):
            if item["type"] == "thread":
                return i
        return 0

    def _move(self, direction: int) -> None:
        if not self._items:
            return
        i = (self._selected + direction) % len(self._items)
        steps = 0
        while self._items[i]["type"] != "thread" and steps < len(self._items):
            i = (i + direction) % len(self._items)
            steps += 1
        if self._items[i]["type"] == "thread":
            self._selected = i
            self._update_rows()

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="picker-title")
        with Container(classes="picker-rows"):
            for item in self._items:
                css = {
                    "header": "picker-header",
                    "subheader": "picker-subheader",
                    "thread": "picker-row",
                }.get(item["type"], "picker-row")
                widget = Static("", classes=css)
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
        for i, (item, widget) in enumerate(
            zip(self._items, self._row_widgets, strict=False)
        ):
            widget.remove_class("picker-row-selected")
            if item["type"] == "header":
                widget.update(build_header_text(item["label"]))
            elif item["type"] == "subheader":
                widget.update(build_subheader_text(item["label"]))
            else:
                thread = item["thread"]
                is_selected = i == self._selected
                text = build_row_text(
                    thread,
                    selected=is_selected,
                    current=thread["thread_id"] == self._current_thread,
                    indented=item.get("indented", False),
                )
                widget.update(text)
                if is_selected:
                    widget.add_class("picker-row-selected")
                    widget.scroll_visible()

    def action_move_up(self) -> None:
        self._move(-1)

    def action_move_down(self) -> None:
        self._move(1)

    def action_select(self) -> None:
        if not self._items or self._selected >= len(self._items):
            self.post_message(self.Cancelled())
            return
        item = self._items[self._selected]
        if item["type"] == "thread":
            self.post_message(self.Picked(item["thread"]["thread_id"]))
        else:
            self.post_message(self.Cancelled())

    def action_cancel(self) -> None:
        self.post_message(self.Cancelled())

    def on_blur(self, event: events.Blur) -> None:
        self.call_after_refresh(self.focus)
