"""Todo/Task list panel widget.

Renders a Rich Panel matching the Rich CLI's ``_render_todo_panel()`` style:
cyan border, centered "Task List" title, status icons per item.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.text import Text
from textual.widgets import Static


class TodoWidget(Static):
    """Displays a task list panel matching the Rich CLI style.

    Usage::

        w = TodoWidget(items)
        await container.mount(w)
        w.update_items(new_items)  # re-render in place
    """

    DEFAULT_CSS = """
    TodoWidget {
        height: auto;
        margin: 1 0;
    }
    """

    def __init__(self, items: list[dict] | None = None) -> None:
        super().__init__("")
        self._items = items or []

    def on_mount(self) -> None:
        self._refresh_display()

    def update_items(self, items: list[dict]) -> None:
        """Replace the task list and re-render."""
        self._items = items
        self._refresh_display()

    def _refresh_display(self) -> None:
        if not self._items:
            self.update(Text("No tasks", style="dim"))
            return

        lines = Text()
        for i, item in enumerate(self._items):
            if i > 0:
                lines.append("\n")
            status = str(item.get("status", "todo")).lower()
            content = str(item.get("content", item.get("task", item.get("title", ""))))

            if status in ("done", "completed", "complete"):
                symbol = "\u2713"
                style = "green dim"
            elif status in ("active", "in_progress", "in-progress", "working"):
                symbol = "\u23f3"
                style = "yellow"
            else:
                symbol = "\u25a1"
                style = "dim"

            lines.append(f"{symbol} ", style=style)
            lines.append(content, style=style)

        self.update(
            Panel(
                lines,
                title="Task List",
                title_align="center",
                border_style="cyan",
                padding=(0, 1),
            )
        )
