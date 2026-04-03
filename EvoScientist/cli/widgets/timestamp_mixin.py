"""Mixin that shows a timestamp toast when a message widget is clicked."""

from __future__ import annotations

import time
from datetime import UTC, datetime


def show_timestamp_toast(widget) -> None:
    """Show a toast with the widget's creation timestamp.

    Silently no-ops if the widget is not mounted or has no ``_created_at``.
    """
    try:
        app = widget.app
    except Exception:
        return
    created_at = getattr(widget, "_created_at", None)
    if created_at is None:
        return
    dt = datetime.fromtimestamp(created_at, tz=UTC).astimezone()
    label = f"{dt:%b} {dt.day}, {dt.hour % 12 or 12}:{dt:%M:%S} {dt:%p}"
    app.notify(label, timeout=3)


class TimestampClickMixin:
    """Mixin that shows a timestamp toast on click.

    Add to any message widget that should display its creation timestamp
    when clicked.  Widgets with custom ``on_click`` (e.g. ToolCallWidget)
    should call :func:`show_timestamp_toast` directly instead.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._created_at: float = time.time()

    def on_click(self, event) -> None:
        """Show timestamp toast on click."""
        # TODO: re-enable after UX review
        # show_timestamp_toast(self)
