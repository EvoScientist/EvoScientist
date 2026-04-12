"""Shared timer-backed base class for transient TUI status widgets."""

from __future__ import annotations

from textual.widgets import Static


class TimedStatusWidget(Static):
    """Static widget with a simple elapsed-time timer.

    Subclasses implement ``_refresh_display()`` and can override
    ``_should_tick()`` when the timer should pause after a state transition.
    """

    TICK_SECONDS = 0.1

    def __init__(self) -> None:
        super().__init__("")
        self._elapsed = 0.0
        self._timer_handle = None

    def on_mount(self) -> None:
        self._timer_handle = self.set_interval(self.TICK_SECONDS, self._tick)
        self._refresh_display()

    def on_unmount(self) -> None:
        self._stop_timer()

    def _tick(self) -> None:
        if self._should_tick():
            self._elapsed += self.TICK_SECONDS
            self._refresh_display()

    def _should_tick(self) -> bool:
        """Return whether the timer should continue advancing."""
        return True

    def _stop_timer(self) -> None:
        if self._timer_handle is not None:
            self._timer_handle.stop()
            self._timer_handle = None

    @property
    def elapsed_seconds(self) -> int:
        return int(self._elapsed)

    def _refresh_display(self) -> None:
        """Update the widget's rendered content."""
        raise NotImplementedError
