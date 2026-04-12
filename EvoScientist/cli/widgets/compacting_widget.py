"""Transient widget shown while manual /compact is running in the TUI."""

from __future__ import annotations

from .timed_status_widget import TimedStatusWidget


class CompactingWidget(TimedStatusWidget):
    """Timer-backed status line for an in-progress manual compact."""

    DEFAULT_CSS = """
    CompactingWidget {
        height: auto;
        color: #f59e0b;
        padding: 0 0;
        margin: 0 0 1 0;
    }
    """

    def __init__(self) -> None:
        super().__init__()

    def _refresh_display(self) -> None:
        self.update(f"Compacting conversation... ({self.elapsed_seconds}s)")

    async def cleanup(self) -> None:
        """Stop timer and remove from DOM."""
        self._stop_timer()
        if self.is_mounted:
            await self.remove()
