"""Transient widget shown while /new or /resume moves the langgraph dev subprocess.

Mirrors ``CompactingWidget`` — a timer-backed status line that ticks elapsed
seconds so the user has live feedback during the up-to-60s langgraph dev
workspace sync (subprocess stop + restart so deployed sub-agents see the
session's new folders).
"""

from __future__ import annotations

from .timed_status_widget import TimedStatusWidget


class WorkspaceSyncWidget(TimedStatusWidget):
    """Timer-backed status line for an in-progress workspace sync."""

    DEFAULT_CSS = """
    WorkspaceSyncWidget {
        height: auto;
        color: #94a3b8;
        padding: 0 0;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def _refresh_display(self) -> None:
        self.update(f"{self._message}... ({self.elapsed_seconds}s)")

    async def cleanup(self) -> None:
        """Stop timer and remove from DOM."""
        self._stop_timer()
        if self.is_mounted:
            await self.remove()
