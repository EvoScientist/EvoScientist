"""Loading spinner widget shown while waiting for the first token."""

from __future__ import annotations

from .timed_status_widget import TimedStatusWidget

_SPINNER_FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"


class LoadingWidget(TimedStatusWidget):
    """Spinner + 'Thinking...' with elapsed time counter.

    Mount when a turn starts; call ``remove()`` when the first
    thinking/text/tool_call event arrives.
    """

    DEFAULT_CSS = """
    LoadingWidget {
        height: auto;
        color: #22d3ee;
        padding: 0 0;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._frame = 0

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_SPINNER_FRAMES)
        super()._tick()

    def _refresh_display(self) -> None:
        char = _SPINNER_FRAMES[self._frame]
        self.update(f"{char} Thinking... ({self.elapsed_seconds}s)")

    async def cleanup(self) -> None:
        """Stop timer and remove from DOM."""
        self._stop_timer()
        await self.remove()
