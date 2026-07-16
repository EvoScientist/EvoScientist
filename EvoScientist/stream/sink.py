"""Session-owned event sink.

A single sink instance, created and owned by a frontend or local stream run,
that both **records** tool-selection state for the stream suppressor and
optionally **renders** model-fallback notices. It is injected into the agent's
middleware (write side) and read by ``stream/tool_selection.py`` (read side).

The tool-selection state machine — active / total / pending / last-emitted with
consume-once + dedup semantics — lives here, in the session sink, replacing the
process-global module variables that used to live in ``middleware/tool_selector``.
The state is guarded by a lock because middleware hooks fire on worker threads
while the stream suppressor reads on the runtime thread (see the threading
contract in :mod:`EvoScientist.middleware.events`).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from .console import console

logger = logging.getLogger(__name__)


class SessionEventSink:
    """Tool-selection state holder + model-fallback renderer for a session.

    Args:
        fallback_display: Optional ``(text, style)`` callback the frontend
            supplies to render a model-fallback notice (Rich: ``console.print``;
            TUI: append a system message). ``None`` renders to the shared
            Rich console.
    """

    def __init__(
        self, fallback_display: Callable[[str, str], None] | None = None
    ) -> None:
        self._lock = threading.Lock()
        self._active = False
        self._total = 0
        self._pending: list[str] | None = None
        self._last_emitted: list[str] = []
        self._fallback_display = fallback_display

    def set_fallback_display(
        self, fallback_display: Callable[[str, str], None] | None
    ) -> None:
        """(Re)bind the fallback-notice display sink.

        Frontends whose display target only exists after construction (the TUI
        binds its ``_append_system`` on mount, clears it on exit) use this
        instead of the constructor argument.
        """
        with self._lock:
            self._fallback_display = fallback_display

    def _display_fallback(self, text: str, style: str) -> None:
        with self._lock:
            fallback_display = self._fallback_display
        if fallback_display is None:
            console.print(text, style=style)
            return
        try:
            fallback_display(text, style)
        except Exception:
            logger.warning("Fallback display callback failed", exc_info=True)
            console.print(text, style=style)

    # --- MiddlewareEventSink write side (any thread) ---------------------
    def on_tool_selection_started(self, total_tools: int) -> None:
        with self._lock:
            self._active = True
            self._total = total_tools

    def on_tool_selection(self, selected: list[str], total_tools: int) -> None:
        with self._lock:
            self._pending = list(selected)
            self._total = total_tools

    def on_tool_selection_ended(self) -> None:
        with self._lock:
            self._active = False

    def emit_fallback_notice(self, text: str, style: str = "yellow") -> None:
        """Render a fallback lifecycle line verbatim."""
        self._display_fallback(text, style)

    # --- ToolSelectionView read side (runtime thread) -------------------
    @property
    def tool_selection_active(self) -> bool:
        with self._lock:
            return self._active

    def tool_selection_pending(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def consume_tool_selection(self) -> tuple[bool, list[str] | None]:
        with self._lock:
            pending = self._pending
            if not pending:
                return (False, None)
            # Consume-once: clear before deciding whether to render.
            self._pending = None
            # Only surface when the selection actually filtered tools and it
            # differs from the last selection already shown to the user.
            if len(pending) < self._total and sorted(pending) != sorted(
                self._last_emitted
            ):
                self._last_emitted = list(pending)
                return (True, list(pending))
            return (True, None)
