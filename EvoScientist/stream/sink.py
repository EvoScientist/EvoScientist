"""Frontend-owned event sink.

A single sink instance, created and owned by an interactive frontend (Rich CLI
or TUI), that both **records** tool-selection state for the stream suppressor
and **renders** model-fallback notices. It is injected into the agent's
middleware (write side) and read by ``stream/tool_selection.py`` (read side).

The tool-selection state machine — active / total / pending / last-emitted with
consume-once + dedup semantics — lives here, in the frontend, replacing the
process-global module variables that used to live in ``middleware/tool_selector``.
The state is guarded by a lock because middleware hooks fire on worker threads
while the stream suppressor reads on the runtime thread (see the threading
contract in :mod:`EvoScientist.middleware.events`).
"""

from __future__ import annotations

import threading
from collections.abc import Callable


class FrontendEventSink:
    """Tool-selection state holder + model-fallback renderer for a frontend.

    Args:
        fallback_display: Optional ``(text, style)`` callback the frontend
            supplies to render a model-fallback notice (Rich: ``console.print``;
            TUI: append a system message). ``None`` drops fallback notices.
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

    def on_model_fallback(self, from_model: str, to_model: str, reason: str) -> None:
        # Formatting lives in the frontend: reproduce the transition line the
        # fallback middleware used to emit itself. ``to_model`` already carries
        # the provider suffix and ``reason`` the exception text.
        if self._fallback_display is None:
            return
        self._fallback_display(
            f"  -> Falling back to {to_model} due to: {reason}", "yellow"
        )

    def emit_fallback_notice(self, text: str, style: str = "yellow") -> None:
        """Render a fallback lifecycle line (header / outcome) verbatim.

        The fallback middleware still narrates the non-transition lines
        (primary-model failure, per-attempt outcome, exhaustion,
        non-fallbackable rejection) as pre-formatted text; the frontend owns
        where they land. This preserves the exact user-facing narration while
        the structured :meth:`on_model_fallback` covers the transition itself.
        """
        if self._fallback_display is None:
            return
        self._fallback_display(text, style)

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
