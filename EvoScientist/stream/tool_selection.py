"""Tool-selection stream suppression helpers."""

from typing import TYPE_CHECKING, Any

from .emitter import StreamEventEmitter

if TYPE_CHECKING:
    from ..middleware.events import ToolSelectionView


class _ToolSelectionSuppressor:
    """Suppress selector model JSON while preserving the UI selection event.

    Reads its selection state from the injected frontend sink (a
    :class:`~EvoScientist.middleware.events.ToolSelectionView`) rather than
    process globals: ``tool_selection_active`` gates chatter suppression and
    ``consume_tool_selection`` yields the deduped list to surface. The sink
    owns the state; this class only observes the stream and asks the sink.
    """

    def __init__(self, emitter: StreamEventEmitter, sink: "ToolSelectionView") -> None:
        self._emitter = emitter
        self._sink = sink
        self._buffering = False
        self._buffer = ""
        self._was_active = False

    def observe_tool_block(self, name: str) -> bool:
        if name == "ToolSelectionResponse":
            self._was_active = True
            return True
        return False

    def process_text(self, text: str) -> tuple[bool, list[dict[str, Any]], str]:
        events = self._emit_selection_if_ready(text)
        if not text:
            return False, events, ""

        if self._buffering:
            self._buffer += text
            json_kind = self._json_buffer_kind(self._buffer)
            if json_kind == "selector" and self._selector_context_active():
                self._was_active = True
                self._buffering = False
                self._buffer = ""
                return True, events, ""
            if json_kind == "complete":
                replay = self._buffer
                self._buffering = False
                self._buffer = ""
                return False, events, replay
            if len(self._buffer) <= 10000:
                return True, events, ""
            replay = self._buffer
            self._buffering = False
            self._buffer = ""
            return False, events, replay

        stripped = text.strip()
        if (
            self._selector_context_active()
            and stripped.startswith("{")
            and ('"tools"' in stripped or len(stripped) <= 10)
        ):
            json_kind = self._json_buffer_kind(stripped)
            if json_kind == "selector":
                self._was_active = True
                return True, events, ""
            if json_kind == "complete":
                return False, events, text
            self._buffering = True
            self._buffer = text
            return True, events, ""

        return False, events, text

    def process_thinking(self, text: str) -> tuple[bool, list[dict[str, Any]]]:
        """Suppress selector-model reasoning while preserving pending UI events."""
        events = self._emit_selection_if_ready(text)
        if not text:
            return False, events
        if self._selector_call_active():
            self._was_active = True
            return True, events
        return False, events

    @staticmethod
    def _json_buffer_kind(text: str) -> str:
        try:
            import json

            parsed = json.loads(text.strip())
        except (TypeError, ValueError):
            stripped = text.strip()
            if '"tools"' in stripped and stripped.endswith("}"):
                return "selector"
            return "incomplete"
        if isinstance(parsed, dict) and "tools" in parsed:
            return "selector"
        return "complete"

    def _selector_context_active(self) -> bool:
        if self._was_active:
            return True
        return self._selector_call_active() or self._selection_pending()

    def _selector_call_active(self) -> bool:
        return bool(self._sink.tool_selection_active)

    def _selection_pending(self) -> bool:
        return bool(self._sink.tool_selection_pending())

    def flush_selection(self) -> list[dict[str, Any]]:
        return self._emit_selection_if_ready("")

    def flush_pending_text(self) -> str:
        if not self._buffering:
            return ""
        replay = self._buffer
        self._buffering = False
        self._buffer = ""
        return replay

    def _emit_selection_if_ready(self, text: str) -> list[dict[str, Any]]:
        if not self._was_active:
            return []
        had_pending, render = self._sink.consume_tool_selection()
        if had_pending:
            # Pending consumed (once) — clear our observation flag regardless of
            # whether the sink chose to render it (dedup / kept-all cases).
            self._was_active = False
            if render is not None:
                return [self._emitter.tool_selection(render).data]
        elif text:
            self._was_active = False
        return []
