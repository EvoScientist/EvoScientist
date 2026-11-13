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

    def snapshot(self) -> tuple[bool, str]:
        return (self._buffering, self._buffer)

    def restore(self, snapshot: tuple[bool, str]) -> None:
        self._buffering, self._buffer = snapshot

    def observe_tool_block(self, name: str) -> bool:
        return name == "ToolSelectionResponse"

    def process_text(self, text: str) -> tuple[bool, list[dict[str, Any]], str]:
        events = self.flush_selection()
        if not text:
            return False, events, ""

        if self._buffering:
            self._buffer += text
            json_kind = self._json_buffer_kind(self._buffer)
            if json_kind == "selector" and self._selector_call_active():
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
            self._selector_call_active()
            and stripped.startswith("{")
            and ('"tools"' in stripped or len(stripped) <= 10)
        ):
            json_kind = self._json_buffer_kind(stripped)
            if json_kind == "selector":
                return True, events, ""
            if json_kind == "complete":
                return False, events, text
            self._buffering = True
            self._buffer = text
            return True, events, ""

        return False, events, text

    def process_thinking(self, text: str) -> tuple[bool, list[dict[str, Any]]]:
        """Suppress selector-model reasoning while preserving pending UI events."""
        events = self.flush_selection()
        if not text:
            return False, events
        return self._selector_call_active(), events

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

    def _selector_call_active(self) -> bool:
        return bool(self._sink.tool_selection_active)

    def flush_selection(self) -> list[dict[str, Any]]:
        # The sink alone says when a selection is ready: the selector's own
        # model output is kept out of the stream, so it cannot be waited for.
        _had_pending, render = self._sink.consume_tool_selection()
        if render is None:
            return []
        return [self._emitter.tool_selection(render).data]

    def flush_pending_text(self) -> str:
        if not self._buffering:
            return ""
        replay = self._buffer
        self._buffering = False
        self._buffer = ""
        return replay
