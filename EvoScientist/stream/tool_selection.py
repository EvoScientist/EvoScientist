"""Tool-selection stream suppression helpers."""

from typing import Any

from .emitter import StreamEventEmitter


class _ToolSelectionSuppressor:
    """Suppress selector model JSON while preserving the UI selection event."""

    def __init__(self, emitter: StreamEventEmitter) -> None:
        self._emitter = emitter
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
        import EvoScientist.middleware.tool_selector as selector_mod

        return bool(
            selector_mod._selector_active or selector_mod._current_selected_tools
        )

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
        import EvoScientist.middleware.tool_selector as selector_mod

        if selector_mod._current_selected_tools:
            self._was_active = False
            selected = selector_mod._current_selected_tools
            selector_mod._current_selected_tools = []
            if len(selected) < selector_mod._total_tools_count and sorted(
                selected
            ) != sorted(selector_mod._last_emitted_tools):
                selector_mod._last_emitted_tools = list(selected)
                return [self._emitter.tool_selection(list(selected)).data]
        elif text:
            self._was_active = False
        return []
