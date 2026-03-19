"""
ToolCallTracker - manages incremental JSON parsing for tool parameters.

Handles tool_use blocks where arguments arrive in fragments via input_json_delta.
"""

import json
from dataclasses import dataclass, field


@dataclass
class ToolCallInfo:
    """Tool call information."""

    id: str
    name: str
    args: dict = field(default_factory=dict)
    emitted: bool = False
    args_complete: bool = False
    _json_buffer: str = ""


class ToolCallTracker:
    """Tool call tracker for incremental argument parsing.

    Usage:
        tracker = ToolCallTracker()
        tracker.update(tool_id, name="execute")
        tracker.append_json_delta('{"command')
        tracker.append_json_delta('": "ls"}')
        tracker.finalize_all()
        info = tracker.get(tool_id)
        yield emitter.tool_call(info.name, info.args)
    """

    def __init__(self):
        self._calls: dict[str, ToolCallInfo] = {}
        self._last_tool_id: str | None = None

    def update(
        self,
        tool_id: str,
        name: str | None = None,
        args: dict | None = None,
        args_complete: bool = False,
    ) -> None:
        """Update tool call info (accumulative)."""
        if tool_id not in self._calls:
            self._calls[tool_id] = ToolCallInfo(
                id=tool_id,
                name=name or "",
                args=args or {},
                args_complete=args_complete,
            )
            self._last_tool_id = tool_id
        else:
            info = self._calls[tool_id]
            if name:
                info.name = name
            if args:
                info.args = args
            if args_complete:
                info.args_complete = True

    def append_json_delta(self, partial_json: str, index: int = 0) -> None:
        """Accumulate input_json_delta fragment."""
        tool_id = self._last_tool_id
        if tool_id and tool_id in self._calls:
            self._calls[tool_id]._json_buffer += partial_json

    def finalize_all(self) -> None:
        """Finalize all tool calls: parse accumulated JSON and mark complete."""
        for info in self._calls.values():
            if info._json_buffer:
                try:
                    info.args = json.loads(info._json_buffer)
                except json.JSONDecodeError:
                    pass
                info._json_buffer = ""
            info.args_complete = True

    def is_ready(self, tool_id: str) -> bool:
        """Check if a tool call is ready to emit (has name and not yet emitted)."""
        if tool_id not in self._calls:
            return False
        info = self._calls[tool_id]
        return bool(info.name) and not info.emitted

    def get_all(self) -> list[ToolCallInfo]:
        """Get all tool calls."""
        return list(self._calls.values())

    def mark_emitted(self, tool_id: str) -> None:
        """Mark a tool call as emitted."""
        if tool_id in self._calls:
            self._calls[tool_id].emitted = True

    def get(self, tool_id: str) -> ToolCallInfo | None:
        """Get tool call info by ID."""
        return self._calls.get(tool_id)

    def get_pending(self) -> list[ToolCallInfo]:
        """Get all unemitted tool calls."""
        return [info for info in self._calls.values() if not info.emitted]

    def emit_all_pending(self) -> list[ToolCallInfo]:
        """Emit all pending tool calls and mark them."""
        pending = self.get_pending()
        for info in pending:
            info.emitted = True
        return pending

    def clear(self) -> None:
        """Clear all tracked tool calls."""
        self._calls.clear()
