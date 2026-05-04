"""Session audit log — append-only JSONL record of agent operations.

Each session writes to ~/.evoscientist/sessions/<thread_id>/audit.jsonl.
All I/O errors are silently swallowed so audit failures never crash the TUI.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from . import paths as _paths

logger = logging.getLogger(__name__)

_LOGGABLE_EVENTS = frozenset(
    {
        "tool_call",
        "tool_result",
        "subagent_start",
        "subagent_tool_call",
        "subagent_tool_result",
        "subagent_end",
    }
)
_CONTENT_TRUNCATE = 2000
_ARGS_TRUNCATE = 2000


class AuditLogger:
    """Append-only JSONL audit logger for a single session.

    Args:
        thread_id: The LangGraph thread ID for this session.
        sessions_dir: Override the base directory (default: ``paths.SESSIONS_DIR``).
            Primarily used in tests.
    """

    def __init__(
        self,
        thread_id: str,
        sessions_dir: Path | None = None,
    ) -> None:
        self._thread_id = thread_id
        base = sessions_dir if sessions_dir is not None else _paths.SESSIONS_DIR
        self._log_path = base / thread_id / "audit.jsonl"
        self._dir_ready = False
        self._session_started = False
        self._session_ended = False
        self._pending_tool_calls: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._logged_tool_calls: set[tuple[str, str, str]] = set()

    @property
    def log_path(self) -> Path:
        return self._log_path

    def log_session_start(self, *, model: str) -> None:
        if self._session_started:
            return
        self._session_started = True
        self._append(
            {
                "kind": "session_start",
                "ts": time.time(),
                "thread_id": self._thread_id,
                "model": model,
            }
        )

    def log_session_end(self) -> None:
        if not self._session_started or self._session_ended:
            return
        self._flush_pending_tool_calls()
        self._session_ended = True
        self._append({"kind": "session_end", "ts": time.time()})

    def log_event(self, event_type: str, event: dict[str, Any]) -> None:
        if event_type not in _LOGGABLE_EVENTS:
            return
        if self._session_ended:
            return
        if event_type in ("tool_call", "subagent_tool_call"):
            self._log_tool_call(event_type, event)
            return
        if event_type in ("tool_result", "subagent_tool_result"):
            self._flush_pending_tool_calls()

        entry = self._build_entry(event_type, event)
        self._append(entry)

    def _build_entry(self, event_type: str, event: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {"kind": event_type, "ts": time.time()}
        if event_type in ("tool_call", "subagent_tool_call"):
            entry["tool"] = event.get("name", "unknown")
            entry["args"] = _cap_json_payload(event.get("args", {}), _ARGS_TRUNCATE)
            entry["tool_id"] = event.get("id", "")
            if event_type == "subagent_tool_call":
                entry["subagent"] = event.get("subagent", "sub-agent")
        elif event_type in ("tool_result", "subagent_tool_result"):
            entry["tool"] = event.get("name", "unknown")
            entry["success"] = event.get("success", True)
            content = str(event.get("content", ""))
            entry["content"] = content[:_CONTENT_TRUNCATE]
            tool_id = event.get("id", "")
            if tool_id:
                entry["tool_id"] = tool_id
            if event_type == "subagent_tool_result":
                entry["subagent"] = event.get("subagent", "sub-agent")
        elif event_type == "subagent_start":
            entry["name"] = event.get("name", "unknown")
            entry["description"] = event.get("description", "")
        elif event_type == "subagent_end":
            entry["name"] = event.get("name", "unknown")
        return entry

    def _log_tool_call(self, event_type: str, event: dict[str, Any]) -> None:
        entry = self._build_entry(event_type, event)
        tool_id = str(entry.get("tool_id", "") or "")
        subagent = str(entry.get("subagent", "") or "")
        key = (event_type, subagent, tool_id)
        if tool_id and key in self._logged_tool_calls:
            return

        args = event.get("args", {})
        if tool_id and not args:
            self._pending_tool_calls.setdefault(key, entry)
            return

        self._pending_tool_calls.pop(key, None)
        if tool_id:
            self._logged_tool_calls.add(key)
        self._append(entry)

    def _flush_pending_tool_calls(self) -> None:
        pending = list(self._pending_tool_calls.items())
        self._pending_tool_calls.clear()
        for key, entry in pending:
            if key in self._logged_tool_calls:
                continue
            self._logged_tool_calls.add(key)
            self._append(entry)

    def _append(self, entry: dict[str, Any]) -> None:
        try:
            if not self._dir_ready:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                self._dir_ready = True
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.debug("audit log write failed", exc_info=True)


def _cap_json_payload(value: Any, limit: int) -> Any:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        serialized = str(value)

    if len(serialized) > limit:
        return serialized[:limit]

    try:
        return json.loads(serialized)
    except Exception:
        return serialized
