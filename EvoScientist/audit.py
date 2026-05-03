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
    {"tool_call", "tool_result", "subagent_start", "subagent_end"}
)
_CONTENT_TRUNCATE = 2000


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

    @property
    def log_path(self) -> Path:
        return self._log_path

    def log_session_start(self, *, model: str) -> None:
        self._append(
            {
                "kind": "session_start",
                "ts": time.time(),
                "thread_id": self._thread_id,
                "model": model,
            }
        )

    def log_session_end(self) -> None:
        self._append({"kind": "session_end", "ts": time.time()})

    def log_event(self, event_type: str, event: dict[str, Any]) -> None:
        if event_type not in _LOGGABLE_EVENTS:
            return
        entry: dict[str, Any] = {"kind": event_type, "ts": time.time()}
        if event_type == "tool_call":
            entry["tool"] = event.get("name", "unknown")
            entry["args"] = event.get("args", {})
            entry["tool_id"] = event.get("id", "")
        elif event_type == "tool_result":
            entry["tool"] = event.get("name", "unknown")
            entry["success"] = event.get("success", True)
            content = str(event.get("content", ""))
            entry["content"] = content[:_CONTENT_TRUNCATE]
        elif event_type in ("subagent_start", "subagent_end"):
            entry["name"] = event.get("name", "unknown")
            entry["agent_id"] = event.get("agent_id", "")
        self._append(entry)

    def _append(self, entry: dict[str, Any]) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("audit log write failed", exc_info=True)
