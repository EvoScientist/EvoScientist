"""Session audit log — append-only JSONL record of agent operations.

Each session writes to ~/.evoscientist/sessions/<thread_id>/audit.jsonl.
All I/O errors are silently swallowed so audit failures never crash the TUI.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
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
        self._pending_tool_calls: dict[str, dict[str, Any]] = {}
        self._logged_tool_call_ids: set[str] = set()

    @property
    def log_path(self) -> Path:
        return self._log_path

    def log_session_start(self, *, model: str, provider: str) -> None:
        if self._session_started:
            return
        self._session_started = True
        self._append(
            {
                "kind": "session_start",
                "ts": _now_iso(),
                "thread_id": self._thread_id,
                "model": model,
                "provider": provider,
            }
        )

    def log_session_end(self) -> None:
        if not self._session_started or self._session_ended:
            return
        self._flush_pending_tool_calls()
        self._session_ended = True
        self._append({"kind": "session_end", "ts": _now_iso()})

    def log_event(self, event_type: str, event: dict[str, Any]) -> None:
        if event_type not in _LOGGABLE_EVENTS:
            return
        if self._session_ended:
            return
        if event_type == "subagent_start" and _is_placeholder_subagent_start(event):
            return
        if event_type in ("tool_call", "subagent_tool_call"):
            self._log_tool_call(event_type, event)
            return
        if event_type in ("tool_result", "subagent_tool_result"):
            self._flush_pending_tool_calls_for_result(event_type, event)
            self._prune_logged_tool_call(event_type, event)

        entry = self._build_entry(event_type, event)
        self._append(entry)

    def _build_entry(self, event_type: str, event: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {"kind": event_type, "ts": _now_iso()}
        if event_type in ("tool_call", "subagent_tool_call"):
            entry["tool"] = event.get("name", "unknown")
            entry["args"] = _cap_json_payload(event.get("args", {}), _ARGS_TRUNCATE)
            entry["tool_id"] = _tool_id(event)
            if event_type == "subagent_tool_call":
                entry["subagent"] = event.get("subagent", "sub-agent")
        elif event_type in ("tool_result", "subagent_tool_result"):
            entry["tool"] = event.get("name", "unknown")
            entry["success"] = event.get("success", True)
            content = str(event.get("content", ""))
            entry["content"] = content[:_CONTENT_TRUNCATE]
            tool_id = _tool_id(event)
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
        if tool_id in self._logged_tool_call_ids:
            return

        args = event.get("args")
        if tool_id and not _has_richer_args(args):
            if tool_id in self._pending_tool_calls:
                self._pending_tool_calls[tool_id] = _merge_pending_tool_call(
                    self._pending_tool_calls[tool_id],
                    entry,
                )
            else:
                self._pending_tool_calls[tool_id] = entry
            return

        self._pending_tool_calls.pop(tool_id, None)
        if tool_id:
            self._track_logged_tool_call(tool_id)
        self._append(entry)

    def _flush_pending_tool_calls(self) -> None:
        pending = list(self._pending_tool_calls.items())
        self._pending_tool_calls.clear()
        for tool_id, entry in pending:
            self._append_pending_tool_call(tool_id, entry)

    def _flush_pending_tool_calls_for_result(
        self,
        event_type: str,
        event: dict[str, Any],
    ) -> None:
        for tool_id in self._pending_tool_ids_for_result(event_type, event):
            entry = self._pending_tool_calls.pop(tool_id)
            self._append_pending_tool_call(tool_id, entry)

    def _append_pending_tool_call(
        self,
        tool_id: str,
        entry: dict[str, Any],
    ) -> None:
        if tool_id in self._logged_tool_call_ids:
            return
        self._track_logged_tool_call(tool_id)
        self._append({**entry, "ts": _now_iso()})

    def _track_logged_tool_call(self, tool_id: str) -> None:
        self._logged_tool_call_ids.add(tool_id)

    def _pending_tool_ids_for_result(
        self,
        event_type: str,
        event: dict[str, Any],
    ) -> list[str]:
        call_type = (
            "subagent_tool_call"
            if event_type == "subagent_tool_result"
            else "tool_call"
        )
        tool_id = _tool_id(event)
        if tool_id:
            return [tool_id] if tool_id in self._pending_tool_calls else []

        tool_name = event.get("name", "unknown")
        matching_ids = [
            pending_id
            for pending_id, entry in self._pending_tool_calls.items()
            if entry.get("kind") == call_type
            and entry.get("tool") == tool_name
            and (
                event_type != "subagent_tool_result"
                or entry.get("subagent")
                == str(event.get("subagent", "sub-agent") or "sub-agent")
            )
        ]
        if len(matching_ids) == 1:
            return matching_ids
        return []

    def _prune_logged_tool_call(
        self,
        event_type: str,
        event: dict[str, Any],
    ) -> None:
        tool_id = _tool_id(event)
        if not tool_id:
            return
        self._logged_tool_call_ids.discard(tool_id)

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
        return {
            "truncated": True,
            "original_type": type(value).__name__,
            "serialized": serialized[:limit],
        }

    try:
        return json.loads(serialized)
    except Exception:
        return serialized


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _tool_id(event: dict[str, Any]) -> str:
    return str(event.get("id") or event.get("tool_call_id") or "")


def _has_richer_args(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return value != {}
    return True


def _merge_pending_tool_call(
    current: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(current)
    if incoming.get("tool") and incoming.get("tool") != "unknown":
        merged["tool"] = incoming["tool"]
    if incoming.get("tool_id"):
        merged["tool_id"] = incoming["tool_id"]
    if incoming.get("subagent") and not merged.get("subagent"):
        merged["subagent"] = incoming["subagent"]
    return merged


def _is_placeholder_subagent_start(event: dict[str, Any]) -> bool:
    return event.get("name", "sub-agent") == "sub-agent" and not event.get(
        "description", ""
    )
