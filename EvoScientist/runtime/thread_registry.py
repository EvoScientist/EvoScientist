"""In-process runtime registry shared across interactive surfaces.

This stores live, thread-scoped state (thinking/subagents/activity/HITL prompts)
so TUI and WebUI can observe the same run state in real time.
"""

from __future__ import annotations

import copy
import json
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

_MAX_ACTIVITY_ITEMS = 200
_MAX_EVENT_BUFFER = 2000
_MAX_TOOL_RESULT_CHARS = 8000
_UNSET = object()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_tool_result(value: Any) -> str:
    """Serialize tool result payloads for UI consumption."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:_MAX_TOOL_RESULT_CHARS]
    try:
        rendered = json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        rendered = str(value)
    return rendered[:_MAX_TOOL_RESULT_CHARS]


@dataclass
class _ThreadRuntimeState:
    state: dict[str, Any]
    seq: int
    events: deque[tuple[int, dict[str, Any]]]


class ThreadRuntimeRegistry:
    """Thread-safe, in-memory registry for live runtime state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: dict[str, _ThreadRuntimeState] = {}
        self._pending_approval_decisions: dict[str, deque[str]] = {}
        self._pending_ask_user_replies: dict[str, deque[str]] = {}
        self._pending_cancel_requests: set[str] = set()

    def _ensure(self, thread_id: str) -> _ThreadRuntimeState:
        runtime = self._threads.get(thread_id)
        if runtime is None:
            runtime = _ThreadRuntimeState(
                state={
                    "status": None,
                    "isRunning": False,
                    "subagents": [],
                    "activity": [],
                    "approval": None,
                    "askUser": None,
                    "updatedAt": _utc_now_iso(),
                },
                seq=0,
                events=deque(maxlen=_MAX_EVENT_BUFFER),
            )
            self._threads[thread_id] = runtime
        return runtime

    def _append_event_locked(
        self,
        runtime: _ThreadRuntimeState,
        event_type: str,
        payload: dict[str, Any],
    ) -> int:
        runtime.seq += 1
        event = {
            "type": event_type,
            "threadId": payload.get("threadId"),
            "payload": copy.deepcopy(payload),
            "createdAt": _utc_now_iso(),
        }
        runtime.events.append((runtime.seq, event))
        runtime.state["updatedAt"] = event["createdAt"]
        return runtime.seq

    def snapshot(self, thread_id: str) -> dict[str, Any]:
        with self._lock:
            runtime = self._threads.get(thread_id)
            if runtime is None:
                return {}
            return copy.deepcopy(runtime.state)

    def thread_ids(self) -> list[str]:
        with self._lock:
            return list(self._threads.keys())

    def events_since(
        self,
        thread_id: str,
        cursor: int = 0,
        *,
        limit: int = 200,
    ) -> tuple[int, list[dict[str, Any]]]:
        with self._lock:
            runtime = self._threads.get(thread_id)
            if runtime is None:
                return cursor, []

            output: list[dict[str, Any]] = []
            latest = cursor
            for seq, event in runtime.events:
                if seq <= cursor:
                    continue
                output.append({"seq": seq, **copy.deepcopy(event)})
                latest = seq
                if len(output) >= limit:
                    break
            return latest, output

    def set_prompt(
        self,
        thread_id: str,
        *,
        approval: dict[str, Any] | None | object = _UNSET,
        ask_user: dict[str, Any] | None | object = _UNSET,
    ) -> None:
        with self._lock:
            runtime = self._ensure(thread_id)
            if approval is not _UNSET:
                runtime.state["approval"] = copy.deepcopy(approval)
                self._append_event_locked(
                    runtime,
                    "approval",
                    {"threadId": thread_id, "approval": approval},
                )
            if ask_user is not _UNSET:
                runtime.state["askUser"] = copy.deepcopy(ask_user)
                self._append_event_locked(
                    runtime,
                    "ask_user",
                    {"threadId": thread_id, "askUser": ask_user},
                )

    def submit_approval_decision(self, thread_id: str, decision: str) -> None:
        with self._lock:
            queue = self._pending_approval_decisions.setdefault(thread_id, deque())
            queue.append(str(decision))
            runtime = self._ensure(thread_id)
            self._append_event_locked(
                runtime,
                "approval_reply",
                {"threadId": thread_id, "decision": str(decision)},
            )

    def pop_approval_decision(self, thread_id: str) -> str | None:
        with self._lock:
            queue = self._pending_approval_decisions.get(thread_id)
            if not queue:
                return None
            decision = queue.popleft()
            if not queue:
                self._pending_approval_decisions.pop(thread_id, None)
            return decision

    def submit_ask_user_reply(self, thread_id: str, reply: str) -> None:
        with self._lock:
            queue = self._pending_ask_user_replies.setdefault(thread_id, deque())
            queue.append(str(reply))
            runtime = self._ensure(thread_id)
            self._append_event_locked(
                runtime,
                "ask_user_reply",
                {"threadId": thread_id, "reply": str(reply)},
            )

    def pop_ask_user_reply(self, thread_id: str) -> str | None:
        with self._lock:
            queue = self._pending_ask_user_replies.get(thread_id)
            if not queue:
                return None
            reply = queue.popleft()
            if not queue:
                self._pending_ask_user_replies.pop(thread_id, None)
            return reply

    def request_cancel(self, thread_id: str) -> None:
        with self._lock:
            runtime = self._ensure(thread_id)
            self._pending_cancel_requests.add(thread_id)
            runtime.state["status"] = "Stopping..."
            runtime.state["isRunning"] = True
            self._append_event_locked(
                runtime,
                "cancel_requested",
                {
                    "threadId": thread_id,
                    "status": runtime.state["status"],
                    "isRunning": runtime.state["isRunning"],
                },
            )

    def is_cancel_requested(self, thread_id: str) -> bool:
        with self._lock:
            return thread_id in self._pending_cancel_requests

    def clear_cancel_request(self, thread_id: str) -> None:
        with self._lock:
            self._pending_cancel_requests.discard(thread_id)

    def set_status(
        self,
        thread_id: str,
        *,
        status: str | None,
        is_running: bool | None = None,
    ) -> None:
        with self._lock:
            runtime = self._ensure(thread_id)
            runtime.state["status"] = status
            if is_running is not None:
                runtime.state["isRunning"] = bool(is_running)
            self._append_event_locked(
                runtime,
                "status",
                {
                    "threadId": thread_id,
                    "status": status,
                    "isRunning": runtime.state.get("isRunning", False),
                },
            )

    def set_running(self, thread_id: str, is_running: bool) -> None:
        with self._lock:
            runtime = self._ensure(thread_id)
            runtime.state["isRunning"] = bool(is_running)
            self._append_event_locked(
                runtime,
                "running",
                {"threadId": thread_id, "isRunning": bool(is_running)},
            )

    def begin_run(self, thread_id: str) -> None:
        """Reset live run-scoped fields for a new generation cycle."""
        with self._lock:
            runtime = self._ensure(thread_id)
            self._pending_cancel_requests.discard(thread_id)
            runtime.state["status"] = None
            runtime.state["isRunning"] = True
            runtime.state["subagents"] = []
            runtime.state["activity"] = []
            runtime.state["approval"] = None
            runtime.state["askUser"] = None
            self._append_event_locked(
                runtime,
                "run_start",
                {"threadId": thread_id, "isRunning": True},
            )

    def update_subagents_from_stream_state(
        self, thread_id: str, stream_state: Any
    ) -> None:
        thinking_text = str(getattr(stream_state, "thinking_text", "") or "").rstrip()
        is_thinking = bool(getattr(stream_state, "is_thinking", False))
        status_value = thinking_text if is_thinking and thinking_text else None

        next_subagents: list[dict[str, Any]] = []
        for subagent in getattr(stream_state, "subagents", []) or []:
            name = str(getattr(subagent, "name", "sub-agent") or "sub-agent")
            description = str(getattr(subagent, "description", "") or "")
            is_active = bool(getattr(subagent, "is_active", False))
            tool_calls = list(getattr(subagent, "tool_calls", []) or [])
            detail = ""
            if tool_calls:
                last_tool = tool_calls[-1]
                detail = f"Tool: {last_tool.get('name', 'unknown')}"

            next_subagents.append(
                {
                    "id": name,
                    "name": name,
                    "description": description,
                    "status": "running" if is_active else "complete",
                    "detail": detail,
                }
            )

        with self._lock:
            runtime = self._ensure(thread_id)
            runtime.state["status"] = status_value
            runtime.state["subagents"] = next_subagents
            runtime.state["isRunning"] = bool(
                runtime.state.get("isRunning", False) or bool(next_subagents)
            )
            self._append_event_locked(
                runtime,
                "stream_state",
                {
                    "threadId": thread_id,
                    "status": status_value,
                    "subagents": next_subagents,
                    "isRunning": runtime.state.get("isRunning", False),
                },
            )

    def apply_subagent_event(
        self,
        thread_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        raw_name = payload.get("subagent") or payload.get("name") or "sub-agent"
        subagent_id = str(payload.get("instance_id") or raw_name)
        subagent_name = str(raw_name)

        with self._lock:
            runtime = self._ensure(thread_id)
            subagents = runtime.state.setdefault("subagents", [])
            if not isinstance(subagents, list):
                subagents = []
                runtime.state["subagents"] = subagents

            def _upsert_subagent(**patch: Any) -> None:
                for subagent in subagents:
                    if isinstance(subagent, dict) and subagent.get("id") == subagent_id:
                        subagent.update(
                            {k: v for k, v in patch.items() if v is not None}
                        )
                        return
                subagents.append(
                    {
                        "id": subagent_id,
                        "name": patch.get("name") or subagent_name,
                        "description": patch.get("description") or "",
                        "status": patch.get("status") or "running",
                        "detail": patch.get("detail") or "",
                    }
                )

            activity = runtime.state.setdefault("activity", [])
            if not isinstance(activity, list):
                activity = []
                runtime.state["activity"] = activity

            activity_entry: dict[str, Any] | None = None
            if event_type == "subagent_start":
                description = str(payload.get("description", ""))
                _upsert_subagent(
                    name=subagent_name,
                    description=description,
                    status="running",
                    detail="Delegated",
                )
                activity_entry = {
                    "id": f"activity_{uuid.uuid4().hex}",
                    "type": "subagent_start",
                    "agent": subagent_name,
                    "title": f"Started {subagent_name}",
                    "detail": description,
                    "status": "running",
                    "createdAt": _utc_now_iso(),
                }
            elif event_type == "subagent_tool_call":
                tool_name = str(payload.get("name", "unknown"))
                _upsert_subagent(
                    name=subagent_name,
                    status="running",
                    detail=f"Tool: {tool_name}",
                )
                activity_entry = {
                    "id": f"activity_{uuid.uuid4().hex}",
                    "type": "subagent_tool_call",
                    "agent": subagent_name,
                    "title": f"{subagent_name} called {tool_name}",
                    "detail": "",
                    "status": "running",
                    "createdAt": _utc_now_iso(),
                }
            elif event_type == "subagent_tool_result":
                tool_name = str(payload.get("name", "unknown"))
                success = bool(payload.get("success", True))
                result_content = _normalize_tool_result(payload.get("content", ""))
                _upsert_subagent(
                    name=subagent_name,
                    status="complete" if success else "error",
                    detail=f"Finished: {tool_name}",
                )
                activity_entry = {
                    "id": f"activity_{uuid.uuid4().hex}",
                    "type": "subagent_tool_result",
                    "agent": subagent_name,
                    "title": f"{subagent_name} finished {tool_name}",
                    "detail": "",
                    "result": result_content,
                    "status": "complete" if success else "error",
                    "createdAt": _utc_now_iso(),
                }
            elif event_type == "subagent_end":
                _upsert_subagent(
                    name=subagent_name,
                    status="complete",
                    detail="Completed",
                )
                activity_entry = {
                    "id": f"activity_{uuid.uuid4().hex}",
                    "type": "subagent_end",
                    "agent": subagent_name,
                    "title": f"Completed {subagent_name}",
                    "detail": "",
                    "status": "complete",
                    "createdAt": _utc_now_iso(),
                }
            elif event_type == "subagent_text":
                content = str(payload.get("content", ""))[:240]
                _upsert_subagent(
                    name=subagent_name,
                    status="running",
                    detail=content[:160],
                )
                activity_entry = {
                    "id": f"activity_{uuid.uuid4().hex}",
                    "type": "subagent_text",
                    "agent": subagent_name,
                    "title": subagent_name,
                    "detail": content,
                    "status": "running",
                    "createdAt": _utc_now_iso(),
                }

            if activity_entry:
                activity.append(activity_entry)
                if len(activity) > _MAX_ACTIVITY_ITEMS:
                    del activity[: len(activity) - _MAX_ACTIVITY_ITEMS]

            runtime.state["isRunning"] = True
            self._append_event_locked(
                runtime,
                event_type,
                {"threadId": thread_id, "event": copy.deepcopy(payload)},
            )

    def clear_thread(self, thread_id: str) -> None:
        with self._lock:
            self._threads.pop(thread_id, None)
            self._pending_cancel_requests.discard(thread_id)


_GLOBAL_THREAD_RUNTIME_REGISTRY = ThreadRuntimeRegistry()


def get_thread_runtime_registry() -> ThreadRuntimeRegistry:
    return _GLOBAL_THREAD_RUNTIME_REGISTRY
