"""In-process runtime registry shared across interactive surfaces.

This stores live, thread-scoped state (thinking/subagents/activity/HITL prompts)
so TUI and WebUI can observe the same run state in real time.
"""

from __future__ import annotations

import copy
import hashlib
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
_MAX_SEEN_TOOL_IDS = 500
_UNSET = object()
_FILE_TOOL_NAMES = {"write_file", "edit_file"}


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


def _extract_file_tool_path(payload: dict[str, Any]) -> str:
    args = payload.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return str(args.get("path") or args.get("file_path") or "").strip()


def _dedupe_key_for_file_activity(
    tool_name: str,
    file_path: str,
    tool_call_id: str,
    payload: dict[str, Any],
) -> str:
    if tool_call_id:
        return f"id:{tool_call_id}"
    try:
        rendered = json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        rendered = str(payload)
    digest = hashlib.sha1(rendered.encode("utf-8")).hexdigest()[:16]
    return f"payload:{tool_name}:{file_path}:{digest}"


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
                    "messages": [],
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

    def seed_messages(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
        *,
        replace: bool = False,
    ) -> None:
        """Seed the live transcript for a thread from durable history."""
        normalized: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "") or "")
            content = str(message.get("content", "") or "")
            message_id = str(message.get("id", "") or "")
            if role not in {"user", "assistant"} or not message_id:
                continue
            normalized_message = {
                "id": message_id,
                "role": role,
                "content": content,
            }
            created_at = str(message.get("createdAt", "") or "").strip()
            if created_at:
                normalized_message["createdAt"] = created_at
            normalized.append(normalized_message)

        if not normalized:
            return

        with self._lock:
            runtime = self._ensure(thread_id)
            existing = runtime.state.get("messages")
            if not replace and isinstance(existing, list) and existing:
                return
            runtime.state["messages"] = copy.deepcopy(normalized)
            runtime.state["_activeAssistantMessageId"] = None
            self._append_event_locked(
                runtime,
                "messages_seeded",
                {"threadId": thread_id, "messages": normalized},
            )

    def append_user_message(
        self,
        thread_id: str,
        content: str,
        *,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        message = {
            "id": message_id or f"user_{uuid.uuid4().hex}",
            "role": "user",
            "content": str(content),
            "createdAt": _utc_now_iso(),
        }
        with self._lock:
            runtime = self._ensure(thread_id)
            messages = runtime.state.setdefault("messages", [])
            if not isinstance(messages, list):
                messages = []
                runtime.state["messages"] = messages
            messages.append(copy.deepcopy(message))
            runtime.state["_activeAssistantMessageId"] = None
            self._append_event_locked(
                runtime,
                "user_message",
                {"threadId": thread_id, "message": message},
            )
        return message

    def start_assistant_message(
        self,
        thread_id: str,
        *,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            runtime = self._ensure(thread_id)
            messages = runtime.state.setdefault("messages", [])
            if not isinstance(messages, list):
                messages = []
                runtime.state["messages"] = messages

            active_id = str(runtime.state.get("_activeAssistantMessageId") or "")
            if active_id:
                for existing in messages:
                    if (
                        isinstance(existing, dict)
                        and existing.get("id") == active_id
                        and existing.get("role") == "assistant"
                    ):
                        return copy.deepcopy(existing)

            message = {
                "id": message_id or f"assistant_{uuid.uuid4().hex}",
                "role": "assistant",
                "content": "",
                "createdAt": _utc_now_iso(),
            }
            messages.append(copy.deepcopy(message))
            runtime.state["_activeAssistantMessageId"] = message["id"]
            self._append_event_locked(
                runtime,
                "assistant_start",
                {"threadId": thread_id, "message": message},
            )
        return message

    def append_assistant_delta(
        self,
        thread_id: str,
        delta: str,
        *,
        message_id: str | None = None,
    ) -> None:
        if not delta:
            return
        with self._lock:
            runtime = self._ensure(thread_id)
            messages = runtime.state.setdefault("messages", [])
            if not isinstance(messages, list):
                messages = []
                runtime.state["messages"] = messages

            target_id = message_id or str(
                runtime.state.get("_activeAssistantMessageId") or ""
            )
            target: dict[str, Any] | None = None
            for message in reversed(messages):
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                if not target_id or message.get("id") == target_id:
                    target = message
                    break

            if target is None:
                target = {
                    "id": target_id or f"assistant_{uuid.uuid4().hex}",
                    "role": "assistant",
                    "content": "",
                    "createdAt": _utc_now_iso(),
                }
                messages.append(target)

            target["content"] = str(target.get("content", "")) + str(delta)
            runtime.state["_activeAssistantMessageId"] = target["id"]
            self._append_event_locked(
                runtime,
                "assistant_delta",
                {
                    "threadId": thread_id,
                    "messageId": target["id"],
                    "delta": str(delta),
                    "content": target["content"],
                },
            )

    def set_assistant_message_content(
        self,
        thread_id: str,
        content: str,
        *,
        message_id: str | None = None,
    ) -> None:
        with self._lock:
            runtime = self._ensure(thread_id)
            messages = runtime.state.setdefault("messages", [])
            if not isinstance(messages, list):
                messages = []
                runtime.state["messages"] = messages

            target_id = message_id or str(
                runtime.state.get("_activeAssistantMessageId") or ""
            )
            target: dict[str, Any] | None = None
            for message in reversed(messages):
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                if not target_id or message.get("id") == target_id:
                    target = message
                    break

            if target is None:
                target = {
                    "id": target_id or f"assistant_{uuid.uuid4().hex}",
                    "role": "assistant",
                    "content": "",
                    "createdAt": _utc_now_iso(),
                }
                messages.append(target)

            target["content"] = str(content)
            runtime.state["_activeAssistantMessageId"] = target["id"]
            self._append_event_locked(
                runtime,
                "assistant_set",
                {
                    "threadId": thread_id,
                    "messageId": target["id"],
                    "content": target["content"],
                },
            )

    def finish_assistant_message(
        self,
        thread_id: str,
        *,
        content: str | None = None,
        message_id: str | None = None,
    ) -> None:
        if content is not None:
            self.set_assistant_message_content(
                thread_id,
                content,
                message_id=message_id,
            )
        with self._lock:
            runtime = self._ensure(thread_id)
            target_id = message_id or str(
                runtime.state.get("_activeAssistantMessageId") or ""
            )
            runtime.state["_activeAssistantMessageId"] = None
            self._append_event_locked(
                runtime,
                "assistant_done",
                {"threadId": thread_id, "messageId": target_id},
            )

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
            runtime.state["_activeAssistantMessageId"] = None
            runtime.state["_pendingFileToolCallsById"] = {}
            runtime.state["_pendingFileToolCallQueue"] = []
            runtime.state["_seenFileToolActivityKeys"] = []
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

    def apply_tool_event(
        self,
        thread_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        tool_name = str(payload.get("name", "") or "").strip()
        if tool_name not in _FILE_TOOL_NAMES:
            return

        if event_type not in {"tool_call", "tool_result"}:
            return

        with self._lock:
            runtime = self._ensure(thread_id)
            tool_call_id = str(
                payload.get("id") or payload.get("tool_call_id") or ""
            ).strip()

            if event_type == "tool_call":
                file_path = _extract_file_tool_path(payload)
                self._remember_file_tool_call_locked(
                    runtime,
                    tool_name=tool_name,
                    file_path=file_path,
                    tool_call_id=tool_call_id,
                )
                self._append_event_locked(
                    runtime,
                    "tool_call",
                    {
                        "threadId": thread_id,
                        "event": copy.deepcopy(payload),
                        "filePath": file_path,
                    },
                )
                return

            if not bool(payload.get("success", True)):
                return

            file_path = self._resolve_completed_file_tool_locked(
                runtime,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                payload=payload,
            )
            if not file_path:
                return

            activity_key = _dedupe_key_for_file_activity(
                tool_name,
                file_path,
                tool_call_id,
                payload,
            )
            seen_keys = runtime.state.setdefault("_seenFileToolActivityKeys", [])
            if not isinstance(seen_keys, list):
                seen_keys = []
                runtime.state["_seenFileToolActivityKeys"] = seen_keys
            if activity_key in seen_keys:
                return
            seen_keys.append(activity_key)
            if len(seen_keys) > _MAX_SEEN_TOOL_IDS:
                del seen_keys[: len(seen_keys) - _MAX_SEEN_TOOL_IDS]

            activity = runtime.state.setdefault("activity", [])
            if not isinstance(activity, list):
                activity = []
                runtime.state["activity"] = activity

            activity_entry = {
                "id": f"activity_{uuid.uuid4().hex}",
                "type": tool_name,
                "agent": "main",
                "title": f"{tool_name}({file_path})",
                "detail": file_path,
                "filePath": file_path,
                "status": "complete",
                "createdAt": _utc_now_iso(),
            }
            activity.append(activity_entry)
            if len(activity) > _MAX_ACTIVITY_ITEMS:
                del activity[: len(activity) - _MAX_ACTIVITY_ITEMS]

            runtime.state["isRunning"] = bool(runtime.state.get("isRunning", False))
            self._append_event_locked(
                runtime,
                "tool_result",
                {
                    "threadId": thread_id,
                    "event": copy.deepcopy(payload),
                    "filePath": file_path,
                    "activity": copy.deepcopy(activity_entry),
                },
            )

    def _remember_file_tool_call_locked(
        self,
        runtime: _ThreadRuntimeState,
        *,
        tool_name: str,
        file_path: str,
        tool_call_id: str,
    ) -> None:
        if not tool_call_id and not file_path:
            return

        pending_by_id = runtime.state.setdefault("_pendingFileToolCallsById", {})
        if not isinstance(pending_by_id, dict):
            pending_by_id = {}
            runtime.state["_pendingFileToolCallsById"] = pending_by_id

        pending_queue = runtime.state.setdefault("_pendingFileToolCallQueue", [])
        if not isinstance(pending_queue, list):
            pending_queue = []
            runtime.state["_pendingFileToolCallQueue"] = pending_queue

        call = {
            "id": tool_call_id,
            "name": tool_name,
            "filePath": file_path,
        }

        if tool_call_id:
            existing = pending_by_id.get(tool_call_id)
            if isinstance(existing, dict):
                existing["name"] = tool_name or existing.get("name", "")
                if file_path:
                    existing["filePath"] = file_path
                call = {
                    "id": tool_call_id,
                    "name": str(existing.get("name", "") or tool_name),
                    "filePath": str(existing.get("filePath", "") or file_path),
                }
            pending_by_id[tool_call_id] = call

        for existing in pending_queue:
            if not isinstance(existing, dict):
                continue
            if tool_call_id and existing.get("id") == tool_call_id:
                existing["name"] = tool_name or existing.get("name", "")
                if file_path:
                    existing["filePath"] = file_path
                return

        pending_queue.append(call)
        if len(pending_queue) > _MAX_SEEN_TOOL_IDS:
            del pending_queue[: len(pending_queue) - _MAX_SEEN_TOOL_IDS]

    def _resolve_completed_file_tool_locked(
        self,
        runtime: _ThreadRuntimeState,
        *,
        tool_name: str,
        tool_call_id: str,
        payload: dict[str, Any],
    ) -> str:
        file_path = _extract_file_tool_path(payload)
        pending_by_id = runtime.state.get("_pendingFileToolCallsById", {})
        if not isinstance(pending_by_id, dict):
            pending_by_id = {}

        pending_queue = runtime.state.get("_pendingFileToolCallQueue", [])
        if not isinstance(pending_queue, list):
            pending_queue = []

        if tool_call_id:
            pending = pending_by_id.pop(tool_call_id, None)
            if isinstance(pending, dict):
                file_path = file_path or str(pending.get("filePath", "") or "")

        matched_index: int | None = None
        if not file_path or tool_call_id:
            for index, pending in enumerate(pending_queue):
                if not isinstance(pending, dict):
                    continue
                pending_id = str(pending.get("id", "") or "")
                pending_name = str(pending.get("name", "") or "")
                if tool_call_id and pending_id != tool_call_id:
                    continue
                if not tool_call_id and pending_name != tool_name:
                    continue
                file_path = file_path or str(pending.get("filePath", "") or "")
                matched_index = index
                break

        if matched_index is not None:
            del pending_queue[matched_index]
        elif not tool_call_id:
            for index, pending in enumerate(pending_queue):
                if not isinstance(pending, dict):
                    continue
                if str(pending.get("name", "") or "") != tool_name:
                    continue
                pending_path = str(pending.get("filePath", "") or "")
                if not pending_path:
                    continue
                file_path = file_path or pending_path
                del pending_queue[index]
                break

        return file_path.strip()

    def clear_thread(self, thread_id: str) -> None:
        with self._lock:
            self._threads.pop(thread_id, None)
            self._pending_cancel_requests.discard(thread_id)


_GLOBAL_THREAD_RUNTIME_REGISTRY = ThreadRuntimeRegistry()


def get_thread_runtime_registry() -> ThreadRuntimeRegistry:
    return _GLOBAL_THREAD_RUNTIME_REGISTRY
