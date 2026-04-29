"""Shared runtime bridge for interactive surfaces and channel adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .thread_registry import ThreadRuntimeRegistry, get_thread_runtime_registry


def _format_todo_list(todos: list[dict[str, Any]]) -> str:
    lines = ["Todo List", ""]
    for i, item in enumerate(todos, 1):
        content = str(item.get("content", "") or "").strip()
        lines.append(f"{i}. {content}")
    lines.append("")
    lines.append(f"{len(todos)} tasks")
    return "\n".join(lines)


def _approval_payload(action_requests: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "kind": "tool_approval",
        "title": "Approval Required",
        "options": [
            {"id": "approve", "label": "Approve"},
            {"id": "reject", "label": "Reject"},
            {"id": "auto", "label": "Approve All"},
        ],
        "actions": [
            {
                "name": (
                    req.get("name", "")
                    if isinstance(req, dict)
                    else getattr(req, "name", "")
                ),
                "command": (
                    (req.get("args", {}) or {}).get("command", "")
                    if isinstance(req, dict)
                    else getattr(req, "args", {}).get("command", "")
                ),
            }
            for req in action_requests
        ],
    }


def _ask_user_payload(ask_user_data: dict[str, Any]) -> dict[str, Any] | None:
    questions = ask_user_data.get("questions", [])
    if not questions:
        return None
    first = questions[0]
    q_type = str(first.get("type", "text") or "text")
    choices = first.get("choices", [])
    return {
        "title": (
            "Quick check-in from EvoScientist"
            if len(questions) == 1
            else f"Question 1/{len(questions)}"
        ),
        "question": str(first.get("question", "")),
        "required": bool(first.get("required", True)),
        "type": q_type,
        "choices": [
            {"value": str(choice.get("value", str(choice)))}
            for choice in choices
            if isinstance(choice, dict) or isinstance(choice, str)
        ],
    }


@dataclass
class RuntimeBridgeCallbacks:
    """Reusable callbacks for run_streaming / stream_agent_events loops."""

    on_thinking: Callable[[str], None] | None
    on_todo: Callable[[list[dict[str, Any]]], None] | None
    on_file_write: Callable[[str], None] | None
    on_stream_event: Callable[[str, Any], None] | None
    on_raw_stream_event: Callable[[dict[str, Any]], None] | None
    hitl_prompt_fn: Callable[[list], list[dict] | None] | None
    ask_user_prompt_fn: Callable[[dict], dict] | None


def mark_run_started(
    thread_id: str,
    *,
    registry: ThreadRuntimeRegistry | None = None,
) -> None:
    runtime_registry = registry or get_thread_runtime_registry()
    runtime_registry.begin_run(thread_id)


def mark_run_finished(
    thread_id: str,
    *,
    registry: ThreadRuntimeRegistry | None = None,
) -> None:
    runtime_registry = registry or get_thread_runtime_registry()
    runtime_registry.clear_cancel_request(thread_id)
    runtime_registry.set_status(thread_id, status=None, is_running=False)
    runtime_registry.set_prompt(thread_id, approval=None, ask_user=None)


def build_runtime_bridge(
    *,
    thread_id: str,
    channel: Any | None = None,
    metadata: dict[str, Any] | None = None,
    send_async: Callable[[Any, str, int], None] | None = None,
    fallback_hitl_prompt: Callable[[list], list[dict] | None] | None = None,
    fallback_ask_user_prompt: Callable[[dict], dict] | None = None,
    registry: ThreadRuntimeRegistry | None = None,
) -> RuntimeBridgeCallbacks:
    """Build channel/runtime callbacks for a streaming run.

    The resulting callbacks:
    - update the shared runtime registry (TUI/WebUI parity),
    - publish to channel-specific methods when available.
    """

    runtime_registry = registry or get_thread_runtime_registry()
    meta = dict(metadata or {})
    if "chat_id" not in meta:
        meta["chat_id"] = thread_id

    def _send_to_channel(coro: Any, label: str, timeout: int = 15) -> None:
        if send_async is None:
            return
        send_async(coro, label, timeout)

    def _thinking_callback(thinking: str) -> None:
        runtime_registry.set_status(thread_id, status=thinking, is_running=True)
        if channel is None:
            return
        if getattr(channel, "send_thinking", False):
            if hasattr(channel, "send_thinking_message_nowait"):
                channel.send_thinking_message_nowait(
                    sender=thread_id,
                    thinking=thinking,
                    metadata=meta,
                )
                return
            if hasattr(channel, "send_thinking_message"):
                _send_to_channel(
                    channel.send_thinking_message(
                        sender=thread_id,
                        thinking=thinking,
                        metadata=meta,
                    ),
                    "Thinking",
                )

    def _todo_callback(items: list[dict[str, Any]]) -> None:
        content = _format_todo_list(items)
        runtime_registry.set_status(thread_id, status=content, is_running=True)
        if channel is None:
            return
        if hasattr(channel, "send_todo_message"):
            _send_to_channel(
                channel.send_todo_message(
                    sender=thread_id,
                    content=content,
                    metadata=meta,
                ),
                "Todo",
            )

    def _media_callback(file_path: str) -> None:
        if channel is None:
            return
        if hasattr(channel, "send_media"):
            _send_to_channel(
                channel.send_media(
                    recipient=thread_id,
                    file_path=file_path,
                    metadata=meta,
                ),
                "Media",
                30,
            )

    def _stream_state_callback(_event_type: str, stream_state: Any) -> None:
        runtime_registry.update_subagents_from_stream_state(thread_id, stream_state)
        if channel is None:
            return
        if hasattr(channel, "send_stream_state_nowait"):
            channel.send_stream_state_nowait(
                sender=thread_id,
                stream_state=stream_state,
                metadata=meta,
            )
            return
        if hasattr(channel, "send_stream_state"):
            _send_to_channel(
                channel.send_stream_state(
                    sender=thread_id,
                    stream_state=stream_state,
                    metadata=meta,
                ),
                "StreamState",
            )

    def _raw_stream_callback(event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type.startswith("subagent_"):
            runtime_registry.apply_subagent_event(thread_id, event_type, event)
            if channel is None:
                return
            if hasattr(channel, "send_subagent_event_nowait"):
                channel.send_subagent_event_nowait(
                    sender=thread_id,
                    event_type=event_type,
                    payload=event,
                    metadata=meta,
                )
                return
            if hasattr(channel, "send_subagent_event"):
                _send_to_channel(
                    channel.send_subagent_event(
                        sender=thread_id,
                        event_type=event_type,
                        payload=event,
                        metadata=meta,
                    ),
                    "SubagentEvent",
                )
            return

        if event_type in {"tool_call", "tool_result"}:
            if channel is None:
                runtime_registry.apply_tool_event(thread_id, event_type, event)
                return
            if hasattr(channel, "send_tool_event_nowait"):
                channel.send_tool_event_nowait(
                    sender=thread_id,
                    event_type=event_type,
                    payload=event,
                    metadata=meta,
                )
                return
            if hasattr(channel, "send_tool_event"):
                _send_to_channel(
                    channel.send_tool_event(
                        sender=thread_id,
                        event_type=event_type,
                        payload=event,
                        metadata=meta,
                    ),
                    "ToolEvent",
                )
                return
            runtime_registry.apply_tool_event(thread_id, event_type, event)

    def _hitl_prompt(action_requests: list) -> list[dict] | None:
        if channel is not None and hasattr(channel, "prompt_approval"):
            return channel.prompt_approval(
                thread_id=thread_id,
                action_requests=action_requests,
            )
        runtime_registry.set_prompt(
            thread_id,
            approval=_approval_payload(action_requests),
        )
        if fallback_hitl_prompt is not None:
            try:
                return fallback_hitl_prompt(action_requests)
            finally:
                runtime_registry.set_prompt(thread_id, approval=None)
        runtime_registry.set_prompt(thread_id, approval=None)
        return None

    def _ask_user_prompt(ask_user_data: dict) -> dict:
        if channel is not None and hasattr(channel, "prompt_ask_user"):
            return channel.prompt_ask_user(
                thread_id=thread_id,
                ask_user_data=ask_user_data,
            )
        prompt_payload = _ask_user_payload(ask_user_data)
        if prompt_payload is not None:
            runtime_registry.set_prompt(
                thread_id,
                ask_user=prompt_payload,
            )
        if fallback_ask_user_prompt is not None:
            try:
                return fallback_ask_user_prompt(ask_user_data)
            finally:
                if prompt_payload is not None:
                    runtime_registry.set_prompt(thread_id, ask_user=None)
        if prompt_payload is not None:
            runtime_registry.set_prompt(thread_id, ask_user=None)
        return {"status": "cancelled"}

    return RuntimeBridgeCallbacks(
        on_thinking=_thinking_callback,
        on_todo=_todo_callback,
        on_file_write=_media_callback,
        on_stream_event=_stream_state_callback,
        on_raw_stream_event=_raw_stream_callback,
        hitl_prompt_fn=_hitl_prompt,
        ask_user_prompt_fn=_ask_user_prompt,
    )


def make_threadsafe_sender(
    loop: asyncio.AbstractEventLoop | None,
) -> Callable[[Any, str, int], None]:
    """Create a helper that runs channel coroutines on a target loop."""

    def _send(coro: Any, _label: str, timeout: int = 15) -> None:
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)

    return _send
