"""Shared source-run context for post-run memory agents."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

from langchain.agents.middleware.types import AgentState
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, filter_messages
from langchain_core.messages.tool import ToolCall
from langgraph.runtime import Runtime

from .types import MemorySourceType


class CompactMessage(TypedDict, total=False):
    """Minimal serializable message shape passed to memory agents."""

    role: str
    content: str
    name: NotRequired[str]
    tool_calls: NotRequired[list[ToolCall]]
    tool_call_id: NotRequired[str]
    status: NotRequired[str]


@dataclass(frozen=True)
class MemorySourceContext:
    """Captured source-run data shared by post-run memory agents."""

    source_type: MemorySourceType
    memory_dir: Path
    workspace_dir: Path
    project_id: str
    source_agent: str
    session_id: str
    trajectory: list[CompactMessage]
    trajectory_digest: str


def _task_tool_call_ids(messages: list[BaseMessage]) -> set[str]:
    """Return ids for subagent delegation tool calls."""
    ids: set[str] = set()
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            if call["name"] == "task" and call["id"]:
                ids.add(call["id"])
    return ids


def _source_agent_direct_tool_call_ids(
    messages: Sequence[BaseMessage],
    *,
    source_agent: str,
) -> set[str]:
    """Return non-delegation tool call ids made by the source agent."""
    ids: set[str] = set()
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        if message.name and message.name != source_agent:
            continue
        for call in message.tool_calls:
            if call["name"] != "task" and call["id"]:
                ids.add(call["id"])
    return ids


def _compact_message(
    message: BaseMessage,
    *,
    omit_task_results: bool,
    task_tool_call_ids: set[str],
) -> CompactMessage:
    """Convert one LangChain message to the worker trajectory format."""
    role = message.type
    content = str(message.text)
    item: CompactMessage = {"role": role, "content": content}
    if message.name:
        item["name"] = message.name
    if isinstance(message, AIMessage):
        tool_calls = list(message.tool_calls)
        if omit_task_results:
            tool_calls = [call for call in tool_calls if call["name"] != "task"]
        if tool_calls:
            item["tool_calls"] = tool_calls
    if isinstance(message, ToolMessage):
        item["tool_call_id"] = message.tool_call_id
        item["status"] = message.status
        if omit_task_results and message.tool_call_id in task_tool_call_ids:
            item["content"] = (
                "[subagent result omitted; subagent memory worker handles it]"
            )
    return item


def _compact_messages(
    messages: Sequence[BaseMessage],
    *,
    omit_task_results: bool = False,
) -> list[CompactMessage]:
    """Convert a run history into the serializable worker trajectory."""
    task_ids = _task_tool_call_ids(list(messages)) if omit_task_results else set()
    items: list[CompactMessage] = []
    for message in messages:
        item = _compact_message(
            message,
            omit_task_results=omit_task_results,
            task_tool_call_ids=task_ids,
        )
        items.append(item)
    return items


def _latest_user_turn_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Return messages from the latest user turn onward."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].type == "human":
            return list(messages[index:])
    return list(messages)


def _compact_turn_messages(
    messages: Sequence[BaseMessage],
    *,
    source_agent: str,
) -> list[CompactMessage]:
    """Build the orchestrator-only trajectory for the turn memory worker.

    LangChain's message filter removes task tool calls and their results, so
    the turn worker never receives subagent instructions or result bodies.
    """

    turn_messages = _latest_user_turn_messages(messages)
    task_ids = _task_tool_call_ids(turn_messages)
    direct_tool_ids = _source_agent_direct_tool_call_ids(
        turn_messages,
        source_agent=source_agent,
    )
    items: list[CompactMessage] = []
    filtered = filter_messages(turn_messages, exclude_tool_calls=task_ids)
    for message in filtered:
        if isinstance(message, ToolMessage):
            if message.tool_call_id not in direct_tool_ids:
                continue
        elif message.name and message.name != source_agent:
            continue

        items.append(
            _compact_message(
                message,
                omit_task_results=False,
                task_tool_call_ids=set(),
            )
        )
    return items


def _state_messages(state: AgentState[object]) -> list[BaseMessage]:
    """Read valid LangChain messages from agent state."""
    messages = state.get("messages", [])
    if not isinstance(messages, list):
        return []
    return [message for message in messages if isinstance(message, BaseMessage)]


def _stable_json(value: object) -> str:
    """Serialize values deterministically for hashing."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _pretty_json(value: object) -> str:
    """Serialize values readably for worker prompts."""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _trajectory_digest(trajectory: list[CompactMessage]) -> str:
    """Return the stable digest for a compact trajectory."""
    return _short_hash(_stable_json(trajectory))


def _trajectory_for_prompt(trajectory: list[CompactMessage]) -> str:
    """Serialize the full compact trajectory for worker prompts."""
    return _pretty_json(trajectory)


def _runtime_thread_id(runtime: Runtime | None) -> str | None:
    """Return the active LangGraph thread id when available."""
    if runtime and runtime.execution_info and runtime.execution_info.thread_id:
        return str(runtime.execution_info.thread_id)
    return None


def _short_hash(text: str) -> str:
    """Return the short hash fragment used in generated ids."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_memory_source_context(
    *,
    state: AgentState[object],
    runtime: Runtime | None,
    memory_dir: str | Path,
    workspace_dir: str | Path,
    project_id: str,
    source_type: MemorySourceType,
    source_agent: str,
) -> MemorySourceContext | None:
    """Capture the current source run as a reusable memory context."""
    session_id = _runtime_thread_id(runtime)
    if session_id is None:
        return None
    if source_type == MemorySourceType.TURN:
        trajectory = _compact_turn_messages(
            _state_messages(state),
            source_agent=source_agent,
        )
    else:
        trajectory = _compact_messages(_state_messages(state))

    if not trajectory:
        return None

    return MemorySourceContext(
        source_type=source_type,
        memory_dir=Path(memory_dir).expanduser(),
        workspace_dir=Path(workspace_dir).expanduser(),
        project_id=project_id,
        source_agent=source_agent,
        session_id=session_id,
        trajectory=trajectory,
        trajectory_digest=_trajectory_digest(trajectory),
    )
