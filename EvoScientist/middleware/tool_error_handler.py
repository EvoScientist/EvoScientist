"""Middleware that catches tool execution exceptions and converts them to error ToolMessages.

Without this, an MCP tool (or any tool) that raises an exception at runtime
crashes the entire agent loop because LangGraph's default ToolNode error handler
only catches argument-validation errors (ToolInvocationError), not execution
errors.

With this middleware, the exception is caught and surfaced to the agent as a
ToolMessage with ``status="error"`` containing the traceback.  The agent can
then decide to retry, use a different tool, or yield to the user.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

# GraphInterrupt must propagate — never catch it as a tool error.
try:
    from langgraph.errors import GraphInterrupt as _GraphInterrupt
except ImportError:  # older langgraph versions
    _GraphInterrupt = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ToolCallRequest

logger = logging.getLogger(__name__)

_INVALID_TOOL_CALL_FEEDBACK_PREFIX = "[INVALID TOOL CALL]"
_MAX_INVALID_TOOL_ARG_CHARS = 1200


class ToolErrorHandlerMiddleware(AgentMiddleware):
    """Catch tool execution exceptions and return them as error ToolMessages."""

    name = "tool_error_handler"

    def __init__(self, *, max_invalid_tool_call_retries: int = 2) -> None:
        super().__init__()
        self._max_invalid_tool_call_retries = max_invalid_tool_call_retries

    @hook_config(can_jump_to=["model", "end"])
    def after_model(
        self, state: Mapping[str, object], runtime: object
    ) -> dict[str, Any] | None:
        """Ask the model to retry malformed tool calls before the agent exits."""
        messages = state.get("messages")
        if not isinstance(messages, Sequence) or not messages:
            return None
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return None

        invalid_tool_calls = _invalid_tool_calls(last_message)
        if not invalid_tool_calls:
            return None

        retry_count = _invalid_tool_call_feedback_count(messages)
        if retry_count >= self._max_invalid_tool_call_retries:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I could not execute the requested tool because I "
                            "repeatedly emitted malformed tool-call arguments. "
                            "I should explain this failure instead of assuming "
                            "the tool succeeded."
                        )
                    )
                ],
                "jump_to": "end",
            }

        return {
            "messages": [
                HumanMessage(
                    content=_invalid_tool_call_feedback(invalid_tool_calls),
                    name="tool_call_validator",
                )
            ],
            "jump_to": "model",
        }

    @hook_config(can_jump_to=["model", "end"])
    async def aafter_model(
        self, state: Mapping[str, object], runtime: object
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        try:
            return handler(request)
        except Exception as exc:
            if _GraphInterrupt is not None and isinstance(exc, _GraphInterrupt):
                raise
            return _build_error_message(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        try:
            return await handler(request)
        except Exception as exc:
            if _GraphInterrupt is not None and isinstance(exc, _GraphInterrupt):
                raise
            return _build_error_message(request)


def _invalid_tool_calls(message: AIMessage) -> list[Mapping[str, object]]:
    """Return malformed tool calls from dedicated fields and content blocks."""
    calls: list[Mapping[str, object]] = []
    seen: set[str] = set()

    def add_call(value: object) -> None:
        if not isinstance(value, Mapping):
            return
        if value.get("type") != "invalid_tool_call":
            return
        call_id = str(value.get("id") or "")
        key = call_id or repr(value)
        if key in seen:
            return
        seen.add(key)
        calls.append(value)

    for call in message.invalid_tool_calls:
        add_call(call)
    if isinstance(message.content, list):
        for block in message.content:
            add_call(block)
    return calls


def _invalid_tool_call_feedback_count(messages: Sequence[object]) -> int:
    """Count retry feedback messages since the last real user message."""
    count = 0
    for message in reversed(messages[:-1]):
        if isinstance(message, HumanMessage):
            content = message.text
            if content.startswith(_INVALID_TOOL_CALL_FEEDBACK_PREFIX):
                count += 1
                continue
            break
    return count


def _truncate_tool_args(value: object) -> str:
    text = str(value or "").strip()
    if len(text) <= _MAX_INVALID_TOOL_ARG_CHARS:
        return text
    return text[:_MAX_INVALID_TOOL_ARG_CHARS] + "...<truncated>"


def _invalid_tool_call_feedback(
    invalid_tool_calls: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        _INVALID_TOOL_CALL_FEEDBACK_PREFIX,
        "The previous assistant message attempted malformed tool call(s). "
        "They were not executed. Retry the needed tool call with valid JSON "
        "arguments that match the tool schema; do not answer as if the tool "
        "succeeded.",
    ]
    for index, call in enumerate(invalid_tool_calls, 1):
        tool_name = str(call.get("name") or "unknown_tool")
        error = str(call.get("error") or "invalid tool-call arguments")
        raw_args = _truncate_tool_args(call.get("args"))
        lines.append(
            f"{index}. `{tool_name}` failed to parse: {error}. "
            f"Raw arguments: {raw_args}"
        )
    return "\n".join(lines)


def _build_error_message(request: ToolCallRequest) -> ToolMessage:
    tb = traceback.format_exc()
    tool_name = request.tool_call.get("name", "unknown_tool")
    logger.error("Tool %r raised an exception:\n%s", tool_name, tb)
    content = (
        f"[TOOL ERROR] Tool '{tool_name}' failed with an exception:\n\n{tb}\n"
        "You may retry the tool call, try an alternative approach, "
        "or inform the user about the failure."
    )
    return ToolMessage(
        content=content,
        tool_call_id=request.tool_call["id"],
        name=tool_name,
        status="error",
    )
