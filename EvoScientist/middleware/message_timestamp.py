"""Stamp every message in the thread with the time it was created.

Most providers put no time on a reply, and no human or tool message carries
one, so clients have nothing to show next to a message. This middleware sets
``response_metadata["evo_created_at"]`` (ISO-8601 UTC, milliseconds) once per
message and never rewrites it:

- AI messages when the model call returns (``wrap_model_call``);
- tool messages when the tool returns (``wrap_tool_call``);
- everything else that enters state between model calls (user input,
  async-task notifications, harness-injected messages) in ``before_agent``
  and ``before_model``.

``response_metadata`` is not sent back to providers, so the key can't reach a
model request.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import Command

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest, ToolCallRequest

CREATED_AT_KEY = "evo_created_at"


def created_at_now() -> str:
    """The current time in the ``evo_created_at`` format."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def with_created_at(message: Any, now: str) -> Any:
    """A copy of *message* with the key set; anything else is returned as is.

    For code that writes messages into thread state without going through the
    agent's hooks (``update_state`` callers).
    """
    if (
        not isinstance(message, BaseMessage)
        or CREATED_AT_KEY in message.response_metadata
    ):
        return message
    # A copy, never in place: the metadata dict is shared with checkpointed state.
    return message.model_copy(
        update={"response_metadata": {**message.response_metadata, CREATED_AT_KEY: now}}
    )


def _stamped_command(command: Command, now: str) -> Command:
    update = command.update
    if not isinstance(update, dict) or not isinstance(update.get("messages"), list):
        return command
    messages = [with_created_at(message, now) for message in update["messages"]]
    return replace(command, update={**update, "messages": messages})


def _stamped_model_result(result: Any, now: str) -> Any:
    if isinstance(result, AIMessage):
        return with_created_at(result, now)
    if isinstance(result, ModelResponse):
        return replace(result, result=[with_created_at(m, now) for m in result.result])
    if isinstance(result, ExtendedModelResponse):
        command = result.command
        return replace(
            result,
            model_response=_stamped_model_result(result.model_response, now),
            command=_stamped_command(command, now) if command is not None else None,
        )
    return result


def _stamped_tool_result(result: Any, now: str) -> Any:
    if isinstance(result, Command):
        return _stamped_command(result, now)
    return with_created_at(result, now)


def stamp_new_messages(messages: Sequence[Any]) -> dict[str, Any] | None:
    """State update stamping the unstamped messages of the current run, or ``None``.

    Walks back from the newest message and stops at the first stamped one, or
    at an unstamped AI message: every reply made since this middleware exists
    is stamped when its model call returns, so an unstamped one predates it.
    Older history of such a thread stays unstamped instead of getting the time
    it was resumed.
    """
    now: str | None = None
    stamped: list[BaseMessage] = []
    for message in reversed(messages):
        if not isinstance(message, BaseMessage):
            continue
        if CREATED_AT_KEY in message.response_metadata or isinstance(
            message, AIMessage
        ):
            break
        # Without an id the reducer would append the copy instead of replacing.
        if message.id is None:
            continue
        now = now or created_at_now()
        stamped.append(with_created_at(message, now))
    if not stamped:
        return None
    return {"messages": stamped[::-1]}


class MessageTimestampMiddleware(AgentMiddleware):
    """Set ``response_metadata["evo_created_at"]`` once on every message."""

    name = "message_timestamp"

    def before_agent(self, state, runtime) -> dict[str, Any] | None:
        return stamp_new_messages(state.get("messages") or [])

    async def abefore_agent(self, state, runtime) -> dict[str, Any] | None:
        return stamp_new_messages(state.get("messages") or [])

    def before_model(self, state, runtime) -> dict[str, Any] | None:
        return stamp_new_messages(state.get("messages") or [])

    async def abefore_model(self, state, runtime) -> dict[str, Any] | None:
        return stamp_new_messages(state.get("messages") or [])

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        return _stamped_model_result(handler(request), created_at_now())

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[Any]],
    ) -> Any:
        return _stamped_model_result(await handler(request), created_at_now())

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return _stamped_tool_result(handler(request), created_at_now())

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        return _stamped_tool_result(await handler(request), created_at_now())
