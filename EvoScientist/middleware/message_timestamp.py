"""Stamp messages with the time they were created.

Most providers put no time on a reply, and no human or tool message carries
one, so clients have nothing to show next to a message. This middleware sets
``response_metadata["evo_created_at"]`` (ISO-8601 UTC, milliseconds) once per
message and never rewrites it:

- the run's input (user turns, async-task notifications) in ``before_agent``;
- AI messages when the model call returns (``wrap_model_call``);
- tool messages when the tool returns (``wrap_tool_call``);
- messages other hooks add to state mid-run (todo errors, HITL rejections,
  rubric revisions), when the next model call returns: ``wrap_model_call``
  sends their stamped copies in the same state update as the reply, so no
  extra checkpoint is written for them.

Code that writes into thread state outside the agent (``update_state``) stamps
its messages itself with :func:`created_at_now` and :func:`with_created_at`.

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


def _stamped_tool_result(result: Any, now: str) -> Any:
    if isinstance(result, Command):
        return _stamped_command(result, now)
    return with_created_at(result, now)


def _unstamped_tail(messages: Sequence[Any], now: str) -> list[BaseMessage]:
    """Stamped copies of the current run's unstamped messages, oldest first.

    Walks back from the newest message and stops at the first stamped one, or
    at an unstamped AI message: every reply made since this middleware exists
    is stamped when its model call returns, so an unstamped one predates it.
    Older history of such a thread stays unstamped instead of getting the time
    it was resumed.
    """
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
        stamped.append(with_created_at(message, now))
    return stamped[::-1]


def stamp_new_messages(messages: Sequence[Any]) -> dict[str, Any] | None:
    """State update stamping the current run's unstamped messages, or ``None``."""
    stamped = _unstamped_tail(messages, created_at_now())
    return {"messages": stamped} if stamped else None


def _stamped_model_response(
    request: ModelRequest, response: ModelResponse, now: str
) -> ModelResponse | ExtendedModelResponse:
    """Stamp the reply, plus any unstamped message added to state since the
    last model call; those copies keep their ids, so they replace in place."""
    response = replace(
        response, result=[with_created_at(m, now) for m in response.result]
    )
    state = getattr(request, "state", None) or {}
    pending = _unstamped_tail(state.get("messages") or [], now)
    if not pending:
        return response
    return ExtendedModelResponse(
        model_response=response, command=Command(update={"messages": pending})
    )


class MessageTimestampMiddleware(AgentMiddleware):
    """Set ``response_metadata["evo_created_at"]`` once per message."""

    name = "message_timestamp"

    def before_agent(self, state, runtime) -> dict[str, Any] | None:
        return stamp_new_messages(state.get("messages") or [])

    async def abefore_agent(self, state, runtime) -> dict[str, Any] | None:
        return stamp_new_messages(state.get("messages") or [])

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        response = handler(request)
        return _stamped_model_response(request, response, created_at_now())

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        response = await handler(request)
        return _stamped_model_response(request, response, created_at_now())

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
