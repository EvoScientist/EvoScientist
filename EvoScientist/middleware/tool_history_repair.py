"""Repair interrupted tool-call history before provider requests.

Strict providers (OpenAI, etc.) reject a message thread in which an assistant
tool call has no matching tool result. That happens whenever a run is
interrupted (cancelled, crashed, timed out) after the model emitted tool calls
but before those tools produced results. This middleware rewrites the outgoing
request so every dangling tool call is closed with a synthetic error result and
every orphan ``ToolMessage`` (a result whose originating call is gone) is
dropped.

It covers cases that deepagents' ``PatchToolCallsMiddleware`` does not:

1. Orphan ``ToolMessage`` dropping -- a tool result whose originating tool call
   is no longer present in history is removed, rather than left to trip strict
   providers.
2. Mid-run coverage -- repair runs at the model boundary on every request
   (including malformed / ``invalid_tool_calls``), not only at agent start, so
   interruptions that happen partway through a run are healed too.
3. Nameless function call dropping -- a tool call whose ``function.name`` is
   missing cannot be serialized into a valid request, so it is removed from the
   assistant message entirely (along with any raw ``additional_kwargs``
   counterpart) before the history is closed off.

Because the middleware only rewrites the request and cannot mutate thread
state, the repaired synthetic results are recomputed on every model call. To
avoid re-logging the same repair forever, warnings are deduplicated per unique
tool-call id via a ``warned`` set owned by the middleware instance.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

logger = logging.getLogger(__name__)
_INTERRUPTED_RESULT = "Tool execution was interrupted before completion."


def _drop_nameless_calls(message: AIMessage) -> tuple[AIMessage, list[str]]:
    """Return the message without function tool calls that have no name.

    A call with no ``name`` cannot be serialized into a valid
    ``function.name``, so strict providers reject the whole request. Only
    nameless function calls are removed; valid calls in the same message and
    non-function call types are preserved.
    """
    raw_calls = message.additional_kwargs.get("tool_calls")
    nameless_raw = [
        call
        for call in raw_calls or []
        if isinstance(call, dict)
        and call.get("type") == "function"
        and (
            not isinstance(call.get("function"), dict)
            or not call["function"].get("name")
        )
    ]
    nameless_parsed = [
        call
        for call in list(message.tool_calls)
        + list(getattr(message, "invalid_tool_calls", []) or [])
        if not call.get("name")
    ]
    if not nameless_raw and not nameless_parsed:
        return message, []

    dropped_ids = [
        call.get("id") or ""
        for call in nameless_parsed + nameless_raw  # type: ignore[union-attr]
    ]
    update: dict[str, object] = {
        "tool_calls": [call for call in message.tool_calls if call.get("name")],
        "invalid_tool_calls": [
            call
            for call in getattr(message, "invalid_tool_calls", []) or []
            if call.get("name")
        ],
    }
    if raw_calls is not None:
        additional_kwargs = dict(message.additional_kwargs)
        remaining = [call for call in raw_calls if call not in nameless_raw]
        if remaining:
            additional_kwargs["tool_calls"] = remaining
        else:
            additional_kwargs.pop("tool_calls", None)
        update["additional_kwargs"] = additional_kwargs
    return message.model_copy(update=update), [tid for tid in dropped_ids if tid]


def repair_tool_history(
    messages: Sequence[AnyMessage],
    warned: set[str] | None = None,
) -> list[AnyMessage]:
    """Return provider-valid history, preserving every complete tool exchange.

    When ``warned`` is provided, repair warnings are emitted only for tool-call
    ids not already present in it; newly-warned ids are added. This keeps the
    warning to once per unique interrupted/malformed call even though the
    middleware re-runs on every model call.
    """
    repaired: list[AnyMessage] = []
    pending: dict[str, str | None] = {}
    synthesized: list[str] = []
    dropped: list[str] = []
    unnamed: list[str] = []

    def close_pending() -> None:
        for tool_call_id, tool_name in pending.items():
            repaired.append(
                ToolMessage(
                    content=_INTERRUPTED_RESULT,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="error",
                )
            )
            synthesized.append(tool_call_id)
        pending.clear()

    for message in messages:
        if isinstance(message, ToolMessage):
            tool_call_id = message.tool_call_id
            if tool_call_id in pending:
                repaired.append(message)
                pending.pop(tool_call_id)
            else:
                dropped.append(tool_call_id)
            continue

        if pending:
            close_pending()
        if isinstance(message, AIMessage):
            message, nameless_ids = _drop_nameless_calls(message)
            unnamed.extend(nameless_ids)
        repaired.append(message)
        if isinstance(message, AIMessage):
            all_calls = list(message.tool_calls) + list(
                getattr(message, "invalid_tool_calls", []) or []
            )
            for call in all_calls:
                if tool_call_id := call.get("id"):
                    pending[tool_call_id] = call.get("name")

    if pending:
        close_pending()

    if warned is not None:
        synthesized = [tid for tid in synthesized if tid not in warned]
        dropped = [tid for tid in dropped if tid not in warned]
        unnamed = [tid for tid in unnamed if tid not in warned]
        warned.update(synthesized)
        warned.update(dropped)
        warned.update(unnamed)

    if synthesized or dropped or unnamed:
        logger.warning(
            "Repaired interrupted tool history: synthesized=%s dropped=%s unnamed=%s",
            synthesized,
            dropped,
            unnamed,
        )
    return repaired


class ToolHistoryRepairMiddleware(AgentMiddleware):
    """Repair dangling calls and orphan results at the model boundary."""

    name = "tool_history_repair"

    def __init__(self) -> None:
        super().__init__()
        self._warned: set[str] = set()

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        messages = repair_tool_history(request.messages, warned=self._warned)
        return request.override(messages=messages)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self.modify_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self.modify_request(request))
