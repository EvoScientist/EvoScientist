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
        warned.update(synthesized)
        warned.update(dropped)

    if synthesized or dropped:
        logger.warning(
            "Repaired interrupted tool history: synthesized=%s dropped=%s",
            synthesized,
            dropped,
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
