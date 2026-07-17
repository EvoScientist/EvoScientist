"""Repair interrupted tool-call history before provider requests."""

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


def repair_tool_history(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Return provider-valid history, preserving every complete tool exchange."""
    repaired: list[AnyMessage] = []
    pending: dict[str, None] = {}
    synthesized: list[str] = []
    dropped: list[str] = []

    def close_pending() -> None:
        for tool_call_id in pending:
            repaired.append(
                ToolMessage(
                    content=_INTERRUPTED_RESULT,
                    tool_call_id=tool_call_id,
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
            for call in message.tool_calls:
                if tool_call_id := call.get("id"):
                    pending[tool_call_id] = None

    if pending:
        close_pending()

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

    @staticmethod
    def modify_request(request: ModelRequest) -> ModelRequest:
        messages = repair_tool_history(request.messages)
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
