"""Middleware that strips orphaned ToolMessages before the model call.

After summarization replaces old messages with a summary, ToolMessages may
reference ``tool_call_id``s whose originating ``AIMessage`` was evicted.
OpenAI's Responses API validates that every ``function_call_output`` has a
matching ``function_call`` and returns HTTP 400 when the match is missing.

This middleware runs as the innermost ``wrap_model_call`` (closest to the
actual API call) and removes any ToolMessage whose ``tool_call_id`` does not
appear in any preceding ``AIMessage.tool_calls``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

logger = logging.getLogger(__name__)


def _strip_orphaned_tool_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Remove ToolMessages that have no matching AIMessage tool_call.

    Returns a new list with orphaned ToolMessages removed. The original list
    is not mutated.
    """
    known_call_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls or []:
                if tc_id := tc.get("id"):
                    known_call_ids.add(tc_id)
            for tc in msg.invalid_tool_calls or []:
                if tc_id := tc.get("id"):
                    known_call_ids.add(tc_id)

    cleaned: list[AnyMessage] = []
    dropped = 0
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id not in known_call_ids:
            dropped += 1
            logger.warning(
                "Stripping orphaned ToolMessage (tool_call_id=%s, name=%s) — "
                "no matching AIMessage found in the conversation. "
                "This typically happens after context summarization.",
                msg.tool_call_id,
                getattr(msg, "name", None),
            )
            continue
        cleaned.append(msg)

    if dropped:
        logger.warning(
            "Removed %d orphaned ToolMessage(s) from %d total messages",
            dropped,
            len(messages),
        )

    return cleaned


class OrphanedToolMessageMiddleware(AgentMiddleware):
    """Strip orphaned ToolMessages before the model call.

    Prevents OpenAI Responses API HTTP 400 errors caused by
    ``function_call_output`` items that reference evicted
    ``function_call`` items after context summarization.
    """

    name = "orphaned_tool_message_sanitizer"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        cleaned = _strip_orphaned_tool_messages(list(request.messages))
        if len(cleaned) != len(request.messages):
            return handler(request.override(messages=cleaned))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        cleaned = _strip_orphaned_tool_messages(list(request.messages))
        if len(cleaned) != len(request.messages):
            return await handler(request.override(messages=cleaned))
        return await handler(request)
