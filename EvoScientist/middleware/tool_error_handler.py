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
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

# GraphInterrupt must propagate — never catch it as a tool error.
try:
    from langgraph.errors import GraphInterrupt as _GraphInterrupt
except ImportError:  # older langgraph versions
    _GraphInterrupt = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ToolCallRequest

logger = logging.getLogger(__name__)


class ToolErrorHandlerMiddleware(AgentMiddleware):
    """Catch tool execution exceptions and return them as error ToolMessages."""

    name = "tool_error_handler"

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
