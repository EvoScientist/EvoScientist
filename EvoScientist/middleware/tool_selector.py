"""LLMToolSelectorMiddleware configuration for EvoScientist.

Wraps LangChain's built-in ``LLMToolSelectorMiddleware`` with project-specific
defaults and a tracker that captures which tools were selected.

The selector only activates when the agent has more than ``threshold`` tools
(default 20).  Below that, the extra LLM call isn't worth the token savings.

Usage::

    from EvoScientist.middleware import create_tool_selector_middleware

    middleware = create_tool_selector_middleware()  # returns [selector, tracker]
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)

# Module-level storage for tool selection state.
# Updated by _ToolSelectionTrackerMiddleware; read by stream/events.py.
_current_selected_tools: list[str] = []
_last_emitted_tools: list[str] = []  # last selection shown to user
_total_tools_count: int = 0  # total tools before selection
_selector_active: bool = False

# Default threshold: only run tool selection when tools exceed this count.
# Base tools are ~14; selector activates when MCP tools push count above 20.
DEFAULT_TOOL_THRESHOLD = 20


class _ConditionalToolSelectorMiddleware(AgentMiddleware):
    """Wraps LLMToolSelectorMiddleware with a tool-count threshold.

    Skips the selection LLM call when ``len(request.tools) <= threshold``,
    avoiding unnecessary overhead for agents with few tools.

    Sets ``_selector_active`` flag during the selector's internal LLM call
    so the streaming layer can suppress its output.
    """

    name = "conditional_tool_selector"

    def __init__(
        self, selector: AgentMiddleware, threshold: int = DEFAULT_TOOL_THRESHOLD
    ):
        super().__init__()
        self._selector = selector
        self._threshold = threshold

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if len(request.tools or []) <= self._threshold:
            return handler(request)

        global _selector_active, _total_tools_count
        _selector_active = True
        _total_tools_count = len(request.tools or [])

        # Track whether handler was called — if so, any exception is from
        # the downstream model, not the selector, and must propagate.
        _handler_called = False

        def _handler_after_selection(req: ModelRequest) -> ModelResponse:
            nonlocal _handler_called
            global _selector_active
            _handler_called = True
            _selector_active = False
            return handler(req)

        try:
            return self._selector.wrap_model_call(request, _handler_after_selection)
        except Exception:
            if _handler_called:
                raise  # Error from downstream model — don't retry
            # Selector itself failed (e.g., structured output not supported).
            logger.warning("Tool selector failed, using all tools", exc_info=True)
            _selector_active = False
            return handler(request)
        finally:
            _selector_active = False

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if len(request.tools or []) <= self._threshold:
            return await handler(request)

        global _selector_active, _total_tools_count
        _selector_active = True
        _total_tools_count = len(request.tools or [])

        _handler_called = False

        async def _handler_after_selection(req: ModelRequest) -> ModelResponse:
            nonlocal _handler_called
            global _selector_active
            _handler_called = True
            _selector_active = False
            return await handler(req)

        try:
            return await self._selector.awrap_model_call(
                request, _handler_after_selection
            )
        except Exception:
            if _handler_called:
                raise
            logger.warning("Tool selector failed, using all tools", exc_info=True)
            _selector_active = False
            return await handler(request)
        finally:
            _selector_active = False


class _ToolSelectionTrackerMiddleware(AgentMiddleware):
    """Captures which tools the model actually receives after filtering.

    Sits right AFTER the selector in the middleware chain (more inner),
    so ``request.tools`` already contains only the selected tools when
    this middleware's ``wrap_model_call`` runs.
    """

    name = "tool_selection_tracker"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        global _current_selected_tools
        tools = [t.name for t in request.tools if hasattr(t, "name")]
        _current_selected_tools = tools
        if tools:
            logger.debug("Selected tools: %s", tools)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        global _current_selected_tools
        tools = [t.name for t in request.tools if hasattr(t, "name")]
        _current_selected_tools = tools
        if tools:
            logger.debug("Selected tools: %s", tools)
        return await handler(request)


def create_tool_selector_middleware(threshold: int = DEFAULT_TOOL_THRESHOLD):
    """Build LLMToolSelectorMiddleware + tracker with EvoScientist defaults.

    Returns a list of two middleware:
    1. Conditional wrapper around ``LLMToolSelectorMiddleware`` — only
       activates when ``len(tools) > threshold``
    2. ``_ToolSelectionTrackerMiddleware`` — captures selected tool names

    Args:
        threshold: Minimum number of tools to trigger selection.
            Default 20.  Set to 0 to always run selection.

    ``think_tool`` and ``task`` are always included because:

    - ``think_tool``: required every step for structured reflection
    - ``task``: core delegation mechanism; tested and confirmed the selector
      model never auto-selects it (0/5 complex queries)
    """
    from langchain.agents.middleware import LLMToolSelectorMiddleware

    # Resolve model and strip thinking/reasoning for structured output compat.
    from EvoScientist.EvoScientist import _ensure_chat_model

    from .utils import disable_thinking

    model = _ensure_chat_model()
    safe_model = disable_thinking(model)

    selector = LLMToolSelectorMiddleware(
        model=safe_model,
        system_prompt=(
            "You are selecting tools for a scientific research agent. "
            "Tasks often involve multi-step workflows. "
            "Select tools that cover both the immediate need and "
            "likely follow-up steps. "
            "If the query is broad or all tools seem relevant, "
            "select all of them — filtering is not always necessary."
        ),
        always_include=["think_tool", "task"],
    )

    return [
        _ConditionalToolSelectorMiddleware(selector, threshold=threshold),
        _ToolSelectionTrackerMiddleware(),
    ]
