"""LLMToolSelectorMiddleware configuration for EvoScientist.

Wraps LangChain's built-in ``LLMToolSelectorMiddleware`` with project-specific
defaults and an optional stream tracker that captures which tools were selected.

The selector only activates when the agent has more than ``threshold`` tools
(default 20).  Below that, the extra LLM call isn't worth the token savings.

Usage::

    from EvoScientist.middleware import create_tool_selector_middleware

    middleware = create_tool_selector_middleware()  # returns [selector, tracker]
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypeAlias

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AIMessage,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# Module-level storage for main-agent tool-selection UI state.
# Updated only when stream tracking is enabled; read by stream/events.py.
_current_selected_tools: list[str] = []
_last_emitted_tools: list[str] = []  # last selection shown to user
_total_tools_count: int = 0  # total tools before selection
_selector_active: bool = False

# Default threshold: only run tool selection when tools exceed this count.
# Base tools are ~14; selector activates when MCP tools push count above 26.
DEFAULT_TOOL_THRESHOLD = 26
DEFAULT_ALWAYS_INCLUDE_TOOLS: frozenset[str] = frozenset(
    {
        "think_tool",
        "task",
        "read_memory",
        "record_observation",
        "search_memory",
    }
)


ToolDefinition: TypeAlias = BaseTool | dict[str, Any]
_ModelCallResult: TypeAlias = (
    ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]
)


def _available_always_include(
    tools: Iterable[ToolDefinition] | None,
    candidates: frozenset[str],
) -> list[str]:
    """Return mandatory tool names that exist as LangChain tools on this request."""
    available_names = {
        tool.name for tool in tools or [] if isinstance(tool, BaseTool) and tool.name
    }
    return sorted(candidates & available_names)


_SelectorFactory = Callable[[list[str]], AgentMiddleware]


class _ConditionalToolSelectorMiddleware(AgentMiddleware):
    """Wraps LLMToolSelectorMiddleware with a tool-count threshold.

    Skips the selection LLM call when ``len(request.tools) <= threshold``,
    avoiding unnecessary overhead for agents with few tools.

    When stream tracking is enabled, sets ``_selector_active`` during the
    selector's internal LLM call so the streaming layer can suppress its output.
    """

    name = "conditional_tool_selector"

    def __init__(
        self,
        selector_factory: _SelectorFactory,
        threshold: int = DEFAULT_TOOL_THRESHOLD,
        *,
        always_include: frozenset[str] | None = None,
        track_stream_selection: bool = True,
    ):
        super().__init__()
        self._selector_factory = selector_factory
        self._threshold = threshold
        self._always_include = always_include or frozenset()
        self._track_stream_selection = track_stream_selection
        # An agent's tools are fixed when its graph is compiled, so the resolved
        # always-include set is invariant across requests for this instance.
        self._selector_cache: dict[tuple[str, ...], AgentMiddleware] = {}

    def _selector_for_request(self, request: ModelRequest) -> AgentMiddleware:
        names = tuple(_available_always_include(request.tools, self._always_include))
        selector = self._selector_cache.get(names)
        if selector is None:
            selector = self._selector_factory(list(names))
            self._selector_cache[names] = selector
        return selector

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> _ModelCallResult:
        if len(request.tools or []) <= self._threshold:
            return handler(request)

        if self._track_stream_selection:
            global _selector_active, _total_tools_count
            _selector_active = True
            _total_tools_count = len(request.tools or [])

        # Track whether handler was called — if so, any exception is from
        # the downstream model, not the selector, and must propagate.
        _handler_called = False

        def _handler_after_selection(req: ModelRequest) -> ModelResponse:
            nonlocal _handler_called
            _handler_called = True
            if self._track_stream_selection:
                global _selector_active
                _selector_active = False
            return handler(req)

        try:
            return self._selector_for_request(request).wrap_model_call(
                request, _handler_after_selection
            )
        except Exception:
            if _handler_called:
                raise  # Error from downstream model — don't retry
            # Selector itself failed (e.g., structured output not supported).
            logger.debug("Tool selector failed, using all tools", exc_info=True)
            if self._track_stream_selection:
                _selector_active = False
            return handler(request)
        finally:
            if self._track_stream_selection:
                _selector_active = False

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> _ModelCallResult:
        if len(request.tools or []) <= self._threshold:
            return await handler(request)

        if self._track_stream_selection:
            global _selector_active, _total_tools_count
            _selector_active = True
            _total_tools_count = len(request.tools or [])

        _handler_called = False

        async def _handler_after_selection(req: ModelRequest) -> ModelResponse:
            nonlocal _handler_called
            _handler_called = True
            if self._track_stream_selection:
                global _selector_active
                _selector_active = False
            return await handler(req)

        try:
            return await self._selector_for_request(request).awrap_model_call(
                request, _handler_after_selection
            )
        except Exception:
            if _handler_called:
                raise
            logger.debug("Tool selector failed, using all tools", exc_info=True)
            if self._track_stream_selection:
                _selector_active = False
            return await handler(request)
        finally:
            if self._track_stream_selection:
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
        tools = [tool.name for tool in request.tools if isinstance(tool, BaseTool)]
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
        tools = [tool.name for tool in request.tools if isinstance(tool, BaseTool)]
        _current_selected_tools = tools
        if tools:
            logger.debug("Selected tools: %s", tools)
        return await handler(request)


def create_tool_selector_middleware(
    threshold: int = DEFAULT_TOOL_THRESHOLD,
    *,
    model: BaseChatModel | None = None,
    track_stream_selection: bool = True,
):
    """Build LLMToolSelectorMiddleware + tracker with EvoScientist defaults.

    Returns middleware for adaptive tool selection:
    1. Conditional wrapper around ``LLMToolSelectorMiddleware`` — only
       activates when ``len(tools) > threshold``
    2. Optional ``_ToolSelectionTrackerMiddleware`` — captures selected tool
       names for the main-agent stream UI when ``track_stream_selection`` is true

    Args:
        model: Chat model for tool selection.  If *None*, the default
            model is resolved via ``_ensure_chat_model()``.
        threshold: Minimum number of tools to trigger selection.
            Default 20.  Set to 0 to always run selection.
        track_stream_selection: Whether to update process-global stream/UI
            state. Disable for async sub-agents that should still select tools
            but should not drive the main-agent tool-selection widget.

    ``think_tool``, ``task``, and memory tools are always included when present
    on the request because:

    - ``think_tool``: required every step for structured reflection
    - ``task``: core delegation mechanism; tested and confirmed the selector
      model never auto-selects it (0/5 complex queries)
    - memory tools: referenced by memory prompts; filtering them makes the
      agent unable to use memory even when the prompt tells it to

    The list is filtered per request because LangChain validates
    ``always_include`` names against the tools available on that request, and
    memory tools are absent in configurations where memory is disabled.
    """
    from langchain.agents.middleware import LLMToolSelectorMiddleware

    from .utils import disable_thinking

    if model is None:
        from EvoScientist.EvoScientist import _ensure_chat_model

        model = _ensure_chat_model()
    safe_model = disable_thinking(model)

    system_prompt = (
        "You are selecting tools for a scientific research agent. "
        "Tasks often involve multi-step workflows. "
        "Select tools that cover both the immediate need and "
        "likely follow-up steps. "
        "If the query is broad or all tools seem relevant, "
        "select all of them — filtering is not always necessary."
    )

    def selector_factory(always_include: list[str]) -> AgentMiddleware:
        return LLMToolSelectorMiddleware(
            model=safe_model,
            system_prompt=system_prompt,
            always_include=always_include,
        )

    middleware: list[AgentMiddleware] = [
        _ConditionalToolSelectorMiddleware(
            selector_factory=selector_factory,
            threshold=threshold,
            always_include=DEFAULT_ALWAYS_INCLUDE_TOOLS,
            track_stream_selection=track_stream_selection,
        ),
    ]
    if track_stream_selection:
        middleware.append(_ToolSelectionTrackerMiddleware())
    return middleware


def reset_tool_selection_state_for_tests() -> None:
    """Reset the process-global tool-selection state.

    The selector/tracker record the last selected tools and the selector-active
    flag in module globals that ``stream/tool_selection.py`` reads to suppress
    selector chatter. Tests that drive the selector must not leak that state
    into later tests; an autouse fixture resets it around every test.
    """
    global _current_selected_tools, _last_emitted_tools
    global _total_tools_count, _selector_active

    _current_selected_tools = []
    _last_emitted_tools = []
    _total_tools_count = 0
    _selector_active = False
