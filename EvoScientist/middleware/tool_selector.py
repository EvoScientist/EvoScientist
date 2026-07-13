"""LLMToolSelectorMiddleware configuration for EvoScientist.

Wraps LangChain's built-in ``LLMToolSelectorMiddleware`` with project-specific
defaults. The wrapper reports what it did through an injected
:class:`~EvoScientist.middleware.events.MiddlewareEventSink`; the frontend sink
owns any display state (there are no process-global variables here).

The selector only activates when the agent has more than ``threshold`` tools
(default 26). Below that, the extra LLM call isn't worth the token savings.

Usage::

    from EvoScientist.middleware import create_tool_selector_middleware

    middleware = create_tool_selector_middleware(events=sink)
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AIMessage,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .events import NO_OP_SINK, MiddlewareEventSink

logger = logging.getLogger(__name__)

# Default threshold: only run tool selection when tools exceed this count.
# Base tools are ~14; selector activates when MCP tools push count above 26.
DEFAULT_TOOL_THRESHOLD = 26
DEFAULT_ALWAYS_INCLUDE_TOOLS: frozenset[str] = frozenset(
    {
        "think_tool",
        "task",
        "read_memory",
        "record_observation",
        "search_observations",
    }
)


def _tool_name(tool: BaseTool | dict[str, Any]) -> str | None:
    if isinstance(tool, BaseTool):
        return tool.name or None
    name = tool.get("name")
    return name if isinstance(name, str) and name else None


def _available_always_include(
    tools: Iterable[BaseTool | dict[str, Any]],
    candidates: frozenset[str],
) -> list[str]:
    """Return mandatory BaseTool names that exist on this request."""
    available_names = {tn for tool in tools if (tn := _tool_name(tool))}
    return sorted(candidates & available_names)


class _ConditionalToolSelectorMiddleware(AgentMiddleware):
    """Wraps LLMToolSelectorMiddleware with a tool-count threshold.

    Skips the selection LLM call when ``len(request.tools) <= threshold``,
    avoiding unnecessary overhead for agents with few tools.

    When selection runs, reports the lifecycle to the injected sink:
    ``on_tool_selection_started`` before the selector call, ``on_tool_selection``
    with the surviving tools once the selector hands off the filtered request,
    and ``on_tool_selection_ended`` when the call finishes (or fails). The sink
    (a frontend one, or :class:`NoOpSink` for subagent / headless stacks) owns
    all display state.
    """

    name = "conditional_tool_selector"

    def __init__(
        self,
        selector_factory: Callable[[list[str]], AgentMiddleware],
        threshold: int = DEFAULT_TOOL_THRESHOLD,
        *,
        always_include: frozenset[str] | None = None,
        events: MiddlewareEventSink | None = None,
    ):
        super().__init__()
        self._selector_factory = selector_factory
        self._threshold = threshold
        self._always_include = always_include or frozenset()
        self._events = events or NO_OP_SINK
        # Agent tools are fixed after graph construction, so the filtered
        # always-include set is stable for this middleware instance.
        self._selector: AgentMiddleware | None = None

    def _build_selector(self, request: ModelRequest) -> AgentMiddleware:
        if self._selector is None:
            names = _available_always_include(request.tools, self._always_include)
            self._selector = self._selector_factory(names)
        return self._selector

    @staticmethod
    def _selected_names(request: ModelRequest) -> list[str]:
        return [name for tool in request.tools if (name := _tool_name(tool))]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage | ExtendedModelResponse:
        if len(request.tools) <= self._threshold:
            return handler(request)

        total = len(request.tools)
        self._events.on_tool_selection_started(total)

        # Track whether handler was called — if so, any exception is from
        # the downstream model, not the selector, and must propagate.
        _handler_called = False
        _selection_open = True

        def _end_selection() -> None:
            nonlocal _selection_open
            if _selection_open:
                self._events.on_tool_selection_ended()
                _selection_open = False

        def _handler_after_selection(req: ModelRequest) -> ModelResponse:
            nonlocal _handler_called
            _handler_called = True
            # ``req.tools`` is the selector-filtered set here.
            selected = self._selected_names(req)
            self._events.on_tool_selection(selected, total)
            if selected:
                logger.debug("Selected tools: %s", selected)
            _end_selection()
            return handler(req)

        try:
            return self._build_selector(request).wrap_model_call(
                request, _handler_after_selection
            )
        except Exception as exc:
            if _handler_called:
                raise  # Error from downstream model — don't retry
            from ..llm.errors import ProviderStreamError
            from .error_normalization import _is_provider_error

            if isinstance(exc, ProviderStreamError) or _is_provider_error(exc):
                # Auth / quota / connection failures on the selector's
                # own model. Falling back to "use all tools" would hit
                # the same provider anyway (same client, likely same
                # credentials). Surface it instead so the user sees
                # the real cause.
                raise
            # Structured-output shape / config failure — gracefully
            # degrade to using all tools.
            logger.debug("Tool selector failed, using all tools", exc_info=True)
            _end_selection()
            return handler(request)
        finally:
            _end_selection()

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage | ExtendedModelResponse:
        if len(request.tools) <= self._threshold:
            return await handler(request)

        total = len(request.tools)
        self._events.on_tool_selection_started(total)

        _handler_called = False
        _selection_open = True

        def _end_selection() -> None:
            nonlocal _selection_open
            if _selection_open:
                self._events.on_tool_selection_ended()
                _selection_open = False

        async def _handler_after_selection(req: ModelRequest) -> ModelResponse:
            nonlocal _handler_called
            _handler_called = True
            selected = self._selected_names(req)
            self._events.on_tool_selection(selected, total)
            if selected:
                logger.debug("Selected tools: %s", selected)
            _end_selection()
            return await handler(req)

        try:
            return await self._build_selector(request).awrap_model_call(
                request, _handler_after_selection
            )
        except Exception as exc:
            if _handler_called:
                raise
            from ..llm.errors import ProviderStreamError
            from .error_normalization import _is_provider_error

            if isinstance(exc, ProviderStreamError) or _is_provider_error(exc):
                # See sync path — surface provider errors, degrade only
                # on shape / config failures.
                raise
            logger.debug("Tool selector failed, using all tools", exc_info=True)
            _end_selection()
            return await handler(request)
        finally:
            _end_selection()


def create_tool_selector_middleware(
    threshold: int = DEFAULT_TOOL_THRESHOLD,
    *,
    model: BaseChatModel | None = None,
    events: MiddlewareEventSink | None = None,
):
    """Build the conditional ``LLMToolSelectorMiddleware`` wrapper.

    Returns a single-element middleware list (kept as a list so the assembly
    site can splat it) that adaptively selects tools only when
    ``len(tools) > threshold``. The wrapper reports the selection lifecycle to
    ``events``; pass a frontend sink for the main agent, or omit it (subagent /
    headless stacks) to get the silent :class:`NoOpSink`.

    Args:
        model: Chat model for tool selection.  If *None*, the default
            model is resolved via ``_ensure_chat_model()``.
        threshold: Minimum number of tools to trigger selection.
            Default 26.  Set to 0 to always run selection.
        events: Frontend event sink to report selection to.

    ``think_tool``, ``task``, and memory tools are always included because:

    - ``think_tool``: required every step for structured reflection
    - ``task``: core delegation mechanism; tested and confirmed the selector
      model never auto-selects it (0/5 complex queries)
    - memory tools: referenced by memory prompts; filtering them makes the
      agent unable to use memory even when the prompt tells it to
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

    return [
        _ConditionalToolSelectorMiddleware(
            selector_factory=selector_factory,
            threshold=threshold,
            always_include=DEFAULT_ALWAYS_INCLUDE_TOOLS,
            events=events,
        ),
    ]
