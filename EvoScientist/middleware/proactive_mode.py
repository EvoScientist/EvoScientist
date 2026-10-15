"""ProactiveModeMiddleware: strip all tools on a proactive shadow turn.

A proactive turn (#263) reasons over a real, idle conversation and returns a
push or no-push *text* decision; it must not run any tool against the real
sandbox. This middleware reads ``configurable.proactive_mode`` per model call
and, when it is ``True``, passes ``request.override(tools=[])`` on, so the
model call carries no tools. Every other request passes through unchanged.

The flag rides on the run's ``configurable`` and is read with
``langgraph.config.get_config()``, like ``active_teams`` in
``middleware/active_team.py``: a per-run signal belongs on the run's config,
not on thread state that outlives the run.

Ordering: ``_get_default_middleware`` puts this middleware first in its list.
langchain composes ``wrap_model_call`` so an earlier middleware wraps a later
one, so the tools are stripped before ``tool_selector`` runs; the selector
then sees no tools and skips its own model call, and memory and the rest of
the list see the empty tool list too. deepagents places its own core
middleware (skills, filesystem, sub-agents, summarization, tool-call
patching) outside that list. Of those, only summarization calls a model, to
write a summary once the history crosses its trigger; that call binds no
tools. The agent's own model call therefore carries no tools.

Empty tools also removes ``ask_user`` (it is a tool), so a proactive turn can
never stop on a human-in-the-loop interrupt.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)


def _read_proactive_mode() -> bool:
    """Read ``configurable.proactive_mode`` from the current RunnableConfig.

    ``False`` outside a runnable context, or when the config is malformed or
    the key is missing or not ``True``.
    """
    try:
        from langgraph.config import get_config

        cfg = get_config()
    except Exception:
        # Outside a runnable context (most common in tests): not a proactive turn.
        return False
    if not isinstance(cfg, dict):
        return False
    configurable = cfg.get("configurable") or {}
    if not isinstance(configurable, dict):
        return False
    return configurable.get("proactive_mode") is True


class ProactiveModeMiddleware(AgentMiddleware):
    """Strip all tools from the model request on a proactive turn."""

    name = "proactive_mode"

    def _apply(self, request: ModelRequest) -> ModelRequest:
        if _read_proactive_mode():
            return request.override(tools=[])
        return request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._apply(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._apply(request))


def create_proactive_mode_middleware() -> ProactiveModeMiddleware:
    """Build ProactiveModeMiddleware."""
    return ProactiveModeMiddleware()
