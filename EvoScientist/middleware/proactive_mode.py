"""ProactiveModeMiddleware: strip all tools on a proactive shadow turn.

A proactive turn (issue #263) reasons over a real, idle chat and emits a
push/no-push *text* decision — it must not execute any tool against the real
sandbox. This middleware reads ``configurable.proactive_mode`` fresh per turn
and, when set, returns ``request.override(tools=[])`` so the model call carries
zero tools. When the flag is absent the request passes through unchanged, so a
normal turn is byte-identical to one without this middleware in the stack.

The flag rides on the ``configurable`` primitive (read via
``langgraph.config.get_config()``), not a server-side thread-state store
(CLAUDE.md #5). Template: ``middleware/active_team.py``, which reads
``configurable.active_teams`` the same way.

Ordering: this middleware is placed *outer* of ``tool_selector`` in
``_get_default_middleware`` (earlier in the list). langchain composes
``wrap_model_call`` so the first middleware in the list is the outermost layer
(``langchain/agents/factory.py``: "first in list becomes outermost layer"), so
an outer ProactiveMode strips the tools *before* ``tool_selector`` runs — the
selector then sees zero tools, skips its LLM selection call
(``middleware/tool_selector.py``: short-circuit when ``len(request.tools) <=
threshold``), and forwards the empty-tools request. No other default-stack
middleware adds tools per request, so the strip is airtight in the composed
stack; the proactive shadow runner additionally wraps the shadow model with a
fail-closed spy that asserts zero bound tools, so a future re-adder is caught
rather than silently leaked.

Empty tools also removes ``ask_user`` (it is a tool), so a proactive turn can
never park on a human-in-the-loop interrupt — no special-casing needed.
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

    Returns ``False`` when the config is absent, malformed, the key is missing
    or non-boolean, or the call happens outside a runnable context (most common
    in tests). Same degradation contract as ``active_team._read_active_teams``.
    """
    try:
        from langgraph.config import get_config

        cfg = get_config()
    except Exception:
        # Outside a runnable context or langgraph not importable — treat as off.
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
        """Return a tools-stripped request when proactive mode is on, else the
        original request unchanged (byte-identical passthrough)."""
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
