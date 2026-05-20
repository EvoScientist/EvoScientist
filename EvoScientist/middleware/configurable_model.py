"""Middleware that resolves the chat model from RunnableConfig.configurable per call.

The deployed async sub-agents run in a separate ``langgraph dev`` subprocess
and have their model frozen into the compiled graph at subprocess boot time
(see ``EvoScientist/subagents/_factory.py``). When the user runs ``/model``
in the CLI, only the CLI process's model state changes — the subprocess
graph still uses the boot-time model.

This middleware closes that gap by reading ``model`` / ``model_provider``
from ``RunnableConfig.configurable`` on every model call. The CLI's patched
``start_async_task`` / ``update_async_task`` (see ``llm/patches.py``) injects
those fields into ``client.runs.create(config=...)``; the deployed graph
hits this middleware and re-resolves the chat model fresh.

When ``configurable.model`` is absent, the middleware is a pass-through —
safe to install on the CLI's in-process agent too.

The middleware mirrors the pattern used by ``ModelFallbackMiddleware``:
``request.override(model=new_model)`` does not break tool binding, because
the downstream model-invocation node re-binds tools per request.

**Reading the config**: ``Runtime`` (per its own docstring) does NOT include
``config``. The official path to reach ``RunnableConfig`` from inside any
runnable context (including middleware) is ``langgraph.config.get_config()``,
which reads a context-var that LangGraph populates per node execution. That
is what this middleware uses. ``request.runtime`` is intentionally NOT
relied upon for config — only for diagnostics.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)


def _read_model_override() -> tuple[str | None, str | None]:
    """Pull ``(model, model_provider)`` from the active ``RunnableConfig``.

    Reads via ``langgraph.config.get_config()`` (the documented entry point
    for accessing the per-run ``RunnableConfig`` from inside any runnable
    context — middleware, node, tool). Returns ``(None, None)`` when the
    config has no ``configurable.model`` override or when called outside a
    runnable context.
    """
    try:
        from langgraph.config import get_config

        cfg = get_config()
    except Exception:
        # Outside a runnable context (most common in tests) or
        # langgraph not importable — nothing to override.
        return None, None
    if not isinstance(cfg, dict):
        return None, None
    configurable = cfg.get("configurable") or {}
    if not isinstance(configurable, dict):
        return None, None
    model = configurable.get("model")
    provider = configurable.get("model_provider")
    return (
        model if isinstance(model, str) and model else None,
        provider if isinstance(provider, str) and provider else None,
    )


class ConfigurableModelMiddleware(AgentMiddleware):
    """Re-resolve the chat model from RunnableConfig.configurable on every call.

    Reads ``model`` and ``model_provider`` from the active ``RunnableConfig``
    via ``langgraph.config.get_config()`` — the documented entry point for
    accessing per-run config from any runnable context (middleware, node, tool).
    When the override is present, calls
    ``EvoScientist.llm.get_chat_model(model=..., provider=...)`` and replaces
    ``request.model`` via ``request.override``. When absent, the middleware
    passes through unchanged.

    Note: ``Runtime`` (per its own docstring) does NOT include ``config`` as a
    field — an earlier version of this middleware tried to read
    ``request.runtime.config`` and silently no-op'd because that attribute does
    not exist. Stick with ``get_config()``.

    A per-instance cache keyed by ``(model, provider)`` avoids rebuilding
    identical models within a turn. The cache is a plain dict guarded by a
    ``threading.Lock`` because middleware instances are shared across
    concurrent requests in long-lived deployments (e.g. ``langgraph dev``
    workers).
    """

    name = "configurable_model"

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict[tuple[str, str | None], Any] = {}
        self._lock = threading.Lock()
        # Track the last (model, provider) pair we INFO-logged so we only
        # surface a banner on transition. Without this, every LLM call in a
        # long async run would emit an identical INFO line.
        self._last_logged_key: tuple[str, str | None] | None = None

    def _log_override(self, model_name: str, provider: str | None) -> None:
        """INFO on transition; DEBUG on subsequent calls with same key."""
        key = (model_name, provider)
        with self._lock:
            transitioned = key != self._last_logged_key
            if transitioned:
                self._last_logged_key = key
        if transitioned:
            logger.info(
                "ConfigurableModelMiddleware: overriding model to %s (%s)",
                model_name,
                provider,
            )
        else:
            logger.debug(
                "ConfigurableModelMiddleware: reusing override model=%s provider=%s",
                model_name,
                provider,
            )

    def _resolve(self, model: str, provider: str | None) -> Any:
        """Return a cached or freshly-built chat model for ``(model, provider)``."""
        key = (model, provider)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        # Build outside the lock (network/SDK init can be slow); two
        # concurrent first-time misses for the same key may build twice but
        # the second result simply overwrites the first — both are equivalent.
        from ..llm import get_chat_model

        new_model = get_chat_model(model=model, provider=provider)
        with self._lock:
            self._cache[key] = new_model
        return new_model

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        model_name, provider = _read_model_override()
        if model_name is None:
            return handler(request)
        try:
            new_model = self._resolve(model_name, provider)
        except Exception:
            logger.warning(
                "ConfigurableModelMiddleware failed to resolve model=%r "
                "provider=%r; falling back to compile-time model",
                model_name,
                provider,
                exc_info=True,
            )
            return handler(request)
        self._log_override(model_name, provider)
        return handler(request.override(model=new_model))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        model_name, provider = _read_model_override()
        if model_name is None:
            return await handler(request)
        try:
            # Offload first-call SDK init off the event loop. ``_resolve`` calls
            # ``get_chat_model`` on a cache miss, which can spend hundreds of ms
            # building HTTP clients. Doing this synchronously inside an
            # ``async def`` would block every other coroutine on the same
            # langgraph dev event loop. Cache hits are still fast (a dict
            # lookup); the thread-pool overhead is irrelevant once warm.
            new_model = await asyncio.to_thread(self._resolve, model_name, provider)
        except Exception:
            logger.warning(
                "ConfigurableModelMiddleware failed to resolve model=%r "
                "provider=%r; falling back to compile-time model",
                model_name,
                provider,
                exc_info=True,
            )
            return await handler(request)
        self._log_override(model_name, provider)
        return await handler(request.override(model=new_model))
