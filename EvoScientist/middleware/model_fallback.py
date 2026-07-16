"""Middleware that implements model fallback on LLM call failures.

Uses LangChain's AgentMiddleware to intercept model calls.  When the primary
model raises an exception, the middleware walks the configured fallback chain,
trying each alternative model in order.  Every fallback attempt and its
outcome is reported to the injected event sink as fallback narration, and the
frontend sink renders it.

Errors that indicate a client-side bug (malformed request / HTTP 400) or a
context-length breach are not eligible for fallback and are re-raised
immediately so the correct handler (user or ContextOverflowMapperMiddleware)
can deal with them.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from .events import MiddlewareEventSink

logger = logging.getLogger(__name__)

_fallback_chain_lock = threading.Lock()
_fallback_chain: list[tuple[str, str]] = []
"""Ordered list of ``(model_name, provider)`` fallback entries."""

_CONTEXT_LIMIT_PATTERNS: list[str] = [
    "context_length_exceeded",
    "context length exceeded",
    "too many tokens",
    "maximum context length",
    "output too large",
    "context_window_exceeded",
    "string_too_long",
    "max_tokens_exceeded",
]
"""Substrings that identify a context-length error in provider messages."""

_MALFORMED_REQUEST_PATTERNS: list[str] = [
    "invalid_request_error",
    "invalid request",
    "malformed",
]
"""Substrings that identify a malformed request (client-side bug)."""

_AUTH_ERROR_PATTERNS: list[str] = [
    "invalid_api_key",
    "authentication",
    "permission",
]
"""Substrings that identify auth/permission errors.

These are intentionally *not* treated as non-fallbackable because a different
provider in the chain may have valid credentials."""


def get_fallback_chain() -> list[tuple[str, str]]:
    """Return a snapshot of the current fallback chain.

    Returns:
        List of ``(model_name, provider)`` tuples in priority order.
    """
    with _fallback_chain_lock:
        return list(_fallback_chain)


def set_fallback_chain(chain: list[tuple[str, str]]) -> None:
    """Replace the entire fallback chain.

    Args:
        chain: New list of ``(model_name, provider)`` tuples.
    """
    global _fallback_chain
    with _fallback_chain_lock:
        _fallback_chain = list(chain)


def add_fallback(model: str, provider: str) -> bool:
    """Append a model to the end of the fallback chain.

    Args:
        model: Short model name (e.g. ``"gpt-5.5"``).
        provider: Provider identifier (e.g. ``"openai"``).

    Returns:
        ``True`` if added, ``False`` if the entry was already present.
    """
    entry = (model, provider)
    with _fallback_chain_lock:
        if entry in _fallback_chain:
            return False
        _fallback_chain.append(entry)
        return True


def remove_fallback(model: str) -> bool:
    """Remove all entries matching *model* regardless of provider.

    Args:
        model: Short model name to remove.

    Returns:
        ``True`` if at least one entry was removed.
    """
    global _fallback_chain
    with _fallback_chain_lock:
        before = len(_fallback_chain)
        _fallback_chain = [(m, p) for m, p in _fallback_chain if m != model]
        return len(_fallback_chain) < before


def remove_fallback_at(index: int) -> tuple[str, str] | None:
    """Remove the entry at a 0-based index.

    Args:
        index: Position in the chain (0-based).

    Returns:
        The removed ``(model, provider)`` tuple, or ``None`` if out of range.
    """
    with _fallback_chain_lock:
        if 0 <= index < len(_fallback_chain):
            return _fallback_chain.pop(index)
        return None


def clear_fallbacks() -> None:
    """Remove every entry from the fallback chain."""
    global _fallback_chain
    with _fallback_chain_lock:
        _fallback_chain = []


def serialize_fallback_chain() -> str:
    """Serialize the chain to a config-friendly string.

    Returns:
        Comma-separated ``"model:provider,model:provider"`` string.
    """
    with _fallback_chain_lock:
        return ",".join(f"{m}:{p}" for m, p in _fallback_chain)


def load_fallback_chain(raw: str) -> None:
    """Populate the chain from a serialized config string.

    Args:
        raw: Comma-separated ``"model:provider"`` pairs.  Empty or
            whitespace-only segments are silently skipped.
    """
    global _fallback_chain
    chain: list[tuple[str, str]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            model, provider = part.rsplit(":", 1)
            chain.append((model.strip(), provider.strip()))
    with _fallback_chain_lock:
        _fallback_chain = chain


def _is_non_fallbackable(exc: Exception) -> str | None:
    """Determine whether an exception should bypass the fallback chain.

    Args:
        exc: The exception raised by a model call.

    Returns:
        A human-readable reason string if the error must *not* trigger
        fallback, or ``None`` if fallback should proceed.
    """
    from langchain_core.exceptions import ContextOverflowError

    if isinstance(exc, ContextOverflowError):
        return "context length exceeded"

    err_msg = str(exc).lower()
    is_400 = "400" in err_msg or "bad request" in err_msg

    if is_400 and any(p in err_msg for p in _CONTEXT_LIMIT_PATTERNS):
        return "context length exceeded"

    if is_400 and any(p in err_msg for p in _MALFORMED_REQUEST_PATTERNS):
        return "malformed request (client-side error)"

    return None


async def _try_fallbacks(
    request: ModelRequest,
    invoke: Callable[[ModelRequest], Awaitable[ModelResponse]],
    primary_exc: Exception,
    events: MiddlewareEventSink,
) -> ModelResponse:
    """Walk the fallback chain, trying each model until one succeeds.

    Shared implementation for both sync and async middleware entry points.
    The *invoke* callable is an async function that calls the handler with
    a given request — the sync path wraps the synchronous handler in a
    trivial coroutine so both paths converge here.

    Args:
        request: The original model request.
        invoke: Async callable that invokes the handler on a request.
        primary_exc: The exception raised by the primary model.
        events: Injected event sink for fallback narration.

    Returns:
        The ``ModelResponse`` from the first successful fallback.

    Raises:
        Exception: Re-raises the last exception if all fallbacks fail.
    """
    from ..llm.models import get_chat_model

    events.emit_fallback_notice(
        f"Primary model failed: {type(primary_exc).__name__}: {primary_exc}",
        "yellow",
    )
    logger.warning(
        "Primary model failed: %s: %s", type(primary_exc).__name__, primary_exc
    )

    # Track the request whose model actually raised ``last_exc`` so we
    # can attribute the exception to the failing model, not the
    # original ``request.model``. Without this, a fallback chain
    # ``deepseek → moonshot`` where moonshot exhausts its quota would
    # surface as ``provider: deepseek`` — the model the user never
    # actually saw fail.
    last_exc = primary_exc
    last_failing_request = request

    for model_name, provider in get_fallback_chain():
        events.emit_fallback_notice(
            f"  -> Falling back to {model_name} ({provider}) due to: "
            f"{type(last_exc).__name__}: {last_exc}",
            "yellow",
        )
        try:
            fallback_model = get_chat_model(model=model_name, provider=provider)
            fb_request = request.override(model=fallback_model)
            result = await invoke(fb_request)
            events.emit_fallback_notice(
                f"  Fallback to {model_name} ({provider}) succeeded",
                "green",
            )
            logger.info("Fallback to %s (%s) succeeded", model_name, provider)
            return result
        except Exception as fb_exc:
            reason = _is_non_fallbackable(fb_exc)
            if reason is not None:
                events.emit_fallback_notice(
                    f"  {model_name} hit non-fallbackable error ({reason}) "
                    f"-- aborting fallback chain",
                    "red",
                )
                _raise_normalized(fb_request, fb_exc)
            last_exc = fb_exc
            last_failing_request = fb_request
            events.emit_fallback_notice(
                f"  x {model_name} also failed: {type(fb_exc).__name__}: {fb_exc}",
                "red",
            )
            logger.warning(
                "Fallback %s (provider=%s) failed: %s: %s",
                model_name,
                provider,
                type(fb_exc).__name__,
                fb_exc,
            )

    events.emit_fallback_notice(
        "  All fallbacks exhausted -- re-raising last error", "red"
    )
    _raise_normalized(last_failing_request, last_exc)


def _raise_normalized(request: ModelRequest, exc: Exception) -> None:
    """Wrap *exc* in a ``ProviderStreamError`` attributed to
    ``request.model`` and raise, so the outer chain sees the failure
    tagged with the model that actually raised.

    Falls back to a plain ``raise`` when the model isn't from a
    recognized provider (``_normalize`` returns None) — nothing useful
    to add.
    """
    from .error_normalization import _normalize

    normalized = _normalize(request, exc)
    if normalized is not None:
        raise normalized from exc
    raise exc


def _guard_and_fallback(
    primary_exc: Exception,
    request: ModelRequest,
    invoke: Callable[[ModelRequest], Awaitable[ModelResponse]],
    events: MiddlewareEventSink,
) -> Awaitable[ModelResponse]:
    """Check non-fallbackable conditions, then delegate to ``_try_fallbacks``.

    Args:
        primary_exc: The exception raised by the primary model.
        request: The original model request.
        invoke: Async callable that invokes the handler on a request.
        events: Injected event sink for fallback narration.

    Returns:
        Coroutine that resolves to the fallback ``ModelResponse``.

    Raises:
        Exception: Re-raises immediately for non-fallbackable errors.
    """
    reason = _is_non_fallbackable(primary_exc)
    if reason is not None:
        events.emit_fallback_notice(
            f"Model error ({reason}) -- not eligible for fallback, re-raising",
            "red",
        )
        _raise_normalized(request, primary_exc)
    return _try_fallbacks(request, invoke, primary_exc, events)


class ModelFallbackMiddleware(AgentMiddleware):
    """LangChain AgentMiddleware that retries failed model calls on fallbacks.

    On each invocation the middleware reads the module-level
    ``_fallback_chain`` so that ``/model-fallback add`` takes effect
    immediately without rebuilding the agent.

    Attributes:
        name: Middleware identifier used by the framework.
    """

    name = "model_fallback"

    def __init__(self, events: MiddlewareEventSink | None = None) -> None:
        super().__init__()
        from .events import NO_OP_SINK

        self._events = events or NO_OP_SINK

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if not _fallback_chain:
            return handler(request)
        try:
            return handler(request)
        except Exception as exc:

            async def _sync_invoke(r: ModelRequest) -> ModelResponse:
                return handler(r)

            import asyncio

            return asyncio.run(
                _guard_and_fallback(exc, request, _sync_invoke, self._events)
            )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if not _fallback_chain:
            return await handler(request)
        try:
            return await handler(request)
        except Exception as exc:
            return await _guard_and_fallback(exc, request, handler, self._events)
