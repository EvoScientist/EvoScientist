"""ErrorNormalizationMiddleware — catch provider-SDK exceptions at the
model boundary and re-raise as a normalized non-dataclass wrapper.

Some provider SDKs (openrouter.errors.* today) decorate their exception
classes with ``@dataclass``. When langgraph_api emits an SSE error
frame via ``json_dumpb`` → ``orjson.dumps(obj, default=default,
option=OPT_SERIALIZE_DATACLASS)``, orjson's dataclass fast-path
enumerates the fields directly and skips the ``default=`` hook that
builds our envelope. The wire payload comes out as
``{"message": …, "status_code": …, "body": …, "headers": null,
"raw_response": null, "data": {…}}`` with no ``error`` / ``class`` /
``provider`` envelope and no way for the WebUI to distinguish quota /
auth / rate-limit / model-not-found.

This middleware sits at the model-call boundary. It catches
``BaseException`` from ``handler()``, and if ``request.model`` is a
recognized provider SDK client, wraps the exception in a
:class:`~EvoScientist.llm.errors.ProviderStreamError` (a plain
``Exception`` subclass, not a dataclass). The wrapper carries the SSE
envelope pre-baked on its instance attributes.

Contract: the wrap decision is based on the **model**, not the
exception. Every exception raised inside a call to a recognized
provider model gets wrapped — SDK exceptions, httpx errors,
langchain-wrapper failures, and even builtins like ``RuntimeError``.
At the middleware boundary we can tell which provider was in use, but
not the exception's precise origin; a uniform envelope is more useful
to the WebUI than gambling on the exception class. If the model isn't
from a recognized provider, or the request carries no ``.model``, the
exception re-raises unchanged and upstream's whitelist / catch-all
behavior takes over.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from ..llm.errors import ProviderStreamError


class ModelOutputTruncatedError(RuntimeError):
    """The provider exhausted its output budget before producing an answer."""


_TRUNCATED_FINISH_REASONS = frozenset(
    {
        "length",
        "max_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "incomplete",
    }
)


def _has_answer_content(content: object) -> bool:
    """Return whether message content contains something beyond reasoning."""
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return content is not None

    reasoning_types = {
        "thinking",
        "redacted_thinking",
        "reasoning",
        "reasoning_content",
    }
    text_types = {"text", "output_text"}
    for block in content:
        if isinstance(block, str):
            if block.strip():
                return True
            continue
        if not isinstance(block, dict):
            return True
        block_type = str(block.get("type", "")).lower()
        if block_type in reasoning_types:
            continue
        if block_type in text_types:
            text = block.get("text")
            if isinstance(text, str):
                if text.strip():
                    return True
            elif text:
                return True
            continue
        # Any non-reasoning block is meaningful output (text, image, refusal,
        # server tool result, etc.), even when its provider-specific payload
        # does not use a ``text`` key.
        return True
    return False


def _truncated_empty_message(response: ModelResponse):
    """Return the empty truncated AI message in *response*, if present."""
    from langchain_core.messages import AIMessage

    if getattr(response, "structured_response", None) is not None:
        return None
    messages = getattr(response, "result", None) or []
    message = next(
        (item for item in reversed(messages) if isinstance(item, AIMessage)), None
    )
    if message is None:
        return None
    if _has_answer_content(message.content):
        return None
    if message.tool_calls or getattr(message, "invalid_tool_calls", None):
        return None

    metadata = message.response_metadata or {}
    reasons = {
        str(metadata.get(key, "")).strip().lower()
        for key in ("finish_reason", "stop_reason", "status")
    }
    incomplete_details = metadata.get("incomplete_details")
    if isinstance(incomplete_details, dict):
        reasons.add(str(incomplete_details.get("reason", "")).strip().lower())
    if reasons.isdisjoint(_TRUNCATED_FINISH_REASONS):
        return None
    return message


def _check_truncated_output(response: ModelResponse) -> ModelResponse:
    """Raise a visible error instead of silently accepting an empty answer."""
    message = _truncated_empty_message(response)
    if message is None:
        return response
    metadata = message.response_metadata or {}
    reason = (
        metadata.get("finish_reason")
        or metadata.get("stop_reason")
        or metadata.get("status")
        or "output limit"
    )
    raise ModelOutputTruncatedError(
        "The model exhausted its output budget during reasoning and returned "
        f"no answer (finish reason: {reason}). Lower reasoning_effort, disable "
        "reasoning with none when supported, or increase the provider "
        "output-token limit."
    )


def _should_pass_through(exc: BaseException) -> bool:
    """True if *exc* is a LangGraph-level signal that must propagate
    untouched — either a control-flow signal or a structural error
    that isn't a provider failure.

    Covers everything in ``langgraph.errors.*``:

    - **Control flow** (breaking these would corrupt the interrupt /
      resume protocol): ``GraphBubbleUp`` and its subclasses
      ``GraphInterrupt``, ``NodeInterrupt``, ``ParentCommand``,
      ``GraphDrained``.
    - **Structural** (wrapping would mis-attribute a graph-level
      issue as a provider failure): ``InvalidUpdateError``,
      ``EmptyInputError``, ``EmptyChannelError``, ``TaskNotFound``,
      ``GraphRecursionError``, ``NodeCancelledError``,
      ``NodeTimeoutError``.

    Symmetric with upstream ``langgraph_api.serde.default``'s
    whitelist, which also exposes these classes' ``str(exc)`` untouched
    rather than swallowing them behind a provider envelope.

    ``KeyboardInterrupt``, ``SystemExit``, and ``asyncio.CancelledError``
    are handled implicitly by catching ``Exception`` — they inherit
    from ``BaseException``.
    """
    return (type(exc).__module__ or "").startswith("langgraph.errors")


# Module prefixes for provider SDK exceptions. Consumed by
# ``_is_provider_error`` to decide whether an exception raised inside
# a model call should surface as a provider incident or gracefully
# degrade (used by ``_ConditionalToolSelectorMiddleware``).
#
# Related sibling: ``_HOST_TO_PROVIDER`` in ``llm/errors.py`` — the
# host-side allow-list. Adding a whole new provider SDK means updating
# both; adding a new routed provider (new base_url through an existing
# SDK) only touches ``_HOST_TO_PROVIDER``.
_PROVIDER_EXC_MODULE_PREFIXES: tuple[str, ...] = (
    "openai",
    "anthropic",
    "google.genai",
    "google.api_core",
    "openrouter",
    "langchain_openai",
    "langchain_deepseek",
    "langchain_anthropic",
    "langchain_google_genai",
    "langchain_openrouter",
    "httpx",
)


def _is_provider_error(exc: BaseException) -> bool:
    """True if *exc* looks like it originated inside a provider SDK
    (openai, anthropic, google.genai, openrouter, httpx, or their
    langchain wrappers), as opposed to a shape / config error (structured
    output not supported, malformed schema, missing tool, …).

    Used by callers that need to decide whether an exception from the
    model call is worth surfacing to the user (provider errors) or
    can be silently degraded around (shape errors). Cheap alternative
    to inspecting ``status_code`` / ``request`` because some provider
    errors — connection errors, timeouts — don't carry those attributes.
    """
    module = type(exc).__module__ or ""
    return any(module.startswith(p) for p in _PROVIDER_EXC_MODULE_PREFIXES)


def _normalize(request: ModelRequest, exc: BaseException) -> ProviderStreamError | None:
    """Return a :class:`ProviderStreamError` wrapping *exc* if the model
    on *request* comes from a recognized provider SDK, or ``None`` if
    the caller should re-raise *exc* unchanged.

    Provider is read from ``request.model`` — the definitive config
    the exception was raised under, not inferred from the exception
    class / URL. Status / code / redaction still come from the raised
    exception because those fields are populated by the SDK at raise
    time.

    Returns ``None`` (caller re-raises unchanged) for:

    - Already-normalized wrappers (would double-attribute).
    - LangGraph control-flow / structural errors — see
      ``_should_pass_through``. This gate lives here so every caller
      of ``_normalize`` (not just the wrap sites of this middleware)
      gets the protection automatically. Notably
      ``ModelFallbackMiddleware`` also calls ``_normalize`` at the
      raise point of its fallback chain.
    - ``ContextOverflowError`` — a cross-layer control signal that
      deepagents' ``SummarizationMiddleware`` catches by type from
      **outside** the user middleware stack to compress history and
      retry. Wrapping it here would change the type and break that
      self-healing fallback.
    - Models we don't recognize as a provider SDK.
    """
    from langchain_core.exceptions import ContextOverflowError

    from ..llm.errors import (
        ProviderStreamError,
        _extract_error_type,
        _extract_provider_code,
        _extract_status_code,
        _provider_from_model,
        _redact_api_keys,
    )

    # Already normalized (e.g. by ModelFallbackMiddleware wrapping against
    # the actual failing model rather than the original request's model).
    # Pass through — re-wrapping would double-attribute.
    if isinstance(exc, ProviderStreamError):
        return None

    # LangGraph control-flow / structural signals must propagate
    # untouched, regardless of which caller invoked us.
    if _should_pass_through(exc):
        return None

    # SummarizationMiddleware sits outside our stack and catches this
    # by exact type to trigger reactive history compression + retry.
    if isinstance(exc, ContextOverflowError):
        return None

    provider = _provider_from_model(getattr(request, "model", None))
    if provider is None:
        return None
    cls = type(exc)
    mod = cls.__module__ or ""
    class_qualname = f"{mod}.{cls.__qualname__}" if mod else cls.__qualname__

    request_id_attr = getattr(exc, "request_id", None)
    request_id = (
        request_id_attr
        if isinstance(request_id_attr, str) and request_id_attr
        else None
    )

    return ProviderStreamError(
        provider=provider,
        class_qualname=class_qualname,
        message=_redact_api_keys(str(exc)),
        status_code=_extract_status_code(exc),
        code=_extract_provider_code(exc),
        err_type=_extract_error_type(exc),
        request_id=request_id,
    )


class ErrorNormalizationMiddleware(AgentMiddleware):
    """Wrap the model call in try/except and normalize provider SDK
    exceptions into a non-dataclass envelope wrapper.

    Place this middleware **outermost** in the chain (first in the
    middleware list) so it catches exceptions raised by inner
    middlewares as well as the model handler itself.
    """

    name = "error_normalization"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        try:
            return _check_truncated_output(handler(request))
        except Exception as exc:
            normalized = _normalize(request, exc)
            if normalized is None:
                raise
            raise normalized from exc

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        try:
            return _check_truncated_output(await handler(request))
        except Exception as exc:
            normalized = _normalize(request, exc)
            if normalized is None:
                raise
            raise normalized from exc
