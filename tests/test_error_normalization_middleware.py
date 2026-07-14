"""Tests for ErrorNormalizationMiddleware + ProviderStreamError.

Verifies that provider-SDK exceptions from a chat model call get
wrapped into a non-dataclass ``ProviderStreamError`` at the model
boundary, and that non-provider exceptions pass through unchanged.
The provider tag is derived from ``request.model`` (class + base_url),
not from the raised exception.
"""

from __future__ import annotations

import asyncio
import dataclasses
from types import SimpleNamespace

import pytest

from EvoScientist.llm.errors import ProviderStreamError
from EvoScientist.middleware.error_normalization import (
    ErrorNormalizationMiddleware,
    _normalize,
)

# ---------------------------------------------------------------------------
# Test fixtures — fake chat model instances + requests
# ---------------------------------------------------------------------------


def _fake_model(module: str, cls_name: str, **attrs):
    """Build a fake chat model instance whose ``type(model).__module__``
    matches *module*, carrying arbitrary attributes for ``base_url`` /
    ``openai_api_base`` / ``anthropic_api_url`` lookup.
    """
    cls = type(cls_name, (), {"__module__": module})
    inst = cls()
    for k, v in attrs.items():
        setattr(inst, k, v)
    return inst


def _request(model):
    """Fake ``ModelRequest`` with just the ``.model`` attribute the
    middleware reads.
    """
    return SimpleNamespace(model=model)


def _openai_model(base_url: str | None = None):
    return _fake_model(
        "langchain_openai.chat_models.base",
        "ChatOpenAI",
        openai_api_base=base_url,
    )


def _anthropic_model(base_url: str | None = None):
    return _fake_model(
        "langchain_anthropic.chat_models",
        "ChatAnthropic",
        anthropic_api_url=base_url,
    )


def _openrouter_model():
    return _fake_model("langchain_openrouter.chat_models", "ChatOpenRouter")


def _google_model():
    return _fake_model("langchain_google_genai.chat_models", "ChatGoogleGenerativeAI")


def _make_exc(cls_name: str = "APIError", message: str = "boom", **attrs):
    """Build a plain-Exception subclass carrying arbitrary attributes
    (``status_code``, ``code``, ``type``, ``request_id`` …).
    """
    cls = type(cls_name, (Exception,), attrs)
    return cls(message)


# ---------------------------------------------------------------------------
# _normalize — provider inference from ModelRequest.model
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_openai_native_model_tags_openai(self):
        req = _request(_openai_model())
        exc = _make_exc(message="rate limited", status_code=429)
        wrapped = _normalize(req, exc)
        assert isinstance(wrapped, ProviderStreamError)
        assert wrapped.provider == "openai"
        assert wrapped.status_code == 429

    def test_openai_routed_deepseek_tagged_by_base_url(self):
        req = _request(_openai_model(base_url="https://api.deepseek.com"))
        wrapped = _normalize(req, _make_exc(message="quota exceeded"))
        assert wrapped.provider == "deepseek"

    def test_openai_routed_moonshot_tagged_by_base_url(self):
        req = _request(_openai_model(base_url="https://api.moonshot.cn/v1"))
        assert _normalize(req, _make_exc()).provider == "moonshot"

    def test_unknown_openai_compat_host_tagged_openai_compat(self):
        req = _request(_openai_model(base_url="https://internal.corp/v1"))
        assert _normalize(req, _make_exc()).provider == "openai_compat"

    def test_anthropic_native_model_tags_anthropic(self):
        req = _request(_anthropic_model(base_url="https://api.anthropic.com"))
        assert _normalize(req, _make_exc()).provider == "anthropic"

    def test_anthropic_routed_minimax_tagged_by_base_url(self):
        req = _request(_anthropic_model(base_url="https://api.minimaxi.com/anthropic"))
        assert _normalize(req, _make_exc()).provider == "minimax"

    def test_unknown_anthropic_compat_host_tagged_anthropic_compat(self):
        req = _request(_anthropic_model(base_url="https://internal.corp/v1"))
        assert _normalize(req, _make_exc()).provider == "anthropic_compat"

    def test_openrouter_tagged_from_class_alone(self):
        req = _request(_openrouter_model())
        wrapped = _normalize(req, _make_exc(cls_name="UnauthorizedResponseError"))
        assert wrapped.provider == "openrouter"
        assert wrapped.class_qualname.endswith(".UnauthorizedResponseError")

    def test_google_genai_tagged_from_class_alone(self):
        req = _request(_google_model())
        assert _normalize(req, _make_exc()).provider == "google_genai"

    def test_unrecognized_model_class_returns_none(self):
        req = _request(_fake_model("some.other.pkg", "SomeModel"))
        assert _normalize(req, _make_exc()) is None

    def test_missing_model_on_request_returns_none(self):
        """If the request has no ``.model`` at all (defensive)."""
        assert _normalize(SimpleNamespace(), _make_exc()) is None

    def test_already_normalized_exception_passes_through(self):
        """``ModelFallbackMiddleware`` wraps against the failing model
        before re-raising. The outer chain's ``_normalize`` must NOT
        double-wrap — otherwise attribution flips back to the original
        request's model.
        """
        req = _request(_openrouter_model())
        pre_wrapped = ProviderStreamError(
            provider="moonshot",
            class_qualname="openai.RateLimitError",
            message="quota exceeded",
        )
        assert _normalize(req, pre_wrapped) is None


# ---------------------------------------------------------------------------
# _is_provider_error — used by tool selector to distinguish provider
# failures (surface) from shape / config failures (degrade)
# ---------------------------------------------------------------------------


class TestIsProviderError:
    def test_openai_module_is_provider_error(self):
        from EvoScientist.middleware.error_normalization import _is_provider_error

        assert _is_provider_error(_make_exc(__module__="openai"))

    def test_httpx_timeout_is_provider_error(self):
        from EvoScientist.middleware.error_normalization import _is_provider_error

        assert _is_provider_error(
            _make_exc(cls_name="TimeoutException", __module__="httpx")
        )

    def test_langchain_wrapper_module_is_provider_error(self):
        from EvoScientist.middleware.error_normalization import _is_provider_error

        assert _is_provider_error(
            _make_exc(
                cls_name="BadRequestError",
                __module__="langchain_openai.chat_models",
            )
        )

    def test_pydantic_validation_is_not_provider_error(self):
        """Structured-output shape failures come from pydantic /
        langchain, NOT from a provider SDK — the tool selector's
        graceful-degrade path is right for these.
        """
        from EvoScientist.middleware.error_normalization import _is_provider_error

        assert not _is_provider_error(
            _make_exc(cls_name="ValidationError", __module__="pydantic")
        )

    def test_builtin_is_not_provider_error(self):
        from EvoScientist.middleware.error_normalization import _is_provider_error

        assert not _is_provider_error(RuntimeError("x"))


# ---------------------------------------------------------------------------
# ProviderStreamError envelope
# ---------------------------------------------------------------------------


class TestProviderStreamErrorEnvelope:
    def test_envelope_contains_required_fields(self):
        err = ProviderStreamError(
            provider="deepseek",
            class_qualname="openai.RateLimitError",
            message="quota exceeded",
            status_code=429,
            code="insufficient_quota",
        )
        env = err.as_envelope()
        assert env["error"] == "RateLimitError"
        assert env["class"] == "openai.RateLimitError"
        assert env["message"] == "quota exceeded"
        assert env["provider"] == "deepseek"
        assert env["status_code"] == 429
        assert env["code"] == "insufficient_quota"

    def test_envelope_omits_absent_optional_fields(self):
        err = ProviderStreamError(
            provider="openrouter",
            class_qualname="openrouter.errors.foo.UnauthorizedResponseError",
            message="User not found.",
        )
        env = err.as_envelope()
        assert "status_code" not in env
        assert "code" not in env
        assert "type" not in env
        assert "request_id" not in env

    def test_provider_stream_error_is_not_a_dataclass(self):
        """The whole point of the wrapper — must not be a dataclass so
        orjson's OPT_SERIALIZE_DATACLASS fast-path doesn't fire.
        """
        err = ProviderStreamError("x", "y.Z", "msg")
        assert not dataclasses.is_dataclass(err)
        assert not dataclasses.is_dataclass(type(err))

    def test_model_dump_returns_envelope(self):
        """Upstream ``serde.default`` calls ``model_dump()`` before its
        exception branch — the hook that lets us skip the serde patch.
        """
        err = ProviderStreamError(
            provider="openrouter",
            class_qualname="openrouter.errors.foo.UnauthorizedResponseError",
            message="User not found.",
            status_code=401,
        )
        assert err.model_dump() == err.as_envelope()


# ---------------------------------------------------------------------------
# Middleware behavior
# ---------------------------------------------------------------------------


class TestMiddleware:
    def _run_awrap(self, mw, request, handler):
        async def _go():
            return await mw.awrap_model_call(request=request, handler=handler)

        return asyncio.run(_go())

    def test_awrap_normalizes_provider_exception(self):
        raised = _make_exc(cls_name="UnauthorizedResponseError", message="boom")

        async def handler(_req):
            raise raised

        req = _request(_openrouter_model())
        mw = ErrorNormalizationMiddleware()
        with pytest.raises(ProviderStreamError) as excinfo:
            self._run_awrap(mw, req, handler)
        assert excinfo.value.provider == "openrouter"
        assert excinfo.value.__cause__ is raised

    def test_awrap_passes_through_non_provider_model_exception(self):
        """If the model isn't a recognized provider SDK, the exception
        passes through unwrapped — same as any non-model exception.
        """
        raised = _make_exc(message="boom")

        async def handler(_req):
            raise raised

        req = _request(_fake_model("some.other.pkg", "SomeModel"))
        mw = ErrorNormalizationMiddleware()
        with pytest.raises(Exception, match="boom") as excinfo:
            self._run_awrap(mw, req, handler)
        assert excinfo.value is raised

    def _langgraph_error_samples(self):
        """Instances covering both branches of ``_should_pass_through``:
        control-flow (``GraphBubbleUp`` + subclasses) and structural
        errors. Constructor signatures vary — some need positional
        args — so build each explicitly.
        """
        from langgraph.errors import (
            EmptyInputError,
            GraphBubbleUp,
            GraphInterrupt,
            InvalidUpdateError,
            NodeTimeoutError,
            TaskNotFound,
        )

        return [
            GraphBubbleUp(),
            GraphInterrupt(),
            InvalidUpdateError("bad update"),
            EmptyInputError("no input"),
            TaskNotFound(),
            NodeTimeoutError("node-x", 1.5, kind="run", run_timeout=1.0),
        ]

    def test_awrap_passes_through_langgraph_errors(self):
        """Exceptions from ``langgraph.errors.*`` must propagate
        untouched even when the model is a recognized provider —
        they're either control-flow signals (interrupts, HITL) or
        graph-level structural errors, neither is a provider incident.
        """
        req = _request(_openrouter_model())  # recognized — would normally wrap
        mw = ErrorNormalizationMiddleware()

        for raised in self._langgraph_error_samples():

            async def handler(_req, _r=raised):
                raise _r

            with pytest.raises(type(raised)) as excinfo:
                self._run_awrap(mw, req, handler)
            assert excinfo.value is raised, (
                f"{type(raised).__name__} got wrapped instead of propagated"
            )

    def test_awrap_passes_through_context_overflow_error(self):
        """``ContextOverflowError`` is a cross-layer control signal:
        deepagents' ``SummarizationMiddleware`` sits outside our stack
        and catches it by type to compress history and retry. Wrapping
        it here would change the type and break that self-healing
        fallback — regressing to a user-visible ``ProviderStreamError``
        on any long conversation.
        """
        from langchain_core.exceptions import ContextOverflowError

        raised = ContextOverflowError("context length exceeded")

        async def handler(_req):
            raise raised

        req = _request(_openrouter_model())  # recognized — would normally wrap
        mw = ErrorNormalizationMiddleware()
        with pytest.raises(ContextOverflowError) as excinfo:
            self._run_awrap(mw, req, handler)
        assert excinfo.value is raised

    def test_awrap_wraps_any_exception_from_recognized_model(self):
        """Any exception raised inside a call to a provider-recognized
        model gets wrapped — including builtins like ``RuntimeError``.
        Rationale: at the middleware boundary we can tell the model is
        a provider, but not the exception's origin (SDK vs
        langchain-wrapper vs httpx vs our code). Wrapping uniformly
        gives the WebUI a consistent envelope; upstream's
        ``RuntimeError``-whitelist would emit ``{"error":
        "RuntimeError", "message": str(exc)}`` which isn't more
        useful.
        """
        raised = RuntimeError("internal glitch")

        async def handler(_req):
            raise raised

        req = _request(_openai_model())
        mw = ErrorNormalizationMiddleware()
        with pytest.raises(ProviderStreamError) as excinfo:
            self._run_awrap(mw, req, handler)
        assert excinfo.value.provider == "openai"
        assert excinfo.value.__cause__ is raised
        assert excinfo.value.class_qualname == "builtins.RuntimeError"

    def test_sync_wrap_normalizes_provider_exception(self):
        raised = _make_exc(message="boom")

        def handler(_req):
            raise raised

        req = _request(_openrouter_model())
        mw = ErrorNormalizationMiddleware()
        with pytest.raises(ProviderStreamError) as excinfo:
            mw.wrap_model_call(request=req, handler=handler)
        assert excinfo.value.provider == "openrouter"

    def test_success_path_returns_handler_result(self):
        async def handler(_req):
            return "ok"

        req = _request(_openrouter_model())
        mw = ErrorNormalizationMiddleware()
        assert self._run_awrap(mw, req, handler) == "ok"
