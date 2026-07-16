"""Tests for the model fallback middleware.

Covers error classification (_is_non_fallbackable) and the end-to-end
fallback chain behaviour via _try_fallbacks / _guard_and_fallback.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage

from EvoScientist.middleware.events import NoOpSink
from EvoScientist.middleware.model_fallback import (
    _guard_and_fallback,
    _is_non_fallbackable,
    _try_fallbacks,
    add_fallback,
    clear_fallbacks,
)
from EvoScientist.stream.sink import SessionEventSink

# ── Helpers ──────────────────────────────────────────────────────

# Silent sink for tests that don't assert on the fallback narration.
_SINK = NoOpSink()


def _fake_request():
    """Build a minimal ModelRequest stub with an .override() method."""
    req = MagicMock()
    req.override = MagicMock(side_effect=lambda **kw: req)
    req.messages = [HumanMessage(content="hi")]
    return req


AI_RESPONSE = AIMessage(content="ok")


@pytest.fixture(autouse=True)
def _clean_chain():
    """Ensure a clean fallback chain for every test."""
    clear_fallbacks()
    yield
    clear_fallbacks()


# ═════════════════════════════════════════════════════════════════
# 1. _is_non_fallbackable — error classification
# ═════════════════════════════════════════════════════════════════


class TestIsNonFallbackable:
    """Verify which errors block fallback and which allow it."""

    # ── Context-length errors: must NOT fallback ────────────────

    def test_context_overflow_error_instance(self):
        exc = ContextOverflowError("too long")
        assert _is_non_fallbackable(exc) == "context length exceeded"

    @pytest.mark.parametrize(
        "msg",
        [
            "Error 400: context_length_exceeded",
            "Bad Request: context length exceeded in prompt",
            "400 too many tokens for this model",
            "Bad Request: maximum context length is 128k",
            "Error 400: output too large",
            "400 Bad Request: context_window_exceeded",
            "400: string_too_long",
            "Bad Request: max_tokens_exceeded",
        ],
    )
    def test_context_limit_400_patterns(self, msg):
        assert _is_non_fallbackable(Exception(msg)) == "context length exceeded"

    # ── Malformed request errors: must NOT fallback ─────────────

    @pytest.mark.parametrize(
        "msg",
        [
            "Error 400: invalid_request_error",
            "400 Bad Request: invalid request body",
            "400: malformed JSON in request",
        ],
    )
    def test_malformed_request_400_patterns(self, msg):
        assert (
            _is_non_fallbackable(Exception(msg))
            == "malformed request (client-side error)"
        )

    # ── Auth errors: SHOULD fallback (different provider may work) ──

    @pytest.mark.parametrize(
        "msg",
        [
            "400 Bad Request: invalid_api_key",
            "400: authentication failed",
            "400 Bad Request: permission denied",
        ],
    )
    def test_auth_errors_are_fallbackable(self, msg):
        assert _is_non_fallbackable(Exception(msg)) is None

    # ── Server / transient errors: SHOULD fallback ──────────────

    @pytest.mark.parametrize(
        "msg",
        [
            "Error 500: internal server error",
            "429 Too Many Requests: rate limit exceeded",
            "503 Service Unavailable",
            "Connection timed out",
            "HTTPSConnectionPool: Read timed out",
            "502 Bad Gateway",
            "overloaded_error: the server is temporarily overloaded",
        ],
    )
    def test_server_errors_are_fallbackable(self, msg):
        assert _is_non_fallbackable(Exception(msg)) is None

    # ── Edge: 400 without a known pattern → fallbackable ────────

    def test_400_unknown_pattern_is_fallbackable(self):
        assert _is_non_fallbackable(Exception("400: unknown_field 'foo'")) is None

    # ── Edge: pattern present but no 400 → fallbackable ─────────

    def test_context_pattern_without_400_is_fallbackable(self):
        exc = Exception("context_length_exceeded (warning only)")
        assert _is_non_fallbackable(exc) is None

    def test_malformed_pattern_without_400_is_fallbackable(self):
        exc = Exception("invalid_request_error logged for debugging")
        assert _is_non_fallbackable(exc) is None


# ═════════════════════════════════════════════════════════════════
# 2. _try_fallbacks — chain walk behaviour
# ═════════════════════════════════════════════════════════════════


class TestTryFallbacks:
    """End-to-end tests for the fallback chain traversal."""

    async def test_first_fallback_succeeds(self):
        """When the first fallback model works, return its response."""
        add_fallback("fb-model", "fb-provider")
        req = _fake_request()
        invoke = AsyncMock(return_value=AI_RESPONSE)

        with patch("EvoScientist.llm.models.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            result = await _try_fallbacks(req, invoke, Exception("503 boom"), _SINK)

        assert result is AI_RESPONSE
        invoke.assert_awaited_once()
        mock_gcm.assert_called_once_with(model="fb-model", provider="fb-provider")

    async def test_skips_failing_fallback_tries_next(self):
        """When the first fallback fails, try the second."""
        add_fallback("fb-bad", "prov-a")
        add_fallback("fb-good", "prov-b")
        req = _fake_request()

        call_count = 0

        async def _invoke(r):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 rate limited")
            return AI_RESPONSE

        with patch("EvoScientist.llm.models.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            result = await _try_fallbacks(req, _invoke, Exception("503 boom"), _SINK)

        assert result is AI_RESPONSE
        assert call_count == 2

    async def test_all_fallbacks_exhausted_raises_last(self):
        """When every fallback fails, re-raise the last exception."""
        add_fallback("fb-a", "prov-a")
        add_fallback("fb-b", "prov-b")
        req = _fake_request()

        last_error = Exception("429 from fb-b")

        call_count = 0

        async def _invoke(r):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("500 from fb-a")
            raise last_error

        with patch("EvoScientist.llm.models.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            with pytest.raises(Exception, match="429 from fb-b") as exc_info:
                await _try_fallbacks(req, _invoke, Exception("503 primary"), _SINK)

        assert exc_info.value is last_error

    async def test_non_fallbackable_in_chain_aborts_immediately(self):
        """A non-fallbackable error from a fallback model aborts the chain."""
        add_fallback("fb-a", "prov-a")
        add_fallback("fb-b", "prov-b")  # should never be reached
        req = _fake_request()

        async def _invoke(r):
            raise Exception("400: context_length_exceeded")

        with patch("EvoScientist.llm.models.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            with pytest.raises(Exception, match="context_length_exceeded"):
                await _try_fallbacks(req, _invoke, Exception("503 primary"), _SINK)

        # get_chat_model should only have been called once (for fb-a),
        # fb-b should never be reached.
        assert mock_gcm.call_count == 1

    async def test_exhausted_fallbacks_attribute_to_last_failing_model(self):
        """Regression: when every fallback fails, the raised
        ``ProviderStreamError`` must be attributed to the model that
        ACTUALLY failed last, not the original ``request.model``.
        Prevents a ``deepseek → moonshot`` chain from surfacing as
        ``provider: deepseek`` after moonshot exhausts its quota.
        """
        from EvoScientist.llm.errors import ProviderStreamError

        add_fallback("moonshot-model", "moonshot")
        # Original request's model is openai-shape. Fallback's model
        # will be openai-shape with a moonshot base_url.
        req = _fake_request()

        # ChatOpenAI-shape model instance so ``_provider_from_model``
        # returns a recognized provider.
        def _make_openai_model(base_url=None):
            cls = type(
                "ChatOpenAI",
                (),
                {"__module__": "langchain_openai.chat_models.base"},
            )
            inst = cls()
            inst.openai_api_base = base_url
            return inst

        req.model = _make_openai_model()  # primary
        fallback_model = _make_openai_model(base_url="https://api.moonshot.cn/v1")
        # ``request.override(model=...)`` must return the request with the
        # new model so ``_try_fallbacks`` tracks the failing model.
        req.override = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(model=kw.get("model", req.model))
        )

        async def _invoke(_r):
            raise Exception("429 quota exceeded")

        with patch("EvoScientist.llm.models.get_chat_model") as mock_gcm:
            mock_gcm.return_value = fallback_model
            with pytest.raises(ProviderStreamError) as exc_info:
                await _try_fallbacks(
                    req,
                    _invoke,
                    Exception("openai primary failed"),
                    _SINK,
                )

        # Attribution flipped to moonshot (the failing fallback), not
        # openai (the original request's model).
        assert exc_info.value.provider == "moonshot"
        assert "quota exceeded" in exc_info.value.message

    async def test_langgraph_error_at_fallback_raise_point_passes_through(self):
        """Regression: ``_raise_normalized`` calls ``_normalize``
        directly, so its ``_should_pass_through`` gate must fire even
        without the ``ErrorNormalizationMiddleware`` wrap sites' own
        check. Prevents a ``langgraph.errors.*`` exception hitting the
        fallback chain from being wrapped as a provider incident.
        """
        from langgraph.errors import InvalidUpdateError

        add_fallback("fb-a", "prov-a")
        req = _fake_request()

        # Use a recognized-provider model so ``_provider_from_model``
        # wouldn't short-circuit — the guard has to come from
        # ``_should_pass_through``, not the provider check.
        cls = type(
            "ChatOpenAI", (), {"__module__": "langchain_openai.chat_models.base"}
        )
        model = cls()
        model.openai_api_base = None
        req.model = model
        req.override = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(model=kw.get("model", req.model))
        )

        raised = InvalidUpdateError("state mismatch")

        async def _invoke(_r):
            raise raised

        with patch("EvoScientist.llm.models.get_chat_model") as mock_gcm:
            mock_gcm.return_value = model
            with pytest.raises(InvalidUpdateError) as exc_info:
                await _try_fallbacks(
                    req,
                    _invoke,
                    Exception("primary failed"),
                    _SINK,
                )
        assert exc_info.value is raised


# ═════════════════════════════════════════════════════════════════
# 3. _guard_and_fallback — pre-check before chain walk
# ═════════════════════════════════════════════════════════════════


class TestGuardAndFallback:
    """Verify that non-fallbackable errors are re-raised before trying the chain."""

    async def test_context_overflow_raises_immediately(self):
        add_fallback("fb", "prov")
        req = _fake_request()
        invoke = AsyncMock()

        with pytest.raises(ContextOverflowError):
            await _guard_and_fallback(
                ContextOverflowError("overflow"), req, invoke, _SINK
            )

        invoke.assert_not_awaited()

    async def test_context_overflow_with_provider_model_passes_through_unwrapped(self):
        """Regression: a ``ContextOverflowError`` entering
        ``_guard_and_fallback`` under a recognized-provider model must
        come out unwrapped. Otherwise ``_raise_normalized`` →
        ``_normalize`` would wrap it as a ``ProviderStreamError`` and
        deepagents' ``SummarizationMiddleware`` (which sits outside
        the user middleware stack and catches by exact type) would
        stop compressing history and retrying.
        """
        add_fallback("fb", "prov")
        req = _fake_request()
        # Recognized provider — without the gate in ``_normalize`` this
        # would wrap. With the gate, the raw type propagates.
        cls = type(
            "ChatOpenAI", (), {"__module__": "langchain_openai.chat_models.base"}
        )
        model = cls()
        model.openai_api_base = None
        req.model = model
        invoke = AsyncMock()

        raised = ContextOverflowError("context length exceeded")
        with pytest.raises(ContextOverflowError) as exc_info:
            await _guard_and_fallback(raised, req, invoke, _SINK)

        assert exc_info.value is raised
        invoke.assert_not_awaited()

    async def test_malformed_400_raises_immediately(self):
        add_fallback("fb", "prov")
        req = _fake_request()
        invoke = AsyncMock()

        with pytest.raises(Exception, match="invalid_request_error"):
            await _guard_and_fallback(
                Exception("400: invalid_request_error"), req, invoke, _SINK
            )

        invoke.assert_not_awaited()

    async def test_server_error_proceeds_to_fallback(self):
        add_fallback("fb", "prov")
        req = _fake_request()
        invoke = AsyncMock(return_value=AI_RESPONSE)

        with patch("EvoScientist.llm.models.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            result = await _guard_and_fallback(
                Exception("503 overloaded"), req, invoke, _SINK
            )

        assert result is AI_RESPONSE
        invoke.assert_awaited_once()

    async def test_auth_error_proceeds_to_fallback(self):
        """Auth errors should try the fallback chain (different provider)."""
        add_fallback("fb", "other-prov")
        req = _fake_request()
        invoke = AsyncMock(return_value=AI_RESPONSE)

        with patch("EvoScientist.llm.models.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            result = await _guard_and_fallback(
                Exception("400 Bad Request: invalid_api_key"), req, invoke, _SINK
            )

        assert result is AI_RESPONSE
        invoke.assert_awaited_once()


# ═════════════════════════════════════════════════════════════════
# 4. UI emit callback
# ═════════════════════════════════════════════════════════════════


class TestUiEmit:
    """Verify that fallback narration reaches the injected frontend sink.

    The fallback middleware sends its narration lines through the same
    ``fallback_display`` callback the frontend supplies, so capturing that
    callback exercises the exact user-facing text.
    """

    def _capturing_sink(self):
        messages: list[tuple[str, str]] = []
        sink = SessionEventSink(
            fallback_display=lambda text, style: messages.append((text, style))
        )
        return sink, messages

    async def test_emit_captures_messages(self):
        add_fallback("fb", "prov")
        req = _fake_request()
        invoke = AsyncMock(return_value=AI_RESPONSE)
        sink, messages = self._capturing_sink()

        with patch("EvoScientist.llm.models.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            await _try_fallbacks(req, invoke, Exception("503 down"), sink)

        texts = [t for t, _ in messages]
        assert any("Primary model failed" in t for t in texts)
        assert any("Falling back to fb (prov)" in t for t in texts)
        assert any("succeeded" in t for t in texts)

    async def test_default_sink_prints_to_console(self):
        add_fallback("fb", "prov")
        req = _fake_request()
        invoke = AsyncMock(return_value=AI_RESPONSE)
        sink = SessionEventSink()

        with (
            patch("EvoScientist.llm.models.get_chat_model") as mock_gcm,
            patch("EvoScientist.stream.sink.console.print") as mock_print,
        ):
            mock_gcm.return_value = MagicMock()
            await _try_fallbacks(req, invoke, Exception("503 down"), sink)

        texts = [call.args[0] for call in mock_print.call_args_list]
        assert any("Primary model failed" in t for t in texts)
        assert any("Falling back to fb (prov)" in t for t in texts)
        assert any("succeeded" in t for t in texts)
        assert all(
            call.kwargs == {"style": "yellow"} for call in mock_print.call_args_list[:2]
        )

    async def test_display_failure_does_not_abort_fallback(self, caplog):
        add_fallback("fb", "prov")
        req = _fake_request()
        invoke = AsyncMock(return_value=AI_RESPONSE)
        sink = SessionEventSink(
            fallback_display=MagicMock(side_effect=RuntimeError("ui unavailable"))
        )

        with (
            patch("EvoScientist.llm.models.get_chat_model") as mock_gcm,
            patch("EvoScientist.stream.sink.console.print") as mock_print,
        ):
            mock_gcm.return_value = MagicMock()
            result = await _try_fallbacks(req, invoke, Exception("503 down"), sink)

        assert result is AI_RESPONSE
        invoke.assert_awaited_once()
        assert "Fallback display callback failed" in caplog.text
        texts = [call.args[0] for call in mock_print.call_args_list]
        assert any("Primary model failed" in t for t in texts)
        assert any("Falling back to fb (prov)" in t for t in texts)
        assert any("succeeded" in t for t in texts)

    def test_noopsink_keeps_fallback_notices_silent(self):
        sink = NoOpSink()

        with patch("EvoScientist.stream.sink.console.print") as mock_print:
            sink.emit_fallback_notice("hidden")

        mock_print.assert_not_called()

    async def test_emit_shows_non_fallbackable_rejection(self):
        add_fallback("fb", "prov")
        req = _fake_request()
        invoke = AsyncMock()
        sink, messages = self._capturing_sink()

        with pytest.raises(ContextOverflowError):
            await _guard_and_fallback(
                ContextOverflowError("overflow"), req, invoke, sink
            )

        texts = [t for t, _ in messages]
        assert any("not eligible for fallback" in t for t in texts)
