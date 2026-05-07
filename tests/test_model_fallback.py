"""Tests for the model fallback middleware.

Covers error classification (_is_non_fallbackable) and the end-to-end
fallback chain behaviour via _try_fallbacks / _guard_and_fallback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage

from EvoScientist.middleware.model_fallback import (
    _guard_and_fallback,
    _is_non_fallbackable,
    _try_fallbacks,
    add_fallback,
    clear_fallbacks,
    set_ui_emit,
)
from tests.conftest import run_async as _run

# ── Helpers ──────────────────────────────────────────────────────


def _fake_request():
    """Build a minimal ModelRequest stub with an .override() method."""
    req = MagicMock()
    req.override = MagicMock(side_effect=lambda **kw: req)
    req.messages = [HumanMessage(content="hi")]
    return req


AI_RESPONSE = AIMessage(content="ok")


@pytest.fixture(autouse=True)
def _clean_chain():
    """Ensure a clean fallback chain and no UI callback for every test."""
    clear_fallbacks()
    set_ui_emit(None)
    yield
    clear_fallbacks()
    set_ui_emit(None)


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

    def test_first_fallback_succeeds(self):
        """When the first fallback model works, return its response."""
        add_fallback("fb-model", "fb-provider")
        req = _fake_request()
        invoke = AsyncMock(return_value=AI_RESPONSE)

        with patch("EvoScientist.llm.models.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            result = _run(_try_fallbacks(req, invoke, Exception("503 boom")))

        assert result is AI_RESPONSE
        invoke.assert_awaited_once()
        mock_gcm.assert_called_once_with(model="fb-model", provider="fb-provider")

    def test_skips_failing_fallback_tries_next(self):
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
            result = _run(_try_fallbacks(req, _invoke, Exception("503 boom")))

        assert result is AI_RESPONSE
        assert call_count == 2

    def test_all_fallbacks_exhausted_raises_last(self):
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
                _run(_try_fallbacks(req, _invoke, Exception("503 primary")))

        assert exc_info.value is last_error

    def test_non_fallbackable_in_chain_aborts_immediately(self):
        """A non-fallbackable error from a fallback model aborts the chain."""
        add_fallback("fb-a", "prov-a")
        add_fallback("fb-b", "prov-b")  # should never be reached
        req = _fake_request()

        async def _invoke(r):
            raise Exception("400: context_length_exceeded")

        with patch("EvoScientist.llm.models.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            with pytest.raises(Exception, match="context_length_exceeded"):
                _run(_try_fallbacks(req, _invoke, Exception("503 primary")))

        # get_chat_model should only have been called once (for fb-a),
        # fb-b should never be reached.
        assert mock_gcm.call_count == 1


# ═════════════════════════════════════════════════════════════════
# 3. _guard_and_fallback — pre-check before chain walk
# ═════════════════════════════════════════════════════════════════


class TestGuardAndFallback:
    """Verify that non-fallbackable errors are re-raised before trying the chain."""

    def test_context_overflow_raises_immediately(self):
        add_fallback("fb", "prov")
        req = _fake_request()
        invoke = AsyncMock()

        with pytest.raises(ContextOverflowError):
            _run(_guard_and_fallback(ContextOverflowError("overflow"), req, invoke))

        invoke.assert_not_awaited()

    def test_malformed_400_raises_immediately(self):
        add_fallback("fb", "prov")
        req = _fake_request()
        invoke = AsyncMock()

        with pytest.raises(Exception, match="invalid_request_error"):
            _run(
                _guard_and_fallback(
                    Exception("400: invalid_request_error"), req, invoke
                )
            )

        invoke.assert_not_awaited()

    def test_server_error_proceeds_to_fallback(self):
        add_fallback("fb", "prov")
        req = _fake_request()
        invoke = AsyncMock(return_value=AI_RESPONSE)

        with patch("EvoScientist.llm.models.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            result = _run(_guard_and_fallback(Exception("503 overloaded"), req, invoke))

        assert result is AI_RESPONSE
        invoke.assert_awaited_once()

    def test_auth_error_proceeds_to_fallback(self):
        """Auth errors should try the fallback chain (different provider)."""
        add_fallback("fb", "other-prov")
        req = _fake_request()
        invoke = AsyncMock(return_value=AI_RESPONSE)

        with patch("EvoScientist.llm.models.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            result = _run(
                _guard_and_fallback(
                    Exception("400 Bad Request: invalid_api_key"), req, invoke
                )
            )

        assert result is AI_RESPONSE
        invoke.assert_awaited_once()


# ═════════════════════════════════════════════════════════════════
# 4. UI emit callback
# ═════════════════════════════════════════════════════════════════


class TestUiEmit:
    """Verify that fallback events are surfaced via the registered callback."""

    def test_emit_captures_messages(self):
        add_fallback("fb", "prov")
        req = _fake_request()
        invoke = AsyncMock(return_value=AI_RESPONSE)

        messages: list[tuple[str, str]] = []
        set_ui_emit(lambda text, style: messages.append((text, style)))

        with patch("EvoScientist.llm.models.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            _run(_try_fallbacks(req, invoke, Exception("503 down")))

        texts = [t for t, _ in messages]
        assert any("Primary model failed" in t for t in texts)
        assert any("Falling back to fb (prov)" in t for t in texts)
        assert any("succeeded" in t for t in texts)

    def test_emit_shows_non_fallbackable_rejection(self):
        add_fallback("fb", "prov")
        req = _fake_request()
        invoke = AsyncMock()

        messages: list[tuple[str, str]] = []
        set_ui_emit(lambda text, style: messages.append((text, style)))

        with pytest.raises(ContextOverflowError):
            _run(_guard_and_fallback(ContextOverflowError("overflow"), req, invoke))

        texts = [t for t, _ in messages]
        assert any("not eligible for fallback" in t for t in texts)
