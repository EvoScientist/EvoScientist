"""Tests for ContextOverflowMapperMiddleware."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import HumanMessage

from EvoScientist.middleware.context_overflow import ContextOverflowMapperMiddleware


def test_is_context_limit_error_openai():
    mw = ContextOverflowMapperMiddleware()
    exc = Exception(
        "Error code: 400 - {'error': {'message': 'This model's maximum context length is 8192 tokens. However, your messages resulted in 10000 tokens.', 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}"
    )
    assert mw._is_context_limit_error(exc) is True


def test_is_context_limit_error_anthropic():
    mw = ContextOverflowMapperMiddleware()
    exc = Exception("HTTP 400 Bad Request: Output too large")
    assert mw._is_context_limit_error(exc) is True


def test_is_not_context_limit_error_without_400():
    mw = ContextOverflowMapperMiddleware()
    exc = Exception("context_length_exceeded, but no status code")
    assert mw._is_context_limit_error(exc) is False


def test_is_not_context_limit_error_with_400_but_no_pattern():
    mw = ContextOverflowMapperMiddleware()
    exc = Exception("HTTP 400 Bad Request: Some other API error")
    assert mw._is_context_limit_error(exc) is False


def test_is_not_context_limit_error_other_status():
    mw = ContextOverflowMapperMiddleware()
    exc = Exception("HTTP 401 Unauthorized")
    assert mw._is_context_limit_error(exc) is False


def test_wrap_model_call_raises_context_overflow():
    # Setup mocks
    msgs = [HumanMessage(content=f"msg {i}") for i in range(10)]
    request = ModelRequest(
        messages=msgs,
        model=MagicMock(),
        state={},
        runtime=MagicMock(),
        system_message=MagicMock(),
    )

    # Mock handler that fails with context error
    handler = MagicMock()
    handler.side_effect = Exception("400 Bad Request: context_length_exceeded")

    mw = ContextOverflowMapperMiddleware()

    with pytest.raises(ContextOverflowError) as excinfo:
        mw.wrap_model_call(request, handler)

    assert "context_length_exceeded" in str(excinfo.value)
    assert handler.call_count == 1


@pytest.mark.anyio
async def test_awrap_model_call_raises_context_overflow():
    # Setup mocks
    msgs = [HumanMessage(content=f"msg {i}") for i in range(10)]
    request = ModelRequest(
        messages=msgs,
        model=MagicMock(),
        state={},
        runtime=MagicMock(),
        system_message=MagicMock(),
    )

    # Mock handler that fails with context error
    handler = AsyncMock()
    handler.side_effect = Exception("400 Bad Request: context_length_exceeded")

    mw = ContextOverflowMapperMiddleware()

    with pytest.raises(ContextOverflowError) as excinfo:
        await mw.awrap_model_call(request, handler)

    assert "context_length_exceeded" in str(excinfo.value)
    assert handler.call_count == 1


@pytest.mark.anyio
async def test_awrap_model_call_passes_through_other_errors():
    request = ModelRequest(
        messages=[],
        model=MagicMock(),
        state={},
        runtime=MagicMock(),
        system_message=MagicMock(),
    )
    handler = AsyncMock()
    handler.side_effect = RuntimeError("Something else")

    mw = ContextOverflowMapperMiddleware()

    with pytest.raises(RuntimeError) as excinfo:
        await mw.awrap_model_call(request, handler)

    assert "Something else" in str(excinfo.value)
    assert not isinstance(excinfo.value, ContextOverflowError)
