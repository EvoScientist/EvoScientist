"""Tests for EvoScientist.middleware.proactive_mode."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from EvoScientist.middleware.proactive_mode import (
    ProactiveModeMiddleware,
    _read_proactive_mode,
    create_proactive_mode_middleware,
)


def _request(tools=None):
    """A minimal ModelRequest stand-in exposing ``tools`` and ``.override``.

    ``override`` mirrors the real immutable contract: it returns a *new*
    object with the given attributes replaced, leaving the original untouched.
    """
    tools = [object(), object()] if tools is None else tools
    request = SimpleNamespace(tools=tools)
    request.override = lambda **kwargs: SimpleNamespace(
        tools=kwargs.get("tools", request.tools)
    )
    return request


# ---- unit tests: _read_proactive_mode behavior ------------------------------


@patch("langgraph.config.get_config")
def test_read_returns_true_when_flag_set(mock_get_config):
    mock_get_config.return_value = {"configurable": {"proactive_mode": True}}
    assert _read_proactive_mode() is True


@patch("langgraph.config.get_config")
def test_read_returns_false_when_configurable_missing(mock_get_config):
    mock_get_config.return_value = {}
    assert _read_proactive_mode() is False


@patch("langgraph.config.get_config")
def test_read_returns_false_when_flag_missing(mock_get_config):
    mock_get_config.return_value = {"configurable": {"other_field": "x"}}
    assert _read_proactive_mode() is False


@patch("langgraph.config.get_config")
def test_read_returns_false_when_flag_falsey(mock_get_config):
    mock_get_config.return_value = {"configurable": {"proactive_mode": False}}
    assert _read_proactive_mode() is False


@patch("langgraph.config.get_config")
def test_read_returns_false_for_non_bool_truthy(mock_get_config):
    """Only the literal boolean ``True`` enables the strip; a truthy string
    (e.g. a mis-serialized WebUI value) must not silently arm it."""
    mock_get_config.return_value = {"configurable": {"proactive_mode": "yes"}}
    assert _read_proactive_mode() is False


@patch("langgraph.config.get_config")
def test_read_returns_false_when_configurable_not_dict(mock_get_config):
    mock_get_config.return_value = {"configurable": "not-a-dict"}
    assert _read_proactive_mode() is False


@patch("langgraph.config.get_config", side_effect=RuntimeError("outside context"))
def test_read_returns_false_outside_runnable_context(mock_get_config):
    assert _read_proactive_mode() is False


# ---- unit tests: middleware behavior ----------------------------------------


@patch("langgraph.config.get_config")
def test_strips_all_tools_when_flag_set(mock_get_config):
    mock_get_config.return_value = {"configurable": {"proactive_mode": True}}
    middleware = ProactiveModeMiddleware()
    modified = middleware._apply(_request())
    assert modified.tools == []


@patch("langgraph.config.get_config")
def test_passthrough_unchanged_when_flag_absent(mock_get_config):
    """Flag absent → the exact same request object is returned (byte-identical
    passthrough), so a normal turn is unaffected by the middleware's presence."""
    mock_get_config.return_value = {"configurable": {}}
    middleware = ProactiveModeMiddleware()
    original = _request()
    tools_before = original.tools
    result = middleware._apply(original)
    assert result is original
    assert result.tools is tools_before


@patch("langgraph.config.get_config")
def test_wrap_model_call_passes_stripped_request_to_handler(mock_get_config):
    mock_get_config.return_value = {"configurable": {"proactive_mode": True}}
    middleware = ProactiveModeMiddleware()
    seen = {}

    def handler(req):
        seen["tools"] = req.tools
        return "response"

    result = middleware.wrap_model_call(_request(), handler)
    assert result == "response"
    assert seen["tools"] == []


@patch("langgraph.config.get_config")
def test_awrap_model_call_passes_stripped_request_to_handler(mock_get_config):
    mock_get_config.return_value = {"configurable": {"proactive_mode": True}}
    middleware = ProactiveModeMiddleware()
    seen = {}

    async def handler(req):
        seen["tools"] = req.tools
        return "response"

    result = asyncio.run(middleware.awrap_model_call(_request(), handler))
    assert result == "response"
    assert seen["tools"] == []


# ---- composition tests: _get_default_middleware -----------------------------


def _mock_config():
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_mode = False
    cfg.auto_approve = False
    cfg.model_fallbacks = None
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    cfg.code_interpreter_timeout = 60
    cfg.code_interpreter_max_result_chars = 6000
    return cfg


@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_default_middleware_includes_proactive_mode_for_main_agent(
    mock_config, mock_model
):
    mock_config.return_value = _mock_config()
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})

    selector = MagicMock(name="tool_selector")
    with patch(
        "EvoScientist.middleware.create_tool_selector_middleware",
        return_value=[selector],
    ):
        from EvoScientist.EvoScientist import _get_default_middleware

        middleware = _get_default_middleware()

    assert any(isinstance(m, ProactiveModeMiddleware) for m in middleware)


@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_default_middleware_excludes_proactive_mode_for_async_subagent(
    mock_config, mock_model
):
    mock_config.return_value = _mock_config()
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})

    with patch(
        "EvoScientist.middleware.create_tool_selector_middleware",
        return_value=[MagicMock()],
    ):
        from EvoScientist.EvoScientist import _get_default_middleware

        middleware = _get_default_middleware(for_async_subagent=True)

    assert not any(isinstance(m, ProactiveModeMiddleware) for m in middleware)


@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_proactive_mode_is_outer_of_tool_selector(mock_config, mock_model):
    """Pins the placement decision: ProactiveMode must sit *before* (outer of)
    tool_selector so the selector skips its LLM call on an empty tool set.
    langchain composes the first middleware in the list as the outermost layer,
    so a lower index == an outer wrap."""
    mock_config.return_value = _mock_config()
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})

    selector = MagicMock(name="tool_selector")
    with patch(
        "EvoScientist.middleware.create_tool_selector_middleware",
        return_value=[selector],
    ):
        from EvoScientist.EvoScientist import _get_default_middleware

        middleware = _get_default_middleware()

    proactive_index = next(
        i for i, m in enumerate(middleware) if isinstance(m, ProactiveModeMiddleware)
    )
    selector_index = middleware.index(selector)
    assert proactive_index < selector_index


# ---- factory ----------------------------------------------------------------


def test_factory_returns_middleware_instance():
    assert isinstance(create_proactive_mode_middleware(), ProactiveModeMiddleware)
