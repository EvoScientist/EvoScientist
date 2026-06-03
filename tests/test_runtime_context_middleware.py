from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import SystemMessage

from EvoScientist.middleware.runtime_context import (
    RuntimeContextMiddleware,
    create_runtime_context_middleware,
)


def _request():
    request = SimpleNamespace(
        state={},
        runtime=object(),
        system_message=SystemMessage(content="base system"),
    )
    request.override = lambda **kwargs: SimpleNamespace(
        **{
            "state": request.state,
            "runtime": request.runtime,
            "system_message": kwargs.get("system_message", request.system_message),
        }
    )
    return request


def _system_text(modified) -> str:
    system_message = modified.system_message
    assert system_message is not None
    return str(system_message.content)


def _mock_config():
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_mode = False
    cfg.auto_approve = False
    cfg.model_fallbacks = None
    cfg.code_interpreter_timeout = 60
    cfg.code_interpreter_max_result_chars = 6000
    return cfg


def test_runtime_context_injects_current_date_and_timezone():
    middleware = RuntimeContextMiddleware(
        now_fn=lambda: datetime(2026, 6, 2, 12, 0, tzinfo=UTC)
    )

    modified = middleware.modify_request(_request())
    system_text = _system_text(modified)

    assert "<runtime_context>" in system_text
    assert "Current date: 2026-06-02" in system_text
    assert "Local timezone: UTC (UTC+00:00)" in system_text
    assert "today" in system_text


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    return_value=[MagicMock(), MagicMock()],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_default_middleware_includes_runtime_context(
    mock_config, mock_model, mock_tool_selector
):
    mock_config.return_value = _mock_config()
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})

    from EvoScientist.EvoScientist import _get_default_middleware

    middleware = _get_default_middleware()

    assert any(isinstance(m, RuntimeContextMiddleware) for m in middleware)


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    return_value=[MagicMock(), MagicMock()],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_async_subagent_middleware_includes_runtime_context(
    mock_config, mock_model, mock_tool_selector
):
    mock_config.return_value = _mock_config()
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})

    from EvoScientist.EvoScientist import _get_default_middleware

    middleware = _get_default_middleware(for_async_subagent=True)

    assert any(isinstance(m, RuntimeContextMiddleware) for m in middleware)


@patch("EvoScientist.EvoScientist._ensure_chat_model")
def test_configured_subagent_middleware_includes_runtime_context(mock_model):
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})

    from EvoScientist.EvoScientist import _inject_subagent_middleware

    subs = [{"name": "test-agent"}]
    _inject_subagent_middleware(subs)

    assert any(isinstance(m, RuntimeContextMiddleware) for m in subs[0]["middleware"])


def test_runtime_context_factory_returns_middleware():
    assert isinstance(create_runtime_context_middleware(), RuntimeContextMiddleware)
