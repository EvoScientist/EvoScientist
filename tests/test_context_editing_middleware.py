"""Tests for ContextEditingMiddleware integration and compute_context_editing_trigger."""

from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware import ContextEditingMiddleware

from EvoScientist.middleware.context_editing import compute_context_editing_trigger


# ---------------------------------------------------------------------------
# compute_context_editing_trigger tests
# ---------------------------------------------------------------------------


def test_compute_trigger_with_profile():
    model = MagicMock()
    model.profile = {"max_input_tokens": 200_000}
    assert compute_context_editing_trigger(model) == 100_000  # 50%


def test_compute_trigger_with_1m_profile():
    model = MagicMock()
    model.profile = {"max_input_tokens": 1_000_000}
    assert compute_context_editing_trigger(model) == 500_000  # 50%


def test_compute_trigger_without_profile():
    model = MagicMock()
    model.profile = None
    assert compute_context_editing_trigger(model) == 100_000  # fallback


def test_compute_trigger_no_profile_attr():
    model = MagicMock(spec=[])  # no attributes at all
    assert compute_context_editing_trigger(model) == 100_000  # fallback


def test_compute_trigger_empty_profile():
    model = MagicMock()
    model.profile = {}
    assert compute_context_editing_trigger(model) == 100_000  # fallback


def test_compute_trigger_custom_fraction():
    model = MagicMock()
    model.profile = {"max_input_tokens": 200_000}
    assert compute_context_editing_trigger(model, fraction=0.30) == 60_000


def test_compute_trigger_custom_fallback():
    model = MagicMock()
    model.profile = None
    assert compute_context_editing_trigger(model, fallback=50_000) == 50_000


# ---------------------------------------------------------------------------
# create_context_editing_middleware tests
# ---------------------------------------------------------------------------


def test_create_middleware_configuration():
    from EvoScientist.middleware.context_editing import (
        create_context_editing_middleware,
    )

    model = MagicMock()
    model.profile = {"max_input_tokens": 200_000}
    mw = create_context_editing_middleware(model)
    edit = mw.edits[0]
    assert edit.trigger == 100_000
    assert edit.keep == 5
    assert "think_tool" in edit.exclude_tools


@patch("EvoScientist.EvoScientist._ensure_chat_model")
def test_create_middleware_model_none_fallback(mock_model):
    from EvoScientist.middleware.context_editing import (
        create_context_editing_middleware,
    )

    mock_model.return_value = MagicMock(profile=None)
    mw = create_context_editing_middleware(None)
    edit = mw.edits[0]
    assert edit.trigger == 100_000  # fallback
    mock_model.assert_called_once()


# ---------------------------------------------------------------------------
# Middleware list integration tests
# ---------------------------------------------------------------------------


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    return_value=[MagicMock(), MagicMock()],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_default_middleware_includes_context_editing(mock_config, mock_model, _):
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_approve = False
    mock_config.return_value = cfg

    from EvoScientist.EvoScientist import _get_default_middleware

    mw = _get_default_middleware()
    assert isinstance(mw[0], ContextEditingMiddleware)


@patch("EvoScientist.EvoScientist._ensure_chat_model")
def test_inject_subagent_includes_context_editing(mock_model):
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})

    from EvoScientist.EvoScientist import _inject_subagent_middleware

    subs = [{"name": "test-agent"}]
    _inject_subagent_middleware(subs)

    middleware_types = [type(m) for m in subs[0]["middleware"]]
    assert ContextEditingMiddleware in middleware_types


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    return_value=[MagicMock(), MagicMock()],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_context_editing_before_overflow_mapper(mock_config, mock_model, _):
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_approve = False
    mock_config.return_value = cfg

    from EvoScientist.middleware import ContextOverflowMapperMiddleware
    from EvoScientist.EvoScientist import _get_default_middleware

    mw = _get_default_middleware()
    type_names = [type(m).__name__ for m in mw]

    ce_idx = type_names.index("ContextEditingMiddleware")
    co_idx = type_names.index("ContextOverflowMapperMiddleware")
    assert ce_idx < co_idx, (
        "ContextEditingMiddleware should come before ContextOverflowMapperMiddleware"
    )
