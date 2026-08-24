"""Tests for ContextEditingMiddleware integration and compute_context_editing_trigger."""

from unittest.mock import MagicMock, patch

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


def test_compute_trigger_with_context_length_attr():
    model = MagicMock(spec=["context_length", "profile"])
    model.context_length = 1_000_000
    model.profile = None
    assert compute_context_editing_trigger(model) == 500_000  # 50%


def test_compute_trigger_with_num_ctx():
    model = MagicMock(spec=["num_ctx", "profile"])
    model.num_ctx = 32_768
    model.profile = None
    assert compute_context_editing_trigger(model) == 16_384  # 50%


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


def test_trigger_is_frozen_at_construction_divergence_pin():
    """PIN (known divergence): the trigger integer is computed once from the
    construction-time model's context window. A per-run
    ``configurable.model`` override (server backend) swaps the chat model
    via ConfigurableModelMiddleware but does NOT resize this trigger — a
    run on a model with a different window keeps the construction model's
    trigger. If this pin ever fails because the trigger became per-run,
    update it consciously and drop the divergence note in
    ``create_context_editing_middleware``'s docstring.
    """
    from EvoScientist.middleware.context_editing import (
        create_context_editing_middleware,
    )

    construction_model = MagicMock()
    construction_model.profile = {"max_input_tokens": 200_000}
    mw = create_context_editing_middleware(construction_model)
    trigger_at_construction = mw.edits[0].trigger
    assert trigger_at_construction == 100_000

    # A "per-run override" to a much smaller-window model: the middleware
    # instance is shared per graph, so its trigger stays frozen.
    override_model = MagicMock()
    override_model.profile = {"max_input_tokens": 32_768}
    assert mw.edits[0].trigger == trigger_at_construction
    assert compute_context_editing_trigger(override_model) == 16_384, (
        "sanity: the override model WOULD compute a different trigger"
    )


# ---------------------------------------------------------------------------
# Middleware list integration tests
# ---------------------------------------------------------------------------


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    return_value=[MagicMock(), MagicMock()],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_default_middleware_includes_context_editing(mock_config, mock_model, mock_ts):
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_approve = False
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    mock_config.return_value = cfg

    from EvoScientist.EvoScientist import _get_default_middleware

    mw = _get_default_middleware()
    # ContextEditingMiddleware is present (its absolute position depends on
    # other leading middlewares like ConfigurableModelMiddleware).
    assert any(isinstance(m, ContextEditingMiddleware) for m in mw)


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
def test_context_editing_before_overflow_mapper(mock_config, mock_model, mock_ts):
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_approve = False
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    mock_config.return_value = cfg

    from EvoScientist.EvoScientist import _get_default_middleware

    mw = _get_default_middleware()
    type_names = [type(m).__name__ for m in mw]

    ce_idx = type_names.index("ContextEditingMiddleware")
    co_idx = type_names.index("ContextOverflowMapperMiddleware")
    assert ce_idx < co_idx, (
        "ContextEditingMiddleware should come before ContextOverflowMapperMiddleware"
    )
