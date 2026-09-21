"""Tests for ContextEditingMiddleware integration and compute_context_editing_trigger."""

from unittest.mock import AsyncMock, MagicMock, patch

from langchain.agents.middleware import ContextEditingMiddleware
from langchain_core.messages import HumanMessage

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


def _fake_model_request(model):
    """ModelRequest stub for invoking the middleware's wrap path."""
    request = MagicMock()
    request.model = model
    request.messages = [HumanMessage(content="hi")]
    request.override = MagicMock(side_effect=lambda **kw: request)
    return request


async def test_trigger_resizes_per_run_model():
    """The edit trigger tracks the current run's model, not construction.

    A per-run ``configurable.model`` override swaps ``request.model`` before
    ContextEditingMiddleware runs (ConfigurableModelMiddleware sits earlier),
    so a smaller-window model must get the smaller trigger on that run.
    """
    from EvoScientist.middleware.context_editing import (
        create_context_editing_middleware,
    )

    construction_model = MagicMock()
    construction_model.profile = {"max_input_tokens": 200_000}
    mw = create_context_editing_middleware(construction_model)
    assert mw.edits[0].trigger == 100_000

    small_model = MagicMock()
    small_model.profile = {"max_input_tokens": 32_768}

    request = _fake_model_request(small_model)
    handler = AsyncMock(return_value=MagicMock())
    await mw.awrap_model_call(request, handler)
    handler.assert_awaited_once()
    assert mw.edits[0].trigger == 16_384

    # Switching back to the construction model restores its trigger.
    request = _fake_model_request(construction_model)
    await mw.awrap_model_call(request, AsyncMock(return_value=MagicMock()))
    assert mw.edits[0].trigger == 100_000


async def test_trigger_recompute_is_cached_for_unhashable_models():
    """Real chat models are unhashable pydantic objects; the trigger cache
    must key on the resolved context window, not the model object, and
    alternating runs compute each window's trigger once."""
    from EvoScientist.middleware.context_editing import (
        create_context_editing_middleware,
    )

    class _UnhashableModel:
        """Stands in for real chat models: a profile dict and no hash."""

        def __init__(self, window):
            self.profile = {"max_input_tokens": window}

        def __hash__(self):
            raise TypeError("unhashable type (pydantic model)")

    construction_model = _UnhashableModel(200_000)
    mw = create_context_editing_middleware(construction_model)
    assert mw.edits[0].trigger == 100_000

    small_model = _UnhashableModel(32_768)
    large_model = _UnhashableModel(200_000)

    calls = []
    real = compute_context_editing_trigger

    def _counting(model, *a, **kw):
        calls.append(model)
        return real(model, *a, **kw)

    with patch(
        "EvoScientist.middleware.context_editing.compute_context_editing_trigger",
        side_effect=_counting,
    ):
        # Alternating models defeat the identity short-circuit, so only the
        # per-window cache can keep the compute count at one per window.
        for model in (small_model, large_model, small_model, large_model):
            await mw.awrap_model_call(
                _fake_model_request(model), AsyncMock(return_value=MagicMock())
            )
    assert calls == [small_model, large_model]  # each window computed once
    assert mw.edits[0].trigger == 100_000


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

    # Subclass of langchain's ContextEditingMiddleware (per-run trigger).
    assert any(isinstance(m, ContextEditingMiddleware) for m in subs[0]["middleware"])
    # Per-run model channel reaches sync subagents too (mirrors the main
    # agent's stack so configurable.model swaps the subagent's model).
    from EvoScientist.middleware import ConfigurableModelMiddleware

    assert any(
        isinstance(m, ConfigurableModelMiddleware) for m in subs[0]["middleware"]
    )
    # The model swap must wrap the trigger sync: ConfigurableModelMiddleware
    # comes before the context-editing middleware in the injected list.
    type_names = [type(m).__name__ for m in subs[0]["middleware"]]
    assert type_names.index("ConfigurableModelMiddleware") < type_names.index(
        "_PerRunTriggerContextEditingMiddleware"
    )


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

    ce_idx = type_names.index("_PerRunTriggerContextEditingMiddleware")
    co_idx = type_names.index("ContextOverflowMapperMiddleware")
    assert ce_idx < co_idx, (
        "ContextEditingMiddleware should come before ContextOverflowMapperMiddleware"
    )
