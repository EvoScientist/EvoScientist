"""Tests for per-run context limits in the summarization middleware (#466).

``_PerRunLimitsSummarizationMiddleware`` replaces deepagents' built-in
``SummarizationMiddleware`` (frozen on the construction model's context
window) so the summarization trigger tracks the run's
``configurable.model`` override instead. These tests reuse the harness
patterns from ``tests/test_context_editing_middleware.py`` and
``tests/test_configurable_model_middleware.py``.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from EvoScientist.middleware.summarization import (
    _PerRunLimitsSummarizationMiddleware,
    create_per_run_summarization_middleware,
)

CONSTRUCTION_WINDOW = 200_000


@contextmanager
def _patched_config(configurable: dict | None):
    """Patch ``langgraph.config.get_config`` to return a controlled value."""
    import langgraph.config as _lg_cfg

    if configurable is None:
        with patch.object(
            _lg_cfg,
            "get_config",
            side_effect=RuntimeError("Called get_config outside of a runnable context"),
        ):
            yield
    else:
        with patch.object(
            _lg_cfg,
            "get_config",
            return_value={"configurable": configurable},
        ):
            yield


def _fake_model_request(model, messages=None):
    """ModelRequest stub for invoking the middleware's wrap path.

    ``request.override(**kw)`` returns a fresh stub with the keyword applied
    (mirroring the real dataclass), so assertions can inspect what the inner
    handler actually received.
    """
    if messages is None:
        messages = [HumanMessage(content="hi")]

    def _build(**overrides):
        new = MagicMock()
        new.model = overrides.get("model", model)
        new.messages = overrides.get("messages", messages)
        new.system_message = overrides.get("system_message", None)
        new.tools = overrides.get("tools", [])
        new.model_settings = overrides.get("model_settings", {})
        new.state = overrides.get("state", {})
        new.override = MagicMock(side_effect=_build)
        return new

    return _build()


def _make_mw(window=CONSTRUCTION_WINDOW):
    construction = MagicMock()
    construction.profile = {"max_input_tokens": window}
    mw = create_per_run_summarization_middleware(construction, MagicMock())
    return mw, construction


class _UnhashableModel:
    """Stands in for real chat models: a profile dict and no hash."""

    _llm_type = "fake"

    def __init__(self, window):
        self.profile = {"max_input_tokens": window}

    def with_retry(self):
        return self

    def __hash__(self):
        raise TypeError("unhashable type (pydantic model)")


# ---------------------------------------------------------------------------
# Per-run limit tracking
# ---------------------------------------------------------------------------


async def test_limits_follow_per_run_model():
    """Trigger limits track the run's model, not the construction model."""
    mw, construction = _make_mw()
    # Built with the construction window.
    assert mw._get_profile_limits() == CONSTRUCTION_WINDOW

    async def _run():
        await mw.awrap_model_call(
            _fake_model_request(construction), AsyncMock(return_value=MagicMock())
        )

    small = MagicMock()
    small.profile = {"max_input_tokens": 32_768}
    with (
        _patched_config({"model": "small-model"}),
        patch("EvoScientist.llm.get_chat_model", return_value=small),
    ):
        await _run()
    assert mw._get_profile_limits() == 32_768
    # The delegated langchain helper sees the same window (its model was
    # re-pointed at a profile shim), so fraction clauses resolve per run.
    assert mw._lc_helper._get_profile_limits() == 32_768
    assert mw._lc_helper.model.profile == {"max_input_tokens": 32_768}

    large = MagicMock()
    large.profile = {"max_input_tokens": 1_000_000}
    with (
        _patched_config({"model": "large-model"}),
        patch("EvoScientist.llm.get_chat_model", return_value=large),
    ):
        await _run()
    assert mw._get_profile_limits() == 1_000_000
    assert mw._lc_helper._get_profile_limits() == 1_000_000

    # Override gone -> construction limits restored.
    with _patched_config({}):
        await _run()
    assert mw._get_profile_limits() == CONSTRUCTION_WINDOW
    assert mw._lc_helper._get_profile_limits() == CONSTRUCTION_WINDOW


def test_sync_wrap_path_syncs_limits_too():
    mw, construction = _make_mw()
    small = MagicMock()
    small.profile = {"max_input_tokens": 32_768}
    with (
        _patched_config({"model": "small-model"}),
        patch("EvoScientist.llm.get_chat_model", return_value=small),
    ):
        mw.wrap_model_call(
            _fake_model_request(construction), MagicMock(return_value=MagicMock())
        )
    assert mw._get_profile_limits() == 32_768


def test_shim_delegates_non_profile_attributes():
    """The profile shim falls through to the real model for everything else
    (langchain provider-matches reported tokens via ``_get_ls_params``)."""
    from EvoScientist.middleware.summarization import _ProfileWindowShim

    target = MagicMock()
    shim = _ProfileWindowShim(target, 32_768)
    assert shim.profile == {"max_input_tokens": 32_768}
    assert shim._get_ls_params() is target._get_ls_params()
    assert shim.some_future_attr is target.some_future_attr


def test_no_override_outside_runnable_context():
    """Outside a runnable context the sync is a no-op (construction limits)."""
    mw, _construction = _make_mw()
    with _patched_config(None):
        mw._sync_limits()
    assert mw._get_profile_limits() == CONSTRUCTION_WINDOW


async def test_window_cache_once_per_window_for_unhashable_models():
    """Real chat models are unhashable pydantic objects; the window cache
    must key on the resolved context window, and alternating runs resolve
    each window once."""
    construction = _UnhashableModel(CONSTRUCTION_WINDOW)
    mw = create_per_run_summarization_middleware(construction, MagicMock())

    small = _UnhashableModel(32_768)
    large = _UnhashableModel(CONSTRUCTION_WINDOW)

    calls = []
    from EvoScientist.llm.context_window import resolve_context_window as _real

    def _counting(model, *a, **kw):
        calls.append(model)
        return _real(model, *a, **kw)

    with (
        patch(
            "EvoScientist.llm.get_chat_model",
            side_effect=[small, large, small, large],
        ),
        patch(
            "EvoScientist.middleware.summarization.resolve_context_window",
            side_effect=_counting,
        ),
    ):
        # Alternating override names (each with its own cached resolution)
        # defeat the identity short-circuit, so only the per-window cache can
        # keep the resolve count at one per window.
        for name in ("small-model", "large-model", "small-model", "large-model"):
            with _patched_config({"model": name}):
                await mw.awrap_model_call(
                    _fake_model_request(construction),
                    AsyncMock(return_value=MagicMock()),
                )
    assert calls == [small, large]  # each window resolved once
    assert mw._get_profile_limits() == CONSTRUCTION_WINDOW


async def test_resolution_failure_keeps_construction_limits():
    mw, construction = _make_mw()
    with (
        _patched_config({"model": "broken-model"}),
        patch("EvoScientist.llm.get_chat_model", side_effect=RuntimeError("no key")),
    ):
        await mw.awrap_model_call(
            _fake_model_request(construction), AsyncMock(return_value=MagicMock())
        )
    assert mw._get_profile_limits() == CONSTRUCTION_WINDOW
    assert mw._lc_helper._get_profile_limits() == CONSTRUCTION_WINDOW


def test_summary_model_stays_construction_model():
    """Summaries don't switch models: ``_summary_model`` is captured once at
    construction and is not re-pointed by the per-run sync."""
    mw, construction = _make_mw()
    small = MagicMock()
    small.profile = {"max_input_tokens": 32_768}
    with (
        _patched_config({"model": "small-model"}),
        patch("EvoScientist.llm.get_chat_model", return_value=small),
    ):
        mw._sync_limits()
    assert mw._lc_helper._summary_model is construction.with_retry()
    assert mw._lc_helper.model is not small


# ---------------------------------------------------------------------------
# Registration in the default middleware list
# ---------------------------------------------------------------------------


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    return_value=[MagicMock(), MagicMock()],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_default_middleware_registers_subclass_exactly_once(
    mock_config, mock_model, mock_ts
):
    mock_model.return_value = MagicMock(
        profile={"max_input_tokens": CONSTRUCTION_WINDOW}
    )
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_approve = False
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    mock_config.return_value = cfg

    from EvoScientist.EvoScientist import _get_default_middleware

    mw = _get_default_middleware(backend=MagicMock())
    summ = [m for m in mw if m.name == "SummarizationMiddleware"]
    assert len(summ) == 1, "expected exactly one summarization middleware"
    assert isinstance(summ[0], _PerRunLimitsSummarizationMiddleware)

    # Without a backend (tests, async sub-agent factories) the stock frozen
    # built-in must be left untouched rather than replaced by a shim-backend
    # instance that would offload history to the wrong place.
    mw_plain = _get_default_middleware()
    assert not [m for m in mw_plain if m.name == "SummarizationMiddleware"]


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    return_value=[MagicMock(), MagicMock()],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_deepagents_name_merge_replaces_stock_instance(
    mock_config, mock_model, mock_ts
):
    """deepagents' name-based merge swaps the frozen built-in for our
    subclass in the identical core-stack slot (the fix mechanism for #466)."""
    mock_model.return_value = MagicMock(
        profile={"max_input_tokens": CONSTRUCTION_WINDOW}
    )
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_approve = False
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    mock_config.return_value = cfg

    from deepagents.graph import _apply_custom_middleware

    from EvoScientist.EvoScientist import _get_default_middleware

    user_mw = _get_default_middleware(backend=MagicMock())

    core = [
        MagicMock(name="FilesystemMiddleware"),
        MagicMock(name="SubAgentMiddleware"),
        MagicMock(name="SummarizationMiddleware"),
        MagicMock(name="PatchToolCallsMiddleware"),
    ]
    merged = _apply_custom_middleware(core, user_mw, core_names={m.name for m in core})
    summ = [m for m in merged if m.name == "SummarizationMiddleware"]
    assert len(summ) == 1
    assert isinstance(summ[0], _PerRunLimitsSummarizationMiddleware)


# ---------------------------------------------------------------------------
# End-to-end: smaller-window run summarizes instead of overflowing
# ---------------------------------------------------------------------------


def _e2e_messages():
    """~18k tokens of plain text: 9% of the 200k construction window, but
    between the 85% trigger (17k) and the 95% input budget (19k) of a 20k run
    window — so only the per-run fraction trigger can fire. Sized for the
    default 4 chars/token approximation FakeListChatModel gets."""
    msgs = []
    for i in range(7):
        msgs.append(HumanMessage(content=f"question {i} " + "detail " * 730))
        msgs.append(AIMessage(content=f"answer {i} " + "detail " * 730))
    return msgs


async def test_smaller_window_run_triggers_summarization(tmp_path):
    """A run whose ``configurable.model`` override resolves to a 20k-window
    model must summarize at that window's trigger — not overflow waiting for
    the frozen 200k construction trigger."""
    from deepagents.backends import FilesystemBackend
    from langchain_core.language_models import FakeListChatModel

    construction = FakeListChatModel(responses=["SUMMARY"])
    construction.profile = {"max_input_tokens": CONSTRUCTION_WINDOW}
    backend = FilesystemBackend(root_dir=tmp_path)
    mw = create_per_run_summarization_middleware(construction, backend)

    run_model = FakeListChatModel(responses=["unused"])
    run_model.profile = {"max_input_tokens": 20_000}

    messages = _e2e_messages()
    handler = AsyncMock(return_value=MagicMock())
    with (
        _patched_config({"model": "small-model"}),
        patch("EvoScientist.llm.get_chat_model", return_value=run_model),
    ):
        await mw.awrap_model_call(_fake_model_request(construction, messages), handler)

    assert mw._get_profile_limits() == 20_000
    handler.assert_awaited_once()
    sent = handler.call_args[0][0].messages
    # The history was compacted to a summary message plus the kept tail...
    assert len(sent) < len(messages)
    assert "SUMMARY" in sent[0].content  # generated by the construction model
    assert "/conversation_history/" in sent[0].content
    # ...and the evicted messages were offloaded to the backend.
    history_files = list((tmp_path / "conversation_history").glob("*.md"))
    assert len(history_files) == 1, "offload silently skipped — no history file"
    assert "question 0" in history_files[0].read_text()


async def test_stock_middleware_stays_frozen_under_same_override(tmp_path):
    """Regression contrast: the stock deepagents middleware does NOT adapt to
    the per-run window, so the same run overflows instead of summarizing."""
    from deepagents.backends import FilesystemBackend
    from deepagents.middleware.summarization import create_summarization_middleware
    from langchain_core.language_models import FakeListChatModel

    construction = FakeListChatModel(responses=["SUMMARY"])
    construction.profile = {"max_input_tokens": CONSTRUCTION_WINDOW}
    mw = create_summarization_middleware(
        construction, FilesystemBackend(root_dir=tmp_path)
    )

    run_model = FakeListChatModel(responses=["unused"])
    run_model.profile = {"max_input_tokens": 20_000}

    messages = _e2e_messages()
    handler = AsyncMock(return_value=MagicMock())
    with (
        _patched_config({"model": "small-model"}),
        patch("EvoScientist.llm.get_chat_model", return_value=run_model),
    ):
        await mw.awrap_model_call(_fake_model_request(construction, messages), handler)

    handler.assert_awaited_once()
    sent = handler.call_args[0][0].messages
    # Frozen 200k trigger: the full oversized history is sent untouched.
    assert len(sent) == len(messages)
    assert not list((tmp_path / "conversation_history").glob("*.md"))
