"""Tests for per-run context limits in the summarization middleware (#466).

``_PerRunLimitsSummarizationMiddleware`` replaces deepagents' built-in
``SummarizationMiddleware`` (frozen on the construction model's context
window) so the summarization trigger tracks the run's
``configurable.model`` override instead. These tests reuse the harness
patterns from ``tests/test_context_editing_middleware.py`` and
``tests/test_configurable_model_middleware.py``.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from EvoScientist.middleware.summarization import (
    _PerRunLimitsSummarizationMiddleware,
    _ProfileWindowShim,
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


async def test_interleaved_tasks_keep_isolated_windows():
    """Two interleaved asyncio tasks never cross-read each other's window.

    The core regression test for review point 1: deepagents'
    ``awrap_model_call`` re-reads the window AFTER awaits (model call,
    offload, summary — its overflow fallback consults the limits again
    mid-call), so per-INSTANCE state let a second task's ``_sync_limits``
    corrupt the first task's in-flight thresholds and clip budgets. The
    per-task ``ContextVar`` state keeps every task on its own window.
    """
    mw, construction = _make_mw()

    small = MagicMock()
    small.profile = {"max_input_tokens": 32_768}
    large = MagicMock()
    large.profile = {"max_input_tokens": 1_000_000}

    entered = asyncio.Event()  # set once task A parks in its model handler
    release_a = asyncio.Event()
    reads: dict[str, Any] = {}

    async def handler_a(request):
        entered.set()
        await release_a.wait()  # "model call in flight" while task B syncs
        # Post-await reads — exactly where deepagents' overflow fallback
        # consults the window mid-call.
        reads["a_limits"] = mw._get_profile_limits()
        reads["a_shim_window"] = mw._lc_helper.model.profile["max_input_tokens"]
        return MagicMock()

    async def task_a():
        with (
            _patched_config({"model": "small-model"}),
            patch("EvoScientist.llm.get_chat_model", return_value=small),
        ):
            await mw.awrap_model_call(_fake_model_request(construction), handler_a)

    async def task_b():
        with (
            _patched_config({"model": "large-model"}),
            patch("EvoScientist.llm.get_chat_model", return_value=large),
        ):
            await mw.awrap_model_call(
                _fake_model_request(construction),
                AsyncMock(return_value=MagicMock()),
            )
        # Still inside task B's own context.
        reads["b_limits"] = mw._get_profile_limits()
        reads["b_shim_window"] = mw._lc_helper.model.profile["max_input_tokens"]

    a = asyncio.create_task(task_a())
    await entered.wait()
    shim_after_a_sync = mw._lc_helper.model

    b = asyncio.create_task(task_b())
    await b
    shim_after_b_sync = mw._lc_helper.model

    release_a.set()
    await a

    # Each task kept its own window across the interleaving.
    assert reads["a_limits"] == 32_768
    assert reads["a_shim_window"] == 32_768
    assert reads["b_limits"] == 1_000_000
    assert reads["b_shim_window"] == 1_000_000
    # The shim was installed exactly once: identity never changed between
    # the two syncs (nor after them).
    assert isinstance(shim_after_a_sync, _ProfileWindowShim)
    assert shim_after_a_sync is shim_after_b_sync
    assert mw._lc_helper.model is shim_after_a_sync
    # The parent task — which never synced — still sees construction limits.
    assert mw._get_profile_limits() == CONSTRUCTION_WINDOW


def test_shim_delegates_non_profile_attributes():
    """The profile shim reports the per-task window and delegates every other
    attribute to the per-task target model (langchain provider-matches
    reported tokens via ``_get_ls_params``)."""
    import contextvars

    from EvoScientist.middleware.summarization import _ProfileWindowShim

    state: contextvars.ContextVar[tuple[Any, int] | None] = contextvars.ContextVar(
        "test_shim_state", default=None
    )
    construction = MagicMock()
    shim = _ProfileWindowShim(state, construction, 32_768)
    # Unsynchronized task: construction window + construction delegation.
    assert shim.profile == {"max_input_tokens": 32_768}
    assert shim._get_ls_params() is construction._get_ls_params()
    assert shim.some_future_attr is construction.some_future_attr

    run_model = MagicMock()
    state.set((run_model, 1_000_000))
    assert shim.profile == {"max_input_tokens": 1_000_000}
    assert shim._get_ls_params() is run_model._get_ls_params()
    assert shim._get_ls_params() is not construction._get_ls_params()


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
# Registration in the async sub-agent factories (review point 2)
# ---------------------------------------------------------------------------


def _factory_cfg_mock() -> MagicMock:
    """Config stub covering both ``_get_default_middleware`` and the factory."""
    cfg = MagicMock()
    cfg.recursion_limit = 1_000_000
    cfg.enable_ask_user = False
    cfg.auto_mode = False
    cfg.auto_approve = False
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    return cfg


def _assert_per_run_summarization(middleware: list, backend: Any) -> None:
    """The middleware list handed to ``create_deep_agent`` must contain
    exactly one name=="SummarizationMiddleware" entry — our subclass, built
    on the same backend the agent itself uses (history offload parity)."""
    summ = [m for m in middleware if m.name == "SummarizationMiddleware"]
    assert len(summ) == 1, "expected exactly one summarization middleware"
    assert isinstance(summ[0], _PerRunLimitsSummarizationMiddleware)
    assert summ[0]._backend is backend


@patch("deepagents.create_deep_agent")
@patch("EvoScientist.EvoScientist._load_mcp_tools_cached", return_value={})
@patch("EvoScientist.EvoScientist._get_default_backend")
@patch("EvoScientist.EvoScientist._ensure_config")
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.utils.load_subagents")
@patch("EvoScientist.config.apply_config_to_env")
@patch("EvoScientist.config.get_effective_config")
@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    return_value=[MagicMock()],
)
def test_async_subagent_factory_installs_per_run_summarization(
    mock_ts,
    mock_get_cfg,
    mock_apply_env,
    mock_load_subs,
    mock_chat,
    mock_config,
    mock_backend,
    mock_mcp,
    mock_create,
):
    """``build_async_subagent_graph`` must pass its backend so the per-run
    subclass replaces the frozen-window built-in (scheduler / writing-agent /
    data-analysis graphs take ``configurable.model`` overrides too — #466)."""
    cfg = _factory_cfg_mock()
    mock_get_cfg.return_value = cfg
    mock_config.return_value = cfg
    mock_chat.return_value = MagicMock(profile={"max_input_tokens": 200_000})
    mock_load_subs.return_value = [
        {"name": "writing-agent", "system_prompt": "", "tools": [], "skills": None}
    ]
    mock_create.return_value.with_config.return_value = MagicMock()

    from EvoScientist.subagents._factory import build_async_subagent_graph

    build_async_subagent_graph("writing-agent")

    kwargs = mock_create.call_args.kwargs
    _assert_per_run_summarization(kwargs["middleware"], mock_backend.return_value)
    assert kwargs["backend"] is mock_backend.return_value


@patch("deepagents.create_deep_agent")
@patch("EvoScientist.EvoScientist._get_default_backend")
@patch("EvoScientist.EvoScientist._ensure_config")
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.config.apply_config_to_env")
@patch("EvoScientist.config.get_effective_config")
@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    return_value=[MagicMock()],
)
def test_expert_container_factory_installs_per_run_summarization(
    mock_ts,
    mock_get_cfg,
    mock_apply_env,
    mock_chat,
    mock_config,
    mock_backend,
    mock_create,
):
    """``build_expert_container_async_graph`` must pass its backend too —
    the expert container graph is its own deployed graph with its own frozen
    built-in that the name merge has to replace (#466)."""
    cfg = _factory_cfg_mock()
    mock_get_cfg.return_value = cfg
    mock_config.return_value = cfg
    mock_chat.return_value = MagicMock(profile={"max_input_tokens": 200_000})
    mock_create.return_value.with_config.return_value = MagicMock()

    from EvoScientist.subagents.expert_container_async import (
        build_expert_container_async_graph,
    )

    build_expert_container_async_graph()

    kwargs = mock_create.call_args.kwargs
    _assert_per_run_summarization(kwargs["middleware"], mock_backend.return_value)
    assert kwargs["backend"] is mock_backend.return_value


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
