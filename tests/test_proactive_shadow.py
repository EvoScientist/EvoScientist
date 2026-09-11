"""Tests for EvoScientist.proactive.shadow (shadow runner + tool-strip spy)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from EvoScientist.proactive.shadow import (
    ProactiveToolLeakError,
    _install_tool_strip_spy,
    _shadow_cfg,
    run_shadow_turn,
)


def _minimal_shadow_graph(model):
    """Build a minimal real graph with the strip middleware + spy installed.

    This is deliberately NOT ``build_shadow_graph`` (which routes through
    ``create_cli_agent`` and its MCP/network load). It composes ``create_deep_agent``
    directly with a keyless fake model, the real ``ProactiveModeMiddleware``, and a
    dummy tool — enough to exercise the safety-critical binding path offline.
    """
    from deepagents import create_deep_agent
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver

    from EvoScientist.middleware.proactive_mode import ProactiveModeMiddleware

    @tool
    def dummy_tool(x: str) -> str:
        """A dummy tool that must never be bound during a proactive turn."""
        return x

    return create_deep_agent(
        model=model,
        tools=[dummy_tool],
        middleware=[ProactiveModeMiddleware()],
        checkpointer=InMemorySaver(),
    )


# ---- _shadow_cfg ------------------------------------------------------------


def test_shadow_cfg_flips_side_effect_flags_and_keeps_reasoning():
    from EvoScientist.config.settings import EvoScientistConfig

    cfg = EvoScientistConfig()
    cfg.model = "sentinel-model"
    cfg.provider = "sentinel-provider"

    shadow = _shadow_cfg(cfg)

    # side-effect axis off
    assert shadow.enable_scheduler is False
    assert shadow.memory_workers_enabled is False
    assert shadow.enable_ask_user is False
    assert shadow.auto_mode is True
    assert shadow.auto_approve is True
    assert shadow.model_fallbacks == ""
    # reasoning axis intact
    assert shadow.model == "sentinel-model"
    assert shadow.provider == "sentinel-provider"
    # original untouched (replace returns a copy)
    assert cfg.enable_scheduler is not False or cfg is not shadow
    assert cfg is not shadow


def test_shadow_cfg_gates_out_the_writer_middleware():
    """The write-side neutering is a gate, not just a flag: pin that the shadow
    config feeds the exact predicates `_get_default_middleware` uses to DROP the
    scheduler and memory-worker middleware. Kept surgical (no graph build) to
    avoid real MEMORIES_DIR writes from constructing the memory middleware."""
    from EvoScientist.config.settings import (
        EvoScientistConfig,
        MemoryControls,
        MemoryObservationTarget,
    )

    shadow = _shadow_cfg(EvoScientistConfig())

    # scheduler-middleware gate (EvoScientist.py:996 `if cfg.enable_scheduler ...`)
    assert shadow.enable_scheduler is False
    # memory-worker gate (EvoScientist.py:1001 `if memory_controls.worker_needed(...)`)
    controls = MemoryControls.from_config(shadow)
    assert controls.worker_needed(MemoryObservationTarget.TURN_WORKER) is False
    assert controls.worker_needed(MemoryObservationTarget.SUBAGENT_WORKER) is False
    # first-contact profile bootstrap gate (EvoScientist.py
    # `enable_profile_bootstrap=not for_async_subagent and not bool(cfg.auto_mode)`):
    # a shadow turn must never ask the consent survey or write intro bookkeeping.
    assert (not False and not bool(shadow.auto_mode)) is False


# ---- tool-strip spy ---------------------------------------------------------


def test_spy_raises_on_non_empty_tool_bind():
    model = FakeListChatModel(responses=["ok"])
    spy = _install_tool_strip_spy(model)
    with pytest.raises(ProactiveToolLeakError, match="record_observation"):
        spy.bind_tools([{"name": "record_observation"}])


def test_spy_preserves_isinstance_and_identity():
    model = FakeListChatModel(responses=["ok"])
    spy = _install_tool_strip_spy(model)
    assert spy is model  # installed in place
    assert isinstance(spy, FakeListChatModel)  # subclass, original type preserved
    assert type(spy).__name__.startswith("ToolStripSpy_")


# ---- run_shadow_turn --------------------------------------------------------


class _FakeGraph:
    """Graph stand-in exposing the methods run_shadow_turn touches."""

    def __init__(self, last_message=None):
        self.updates = []
        self.checkpointer = MagicMock()
        self.checkpointer.adelete_thread = AsyncMock()
        self.last_message = last_message

    async def aupdate_state(self, config, values, as_node=None):
        self.updates.append((config, values))

    async def aget_state(self, config):
        msgs = [self.last_message] if self.last_message is not None else []
        return SimpleNamespace(values={"messages": msgs})


def _patch_stream(events=None, raise_exc=None):
    """Patch stream_agent_events with an async generator, capturing its args."""
    captured = {}

    def _factory(agent, message, thread_id, *, configurable_extra=None, **kw):
        captured["message"] = message
        captured["thread_id"] = thread_id
        captured["configurable_extra"] = configurable_extra

        async def _gen():
            for ev in events or []:
                yield ev
            if raise_exc is not None:
                raise raise_exc

        return _gen()

    return patch(
        "EvoScientist.stream.events.stream_agent_events", side_effect=_factory
    ), captured


def test_run_shadow_turn_returns_terminal_done_text():
    graph = _FakeGraph()
    msgs = [{"role": "user", "content": "prior message"}]
    events = [
        {"type": "text", "content": "hel"},
        {"type": "text", "content": "lo"},
        {"type": "done", "content": "hello"},
    ]
    patcher, captured = _patch_stream(events=events)
    with patcher:
        result = asyncio.run(run_shadow_turn(graph, msgs, "<proactive_trigger>"))

    assert result == "hello"
    # seeded history under the messages channel
    assert graph.updates
    assert graph.updates[0][1] == {"messages": msgs}
    # proactive_mode carried into the run config
    assert captured["configurable_extra"] == {"proactive_mode": True}
    assert captured["message"] == "<proactive_trigger>"
    # shadow thread cleaned up
    graph.checkpointer.adelete_thread.assert_awaited_once()


def test_run_shadow_turn_falls_back_to_text_when_done_empty():
    graph = _FakeGraph()
    events = [
        {"type": "text", "content": "streamed "},
        {"type": "text", "content": "answer"},
        {"type": "done", "content": ""},
    ]
    patcher, _ = _patch_stream(events=events)
    with patcher:
        result = asyncio.run(run_shadow_turn(graph, [], "t"))
    assert result == "streamed answer"


def test_run_shadow_turn_returns_none_on_tool_leak():
    graph = _FakeGraph()
    patcher, _ = _patch_stream(
        events=[{"type": "text", "content": "partial"}],
        raise_exc=ProactiveToolLeakError("bound 1 tool(s)"),
    )
    with patcher:
        result = asyncio.run(run_shadow_turn(graph, [], "t"))
    assert result is None
    # cleanup still runs on the abort path
    graph.checkpointer.adelete_thread.assert_awaited_once()


def test_run_shadow_turn_returns_none_on_generic_error():
    graph = _FakeGraph()
    patcher, _ = _patch_stream(raise_exc=RuntimeError("provider blew up"))
    with patcher:
        result = asyncio.run(run_shadow_turn(graph, [], "t"))
    assert result is None
    graph.checkpointer.adelete_thread.assert_awaited_once()


# ---- spy on the real binding path (no network) ------------------------------
# These prove the two safety claims the pure-unit tests above cannot: that the
# spy is actually the object langchain binds against, and that proactive_mode
# genuinely zeroes tools end-to-end through a composed graph. The full
# create_cli_agent path (MCP/network) stays deferred; this covers the interlock.


def test_proactive_turn_binds_zero_tools_and_spy_stays_silent():
    """proactive_mode=True → tools stripped → model.bind (not bind_tools) → the
    turn completes and the spy never fires."""
    model = _install_tool_strip_spy(FakeListChatModel(responses=["I decide: NO_PUSH"]))
    graph = _minimal_shadow_graph(model)
    out = graph.invoke(
        {"messages": [{"role": "user", "content": "hi"}]},
        config={"configurable": {"thread_id": "t-proactive", "proactive_mode": True}},
    )
    assert out["messages"][-1].content == "I decide: NO_PUSH"


def test_spy_fires_when_strip_absent_proving_it_is_on_the_binding_path():
    """Without proactive_mode the graph binds its tools, so the spy MUST fire —
    which is the proof that the spy sits on the real model-binding path (a bug
    that moved it off-path would let this pass silently)."""
    model = _install_tool_strip_spy(FakeListChatModel(responses=["unused"]))
    graph = _minimal_shadow_graph(model)
    with pytest.raises(ProactiveToolLeakError, match="dummy_tool"):
        graph.invoke(
            {"messages": [{"role": "user", "content": "hi"}]},
            config={"configurable": {"thread_id": "t-leak"}},
        )


def test_run_shadow_turn_logs_raw_message_when_reply_is_empty(caplog):
    """An empty reply is the one outcome the caller cannot explain (NO_PUSH is
    literal text), so the raw last message is logged before the thread is deleted."""
    import logging

    from langchain_core.messages import AIMessage

    graph = _FakeGraph(
        last_message=AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "thought about it"},
            response_metadata={"finish_reason": "stop"},
        )
    )
    patcher, _ = _patch_stream(events=[{"type": "done", "content": ""}])
    with patcher, caplog.at_level(logging.INFO, logger="EvoScientist.proactive.shadow"):
        result = asyncio.run(run_shadow_turn(graph, [], "t"))

    assert result == ""
    hits = [r.message for r in caplog.records if "returned no text" in r.message]
    assert len(hits) == 1
    assert "thought about it" in hits[0]
    assert "finish_reason" in hits[0]
    graph.checkpointer.adelete_thread.assert_awaited_once()


def test_run_shadow_turn_does_not_log_empty_diagnostic_when_text_present(caplog):
    import logging

    graph = _FakeGraph()
    patcher, _ = _patch_stream(events=[{"type": "done", "content": "NO_PUSH"}])
    with patcher, caplog.at_level(logging.INFO, logger="EvoScientist.proactive.shadow"):
        asyncio.run(run_shadow_turn(graph, [], "t"))
    assert not [r for r in caplog.records if "returned no text" in r.message]


def test_run_shadow_turn_discards_non_clean_finish(caplog):
    """A truncated or errored generation is never a message to send: the text is
    dropped (None → fail-closed) even though the stream produced some."""
    import logging

    from langchain_core.messages import AIMessage

    for reason in ("length", "error", "content_filter"):
        graph = _FakeGraph(
            last_message=AIMessage(
                content="partial", response_metadata={"finish_reason": reason}
            )
        )
        patcher, _ = _patch_stream(events=[{"type": "done", "content": "partial"}])
        with (
            patcher,
            caplog.at_level(logging.WARNING, logger="EvoScientist.proactive.shadow"),
        ):
            result = asyncio.run(run_shadow_turn(graph, [], "t"))
        assert result is None, reason
        assert any(reason in r.message for r in caplog.records)
        graph.checkpointer.adelete_thread.assert_awaited_once()


def test_run_shadow_turn_keeps_text_on_clean_or_absent_finish():
    from langchain_core.messages import AIMessage

    for metadata in ({"finish_reason": "stop"}, {}, None):
        graph = _FakeGraph(
            last_message=AIMessage(content="msg", response_metadata=metadata or {})
        )
        patcher, _ = _patch_stream(events=[{"type": "done", "content": "msg"}])
        with patcher:
            assert asyncio.run(run_shadow_turn(graph, [], "t")) == "msg"
