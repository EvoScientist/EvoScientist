"""Tests for EvoScientist.proactive.serve_runner (serve-driven tick + apply)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from langchain_core.messages import AIMessage

from EvoScientist.proactive.gate import ProactiveGateSettings
from EvoScientist.proactive.serve_runner import (
    apply_proactive_decision,
    gate_settings_from_config,
    run_serve_proactive_tick,
)
from EvoScientist.proactive.service import ProactiveDecision

_NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


class _FakeThreads:
    def __init__(self, states, threads):
        self.states = states
        self.threads = threads
        self.updates: list[tuple] = []

    async def get_state(self, tid):
        return self.states[tid]

    async def get(self, tid):
        return self.threads[tid]

    async def update_state(self, tid, values, *, as_node=None):
        self.updates.append((tid, values, as_node))


class _FakeClient:
    def __init__(self, threads):
        self.threads = threads


def _settings(**over):
    base = {
        "enabled": True,
        "idle_minutes": 120,
        "quiet_hours": None,
        "timezone": "UTC",
    }
    base.update(over)
    return ProactiveGateSettings(**base)


def _thread(*, minutes_idle=200, status="idle"):
    return {
        "updated_at": (_NOW - timedelta(minutes=minutes_idle)).isoformat(),
        "status": status,
    }


def _state(head="h0", messages=None, interrupts=None):
    state = {"values": {"messages": messages or []}, "checkpoint_id": head}
    if interrupts is not None:
        state["interrupts"] = interrupts
    return state


# ---- gate_settings_from_config ----------------------------------------------


def test_gate_settings_from_config_maps_fields():
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        proactive_enabled=True,
        proactive_idle_minutes=90,
        proactive_quiet_hours="",  # empty -> None
        proactive_timezone="UTC",
    )
    s = gate_settings_from_config(cfg)
    assert s.enabled is True
    assert s.idle_minutes == 90
    assert s.quiet_hours is None
    assert s.timezone == "UTC"


# ---- run_serve_proactive_tick -----------------------------------------------


async def _tick(client, *, reply, settings=None, origin_present=None):
    async def shadow_runner(messages, trigger):
        return reply

    return await run_serve_proactive_tick(
        client,
        candidate_ids=["t1"],
        now=_NOW,
        settings=settings or _settings(),
        shadow_runner=shadow_runner,
        trigger="TRIGGER",
        workspace_dir="/ws",
        model="m",
        gen_id=lambda: "pid1",
        origin_present=origin_present or (lambda tid: True),
    )


async def test_tick_decides_push_for_idle_thread():
    client = _FakeClient(_FakeThreads({"t1": _state("h0")}, {"t1": _thread()}))
    decisions = await _tick(client, reply="a useful update")
    assert len(decisions) == 1
    assert decisions[0].stage == "decided"
    assert decisions[0].head == "h0"


async def test_tick_gate_rejects_busy_thread():
    client = _FakeClient(
        _FakeThreads({"t1": _state("h0")}, {"t1": _thread(status="busy")})
    )
    decisions = await _tick(client, reply="a useful update")
    assert decisions[0].stage == "gate_rejected"
    assert decisions[0].reason == "in_flight"


async def test_tick_gate_rejects_interrupted_status():
    client = _FakeClient(
        _FakeThreads({"t1": _state("h0")}, {"t1": _thread(status="interrupted")})
    )
    decisions = await _tick(client, reply="a useful update")
    assert decisions[0].reason == "in_flight"  # parked HITL, not a push target


async def test_tick_gate_rejects_pending_interrupt():
    client = _FakeClient(
        _FakeThreads({"t1": _state("h0", interrupts=[{"id": "x"}])}, {"t1": _thread()})
    )
    decisions = await _tick(client, reply="a useful update")
    assert decisions[0].reason == "in_flight"


async def test_tick_gates_out_when_origin_forgotten():
    client = _FakeClient(_FakeThreads({"t1": _state("h0")}, {"t1": _thread()}))
    decisions = await _tick(client, reply="x", origin_present=lambda tid: False)
    assert decisions[0].stage == "gate_rejected"
    assert decisions[0].reason == "no_origin"


async def test_tick_isolates_per_thread_errors():
    # get_state raises for the only candidate -> tick continues, returns nothing.
    class _Boom(_FakeThreads):
        async def get_state(self, tid):
            raise RuntimeError("boom")

    client = _FakeClient(_Boom({}, {"t1": _thread()}))
    decisions = await _tick(client, reply="x")
    assert decisions == []


# ---- apply_proactive_decision -----------------------------------------------


def _decided(head="h0"):
    return ProactiveDecision(
        "decided",
        "ok",
        reply="hello",
        message=AIMessage(content="hello", id="proactive-pid1"),
        proactive_id="pid1",
        source_thread_id="t1",
        head=head,
    )


async def _apply(client, decision, *, publish, origin_present=None):
    return await apply_proactive_decision(
        client,
        decision,
        publish=publish,
        origin_present=origin_present or (lambda tid: True),
    )


async def test_apply_writes_once_and_publishes_when_head_matches():
    threads = _FakeThreads({"t1": _state("h0")}, {"t1": _thread()})
    client = _FakeClient(threads)
    published: list[tuple] = []

    ok = await _apply(
        client,
        _decided("h0"),
        publish=lambda tid, c: published.append((tid, c)) or True,
    )

    assert ok is True
    # a single write (the tagged append as the model node); no END clear
    assert len(threads.updates) == 1
    assert threads.updates[0][2] == "model"
    assert threads.updates[0][1]["messages"][0].id == "proactive-pid1"
    assert published == [("t1", "hello")]


async def test_apply_skips_when_head_moved():
    threads = _FakeThreads({"t1": _state("h1")}, {"t1": _thread()})  # head moved
    client = _FakeClient(threads)
    ok = await _apply(client, _decided("h0"), publish=lambda t, c: True)
    assert ok is False
    assert threads.updates == []  # nothing written


async def test_apply_skips_parked_interrupt_without_clobbering():
    threads = _FakeThreads(
        {"t1": _state("h0", interrupts=[{"id": "x"}])}, {"t1": _thread()}
    )
    client = _FakeClient(threads)
    ok = await _apply(client, _decided("h0"), publish=lambda t, c: True)
    assert ok is False
    assert threads.updates == []  # the pending interrupt is preserved


async def test_apply_skips_when_origin_forgotten():
    threads = _FakeThreads({"t1": _state("h0")}, {"t1": _thread()})
    client = _FakeClient(threads)
    ok = await _apply(
        client,
        _decided("h0"),
        publish=lambda t, c: True,
        origin_present=lambda tid: False,
    )
    assert ok is False
    assert (
        threads.updates == []
    )  # no phantom message when there's nothing to deliver to


async def test_apply_returns_false_when_write_lands_but_delivery_fails():
    threads = _FakeThreads({"t1": _state("h0")}, {"t1": _thread()})
    client = _FakeClient(threads)
    ok = await _apply(client, _decided("h0"), publish=lambda t, c: False)
    assert ok is False
    assert len(threads.updates) == 1  # the write still happened; caller is warned


async def test_apply_ignores_non_decided():
    threads = _FakeThreads({}, {})
    client = _FakeClient(threads)
    ok = await _apply(
        client, ProactiveDecision("no_push", "no_push"), publish=lambda t, c: True
    )
    assert ok is False
