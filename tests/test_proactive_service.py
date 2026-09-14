"""Tests for EvoScientist.proactive.service (Shape B decide orchestrator)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from langchain_core.messages import AIMessage

from EvoScientist.proactive.commit import read_proactive_tag
from EvoScientist.proactive.gate import ProactiveGateSettings
from EvoScientist.proactive.service import (
    PROACTIVE_TRIGGER,
    decide_proactive_push,
)

_NOON = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def _settings(**over):
    base = {
        "enabled": True,
        "idle_minutes": 120,
        "quiet_hours": None,
        "timezone": "UTC",
    }
    base.update(over)
    return ProactiveGateSettings(**base)


async def _decide(
    *,
    reply,
    settings=None,
    last_activity=None,
    has_origin=True,
    in_flight=False,
    pre_head="h0",
    current_head="h0",
):
    shadow_calls = []

    async def shadow_runner(messages, trigger):
        shadow_calls.append((messages, trigger))
        return reply

    async def read_current_head():
        return current_head

    decision = await decide_proactive_push(
        settings=settings or _settings(),
        source_thread_id="t1",
        now=_NOON,
        last_activity=last_activity
        if last_activity is not None
        else _NOON - timedelta(minutes=200),
        pre_head=pre_head,
        has_origin=has_origin,
        in_flight=in_flight,
        source_messages=[AIMessage(content="prior")],
        shadow_runner=shadow_runner,
        read_current_head=read_current_head,
        trigger=PROACTIVE_TRIGGER,
        workspace_dir="/ws",
        model="m",
        proactive_id="pid1",
    )
    return decision, shadow_calls


async def test_gate_rejected_skips_shadow():
    decision, shadow_calls = await _decide(reply="ignored", has_origin=False)
    assert decision.stage == "gate_rejected"
    assert decision.reason == "no_origin"
    assert shadow_calls == []  # no model call once the gate rejects


async def test_no_push_when_shadow_returns_sentinel():
    decision, _ = await _decide(reply="NO_PUSH")
    assert decision.stage == "no_push"
    assert decision.message is None


async def test_stale_when_head_moved_during_shadow():
    decision, _ = await _decide(reply="a real update", pre_head="h0", current_head="h1")
    assert decision.stage == "stale"
    assert decision.message is None


async def test_decided_carries_tagged_message():
    decision, shadow_calls = await _decide(reply="here is a useful update")
    assert decision.stage == "decided"
    assert decision.proactive_id == "pid1"
    assert decision.source_thread_id == "t1"
    assert isinstance(decision.message, AIMessage)
    assert read_proactive_tag(decision.message.additional_kwargs) == {
        "is_proactive_push": True,
        "proactive_id": "pid1",
    }
    # the shadow was fed the trigger, not written to the source thread
    assert shadow_calls[0][1] == PROACTIVE_TRIGGER
