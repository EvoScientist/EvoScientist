"""Tests for EvoScientist.proactive.service."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage

from EvoScientist.proactive.commit import build_proactive_tag
from EvoScientist.proactive.gate import ProactiveGateSettings
from EvoScientist.proactive.service import PROACTIVE_TRIGGER, decide_proactive_push

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
MESSAGES = [HumanMessage(content="q", id="m1"), AIMessage(content="a", id="m2")]


def _decide(shadow_reply="Here is the follow-up.", **overrides):
    calls = []

    async def shadow_runner(messages, trigger):
        calls.append((messages, trigger))
        return shadow_reply

    kwargs = {
        "settings": ProactiveGateSettings(
            idle_minutes=120, quiet_hours="", timezone="UTC"
        ),
        "source_thread_id": "t1",
        "now": NOW,
        "armed": True,
        "has_origin": True,
        "in_flight": False,
        "last_activity": NOW - timedelta(hours=3),
        "last_evaluated_id": None,
        "pre_head": "h1",
        "source_messages": MESSAGES,
        "shadow_runner": shadow_runner,
        "read_current_head": lambda: "h1",
        "proactive_id": "p1",
    }
    kwargs.update(overrides)
    return asyncio.run(decide_proactive_push(**kwargs)), calls


def test_decided_push_carries_the_message_and_the_evaluated_id():
    decision, calls = _decide()
    assert decision.stage == "decided"
    assert decision.proactive_id == "p1"
    assert decision.message.content == "Here is the follow-up."
    assert decision.pre_head == "h1"
    assert decision.evaluated_message_id == "m2"
    assert calls == [(MESSAGES, PROACTIVE_TRIGGER)]


def test_gate_rejection_skips_the_shadow():
    decision, calls = _decide(armed=False)
    assert decision.stage == "gate_rejected"
    assert decision.reason == "not_armed"
    assert decision.evaluated_message_id is None
    assert calls == []


def test_unchanged_thread_is_not_evaluated_again():
    decision, calls = _decide(last_evaluated_id="m2")
    assert decision.reason == "no_new_information"
    assert calls == []


def test_a_push_alone_is_not_new_information():
    push = AIMessage(
        content="p", id="proactive-p0", additional_kwargs=build_proactive_tag("p0")
    )
    decision, calls = _decide(last_evaluated_id="m2", source_messages=[*MESSAGES, push])
    assert decision.reason == "no_new_information"
    assert calls == []


def test_no_push_records_the_evaluation():
    decision, _ = _decide(shadow_reply="NO_PUSH")
    assert decision.stage == "no_push"
    assert decision.message is None
    assert decision.proactive_id is None
    assert decision.evaluated_message_id == "m2"


def test_stale_when_the_head_moved_during_the_shadow():
    async def head():
        return "h2"

    decision, _ = _decide(read_current_head=head)
    assert decision.stage == "stale"
    assert decision.message is None
    assert decision.evaluated_message_id is None  # the next check runs again
