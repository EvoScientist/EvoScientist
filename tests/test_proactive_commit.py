"""Tests for EvoScientist.proactive.commit."""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from EvoScientist.proactive.commit import (
    MAX_PUSH_CHARS,
    build_proactive_tag,
    decide_commit,
    last_non_proactive_message_id,
    proactive_message_id,
    read_proactive_tag,
)


def _decide(reply, **overrides):
    kwargs = {
        "source_thread_id": "t1",
        "pre_head": "h1",
        "current_head": "h1",
        "proactive_id": "p1",
    }
    kwargs.update(overrides)
    return decide_commit(reply, **kwargs)


def test_push_is_a_tagged_ai_message_with_the_stripped_text():
    decision = _decide("  Follow-up: the run finished.  ")
    assert decision.action == "push"
    assert decision.message.content == "Follow-up: the run finished."
    assert decision.message.id == proactive_message_id("p1") == "proactive-p1"
    assert read_proactive_tag(decision.message.additional_kwargs) == {
        "is_proactive_push": True,
        "proactive_id": "p1",
    }


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (None, "no_reply"),
        ("", "empty"),
        ("   ", "empty"),
        ("NO_PUSH", "no_push"),
        (" NO_PUSH\n", "no_push"),
        ("NO_PUSH.", "no_push"),
        ("**NO_PUSH**", "no_push"),
        ("`NO_PUSH`", "no_push"),
        ("no_push", "no_push"),
        ("NO_PUSH:", "no_push"),
        ("NO_PUSH\nNothing new since the last reply.", "no_push"),
        ("Nothing new since the last reply.\n\nNO_PUSH", "no_push"),
        ("x" * (MAX_PUSH_CHARS + 1), "too_long"),
        ("<tool_call>execute ls</tool_call>", "tool_call_text"),
    ],
)
def test_skips(reply, reason):
    decision = _decide(reply)
    assert decision.action == "skip"
    assert decision.reason == reason
    assert decision.message is None


def test_a_message_that_mentions_the_sentinel_is_pushed():
    assert (
        _decide("I could answer NO_PUSH here,\nbut the run finished.").action == "push"
    )


def test_length_is_counted_on_the_text_that_would_be_sent():
    padded = "  " + "x" * MAX_PUSH_CHARS + "\n\n\n"
    assert _decide(padded).action == "push"


def test_stale_when_the_head_moved():
    decision = _decide("hello", current_head="h2")
    assert decision.action == "stale"
    assert decision.message is None


def test_logs_never_carry_the_reply_text(caplog):
    secret = "the secret reply text"
    with caplog.at_level(logging.DEBUG, logger="EvoScientist.proactive.commit"):
        _decide(secret)
        _decide(secret, current_head="h2")
        _decide("<tool_call>" + secret)
        _decide(secret * 400)
    assert secret not in caplog.text


@pytest.mark.parametrize(
    "kwargs", [None, "x", {}, {"evoscientist": "x"}, {"evoscientist": {}}]
)
def test_read_proactive_tag_rejects_non_tags(kwargs):
    assert read_proactive_tag(kwargs) is None


def test_last_non_proactive_message_id_skips_pushes_and_id_less_messages():
    push = AIMessage(
        content="push", id="proactive-p1", additional_kwargs=build_proactive_tag("p1")
    )
    messages = [
        HumanMessage(content="q", id="m1"),
        AIMessage(content="a", id="m2"),
        push,
        AIMessage(content="no id"),
    ]
    assert last_non_proactive_message_id(messages) == "m2"
    assert last_non_proactive_message_id([push]) is None
    assert last_non_proactive_message_id([]) is None
