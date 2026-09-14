"""Tests for EvoScientist.proactive.commit (decision + tag helpers)."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from EvoScientist.proactive.commit import (
    NO_PUSH_SENTINEL,
    build_proactive_tag,
    decide_commit,
    proactive_message_id,
    read_proactive_tag,
)


def _decide(reply, *, pre_head="h0", current_head="h0", proactive_id="pid1"):
    return decide_commit(
        reply,
        source_thread_id="t1",
        workspace_dir="/ws",
        model="m",
        pre_head=pre_head,
        current_head=current_head,
        proactive_id=proactive_id,
    )


# ---- tag / id helpers -------------------------------------------------------


def test_proactive_message_id_is_stable():
    assert proactive_message_id("abc") == "proactive-abc"


def test_build_and_read_tag_roundtrip():
    tag = build_proactive_tag("pid1")
    assert tag == {"evoscientist": {"is_proactive_push": True, "proactive_id": "pid1"}}
    assert read_proactive_tag(tag) == {
        "is_proactive_push": True,
        "proactive_id": "pid1",
    }


def test_read_tag_returns_none_for_untagged():
    assert read_proactive_tag({}) is None
    assert read_proactive_tag({"evoscientist": {"is_proactive_push": False}}) is None
    assert read_proactive_tag("not-a-dict") is None
    assert read_proactive_tag({"other": {"is_proactive_push": True}}) is None


# ---- decide_commit skips ----------------------------------------------------


def test_decide_skips_none_empty_and_no_push():
    assert _decide(None).action == "skip"
    assert _decide("   ").action == "skip"
    d = _decide(NO_PUSH_SENTINEL)
    assert (d.action, d.reason) == ("skip", "no_push")


def test_decide_skips_too_long():
    d = _decide("x" * 5000)
    assert (d.action, d.reason) == ("skip", "too_long")


def test_decide_skips_tool_call_text():
    d = _decide("<tool_call>execute ls</tool_call>")
    assert (d.action, d.reason) == ("skip", "tool_call_text")


def test_decide_stale_when_head_moved():
    d = _decide("a real message", pre_head="h0", current_head="h1")
    assert (d.action, d.reason) == ("stale", "stale_source")


# ---- decide_commit would_commit ---------------------------------------------


def test_decide_would_commit_builds_tagged_message():
    d = _decide("here is a useful update", proactive_id="pid9")
    assert d.action == "would_commit"
    assert isinstance(d.message, AIMessage)
    assert d.message.content == "here is a useful update"
    assert d.message.id == "proactive-pid9"
    assert read_proactive_tag(d.message.additional_kwargs) == {
        "is_proactive_push": True,
        "proactive_id": "pid9",
    }
    assert d.as_node == "model"
    assert d.metadata is not None
