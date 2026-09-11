"""Tests for EvoScientist.proactive.commit (decision + conflict-aware commit)."""

from __future__ import annotations

import logging

import httpx
import pytest
from langgraph.graph import END
from langgraph_sdk.errors import ConflictError

from EvoScientist.proactive.commit import (
    NO_PUSH_SENTINEL,
    commit_with_conflict_retry,
    decide_commit,
    make_gateway_commit_fn,
    read_proactive_tag,
)


def _decide(reply, *, pre_head="h1", current_head="h1"):
    return decide_commit(
        reply,
        source_thread_id="src-thread",
        workspace_dir="/ws",
        model="some-model",
        pre_head=pre_head,
        current_head=current_head,
        proactive_id="pid-123",
    )


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (None, "no_reply"),
        ("", "empty"),
        ("   ", "empty"),
        (NO_PUSH_SENTINEL, "no_push"),
        (f"  {NO_PUSH_SENTINEL}  ", "no_push"),
    ],
)
def test_skip_cases(reply, reason):
    d = _decide(reply)
    assert d.action == "skip"
    assert d.reason == reason
    assert d.message is None


def test_stale_when_source_head_changed():
    d = _decide("Hey, checking in!", pre_head="h1", current_head="h2")
    assert d.action == "stale"
    assert d.reason == "stale_source"
    assert d.message is None


def test_would_commit_builds_tagged_message_and_metadata():
    d = _decide("Hey, want to pick this back up?")
    assert d.action == "would_commit"
    assert d.as_node == "model"
    # tagged AIMessage
    assert d.message is not None
    assert d.message.content == "Hey, want to pick this back up?"
    tag = d.message.additional_kwargs["evoscientist"]
    assert tag == {"is_proactive_push": True, "proactive_id": "pid-123"}
    assert read_proactive_tag(d.message.additional_kwargs) == tag
    # metadata carries the agent_name the main-thread filter keys on
    assert d.metadata is not None
    assert d.metadata["agent_name"] == "EvoScientist"
    assert d.metadata["workspace_dir"] == "/ws"
    assert d.metadata["model"] == "some-model"
    assert "updated_at" in d.metadata


def test_would_commit_logs_the_ledger_line(caplog):
    with caplog.at_level(logging.INFO, logger="EvoScientist.proactive.commit"):
        _decide("Hey there")
    assert any(
        "push for thread src-thread: append AIMessage" in r.message
        for r in caplog.records
    )


def test_would_commit_does_not_mutate_or_write():
    # decide_commit is pure: returns a decision object, performs no I/O. The
    # message is a fresh AIMessage each call (no shared state).
    d1 = _decide("first")
    d2 = _decide("second")
    assert d1.message is not d2.message
    assert d1.message.content == "first"
    assert d2.message.content == "second"


# ---- commit_with_conflict_retry (ConflictError defer-and-retry) ----


def _conflict() -> ConflictError:
    resp = httpx.Response(409, request=httpx.Request("POST", "http://test"))
    return ConflictError("run in flight", response=resp, body=None)


class _FakeCommitFn:
    """Injectable commit callable; raises ConflictError on the first ``fail``
    attempts, then records the committed message."""

    def __init__(self, fail: int = 0) -> None:
        self._fail = fail
        self.calls = 0
        self.committed: list = []

    async def __call__(self, message, metadata, as_node) -> None:
        self.calls += 1
        if self.calls <= self._fail:
            raise _conflict()
        self.committed.append((message, metadata, as_node))


class _Head:
    """read_current_head callable that returns ``value`` and can be flipped."""

    def __init__(self, value: str | None) -> None:
        self.value = value

    def __call__(self) -> str | None:
        return self.value


async def _retry(commit_fn, head, *, pre_head="h1", max_retries=3):
    decision = _decide("Hey, want to pick this back up?")
    assert decision.action == "would_commit"
    return await commit_with_conflict_retry(
        decision,
        source_thread_id="src-thread",
        commit_fn=commit_fn,
        read_current_head=head,
        pre_head=pre_head,
        max_retries=max_retries,
    )


async def test_clean_commit_first_try():
    commit_fn = _FakeCommitFn(fail=0)
    outcome = await _retry(commit_fn, _Head("h1"))
    assert outcome.status == "committed"
    assert outcome.attempts == 1
    assert commit_fn.calls == 1
    assert commit_fn.committed[0][0].content == "Hey, want to pick this back up?"
    assert commit_fn.committed[0][2] == "model"


async def test_conflict_once_then_succeeds(caplog):
    commit_fn = _FakeCommitFn(fail=1)
    with caplog.at_level(logging.WARNING, logger="EvoScientist.proactive.commit"):
        outcome = await _retry(commit_fn, _Head("h1"))
    assert outcome.status == "committed"
    assert outcome.attempts == 2
    assert commit_fn.calls == 2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "ConflictError" in warnings[0].message


async def test_conflict_then_head_moved_discards_as_stale():
    commit_fn = _FakeCommitFn(fail=1)
    head = _Head("h1")

    async def move_head_backoff(_attempt: int) -> None:
        head.value = "h2"  # the in-flight user run committed

    decision = _decide("Hey there")
    outcome = await commit_with_conflict_retry(
        decision,
        source_thread_id="src-thread",
        commit_fn=commit_fn,
        read_current_head=head,
        pre_head="h1",
        backoff=move_head_backoff,
    )
    assert outcome.status == "stale"
    assert outcome.reason == "stale_source"
    # committed exactly zero times: the one call raised, the retry saw the moved head
    assert commit_fn.committed == []


async def test_conflict_every_attempt_exhausts_retries(caplog):
    commit_fn = _FakeCommitFn(fail=99)
    with caplog.at_level(logging.WARNING, logger="EvoScientist.proactive.commit"):
        outcome = await _retry(commit_fn, _Head("h1"), max_retries=3)
    assert outcome.status == "conflict_exhausted"
    assert outcome.attempts == 4  # max_retries + 1
    assert commit_fn.calls == 4
    assert commit_fn.committed == []


async def test_head_already_moved_at_entry_is_stale():
    commit_fn = _FakeCommitFn(fail=0)
    outcome = await _retry(commit_fn, _Head("h2"), pre_head="h1")
    assert outcome.status == "stale"
    assert outcome.attempts == 0
    assert commit_fn.calls == 0


async def test_non_would_commit_decision_rejected():
    skip = _decide(NO_PUSH_SENTINEL)
    assert skip.action == "skip"
    with pytest.raises(ValueError, match="would_commit"):
        await commit_with_conflict_retry(
            skip,
            source_thread_id="src-thread",
            commit_fn=_FakeCommitFn(),
            read_current_head=_Head("h1"),
            pre_head="h1",
        )


# ---- make_gateway_commit_fn (server-path commit) ----


class _FakeGateway:
    """Widened update_state_values; raises ConflictError on the first ``fail``
    appends and/or the first ``fail_clear`` clears (clear = values is None)."""

    def __init__(self, *, fail_appends=0, fail_clears=0):
        self._fail_appends = fail_appends
        self._fail_clears = fail_clears
        self._appends = 0
        self._clears = 0
        self.calls = []

    async def update_state_values(
        self, target, thread_id, values, *, as_node=None, metadata=None
    ):
        self.calls.append({"values": values, "as_node": as_node, "metadata": metadata})
        if values is not None:  # append
            self._appends += 1
            if self._appends <= self._fail_appends:
                raise _conflict()
        else:  # clear
            self._clears += 1
            if self._clears <= self._fail_clears:
                raise _conflict()


def _appends(gw):
    return [c for c in gw.calls if c["values"] is not None]


async def _retry_via_gateway(gw, *, max_retries=3):
    decision = _decide("Hey, want to pick this back up?")
    return await commit_with_conflict_retry(
        decision,
        source_thread_id="src-thread",
        commit_fn=make_gateway_commit_fn(gw, "target", "src-thread"),
        read_current_head=_Head("h1"),
        pre_head="h1",
        max_retries=max_retries,
    )


async def test_gateway_commit_fn_appends_then_clears():
    gw = _FakeGateway()
    commit_fn = make_gateway_commit_fn(gw, "target", "src-thread")
    d = _decide("Hello there")
    await commit_fn(d.message, d.metadata, d.as_node)
    assert len(gw.calls) == 2
    append, clear = gw.calls
    assert append["values"] == {"messages": [d.message]}
    assert append["as_node"] == "model"
    assert append["metadata"] == d.metadata
    assert clear["values"] is None
    assert clear["as_node"] == END


async def test_gateway_commit_conflict_on_append_then_succeeds():
    gw = _FakeGateway(fail_appends=1)
    outcome = await _retry_via_gateway(gw)
    assert outcome.status == "committed"
    assert outcome.attempts == 2
    assert len(_appends(gw)) == 2  # first conflicted (uncommitted), second landed


async def test_gateway_commit_append_conflict_exhausts():
    gw = _FakeGateway(fail_appends=99)
    outcome = await _retry_via_gateway(gw, max_retries=2)
    assert outcome.status == "conflict_exhausted"
    assert outcome.attempts == 3


async def test_gateway_commit_clear_conflict_is_swallowed_no_double_append():
    # Every clear conflicts, but the append (the actual commit) must land exactly
    # once — the clear-conflict must NOT trigger a retry that re-appends.
    gw = _FakeGateway(fail_clears=99)
    outcome = await _retry_via_gateway(gw)
    assert outcome.status == "committed"
    assert outcome.attempts == 1
    assert len(_appends(gw)) == 1


def test_reply_over_the_length_cap_is_skipped_as_too_long():
    d = _decide("x" * 4001)
    assert (d.action, d.reason) == ("skip", "too_long")
    assert d.message is None


def test_reply_at_the_length_cap_still_commits():
    d = _decide("x" * 4000)
    assert d.action == "would_commit"


def test_tool_call_envelope_as_text_is_skipped():
    d = _decide("<tool_call>execute<arg_key>command</arg_key></tool_call>")
    assert (d.action, d.reason) == ("skip", "tool_call_text")
    assert d.message is None


def test_committed_push_carries_a_stable_id():
    d = _decide("Hey there")
    assert d.message.id == "proactive-pid-123"


def test_length_cap_is_configurable():
    d = decide_commit(
        "hello world",
        source_thread_id="s",
        workspace_dir=None,
        model=None,
        pre_head=None,
        current_head=None,
        proactive_id="p",
        max_reply_chars=5,
    )
    assert d.reason == "too_long"
