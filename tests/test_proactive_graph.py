"""Tests for the proactive cron helpers and the ``proactive`` graph tick."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_sdk.errors import NotFoundError

from EvoScientist.proactive import cron as proactive_cron
from EvoScientist.proactive import graph as proactive_graph
from EvoScientist.proactive.gate import ProactiveGateSettings

# ---- cron helpers -----------------------------------------------------------


def test_cron_input_is_the_whole_candidate_list():
    assert proactive_cron.cron_input(["a", "b"]) == {
        "candidates": [{"thread_id": "a"}, {"thread_id": "b"}]
    }
    assert proactive_cron.cron_input([]) == {"candidates": []}


def test_create_proactive_cron_binds_a_rejecting_cron_to_the_thread(workspace):
    client = MagicMock()
    proactive_cron.create_proactive_cron(
        client, workspace, thread_id="t-cron", candidates=["t1"], timezone="UTC"
    )
    args = client.crons.create_for_thread.call_args
    assert args.args == ("t-cron", "proactive")
    assert args.kwargs["schedule"] == "*/10 * * * *"
    assert args.kwargs["input"] == {"candidates": [{"thread_id": "t1"}]}
    assert args.kwargs["metadata"]["run_kind"] == "proactive_check"
    assert args.kwargs["metadata"]["workspace_dir"] == workspace.key
    assert args.kwargs["multitask_strategy"] == "reject"
    assert args.kwargs["durability"] == "exit"


def test_create_proactive_thread_is_tagged_with_the_workspace(workspace):
    client = MagicMock()
    client.threads.create.return_value = {"thread_id": "t-cron"}
    assert proactive_cron.create_proactive_thread(client, workspace) == "t-cron"
    assert client.threads.create.call_args.kwargs["metadata"] == {
        "run_kind": "proactive_check",
        "workspace_dir": workspace.key,
    }


def test_list_proactive_crons_keeps_only_this_workspace(workspace, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    client = MagicMock()
    client.crons.search.side_effect = [
        [
            {"cron_id": "c1", "metadata": {"workspace_dir": workspace.key}},
            {"cron_id": "c2", "metadata": {"workspace_dir": other.as_posix()}},
        ],
        [],
    ]
    rows = proactive_cron.list_proactive_crons(client, workspace)
    assert [r["cron_id"] for r in rows] == ["c1"]
    assert client.crons.search.call_args.kwargs["metadata"] == {
        "run_kind": "proactive_check"
    }


def test_sessions_prunes_the_same_graph_id():
    from EvoScientist import sessions

    assert sessions.PROACTIVE_GRAPH_ID == proactive_cron.PROACTIVE_GRAPH_ID


# ---- the tick -----------------------------------------------------------------

NOW = datetime.now(UTC)
QUIET_SINCE = (NOW - timedelta(hours=5)).isoformat()


class _FakeThreads:
    def __init__(self, threads):
        self._threads = threads
        self.state_reads = []

    async def get(self, thread_id):
        if thread_id not in self._threads:
            raise NotFoundError("gone", response=MagicMock(), body=None)
        return self._threads[thread_id]["thread"]

    async def get_state(self, thread_id):
        self.state_reads.append(thread_id)
        return self._threads[thread_id]["state"]


def _thread(workspace, *, armed=True, status="idle", last_id="m2"):
    return {
        "thread": {
            "metadata": {
                "workspace_dir": workspace.key,
                **({"proactive": "on"} if armed else {}),
            },
            "status": status,
            "state_updated_at": QUIET_SINCE,
        },
        "state": {
            "values": {
                "messages": [
                    {"type": "human", "content": "run it", "id": "m1"},
                    {"type": "ai", "content": "I will report back", "id": last_id},
                ]
            },
            "checkpoint": {"checkpoint_id": "h1"},
            "interrupts": [],
        },
    }


@pytest.fixture
def tick(monkeypatch, workspace):
    """A compiled tick with fake config, client and shadow; returns a runner."""
    shadow_calls = []
    reply = {"text": "The benchmark finished: all green."}
    threads = {}

    deps = proactive_graph._Deps(
        key=("k",),
        cfg=MagicMock(),
        settings=ProactiveGateSettings(idle_minutes=60, quiet_hours="", timezone="UTC"),
        url="http://127.0.0.1:1",
    )
    monkeypatch.setattr(proactive_graph, "_load_deps", lambda previous: deps)
    monkeypatch.setattr(proactive_graph, "_build_shadow", lambda d, dirs: object())

    async def fake_shadow(graph, messages, trigger):
        shadow_calls.append([m.id for m in messages])
        if reply.get("error") is not None:
            raise reply.pop("error")
        return reply["text"]

    monkeypatch.setattr("EvoScientist.proactive.shadow.run_shadow_turn", fake_shadow)
    fake_client = MagicMock()
    fake_client.threads = _FakeThreads(threads)
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.sdk.get_langgraph_async_client",
        lambda **kwargs: fake_client,
    )

    graph = proactive_graph.build_proactive_graph(workspace=workspace)
    graph.checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "cron-thread"}}

    def run(candidates):
        return asyncio.run(graph.ainvoke({"candidates": candidates}, config))

    return run, threads, shadow_calls, reply


def _candidates(*ids):
    return [{"thread_id": tid} for tid in ids]


def test_tick_records_a_push_decision(tick, workspace):
    run, threads, shadow_calls, _ = tick
    threads["t1"] = _thread(workspace)

    state = run(_candidates("t1"))

    assert shadow_calls == [["m1", "m2"]]
    [decision] = state["decisions"]
    assert decision["thread_id"] == "t1"
    assert decision["head"] == "h1"
    assert decision["text"] == "The benchmark finished: all green."
    assert decision["proactive_id"]
    assert state["evaluated"] == {"t1": "m2"}
    assert state["last_tick"] == [
        {"thread_id": "t1", "stage": "decided", "reason": "ok"}
    ]


def test_next_tick_skips_a_thread_with_nothing_new_and_clears_old_decisions(
    tick, workspace
):
    run, threads, shadow_calls, _ = tick
    threads["t1"] = _thread(workspace)
    run(_candidates("t1"))

    state = run(_candidates("t1"))

    assert len(shadow_calls) == 1
    assert state["decisions"] == []
    assert state["last_tick"][0]["reason"] == "no_new_information"
    assert state["evaluated"] == {"t1": "m2"}


def test_a_new_message_makes_the_thread_a_candidate_again(tick, workspace):
    run, threads, shadow_calls, reply = tick
    threads["t1"] = _thread(workspace)
    reply["text"] = "NO_PUSH"
    run(_candidates("t1"))
    threads["t1"] = _thread(workspace, last_id="m3")

    state = run(_candidates("t1"))

    assert len(shadow_calls) == 2
    assert state["evaluated"] == {"t1": "m3"}


@pytest.mark.parametrize(
    ("thread_kwargs", "reason"),
    [
        ({"armed": False}, "not_armed"),
        ({"status": "busy"}, "in_flight"),
        ({"status": "interrupted"}, "in_flight"),
    ],
)
def test_gated_threads_get_no_shadow_turn(tick, workspace, thread_kwargs, reason):
    run, threads, shadow_calls, _ = tick
    threads["t1"] = _thread(workspace, **thread_kwargs)

    state = run(_candidates("t1"))

    assert shadow_calls == []
    assert state["decisions"] == []
    assert state["last_tick"] == [
        {"thread_id": "t1", "stage": "gate_rejected", "reason": reason}
    ]


def test_a_thread_of_another_workspace_is_skipped(tick, workspace, tmp_path):
    run, threads, shadow_calls, _ = tick
    other = tmp_path / "other"
    other.mkdir()
    entry = _thread(workspace)
    entry["thread"]["metadata"]["workspace_dir"] = other.as_posix()
    threads["t1"] = entry

    state = run(_candidates("t1"))

    assert shadow_calls == []
    assert state["last_tick"][0]["reason"] == "other_workspace"


def test_a_deleted_thread_is_reported_and_dropped(tick, workspace):
    run, threads, _, _ = tick
    threads["t1"] = _thread(workspace)
    run(_candidates("t1"))
    del threads["t1"]

    state = run(_candidates("t1"))

    assert state["last_tick"] == [{"thread_id": "t1", "stage": "gone", "reason": ""}]
    assert state["evaluated"] == {}


def test_threads_that_leave_the_candidates_drop_out_of_the_record(tick, workspace):
    run, threads, _, _ = tick
    threads["t1"] = _thread(workspace)
    threads["t2"] = _thread(workspace)
    run(_candidates("t1", "t2"))

    state = run(_candidates("t2"))

    assert set(state["evaluated"]) == {"t2"}


def test_a_failed_shadow_turn_is_retried_on_the_next_tick(tick, workspace):
    run, threads, shadow_calls, reply = tick
    threads["t1"] = _thread(workspace)
    reply["error"] = RuntimeError("provider down")

    state = run(_candidates("t1"))

    assert state["last_tick"] == [{"thread_id": "t1", "stage": "error", "reason": ""}]
    assert state["evaluated"] == {}

    state = run(_candidates("t1"))

    assert len(shadow_calls) == 2
    assert state["last_tick"][0]["stage"] == "decided"
    assert state["evaluated"] == {"t1": "m2"}


def test_one_failing_thread_does_not_stop_the_tick(tick, workspace):
    run, threads, _, _ = tick
    threads["t1"] = {"thread": None, "state": None}  # .get() on None raises
    threads["t2"] = _thread(workspace)

    state = run(_candidates("t1", "t2"))

    assert state["last_tick"][0] == {"thread_id": "t1", "stage": "error", "reason": ""}
    assert [d["thread_id"] for d in state["decisions"]] == ["t2"]
