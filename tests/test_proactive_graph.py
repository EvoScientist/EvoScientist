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
from EvoScientist.proactive.commit import build_proactive_tag
from EvoScientist.proactive.gate import ProactiveGateSettings

# ---- settings ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("proactive_timezone", "expected"),
    [("", "Europe/Warsaw"), ("America/New_York", "America/New_York")],
)
def test_quiet_hours_use_the_host_zone_not_the_scheduler_zone(
    monkeypatch, proactive_timezone, expected
):
    monkeypatch.setattr("tzlocal.get_localzone_name", lambda: "Europe/Warsaw")
    cfg = MagicMock(
        proactive_idle_minutes=60,
        proactive_quiet_hours="",
        proactive_timezone=proactive_timezone,
        scheduler_default_timezone="Asia/Tokyo",
    )

    assert proactive_graph._gate_settings(cfg).timezone == expected


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
        if reply.get("always_error") is not None:
            raise reply["always_error"]
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


def test_a_gated_thread_never_builds_the_shadow(tick, workspace, monkeypatch):
    run, threads, _, _ = tick

    def failing_build(deps, dirs):
        raise RuntimeError("MCP load failed")

    monkeypatch.setattr(proactive_graph, "_build_shadow", failing_build)
    threads["t1"] = _thread(workspace, armed=False)
    threads["t2"] = _thread(workspace)

    state = run(_candidates("t1", "t2"))

    assert state["last_tick"] == [
        {"thread_id": "t1", "stage": "gate_rejected", "reason": "not_armed"},
        {"thread_id": "t2", "stage": "error", "reason": ""},
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


def test_a_thread_without_folders_is_skipped(tick, workspace):
    run, threads, shadow_calls, _ = tick
    entry = _thread(workspace)
    del entry["thread"]["metadata"]["workspace_dir"]
    threads["t1"] = entry

    state = run(_candidates("t1"))

    assert shadow_calls == []
    assert state["last_tick"][0]["reason"] == "no_folders"


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


def test_a_thread_that_keeps_failing_is_tried_1_2_4_ticks_apart(tick, workspace):
    run, threads, shadow_calls, reply = tick
    threads["t1"] = _thread(workspace)
    reply["always_error"] = RuntimeError("context window exceeded")

    tried = []
    for n in range(1, 9):
        before = len(shadow_calls)
        state = run(_candidates("t1"))
        if len(shadow_calls) > before:
            tried.append(n)
            assert state["last_tick"][0]["stage"] == "error"
        else:
            assert state["last_tick"][0]["reason"] == "backoff"

    assert tried == [1, 2, 4, 8]
    assert state["failures"]["t1"]["count"] == 4
    assert state["evaluated"] == {}


def test_a_new_message_ends_the_backoff(tick, workspace):
    run, threads, shadow_calls, reply = tick
    threads["t1"] = _thread(workspace)
    reply["always_error"] = RuntimeError("context window exceeded")
    run(_candidates("t1"))
    run(_candidates("t1"))  # second failure: the next tick is skipped
    threads["t1"] = _thread(workspace, last_id="m3")
    del reply["always_error"]

    state = run(_candidates("t1"))

    assert len(shadow_calls) == 3
    assert state["last_tick"][0]["stage"] == "decided"
    assert state["failures"] == {}
    assert state["evaluated"] == {"t1": "m3"}


def test_the_backoff_stops_growing_at_six_hours():
    record = None
    for _ in range(20):
        record = proactive_graph._failure_after(record, "m2")
    assert record["skip"] == proactive_graph.MAX_BACKOFF_TICKS
    assert (record["skip"] + 1) * 10 == 6 * 60
    assert proactive_graph._failure_after(record, "m3") == {
        "message_id": "m3",
        "count": 1,
        "skip": 0,
    }


def test_a_thread_back_in_the_candidates_after_its_push_is_not_pushed_again(
    tick, workspace
):
    run, threads, shadow_calls, _ = tick
    threads["t1"] = _thread(workspace)
    run(_candidates("t1"))
    threads["t1"]["state"]["values"]["messages"].append(
        {
            "type": "ai",
            "content": "The benchmark finished: all green.",
            "id": "proactive-p1",
            "additional_kwargs": build_proactive_tag("p1"),
        }
    )
    run(_candidates())

    state = run(_candidates("t1"))

    assert len(shadow_calls) == 1
    assert state["decisions"] == []
    assert state["last_tick"][0]["reason"] == "no_new_information"


def test_a_decision_is_stamped_when_it_is_made(tick, workspace, monkeypatch):
    run, threads, _, _ = tick
    threads["t1"] = _thread(workspace)
    returned_at = []

    async def slow_shadow(graph, messages, trigger):
        await asyncio.sleep(0.05)
        returned_at.append(datetime.now(UTC))
        return "The benchmark finished: all green."

    monkeypatch.setattr("EvoScientist.proactive.shadow.run_shadow_turn", slow_shadow)

    [decision] = run(_candidates("t1"))["decisions"]

    assert datetime.fromisoformat(decision["decided_at"]) >= returned_at[0]


def test_one_failing_thread_does_not_stop_the_tick(tick, workspace):
    run, threads, _, _ = tick
    threads["t1"] = {"thread": None, "state": None}  # .get() on None raises
    threads["t2"] = _thread(workspace)

    state = run(_candidates("t1", "t2"))

    assert state["last_tick"][0] == {"thread_id": "t1", "stage": "error", "reason": ""}
    assert [d["thread_id"] for d in state["decisions"]] == ["t2"]


def test_a_tick_whose_dependencies_fail_to_load_fails_alone(
    tick, workspace, monkeypatch
):
    run, threads, shadow_calls, _ = tick
    threads["t1"] = _thread(workspace)
    working = proactive_graph._load_deps

    def failing(previous):
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(proactive_graph, "_load_deps", failing)
    with pytest.raises(RuntimeError, match="config unreadable"):
        run(_candidates("t1"))

    monkeypatch.setattr(proactive_graph, "_load_deps", working)
    state = run(_candidates("t1"))

    assert len(shadow_calls) == 1
    assert state["last_tick"][0]["stage"] == "decided"
