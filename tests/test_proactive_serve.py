"""Tests for EvoScientist.proactive.serve (serve's proactive controller)."""

from __future__ import annotations

import itertools
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from langgraph_sdk.errors import NotFoundError

from EvoScientist.paths import Workspace
from EvoScientist.proactive.commit import read_proactive_tag
from EvoScientist.proactive.serve import (
    DECISION_MAX_AGE_SECONDS,
    POLL_INTERVAL_SECONDS,
    STALE_TICK_SECONDS,
    ProactiveServe,
    clear_post_append_next,
    terminal_node,
)

MAIN_GRAPH = {
    "nodes": [],
    "edges": [
        {"source": "__start__", "target": "model"},
        {"source": "model", "target": "HumanInTheLoopMiddleware.after_model"},
        {
            "source": "HumanInTheLoopMiddleware.after_model",
            "target": "Todo.after_model",
        },
        {"source": "Todo.after_model", "target": "tools", "conditional": True},
        {
            "source": "Todo.after_model",
            "target": "memory.after_agent",
            "conditional": True,
        },
        {"source": "tools", "target": "model"},
        {"source": "memory.after_agent", "target": "Interp.after_agent"},
        {"source": "Interp.after_agent", "target": "__end__"},
    ],
}


# ---- terminal node ----------------------------------------------------------


def test_terminal_node_is_the_only_unconditional_edge_to_end():
    assert terminal_node(MAIN_GRAPH) == "Interp.after_agent"


def test_terminal_node_ignores_conditional_edges_to_end():
    graph = {
        "edges": [
            {"source": "__start__", "target": "a"},
            {"source": "a", "target": "__end__", "conditional": True},
        ]
    }
    assert terminal_node(graph) is None


def test_terminal_node_is_none_when_not_unique():
    graph = {
        "edges": [
            {"source": "a", "target": "__end__"},
            {"source": "b", "target": "__end__"},
        ]
    }
    assert terminal_node(graph) is None


# ---- a fake dev server -------------------------------------------------------


class _Crons:
    def __init__(self):
        self.rows = []
        self.created = []
        self.updated = []
        self._ids = itertools.count(1)

    def search(self, *, metadata, limit, offset):
        rows = [
            r
            for r in self.rows
            if all(r["metadata"].get(k) == v for k, v in metadata.items())
        ]
        return rows[offset : offset + limit]

    def delete(self, cron_id):
        self.rows = [r for r in self.rows if r["cron_id"] != cron_id]

    def create_for_thread(self, thread_id, assistant_id, **kwargs):
        row = {
            "cron_id": f"c{next(self._ids)}",
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            **kwargs,
        }
        self.rows.append(row)
        self.created.append(row)
        return row

    def update(self, cron_id, **kwargs):
        if not any(r["cron_id"] == cron_id for r in self.rows):
            raise NotFoundError("no cron", response=MagicMock(), body=None)
        self.updated.append((cron_id, kwargs))


class _Threads:
    def __init__(self):
        self.threads = {}
        self.states = {}
        self.created = []
        self.writes = []

    def get(self, thread_id):
        if thread_id not in self.threads:
            raise NotFoundError("gone", response=MagicMock(), body=None)
        return self.threads[thread_id]

    def get_state(self, thread_id):
        return self.states[thread_id]

    def search(self, *, metadata, limit):
        return [
            {"thread_id": tid}
            for tid, t in self.threads.items()
            if all((t.get("metadata") or {}).get(k) == v for k, v in metadata.items())
        ][:limit]

    def create(self, *, metadata):
        tid = f"cron-thread-{len(self.created) + 1}"
        self.threads[tid] = {"thread_id": tid, "metadata": metadata, "status": "idle"}
        self.created.append(tid)
        return {"thread_id": tid}

    def update(self, thread_id, *, metadata):
        self.threads[thread_id].setdefault("metadata", {}).update(metadata)

    def update_state(self, thread_id, values, *, as_node):
        self.writes.append((thread_id, values, as_node))
        return {"checkpoint": {"checkpoint_id": f"after-{len(self.writes)}"}}


class _Runs:
    def __init__(self):
        self.running = {}
        self.cancelled = []

    def list(self, thread_id, *, status):
        return self.running.get(thread_id, []) if status == "running" else []

    def cancel(self, thread_id, run_id, *, action):
        self.cancelled.append((thread_id, run_id, action))


class _Assistants:
    def __init__(self, graph):
        self.graph = graph

    def get_graph(self, assistant_id):
        assert assistant_id == "EvoScientist"
        return self.graph


class _Client:
    def __init__(self, graph=MAIN_GRAPH):
        self.crons = _Crons()
        self.threads = _Threads()
        self.runs = _Runs()
        self.assistants = _Assistants(graph)


@pytest.fixture
def setup(workspace):
    client = _Client()
    origins = {"t1"}
    published = []
    controller = ProactiveServe(
        workspace=workspace,
        client=client,
        timezone="UTC",
        publish=lambda tid, text: published.append((tid, text)) or True,
        can_deliver=lambda tid: tid in origins,
    )
    return controller, client, origins, published


def _source(client, tid="t1", *, head="h1", status="idle", interrupts=()):
    client.threads.threads[tid] = {"thread_id": tid, "status": status, "metadata": {}}
    client.threads.states[tid] = {
        "checkpoint": {"checkpoint_id": head},
        "next": [],
        "interrupts": list(interrupts),
    }


def _tick(client, controller, *, decisions=(), age=0.0, status="idle"):
    """Make the cron thread look like it just ticked (``age`` seconds ago)."""
    tid = controller._cron_thread_id
    ticked_at = (datetime.now(UTC) - timedelta(seconds=age)).isoformat()
    client.threads.threads[tid]["status"] = status
    client.threads.threads[tid]["updated_at"] = ticked_at
    client.threads.states[tid] = {
        "created_at": ticked_at,
        "values": {"decisions": list(decisions)},
    }


def _decision(pid="p1", tid="t1", head="h1", text="The run finished.", age=0.0):
    return {
        "proactive_id": pid,
        "thread_id": tid,
        "head": head,
        "text": text,
        "decided_at": (datetime.now(UTC) - timedelta(seconds=age)).isoformat(),
    }


# ---- unavailable ---------------------------------------------------------------


def test_unavailable_controller_does_nothing(workspace):
    controller = ProactiveServe(
        workspace=workspace, client=None, unavailable_reason="local backend"
    )
    controller.armed.add("t1")
    controller.reconcile()
    controller.mirror()
    controller.poll(current_thread_id="t1")
    controller.stop()
    assert controller.available is False


# ---- reconcile / mirror / stop -------------------------------------------------


def test_reconcile_with_nothing_armed_deletes_leftover_crons(setup, workspace):
    controller, client, _, _ = setup
    client.crons.rows.append(
        {
            "cron_id": "old",
            "metadata": {"run_kind": "proactive_check", "workspace_dir": workspace.key},
        }
    )
    controller.reconcile()
    assert client.crons.rows == []
    assert client.crons.created == []


def test_reconcile_creates_one_cron_on_a_new_cron_thread(setup, workspace):
    controller, client, _, _ = setup
    controller.armed.add("t1")
    controller.reconcile()
    [cron] = client.crons.rows
    assert cron["thread_id"] == "cron-thread-1"
    assert cron["input"] == {"candidates": [{"thread_id": "t1"}]}
    assert cron["multitask_strategy"] == "reject"
    assert cron["durability"] == "exit"
    assert cron["metadata"]["workspace_dir"] == workspace.key
    assert client.threads.threads["cron-thread-1"]["metadata"]["run_kind"] == (
        "proactive_check"
    )


def test_reconcile_recreates_instead_of_keeping_and_reuses_the_cron_thread(setup):
    controller, client, _, _ = setup
    controller.armed.add("t1")
    controller.reconcile()
    first = client.crons.rows[0]["cron_id"]
    controller._cron_thread_id = None  # a new serve process

    controller.reconcile()

    assert [r["cron_id"] for r in client.crons.rows] != [first]
    assert len(client.crons.rows) == 1
    assert client.crons.rows[0]["thread_id"] == "cron-thread-1"
    assert client.threads.created == ["cron-thread-1"]


def test_reconcile_cancels_a_tick_left_running(setup):
    controller, client, _, _ = setup
    controller.armed.add("t1")
    controller.reconcile()
    client.runs.running["cron-thread-1"] = [{"run_id": "r9"}]

    controller.reconcile()

    assert client.runs.cancelled == [("cron-thread-1", "r9", "interrupt")]


def test_candidates_are_the_armed_threads_serve_can_deliver_to(setup):
    controller, _, origins, _ = setup
    controller.armed.update({"t1", "t2"})
    assert controller.candidates() == ["t1"]
    origins.add("t2")
    assert controller.candidates() == ["t1", "t2"]


def test_mirror_sends_the_whole_candidate_list(setup):
    controller, client, _, _ = setup
    controller.armed.add("t1")
    controller.reconcile()
    controller.mirror()
    controller.mirror()
    assert (
        client.crons.updated
        == [
            (
                client.crons.rows[0]["cron_id"],
                {"input": {"candidates": [{"thread_id": "t1"}]}},
            )
        ]
        * 2
    )


def test_mirror_creates_the_cron_when_armed_and_none_exists(setup):
    controller, client, _, _ = setup
    controller.armed.add("t1")
    controller.mirror()
    assert len(client.crons.rows) == 1


def test_mirror_with_nothing_armed_and_no_cron_does_nothing(setup):
    controller, client, _, _ = setup
    controller.mirror()
    assert client.crons.rows == []
    assert client.crons.updated == []


def test_mirror_recreates_a_cron_the_server_lost(setup):
    controller, client, _, _ = setup
    controller.armed.add("t1")
    controller.reconcile()
    client.crons.rows.clear()  # a dev-server restart without persistence

    controller.mirror()

    assert len(client.crons.rows) == 1
    assert controller._cron_id == client.crons.rows[0]["cron_id"]


def test_arm_stores_the_chat_with_the_flag(setup):
    controller, client, _, _ = setup
    updates = []
    client.threads.update = lambda tid, *, metadata: updates.append((tid, metadata))
    controller.origin_of = lambda tid: MagicMock(
        channel_type="telegram", chat_id="42", metadata={"thread_ts": "1.2"}
    )

    controller.arm("t1")

    assert updates == [
        (
            "t1",
            {
                "proactive": "on",
                "origin_channel_type": "telegram",
                "origin_chat_id": "42",
                "origin_metadata": '{"thread_ts": "1.2"}',
            },
        )
    ]


def _refuse_crons(client):
    def refuse(*args, **kwargs):
        raise RuntimeError("assistant not found")

    client.crons.create_for_thread = refuse
    client.crons.update = refuse


def test_a_failed_arm_leaves_the_thread_disarmed(setup):
    controller, client, _, _ = setup
    _source(client)
    _refuse_crons(client)

    with pytest.raises(RuntimeError, match="assistant not found"):
        controller.arm("t1")

    assert not controller.is_armed("t1")
    assert client.threads.threads["t1"]["metadata"] == {"proactive": "off"}


def test_a_failed_arm_of_an_armed_thread_keeps_it_armed(setup):
    controller, client, _, _ = setup
    _source(client)
    controller.arm("t1")
    _refuse_crons(client)

    with pytest.raises(RuntimeError, match="assistant not found"):
        controller.arm("t1")

    assert controller.is_armed("t1")
    assert client.threads.threads["t1"]["metadata"] == {"proactive": "on"}


def test_leaving_an_armed_thread_pauses_it_until_serve_resumes_it(setup):
    controller, client, _, _ = setup
    _source(client)
    controller.arm("t1")

    controller.leave("t1")

    assert not controller.is_armed("t1")
    assert client.threads.threads["t1"]["metadata"] == {"proactive": "paused"}
    assert client.crons.rows == []

    controller.adopt("t1")

    assert controller.is_armed("t1")
    assert client.threads.threads["t1"]["metadata"] == {"proactive": "on"}


def test_reconcile_writes_back_a_flag_a_restart_dropped(setup):
    controller, client, _, _ = setup
    _source(client)
    controller.armed.add("t1")

    controller.reconcile()

    assert client.threads.threads["t1"]["metadata"] == {"proactive": "on"}


def test_moving_workspace_leaves_the_old_armed_threads_behind(setup, tmp_path):
    controller, client, origins, _ = setup
    controller.armed.add("t1")
    controller.restored_thread_id = "t1"
    controller.reconcile()
    controller.stop()

    controller.move_to(Workspace(tmp_path / "other"))
    client.threads.threads["t2"] = {"thread_id": "t2", "metadata": {}}
    origins.add("t2")
    controller.arm("t2")
    assert len(client.crons.rows) == 1
    controller.disarm("t2")

    assert controller.armed == set()
    assert controller.restored_thread_id is None
    assert client.crons.rows == []


def test_a_successful_move_ends_an_earlier_failed_move(setup, tmp_path):
    from EvoScientist.proactive.serve import MOVE_FAILED_REASON

    controller, _, _, _ = setup
    controller.unavailable_reason = MOVE_FAILED_REASON

    controller.move_to(Workspace(tmp_path / "other"))

    assert controller.available


def test_a_move_keeps_a_reason_it_did_not_cause(setup, tmp_path):
    controller, _, _, _ = setup
    controller.unavailable_reason = "the local gateway backend runs in-process"

    controller.move_to(Workspace(tmp_path / "other"))

    assert not controller.available


@pytest.mark.parametrize(
    ("metadata", "armed_before", "armed_after"),
    [
        ({"proactive": "on"}, set(), {"t1"}),
        ({"proactive": "off"}, {"t1"}, set()),
        ({}, {"t1"}, {"t1"}),
    ],
)
def test_adopt_takes_the_flag_from_the_resumed_thread(
    setup, metadata, armed_before, armed_after
):
    controller, client, _, _ = setup
    client.threads.threads["t1"] = {"thread_id": "t1", "metadata": metadata}
    controller.armed |= armed_before

    controller.adopt("t1")

    assert controller.armed == armed_after
    assert client.crons.updated == []  # the resume mirrors once, at its end


def test_adopt_ignores_a_thread_the_server_does_not_have(setup):
    controller, _, _, _ = setup
    controller.adopt("gone")
    assert controller.armed == set()


def test_stop_deletes_the_cron(setup):
    controller, client, _, _ = setup
    controller.armed.add("t1")
    controller.reconcile()
    controller.stop()
    assert client.crons.rows == []


# ---- poll: liveness ----------------------------------------------------------


@pytest.fixture
def armed(setup):
    controller, client, origins, published = setup
    controller.armed.add("t1")
    controller.reconcile()
    _source(client)
    return controller, client, origins, published


def test_poll_is_throttled(armed):
    controller, client, _, published = armed
    _tick(client, controller, decisions=[_decision()])
    controller.poll(current_thread_id="t1", now=1000.0)
    _tick(client, controller, decisions=[_decision(pid="p2")])
    controller.poll(current_thread_id="t1", now=1000.0 + POLL_INTERVAL_SECONDS - 1)
    assert [p[1] for p in published] == ["The run finished."]


def test_poll_recreates_a_missing_cron(armed):
    controller, client, _, _ = armed
    client.crons.rows.clear()
    controller.poll(current_thread_id="t1")
    assert len(client.crons.rows) == 1


def test_poll_recreates_the_cron_once_after_a_failed_tick(armed):
    controller, client, _, published = armed
    before = client.crons.rows[0]["cron_id"]
    controller._cron_created_at -= 60
    _tick(client, controller, decisions=[_decision()], status="error")
    controller.poll(current_thread_id="t1", now=1000.0)
    recreated = client.crons.rows[0]["cron_id"]
    assert recreated != before
    assert published == []

    # The thread stays ``error`` until the next tick ends: no second recreate.
    controller.poll(current_thread_id="t1", now=1000.0 + POLL_INTERVAL_SECONDS)
    assert client.crons.rows[0]["cron_id"] == recreated


def test_poll_recreates_a_stalled_cron(armed):
    controller, client, _, _ = armed
    before = client.crons.rows[0]["cron_id"]
    controller._cron_created_at = 0.0
    _tick(client, controller, age=STALE_TICK_SECONDS + 60)
    controller.poll(current_thread_id="t1")
    assert client.crons.rows[0]["cron_id"] != before


def test_poll_does_not_recreate_a_cron_that_has_not_ticked_yet(armed):
    controller, client, _, _ = armed
    before = client.crons.rows[0]["cron_id"]
    _tick(client, controller, age=STALE_TICK_SECONDS + 60)  # cron created just now
    controller.poll(current_thread_id="t1")
    assert client.crons.rows[0]["cron_id"] == before


# ---- poll: applying a push ---------------------------------------------------


def test_push_is_appended_in_one_write_as_the_terminal_node_and_delivered(armed):
    controller, client, _, published = armed
    _tick(client, controller, decisions=[_decision()])

    controller.poll(current_thread_id="t1")

    [(tid, values, as_node)] = client.threads.writes
    assert (tid, as_node) == ("t1", "Interp.after_agent")
    [message] = values["messages"]
    assert message["type"] == "ai"
    assert message["content"] == "The run finished."
    assert message["id"] == "proactive-p1"
    assert read_proactive_tag(message["additional_kwargs"])["proactive_id"] == "p1"
    assert published == [("t1", "The run finished.")]


def test_a_decision_is_applied_once(armed):
    controller, client, _, published = armed
    _tick(client, controller, decisions=[_decision()])
    controller.poll(current_thread_id="t1", now=1000.0)
    controller.poll(current_thread_id="t1", now=1000.0 + POLL_INTERVAL_SECONDS)
    assert len(client.threads.writes) == 1
    assert len(published) == 1


@pytest.mark.parametrize(
    "change",
    ["disarmed", "no_origin", "busy", "interrupted", "interrupt", "stale"],
)
def test_guards_skip_the_write(armed, change):
    controller, client, origins, published = armed
    if change == "disarmed":
        controller.armed.discard("t1")
        controller.armed.add("other")  # keep the poll running
    elif change == "no_origin":
        origins.clear()
    elif change in ("busy", "interrupted"):
        _source(client, status=change)
    elif change == "interrupt":
        _source(client, interrupts=[{"id": "i1"}])
    else:
        _source(client, head="h2")
    _tick(client, controller, decisions=[_decision()])

    controller.poll(current_thread_id="t1")

    assert client.threads.writes == []
    assert published == []


def test_poll_mirrors_a_candidate_whose_origin_appeared_after_the_last_mirror(
    armed,
):
    """A channel ``/resume`` into an armed thread mirrors before the origin is
    re-bound; the next poll sends the thread."""
    controller, client, origins, _ = armed
    controller.armed.add("t2")
    controller.mirror()
    _tick(client, controller)
    origins.add("t2")

    controller.poll(current_thread_id="t2")

    assert client.crons.updated[-1][1] == {
        "input": {"candidates": [{"thread_id": "t1"}, {"thread_id": "t2"}]}
    }
    sent = len(client.crons.updated)
    controller.poll(current_thread_id="t2", now=time.time() + POLL_INTERVAL_SECONDS)
    assert len(client.crons.updated) == sent


def test_a_decision_left_from_before_a_restart_is_not_applied(armed):
    controller, client, _, published = armed
    _tick(
        client,
        controller,
        decisions=[_decision(age=DECISION_MAX_AGE_SECONDS + 60)],
    )

    controller.poll(current_thread_id="t1")

    assert client.threads.writes == []
    assert published == []


def test_a_push_to_another_thread_switches_serve_to_it(armed):
    controller, client, origins, _ = armed
    origins.add("t0")
    controller.armed.add("t0")
    _source(client, "t0")
    _tick(client, controller, decisions=[_decision(tid="t0")])
    switched = []

    controller.poll(current_thread_id="t1", switch_to=switched.append)

    assert switched == ["t0"]


def test_without_a_unique_terminal_node_the_push_is_appended_as_model_and_cleared(
    workspace,
):
    graph = {"edges": [{"source": "model", "target": "after"}]}
    client = _Client(graph=graph)
    controller = ProactiveServe(
        workspace=workspace,
        client=client,
        publish=lambda tid, text: True,
        can_deliver=lambda tid: True,
    )
    controller.armed.add("t1")
    controller.reconcile()
    _source(client)
    _tick(client, controller, decisions=[_decision()])
    original_update = client.threads.update_state

    def update_state(thread_id, values, *, as_node):
        result = original_update(thread_id, values, as_node=as_node)
        if as_node == "model":
            client.threads.states["t1"] = {
                "checkpoint": {"checkpoint_id": result["checkpoint"]["checkpoint_id"]},
                "next": ["after"],
                "interrupts": [],
            }
        return result

    client.threads.update_state = update_state

    controller.poll(current_thread_id="t1")

    assert [(w[2], w[1]) for w in client.threads.writes][1] == ("__end__", None)
    assert client.threads.writes[0][2] == "model"


@pytest.mark.parametrize(
    "state",
    [
        {
            "checkpoint": {"checkpoint_id": "someone-else"},
            "next": ["after"],
            "interrupts": [],
        },
        {"checkpoint": {"checkpoint_id": "w1"}, "next": ["tools"], "interrupts": []},
        {
            "checkpoint": {"checkpoint_id": "w1"},
            "next": ["after"],
            "interrupts": [{"id": "i"}],
        },
    ],
)
def test_the_fallback_clear_leaves_any_other_state_alone(state):
    client = _Client()
    client.threads.states["t1"] = state
    assert (
        clear_post_append_next(
            client, "t1", written_checkpoint_id="w1", after_model=frozenset({"after"})
        )
        is False
    )
    assert client.threads.writes == []


# ---- arming ------------------------------------------------------------------


def test_arm_stores_the_flag_on_the_thread_and_creates_the_cron(setup):
    controller, client, _, _ = setup
    updates = []
    client.threads.update = lambda tid, *, metadata: updates.append((tid, metadata))

    controller.arm("t1")

    assert updates == [("t1", {"proactive": "on"})]
    assert controller.is_armed("t1")
    assert client.crons.rows[0]["input"] == {"candidates": [{"thread_id": "t1"}]}


def test_disarming_the_last_thread_deletes_the_cron(setup):
    controller, client, _, _ = setup
    updates = []
    client.threads.update = lambda tid, *, metadata: updates.append((tid, metadata))
    controller.arm("t1")

    controller.disarm("t1")

    assert updates[-1] == ("t1", {"proactive": "off"})
    assert not controller.is_armed("t1")
    assert client.crons.rows == []


def test_disarming_one_of_two_threads_keeps_the_cron(setup):
    controller, client, origins, _ = setup
    client.threads.update = lambda tid, *, metadata: None
    origins.add("t2")
    controller.arm("t1")
    controller.arm("t2")

    controller.disarm("t1")

    assert len(client.crons.rows) == 1
    assert client.crons.updated[-1][1] == {
        "input": {"candidates": [{"thread_id": "t2"}]}
    }


def test_arming_needs_an_available_controller(workspace):
    controller = ProactiveServe(workspace=workspace, unavailable_reason="local backend")
    with pytest.raises(RuntimeError, match="local backend"):
        controller.arm("t1")


# ---- the real main graph ------------------------------------------------------


def test_on_the_real_main_graph_a_push_leaves_nothing_to_run(workspace, monkeypatch):
    """A middleware that adds a second edge to ``__end__`` fails here, not by
    sending every push down the fallback."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver

    import EvoScientist.EvoScientist as es_mod
    from EvoScientist.config.settings import EvoScientistConfig

    class Fake(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    monkeypatch.setattr(es_mod, "_load_mcp_tools_cached", lambda **kwargs: {})
    graph = es_mod.create_cli_agent(
        work_dir=workspace.root,
        checkpointer=InMemorySaver(),
        config=EvoScientistConfig(),
        chat_model=Fake(messages=iter([AIMessage(content="hello")])),
        workspace=workspace,
    )
    # The shape GET /assistants/{id}/graph serves.
    node = terminal_node(graph.get_graph().to_json())
    assert node is not None

    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"messages": [HumanMessage(content="hi")]}, config)
    graph.update_state(
        config,
        {"messages": [AIMessage(content="push", id="proactive-p1")]},
        as_node=node,
    )

    state = graph.get_state(config)
    assert state.next == ()
    assert state.interrupts == ()
    assert state.values["messages"][-1].id == "proactive-p1"
