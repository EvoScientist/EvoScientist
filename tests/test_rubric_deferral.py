"""Grading deferred while this thread's background sub-agents or processes run."""

from __future__ import annotations

import asyncio
import logging
import socket
import sys

import httpcore
import httpx
import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.async_subagents import AsyncSubAgentState
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_sdk.errors import NotFoundError

from EvoScientist.message_meta import this_run_messages
from EvoScientist.middleware.background import BackgroundState
from EvoScientist.middleware.rubric import create_rubric_middleware
from tests.test_rubric_middleware import FAIL, PASS, _Main, _ScriptedGrader

GRADER_NODE = "EvoRubricMiddleware.after_agent"


class _WorkState(AsyncSubAgentState, BackgroundState):
    """The channels the async and background middleware add on the main agent."""


class _WorkChannels(AgentMiddleware):
    """The work channels, plus sub-agent tasks a turn starts (see ``_launch``)."""

    name = "work_channels"
    state_schema = _WorkState

    def __init__(self):
        super().__init__()
        self.launching: dict | None = None

    def after_model(self, state, runtime):
        tasks, self.launching = self.launching, None
        return {"async_tasks": tasks} if tasks else None


class _Live:
    """Fake live status: one answer per task read; per-process answers, then a table.

    A process "started during the turn" is a record whose live status is
    ``unknown`` when the run starts (so it is not skipped) and ``running``
    when grading is about to begin.
    """

    def __init__(self, task_answers=(), processes=None, process_answers=None):
        self.task_answers = list(task_answers)
        self.processes = dict(processes or {})
        self.process_answers = {k: list(v) for k, v in (process_answers or {}).items()}
        self.calls: list[tuple[str, str]] = []

    async def task_status(self, thread_id, run_id):
        self.calls.append((thread_id, run_id))
        answer = self.task_answers.pop(0) if self.task_answers else "success"
        if isinstance(answer, Exception):
            raise answer
        return answer

    def process_status(self, process_id):
        answers = self.process_answers.get(process_id)
        if answers:
            return answers.pop(0)
        return self.processes.get(process_id, "unknown")


STARTED_THIS_TURN = ["unknown", "running"]


def _agent(tmp_path, verdicts, replies, live, *, max_iterations=3):
    grader = _ScriptedGrader(verdicts=verdicts)
    rubric = create_rubric_middleware(
        model=grader,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        max_iterations=max_iterations,
        prepare_messages=this_run_messages,
        require_run_marker=True,
        task_status=live.task_status,
        process_status=live.process_status,
    )
    work = _WorkChannels()
    agent = create_agent(
        model=_Main(messages=iter([AIMessage(r) for r in replies])),
        tools=[],
        middleware=[work, rubric],
        checkpointer=InMemorySaver(),
    )
    agent.work = work
    return agent, grader


def _launch(agent, tasks):
    """The next turn starts these sub-agent tasks (after its first model call)."""
    agent.work.launching = tasks


def _op(seq, session="s1", **fields):
    return {"session": session, "seq": seq, **fields}


def _task(task_id="t1", run_id="r1", status="running"):
    return {
        task_id: {
            "task_id": task_id,
            "agent_name": "writing-agent",
            "thread_id": task_id,
            "run_id": run_id,
            "status": status,
        }
    }


def _process(process_id="p1", origin="t", status="running"):
    return {
        process_id: {
            "process_id": process_id,
            "name": "job",
            "status": status,
            "origin_thread_id": origin,
        }
    }


async def _run(agent, text, *, mode="thread", changes=(), thread="t", rubric=None):
    cfg = {"configurable": {"thread_id": thread}}
    if mode:
        cfg["configurable"]["rubric_run"] = {"mode": mode, "changes": list(changes)}
    payload = {"messages": [HumanMessage(text)]}
    if rubric is not None:
        payload["rubric"] = rubric
    events = []
    async for _m, data in agent.astream(payload, cfg, stream_mode=["custom"]):
        if isinstance(data, dict):
            events.append(data)
    return events, (await agent.aget_state(cfg)).values


def _results(events):
    return [e.get("result") for e in events if e["type"] == "rubric_evaluation_end"]


def _deferred(events):
    return [
        (e["tasks"], e["processes"])
        for e in events
        if e["type"] == "rubric_evaluation_deferred"
    ]


def _seed(agent, update, thread="t"):
    cfg = {"configurable": {"thread_id": thread}}
    agent.update_state(cfg, update, as_node=GRADER_NODE)
    agent.update_state(cfg, None, as_node="__end__")


async def test_running_task_defers_and_emits_the_event(tmp_path):
    live = _Live(task_answers=["running"])
    agent, grader = _agent(tmp_path, [PASS], ["a", "b"], live)
    _launch(agent, _task())
    events, values = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == []
    assert _deferred(events) == [(1, 0)]
    assert not [e for e in events if e["type"] == "rubric_evaluation_start"]
    assert grader.calls == 0
    assert values["_rubric_grading"]["deferred"] is True
    assert values["rubric"] == "- s"
    assert values["async_tasks"]["t1"]["status"] == "running"
    assert live.calls == [("t1", "r1")]


async def test_stored_running_but_live_finished_grades(tmp_path):
    live = _Live(task_answers=["success"])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _launch(agent, _task())
    events, values = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert _deferred(events) == []
    assert values["async_tasks"]["t1"]["status"] == "running"


async def test_unreadable_status_grades_and_warns(tmp_path, caplog):
    live = _Live(task_answers=[RuntimeError("boom")])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _launch(agent, _task())
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert any("could not read" in r.getMessage() for r in caplog.records)


async def test_not_found_grades_without_a_warning(tmp_path, caplog):
    gone = NotFoundError(
        "gone",
        response=httpx.Response(404, request=httpx.Request("GET", "http://x")),
        body=None,
    )
    live = _Live(task_answers=[gone])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _launch(agent, _task())
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_input_mode_ignores_background_work(tmp_path):
    live = _Live(task_answers=["running"], processes={"p1": "running"})
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _launch(agent, _task())
    _seed(agent, {"bg_processes": _process()})
    events, _ = await _run(agent, "task", mode="input", rubric="- s")
    assert _results(events) == ["satisfied"]
    assert live.calls == []


async def test_no_checklist_emits_no_deferred_event(tmp_path):
    live = _Live(task_answers=["running"])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _launch(agent, _task())
    events, _ = await _run(agent, "chat")
    assert events == []
    assert live.calls == []


async def test_followup_finishes_a_deferred_grading_from_the_task_message(tmp_path):
    live = _Live(task_answers=["running", "success"])
    agent, grader = _agent(tmp_path, [PASS], ["draft", "noted", "x"], live)
    _launch(agent, _task())
    await _run(agent, "the task", changes=[_op(1, next="- n")])
    events, values = await _run(
        agent, "[Async tasks update] t1 finished", mode="followup"
    )
    assert _results(events) == ["satisfied"]
    assert "the task" in grader.last_payload
    assert values["_next_rubric"] is None
    assert values["_rubric_grading"] is None


async def test_followup_with_nothing_deferred_does_not_grade_or_consume_next(tmp_path):
    live = _Live()
    agent, grader = _agent(tmp_path, [PASS], ["noted", "x"], live)
    events, values = await _run(
        agent, "[Async tasks update]", mode="followup", changes=[_op(1, next="- n")]
    )
    assert events == []
    assert grader.calls == 0
    assert values["_next_rubric"] == {"text": "- n", "id": "s1:1"}


async def test_followup_keeps_the_pass_count(tmp_path):
    # Each revision is followed by background work: FAIL, defer, FAIL, defer,
    # then the third pass ends the grading at the limit.
    live = _Live(task_answers=["success", "running", "success", "running", "success"])
    agent, grader = _agent(
        tmp_path, [FAIL, FAIL, FAIL], ["a", "b", "c", "d", "e", "f", "g"], live
    )
    _launch(agent, _task())
    first, _ = await _run(agent, "task", changes=[_op(1, next="- n")])
    assert _results(first) == ["needs_revision"]
    assert _deferred(first) == [(1, 0)]
    second, _ = await _run(agent, "update 1", mode="followup")
    assert _results(second) == ["needs_revision"]
    assert _deferred(second) == [(1, 0)]
    third, values = await _run(agent, "update 2", mode="followup")
    assert _results(third) == ["max_iterations_reached"]
    assert grader.calls == 3
    assert values["_next_rubric"] is None
    assert len(values["_rubric_evaluations"]) == 3
    assert len({e["grading_run_id"] for e in values["_rubric_evaluations"]}) == 1


async def test_followup_with_a_missing_start_returns_grader_error(tmp_path, caplog):
    live = _Live()
    agent, grader = _agent(tmp_path, [PASS], ["noted", "x"], live)
    _seed(
        agent,
        {
            "_next_rubric": {"text": "- n", "id": "s1:1"},
            "_rubric_applied": {"s1": 1},
            "rubric": "- n",
            "_rubric_grading": {
                "source": "next",
                "id": "s1:1",
                "start": "m-gone",
                "deferred": True,
                "skip": [],
            },
        },
    )
    with caplog.at_level(logging.WARNING):
        events, values = await _run(agent, "[Async tasks update]", mode="followup")
    assert _results(events) == ["grader_error"]
    assert grader.calls == 0
    assert values["rubric"] is None
    assert values["_rubric_grading"] is None
    assert values["_next_rubric"] is None
    # Nothing was graded, so nothing may read as a failure of the grader.
    (verdict,) = [e for e in events if e["type"] == "rubric_evaluation_end"]
    assert verdict["explanation"] == (
        "Not graded: the message that started this task is no longer in the thread."
    )
    assert [e["explanation"] for e in values["_rubric_evaluations"]] == [
        verdict["explanation"]
    ]
    assert [(r.name, r.levelname) for r in caplog.records] == [
        ("EvoScientist.middleware.rubric", "WARNING")
    ]
    assert "no longer in the thread" in caplog.records[0].getMessage()
    assert caplog.records[0].exc_info is None


async def test_a_callback_that_raises_does_not_stop_the_ungraded_ending(
    tmp_path, caplog, monkeypatch
):
    """As upstream does for a verdict: the callback's error is logged and the
    grading still ends."""

    def _broken(_evaluation):
        raise RuntimeError("callback broke")

    monkeypatch.setattr(
        "EvoScientist.middleware.rubric._log_rubric_evaluation", _broken
    )
    live = _Live()
    agent, _grader = _agent(tmp_path, [PASS], ["noted", "x"], live)
    _seed(
        agent,
        {
            "_next_rubric": {"text": "- n", "id": "s1:1"},
            "_rubric_applied": {"s1": 1},
            "rubric": "- n",
            "_rubric_grading": {
                "source": "next",
                "id": "s1:1",
                "start": "m-gone",
                "deferred": True,
                "skip": [],
            },
        },
    )
    with caplog.at_level(logging.WARNING):
        events, values = await _run(agent, "[Async tasks update]", mode="followup")
    assert _results(events) == ["grader_error"]
    assert values["_rubric_grading"] is None
    assert any(
        "on_evaluation callback raised" in r.getMessage() and r.exc_info
        for r in caplog.records
    )


async def test_deferred_sticky_grading_leaves_a_later_next_with_equal_text(tmp_path):
    live = _Live(task_answers=["running", "running", "success"])
    agent, _ = _agent(tmp_path, [PASS, PASS], ["a", "b", "c", "d", "e"], live)
    _launch(agent, _task())
    await _run(agent, "task", changes=[_op(1, sticky="- s")])
    stored, values = await _run(
        agent, "[Async tasks update]", mode="followup", changes=[_op(2, next="- s")]
    )
    assert _deferred(stored) == [(1, 0)]
    assert values["_next_rubric"] == {"text": "- s", "id": "s1:2"}
    done, values = await _run(agent, "[Async tasks update]", mode="followup")
    assert _results(done) == ["satisfied"]
    assert values["_next_rubric"] == {"text": "- s", "id": "s1:2"}
    after, values = await _run(agent, "next task")
    assert _results(after) == ["satisfied"]
    assert values["_next_rubric"] is None


async def test_typed_turn_while_deferred_keeps_the_start(tmp_path):
    live = _Live(task_answers=["running", "running", "success"])
    agent, grader = _agent(tmp_path, [PASS], ["a", "b", "c", "d"], live)
    _launch(agent, _task())
    await _run(agent, "the task", changes=[_op(1, sticky="- s")])
    more, values = await _run(agent, "and also this")
    assert _deferred(more) == [(1, 0)]
    assert values["_rubric_grading"]["deferred"] is True
    done, _ = await _run(agent, "[Async tasks update]", mode="followup")
    assert _results(done) == ["satisfied"]
    assert "the task" in grader.last_payload
    assert "and also this" in grader.last_payload


async def test_replaced_checklist_voids_the_deferred_grading(tmp_path):
    live = _Live(task_answers=["running", "success"])
    agent, grader = _agent(tmp_path, [PASS], ["a", "b", "c"], live)
    _launch(agent, _task())
    await _run(agent, "task", changes=[_op(1, sticky="- s")])
    events, values = await _run(
        agent,
        "[Async tasks update]",
        mode="followup",
        changes=[_op(2, sticky=None, next=None)],
    )
    assert events == []
    assert grader.calls == 0
    assert values["_rubric_grading"] is None
    assert values["rubric"] is None


async def test_process_started_this_turn_defers_and_its_exit_followup_grades(tmp_path):
    live = _Live(process_answers={"p1": STARTED_THIS_TURN})
    agent, grader = _agent(tmp_path, [PASS], ["a", "b", "c"], live)
    _seed(agent, {"bg_processes": _process()})
    events, values = await _run(agent, "start the job", changes=[_op(1, sticky="- s")])
    assert _deferred(events) == [(0, 1)]
    assert values["_rubric_grading"]["skip"] == []
    live.processes["p1"] = "success"
    done, _ = await _run(agent, "[Async tasks update] p1 exited", mode="followup")
    assert _results(done) == ["satisfied"]
    assert "start the job" in grader.last_payload


async def test_process_running_before_the_message_is_skipped(tmp_path):
    live = _Live(processes={"p1": "running"}, process_answers={"p1": STARTED_THIS_TURN})
    agent, _ = _agent(tmp_path, [PASS], ["a", "b", "c"], live)
    _seed(agent, {"bg_processes": _process()})
    first, _ = await _run(agent, "start", changes=[_op(1, sticky="- s")])
    assert _deferred(first) == [(0, 1)]
    # The process is still running when the next message is typed: skipped.
    events, values = await _run(agent, "next message")
    assert _results(events) == ["satisfied"]
    assert _deferred(events) == []
    assert values["_rubric_grading"] is None


async def test_foreign_process_does_not_defer(tmp_path):
    live = _Live(processes={"p1": "running"})
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _seed(agent, {"bg_processes": _process(origin="other-thread")})
    events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert _deferred(events) == []


async def test_process_with_an_empty_origin_counts_as_this_thread(tmp_path):
    live = _Live(process_answers={"p1": STARTED_THIS_TURN})
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _seed(agent, {"bg_processes": _process(origin=None)})
    events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _deferred(events) == [(0, 1)]


async def test_unknown_process_does_not_defer(tmp_path):
    live = _Live()  # registry knows nothing: a restarted CLI
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _seed(agent, {"bg_processes": _process()})
    events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]


async def test_task_and_process_report_both_counts(tmp_path):
    live = _Live(task_answers=["running"], process_answers={"p1": STARTED_THIS_TURN})
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _launch(agent, _task())
    _seed(agent, {"bg_processes": _process()})
    events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _deferred(events) == [(1, 1)]


def test_live_task_status_uses_a_fresh_client_and_closes_it(monkeypatch):
    from EvoScientist.langgraph_dev import sdk
    from EvoScientist.middleware import rubric as rubric_mod

    log = []

    class _Runs:
        async def get(self, thread_id, run_id):
            log.append(("get", thread_id, run_id))
            return {"status": "running"}

    class _Client:
        runs = _Runs()

    class _Scoped:
        def __init__(self, url, **_kwargs):
            log.append(("open", url))

        async def __aenter__(self):
            return _Client()

        async def __aexit__(self, *exc):
            log.append(("close",))

    monkeypatch.setattr(sdk, "langgraph_async_client_scoped", _Scoped)
    monkeypatch.setattr(
        sdk, "configured_langgraph_dev_url", lambda: "http://127.0.0.1:1"
    )
    status = asyncio.run(rubric_mod.live_task_status("t1", "r1"))
    assert status == "running"
    assert log == [("open", "http://127.0.0.1:1"), ("get", "t1", "r1"), ("close",)]


def test_scoped_client_has_no_transport_retries_and_closes_on_exit():
    from EvoScientist.langgraph_dev.sdk import langgraph_async_client_scoped

    async def _probe():
        async with langgraph_async_client_scoped("http://127.0.0.1:1") as client:
            http = client.http.client
            transport = http._transport
            return http, getattr(transport, "_pool", None)

    http, pool = asyncio.run(_probe())
    assert http.is_closed
    assert pool is not None
    assert pool._retries == 0


def _connect_error(
    root: BaseException, *, every_address: bool = False
) -> httpx.ConnectError:
    """An ``httpx.ConnectError`` chained the way httpx raises it, down to ``root``.

    ``every_address`` is the async client's shape: anyio tries each address and
    folds the OS errors under one ``OSError``.
    """
    inner: BaseException = root
    if every_address:
        inner = OSError("All connection attempts failed")
        inner.__cause__ = root
    core = httpcore.ConnectError(str(inner))
    core.__cause__ = inner
    exc = httpx.ConnectError(str(inner))
    exc.__cause__ = core
    return exc


async def test_connect_error_grades_without_a_warning(tmp_path, caplog):
    refused = _connect_error(
        ConnectionRefusedError(61, "Connect call failed"), every_address=True
    )
    live = _Live(task_answers=[refused])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _launch(agent, _task())
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_refusal_on_one_address_among_other_failures_is_quiet(tmp_path, caplog):
    # localhost on a host without IPv6: ::1 unreachable, 127.0.0.1 refused. A
    # refusal on any address proves nobody listens there: not a reason to warn.
    mixed = _connect_error(
        ExceptionGroup(
            "multiple connection attempts failed",
            [
                OSError(51, "Network is unreachable"),
                ConnectionRefusedError(61, "Connect call failed"),
            ],
        ),
        every_address=True,
    )
    live = _Live(task_answers=[mixed])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _launch(agent, _task())
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_connect_error_other_than_refusal_grades_with_a_warning(tmp_path, caplog):
    unresolved = _connect_error(socket.gaierror(8, "nodename nor servname provided"))
    live = _Live(task_answers=[unresolved])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _launch(agent, _task())
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert any("could not read" in r.getMessage() for r in caplog.records)


async def test_read_timeout_grades_with_a_warning(tmp_path, caplog):
    live = _Live(task_answers=[httpx.ReadTimeout("slow")])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _launch(agent, _task())
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert any("could not read" in r.getMessage() for r in caplog.records)


async def test_status_reads_are_skipped_when_the_server_is_unavailable(tmp_path):
    live = _Live(task_answers=["running"])
    grader = _ScriptedGrader(verdicts=[PASS])

    async def _down():
        return False

    rubric = create_rubric_middleware(
        model=grader,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        max_iterations=3,
        prepare_messages=this_run_messages,
        require_run_marker=True,
        task_status=live.task_status,
        process_status=live.process_status,
        server_available=_down,
    )
    agent = create_agent(
        model=_Main(messages=iter([AIMessage("a"), AIMessage("b")])),
        tools=[],
        middleware=[work := _WorkChannels(), rubric],
        checkpointer=InMemorySaver(),
    )
    agent.work = work
    _launch(agent, _task())
    events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert live.calls == []


def test_dev_server_reachable_needs_the_flag_and_the_probe(monkeypatch, caplog):
    from EvoScientist.langgraph_dev import manager
    from EvoScientist.middleware import rubric as rubric_mod

    probes = []
    answers = [(True, None)]
    monkeypatch.setattr(manager, "is_async_subagents_available", lambda: False)

    async def _probe(url):
        probes.append(url)
        return answers[0]

    monkeypatch.setattr(rubric_mod, "_probe_dev_server", _probe)
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.sdk.configured_langgraph_dev_url",
        lambda: "http://127.0.0.1:1",
    )
    assert asyncio.run(rubric_mod.dev_server_reachable()) is False
    assert probes == []
    monkeypatch.setattr(manager, "is_async_subagents_available", lambda: True)
    assert asyncio.run(rubric_mod.dev_server_reachable()) is True
    assert probes == ["http://127.0.0.1:1"]
    answers[0] = (False, None)  # refused: quiet
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        assert asyncio.run(rubric_mod.dev_server_reachable()) is False
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    answers[0] = (False, "health check timed out")  # slow: say so
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        assert asyncio.run(rubric_mod.dev_server_reachable()) is False
    assert any("timed out" in r.getMessage() for r in caplog.records)


async def test_probe_dev_server_tells_refusal_from_other_failures():
    from EvoScientist.middleware.rubric import _probe_dev_server

    # Nothing listens on port 1: a refused connection, no reason to report.
    # Windows does not refuse a closed loopback port within the probe's second
    # and reports the timeout instead.
    answers, reason = await _probe_dev_server("http://127.0.0.1:1")
    assert answers is False
    assert reason is None or (sys.platform == "win32" and reason == "ConnectTimeout")

    def _answer(status):
        return httpx.MockTransport(lambda request: httpx.Response(status))

    def _raising(exc):
        def handler(request):
            raise exc

        return httpx.MockTransport(handler)

    probe = "http://dev"
    assert await _probe_dev_server(probe, transport=_answer(200)) == (True, None)
    # An answer other than 200 is a server in trouble, worth a reason.
    assert await _probe_dev_server(probe, transport=_answer(503)) == (
        False,
        "HTTP 503",
    )
    unresolved = _connect_error(socket.gaierror(8, "nodename nor servname provided"))
    assert await _probe_dev_server(probe, transport=_raising(unresolved)) == (
        False,
        "ConnectError: [Errno 8] nodename nor servname provided",
    )
    # Two addresses, the refused one tried first: still nobody listening.
    mixed = _connect_error(
        ExceptionGroup(
            "multiple connection attempts failed",
            [
                ConnectionRefusedError(61, "Connect call failed"),
                OSError(51, "Network is unreachable"),
            ],
        ),
        every_address=True,
    )
    assert await _probe_dev_server(probe, transport=_raising(mixed)) == (False, None)
    # anyio raises a bare TimeoutError: the type name is the whole reason.
    assert await _probe_dev_server(
        probe, transport=_raising(httpx.ReadTimeout(""))
    ) == (
        False,
        "ReadTimeout",
    )


async def test_live_task_status_refusal_is_recognised(monkeypatch):
    from EvoScientist.langgraph_dev import sdk
    from EvoScientist.middleware import rubric as rubric_mod

    monkeypatch.setattr(
        sdk, "configured_langgraph_dev_url", lambda: "http://127.0.0.1:1"
    )
    with pytest.raises(httpx.ConnectError) as info:
        await rubric_mod.live_task_status("t1", "r1")
    assert rubric_mod._connection_refused(info.value)


async def test_availability_runs_only_with_candidate_tasks_and_inside_the_budget(
    tmp_path, monkeypatch, caplog
):
    from EvoScientist.middleware import rubric as rubric_mod

    calls = []
    answered = []
    cut_after = []

    async def _slow_check():
        calls.append("check")
        began = asyncio.get_running_loop().time()
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cut_after.append(asyncio.get_running_loop().time() - began)
            raise
        answered.append(True)
        return True

    monkeypatch.setattr(rubric_mod, "DEFER_QUERY_TIMEOUT_SECONDS", 0.05)
    live = _Live(task_answers=["running"])
    grader = _ScriptedGrader(verdicts=[PASS, PASS])
    rubric = create_rubric_middleware(
        model=grader,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        max_iterations=3,
        prepare_messages=this_run_messages,
        require_run_marker=True,
        task_status=live.task_status,
        process_status=live.process_status,
        server_available=_slow_check,
    )
    agent = create_agent(
        model=_Main(messages=iter([AIMessage(str(i)) for i in range(4)])),
        tools=[],
        middleware=[work := _WorkChannels(), rubric],
        checkpointer=InMemorySaver(),
    )
    agent.work = work
    # No task records: the availability check is not even made.
    events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert calls == []
    # A stored running task: the check runs, but inside the budget.
    _launch(agent, _task())
    started = asyncio.get_running_loop().time()
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        events, _ = await _run(agent, "again")
    elapsed = asyncio.get_running_loop().time() - started
    assert _results(events) == ["satisfied"]
    assert calls == ["check"]
    # The check was cut at its 50 ms budget, not waited for. Its own time is
    # measured apart from the turn's, with room for a slow runner in both.
    assert answered == []
    (cut,) = cut_after
    assert cut < 0.5, cut
    assert elapsed < 2.5, elapsed
    assert any("exceeded" in r.getMessage() for r in caplog.records)


async def test_connect_timeout_grades_with_a_warning(tmp_path, caplog):
    live = _Live(task_answers=[httpx.ConnectTimeout("slow connect")])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _launch(agent, _task())
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert any("could not read" in r.getMessage() for r in caplog.records)


async def test_a_new_grading_starts_with_no_evaluations_even_when_deferred(tmp_path):
    live = _Live(task_answers=["success", "success", "running"])
    agent, _ = _agent(tmp_path, [FAIL, PASS, PASS], ["a", "b", "c", "d"], live)
    _launch(agent, _task())
    await _run(agent, "task one", changes=[_op(1, sticky="- s")])
    _launch(agent, _task("t2", run_id="r2"))
    events, values = await _run(agent, "task two")
    assert _deferred(events) == [(1, 0)]
    assert values["_rubric_evaluations"] == []


@pytest.mark.parametrize("mode", ["thread", "followup"])
async def test_sync_hook_path_is_not_used_by_these_tests(tmp_path, mode):
    """Documented limit: the sync ``after_agent`` grades without deferring."""
    live = _Live(task_answers=["running"])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _launch(agent, _task())
    cfg = {
        "configurable": {
            "thread_id": "t",
            "rubric_run": {
                "mode": mode,
                "changes": [{"session": "s1", "seq": 1, "sticky": "- s"}],
            },
        }
    }
    events = [
        d
        for _m, d in agent.stream(
            {"messages": [HumanMessage("task")]}, cfg, stream_mode=["custom"]
        )
        if isinstance(d, dict)
    ]
    if mode == "thread":
        assert _results(events) == ["satisfied"]
    assert _deferred(events) == []


# -- sub-agent tasks that were running before the grading started --------------


async def test_a_task_running_before_the_grading_started_does_not_hold_it(tmp_path):
    """A long task from before the checklist was set belongs to earlier work."""
    live = _Live(task_answers=["running"])
    agent, grader = _agent(tmp_path, [PASS], ["a", "b"], live)
    _seed(agent, {"async_tasks": _task()})
    events, values = await _run(agent, "one task", changes=[_op(1, next="- n")])
    assert _deferred(events) == []
    assert _results(events) == ["satisfied"]
    assert "one task" in grader.last_payload
    assert live.calls == []
    assert values["_next_rubric"] is None
    assert values["_rubric_grading"] is None


async def test_an_earlier_task_does_not_hold_a_grading_it_did_not_start_either(
    tmp_path,
):
    """The turn's own task holds the grading; the older one is not even read."""
    live = _Live(task_answers=["running"])
    agent, _grader = _agent(tmp_path, [PASS], ["a", "b"], live)
    earlier = {
        **_task("old", run_id="r-old"),
        **_task("done", run_id="r-done", status="success"),
    }
    _seed(agent, {"async_tasks": earlier})
    _launch(agent, _task("new", run_id="r-new"))
    events, values = await _run(agent, "the task", changes=[_op(1, sticky="- s")])
    assert _deferred(events) == [(1, 0)]
    assert live.calls == [("new", "r-new")]
    # Only what was still going is remembered; finished runs need no entry.
    assert values["_rubric_grading"]["skip_runs"] == ["r-old"]


async def test_a_new_run_of_an_earlier_task_holds_the_grading(tmp_path):
    """An earlier task sent on again by this turn is this turn's work."""
    live = _Live(task_answers=["running"])
    agent, _grader = _agent(tmp_path, [PASS], ["a", "b"], live)
    _seed(agent, {"async_tasks": _task("t1", run_id="r1")})
    _launch(agent, _task("t1", run_id="r2"))
    events, _values = await _run(agent, "the task", changes=[_op(1, sticky="- s")])
    assert _deferred(events) == [(1, 0)]
    assert live.calls == [("t1", "r2")]


async def test_a_deferred_grading_keeps_what_it_skipped_at_its_start(tmp_path):
    """The next message goes on with the same grading: its own task still holds."""
    live = _Live(task_answers=["running", "running", "success"])
    agent, grader = _agent(tmp_path, [PASS], ["a", "b", "c", "d"], live)
    _seed(agent, {"async_tasks": _task("old", run_id="r-old")})
    _launch(agent, _task("new", run_id="r-new"))
    await _run(agent, "the task", changes=[_op(1, sticky="- s")])
    more, values = await _run(agent, "and also this")
    assert _deferred(more) == [(1, 0)]
    assert values["_rubric_grading"]["skip_runs"] == ["r-old"]
    done, _ = await _run(agent, "[Async tasks update]", mode="followup")
    assert _results(done) == ["satisfied"]
    assert "the task" in grader.last_payload


# -- runs without the marker while a grading is deferred -----------------------


async def test_an_unmarked_run_keeps_a_deferred_grading(tmp_path):
    """Another client's turn on the thread must not lose the grading that waits."""
    live = _Live(task_answers=["running", "success"])
    agent, grader = _agent(tmp_path, [PASS], ["draft", "other", "noted", "x"], live)
    _launch(agent, _task())
    await _run(agent, "the task", changes=[_op(1, next="- n")])
    _events, values = await _run(agent, "from another client", mode=None)
    assert values["rubric"] is None
    assert values["_rubric_grading"]["deferred"] is True
    assert grader.calls == 0
    events, values = await _run(
        agent, "[Async tasks update] t1 finished", mode="followup"
    )
    assert _results(events) == ["satisfied"]
    assert "the task" in grader.last_payload
    assert values["_next_rubric"] is None
    assert values["_rubric_grading"] is None


async def test_the_next_message_after_an_unmarked_run_grades_the_original_task(
    tmp_path,
):
    """Without a notification the grading runs at the next message, from its start."""
    live = _Live(task_answers=["running", "success"])
    agent, grader = _agent(tmp_path, [PASS], ["draft", "other", "noted", "x"], live)
    _launch(agent, _task())
    await _run(agent, "the task", changes=[_op(1, next="- n")])
    await _run(agent, "from another client", mode=None)
    events, values = await _run(agent, "thx")
    assert _results(events) == ["satisfied"]
    assert "the task" in grader.last_payload
    assert values["_next_rubric"] is None


async def test_an_unmarked_run_still_drops_a_grading_that_was_not_deferred(tmp_path):
    """A marked run parked mid-way and continued elsewhere leaves nothing behind."""
    live = _Live()
    agent, grader = _agent(tmp_path, [PASS], ["a", "b"], live)
    _seed(
        agent,
        {
            "_sticky_rubric": {"text": "- s", "id": "s1:1"},
            "rubric": "- s",
            "_rubric_grading": {
                "source": "sticky",
                "id": "s1:1",
                "start": "m1",
                "deferred": False,
                "skip": [],
            },
        },
    )
    _events, values = await _run(agent, "from another client", mode=None)
    assert values["rubric"] is None
    assert values["_rubric_grading"] is None
    assert grader.calls == 0


async def test_a_rubric_change_made_during_a_deferred_grading_waits_for_the_next_message(
    tmp_path,
):
    from EvoScientist.rubric_session import RubricSession, rubric_configurable_extra

    def changes(mode):
        return rubric_configurable_extra(session, "t", mode)["rubric_run"]["changes"]

    session = RubricSession()
    live = _Live(task_answers=["running", "success"])
    agent, _ = _agent(tmp_path, [PASS] * 3, ["draft", "noted", "next"], live)
    _launch(agent, _task())
    session.record("t", sticky="- A")
    _, v1 = await _run(agent, "the task", changes=changes("thread"))
    assert v1["_rubric_grading"]["deferred"]

    session.record("t", sticky="- B")
    ev, v2 = await _run(
        agent,
        "[Async tasks update] t1 finished",
        mode="followup",
        changes=changes("followup"),
    )
    assert _results(ev) == ["satisfied"]
    assert v2["_sticky_rubric"]["text"] == "- A"

    _, v3 = await _run(agent, "go on", changes=changes("thread"))
    assert v3["_sticky_rubric"]["text"] == "- B"
