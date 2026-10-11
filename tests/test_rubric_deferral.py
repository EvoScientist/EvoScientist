"""Grading deferred while this thread's background sub-agents or processes run."""

from __future__ import annotations

import asyncio
import logging
import socket

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
    name = "work_channels"
    state_schema = _WorkState


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
    agent = create_agent(
        model=_Main(messages=iter([AIMessage(r) for r in replies])),
        tools=[],
        middleware=[_WorkChannels(), rubric],
        checkpointer=InMemorySaver(),
    )
    return agent, grader


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
    _seed(agent, {"async_tasks": _task()})
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
    _seed(agent, {"async_tasks": _task()})
    events, values = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert _deferred(events) == []
    assert values["async_tasks"]["t1"]["status"] == "running"


async def test_unreadable_status_grades_and_warns(tmp_path, caplog):
    live = _Live(task_answers=[RuntimeError("boom")])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _seed(agent, {"async_tasks": _task()})
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
    _seed(agent, {"async_tasks": _task()})
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_input_mode_ignores_background_work(tmp_path):
    live = _Live(task_answers=["running"], processes={"p1": "running"})
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _seed(agent, {"async_tasks": _task(), "bg_processes": _process()})
    events, _ = await _run(agent, "task", mode="input", rubric="- s")
    assert _results(events) == ["satisfied"]
    assert live.calls == []


async def test_no_checklist_emits_no_deferred_event(tmp_path):
    live = _Live(task_answers=["running"])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _seed(agent, {"async_tasks": _task()})
    events, _ = await _run(agent, "chat")
    assert events == []
    assert live.calls == []


async def test_followup_finishes_a_deferred_grading_from_the_task_message(tmp_path):
    live = _Live(task_answers=["running", "success"])
    agent, grader = _agent(tmp_path, [PASS], ["draft", "noted", "x"], live)
    _seed(agent, {"async_tasks": _task()})
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
    _seed(agent, {"async_tasks": _task()})
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


async def test_followup_with_a_missing_start_returns_grader_error(tmp_path):
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
    events, values = await _run(agent, "[Async tasks update]", mode="followup")
    assert _results(events) == ["grader_error"]
    assert grader.calls == 0
    assert values["rubric"] is None
    assert values["_rubric_grading"] is None
    assert values["_next_rubric"] is None


async def test_deferred_sticky_grading_leaves_a_later_next_with_equal_text(tmp_path):
    live = _Live(task_answers=["running", "running", "success"])
    agent, _ = _agent(tmp_path, [PASS, PASS], ["a", "b", "c", "d", "e"], live)
    _seed(agent, {"async_tasks": _task()})
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
    _seed(agent, {"async_tasks": _task()})
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
    _seed(agent, {"async_tasks": _task()})
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
    _seed(agent, {"async_tasks": _task(), "bg_processes": _process()})
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
    _seed(agent, {"async_tasks": _task()})
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
    _seed(agent, {"async_tasks": _task()})
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_connect_error_other_than_refusal_grades_with_a_warning(tmp_path, caplog):
    unresolved = _connect_error(socket.gaierror(8, "nodename nor servname provided"))
    live = _Live(task_answers=[unresolved])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _seed(agent, {"async_tasks": _task()})
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert any("could not read" in r.getMessage() for r in caplog.records)


async def test_read_timeout_grades_with_a_warning(tmp_path, caplog):
    live = _Live(task_answers=[httpx.ReadTimeout("slow")])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _seed(agent, {"async_tasks": _task()})
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
        middleware=[_WorkChannels(), rubric],
        checkpointer=InMemorySaver(),
    )
    _seed(agent, {"async_tasks": _task()})
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
    assert await _probe_dev_server("http://127.0.0.1:1") == (False, None)

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

    async def _slow_check():
        calls.append("check")
        await asyncio.sleep(0.2)
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
        middleware=[_WorkChannels(), rubric],
        checkpointer=InMemorySaver(),
    )
    # No task records: the availability check is not even made.
    events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert calls == []
    # A stored running task: the check runs, but inside the budget.
    _seed(agent, {"async_tasks": _task()})
    started = asyncio.get_running_loop().time()
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        events, _ = await _run(agent, "again")
    elapsed = asyncio.get_running_loop().time() - started
    assert _results(events) == ["satisfied"]
    assert calls == ["check"]
    assert elapsed < 0.15, elapsed
    assert any("exceeded" in r.getMessage() for r in caplog.records)


async def test_connect_timeout_grades_with_a_warning(tmp_path, caplog):
    live = _Live(task_answers=[httpx.ConnectTimeout("slow connect")])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _seed(agent, {"async_tasks": _task()})
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        events, _ = await _run(agent, "task", changes=[_op(1, sticky="- s")])
    assert _results(events) == ["satisfied"]
    assert any("could not read" in r.getMessage() for r in caplog.records)


async def test_a_new_grading_starts_with_no_evaluations_even_when_deferred(tmp_path):
    live = _Live(task_answers=["success", "success", "running"])
    agent, _ = _agent(tmp_path, [FAIL, PASS, PASS], ["a", "b", "c", "d"], live)
    _seed(agent, {"async_tasks": _task()})
    await _run(agent, "task one", changes=[_op(1, sticky="- s")])
    events, values = await _run(agent, "task two")
    assert _deferred(events) == [(1, 0)]
    assert values["_rubric_evaluations"] == []


@pytest.mark.parametrize("mode", ["thread", "followup"])
async def test_sync_hook_path_is_not_used_by_these_tests(tmp_path, mode):
    """Documented limit: the sync ``after_agent`` grades without deferring."""
    live = _Live(task_answers=["running"])
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"], live)
    _seed(agent, {"async_tasks": _task()})
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
