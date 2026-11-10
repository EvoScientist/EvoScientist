"""Shared acceptance grader: prompt, clearing rule and the revision loop."""

from __future__ import annotations

import logging

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.rubric import GRADER_SYSTEM_PROMPT
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from EvoScientist.middleware.rubric import (
    GRADER_FILE_CHECK_PROMPT,
    create_rubric_middleware,
)
from tests.fakes.rubric import FAIL, PASS, MainModel, ScriptedGrader


class _OtherAfterAgent(AgentMiddleware):
    """Stands in for the memory-lifecycle and code-interpreter hooks."""

    name = "other_after_agent"

    def __init__(self) -> None:
        self.calls = 0

    def after_agent(self, state, runtime):
        self.calls += 1
        return None


def _agent(tmp_path, verdicts, replies, *, max_iterations=3, require_run_marker=False):
    grader = ScriptedGrader(verdicts=verdicts)
    other = _OtherAfterAgent()
    rubric = create_rubric_middleware(
        model=grader,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        max_iterations=max_iterations,
        require_run_marker=require_run_marker,
    )
    agent = create_agent(
        model=MainModel(messages=iter([AIMessage(r) for r in replies])),
        tools=[],
        middleware=[other, rubric],
        checkpointer=InMemorySaver(),
    )
    return agent, grader, other, rubric


def _run(agent, payload, thread_id="t", *, marked=False):
    cfg = {"configurable": {"thread_id": thread_id}}
    if marked:
        cfg["configurable"]["rubric_run"] = {"mode": "input"}
    events = [
        data
        for _mode, data in agent.stream(payload, cfg, stream_mode=["custom"])
        if isinstance(data, dict)
    ]
    results = [e.get("result") for e in events if e["type"] == "rubric_evaluation_end"]
    return results, agent.get_state(cfg).values


def test_grader_prompt_is_upstream_plus_the_file_check(tmp_path):
    *_, rubric = _agent(tmp_path, [PASS], ["a"])
    assert rubric._system_prompt == GRADER_SYSTEM_PROMPT + GRADER_FILE_CHECK_PROMPT
    assert "check that file on disk" in GRADER_FILE_CHECK_PROMPT
    assert "Open only files that a criterion or the transcript names" in (
        GRADER_FILE_CHECK_PROMPT
    )


def test_revision_then_pass_clears_the_rubric(tmp_path):
    agent, grader, other, _rubric = _agent(tmp_path, [FAIL, PASS], ["draft", "revised"])
    results, values = _run(
        agent, {"messages": [HumanMessage("do it")], "rubric": "- c1"}
    )
    assert results == ["needs_revision", "satisfied"]
    assert values["rubric"] is None
    assert values["_rubric_status"] == "satisfied"
    assert grader.calls == 2
    assert other.calls == 1  # skipped on the revision pass, run once at the end
    feedback = [
        m
        for m in values["messages"]
        if isinstance(m, HumanMessage) and m.additional_kwargs.get("lc_source")
    ]
    assert [m.additional_kwargs["lc_source"] for m in feedback] == ["rubric_grader"]


def test_pass_limit_ends_the_run_and_clears_the_rubric(tmp_path):
    agent, _grader, other, _rubric = _agent(tmp_path, [FAIL], ["a", "b", "c", "d"])
    results, values = _run(
        agent, {"messages": [HumanMessage("do it")], "rubric": "- c1"}
    )
    assert results == ["needs_revision", "needs_revision", "max_iterations_reached"]
    assert values["rubric"] is None
    assert other.calls == 1


def test_grader_error_ends_the_run_and_clears_the_rubric(tmp_path):
    agent, _grader, other, _rubric = _agent(tmp_path, [RuntimeError("boom")], ["a"])
    results, values = _run(
        agent, {"messages": [HumanMessage("do it")], "rubric": "- c1"}
    )
    assert results == ["grader_error"]
    assert values["rubric"] is None
    assert values["_rubric_status"] == "grader_error"
    assert other.calls == 1


def test_a_run_without_rubric_is_untouched(tmp_path):
    agent, grader, _other, rubric = _agent(tmp_path, [PASS], ["a"])
    results, _values = _run(
        agent, {"messages": [HumanMessage("do it")], "rubric": None}
    )
    assert results == []
    assert grader.calls == 0
    assert rubric._grader is None


def test_finished_rubric_does_not_grade_the_next_turn(tmp_path):
    agent, grader, _other, _rubric = _agent(tmp_path, [PASS], ["a", "b"])
    _run(agent, {"messages": [HumanMessage("do it")], "rubric": "- c1"})
    results, _values = _run(agent, {"messages": [HumanMessage("thanks")]})
    assert results == []
    assert grader.calls == 1


def _openrouter(model_id: str):
    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(model=model_id, api_key="test-key")


def test_unsupported_grader_warns_on_first_rubric_run_only(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        rubric = create_rubric_middleware(
            model=_openrouter("anthropic/claude-fable-5.1"),
            backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
            max_iterations=3,
        )
        assert caplog.text == ""  # quiet at build
        rubric.before_agent({"messages": []}, None)
        assert caplog.text == ""  # quiet without a rubric
        rubric.before_agent({"messages": [], "rubric": "- x"}, None)
        assert "claude-fable-5.1" in caplog.text
        assert "auxiliary_model" in caplog.text
        caplog.clear()
        rubric.before_agent({"messages": [], "rubric": "- x"}, None)
        assert caplog.text == ""  # once per instance


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic/claude-sonnet-5",
        "anthropic/claude-fable-5",
        "google/gemini-3.8-flash",
    ],
)
def test_supported_grader_never_warns(tmp_path, caplog, model_id):
    with caplog.at_level(logging.WARNING, logger="EvoScientist.middleware.rubric"):
        rubric = create_rubric_middleware(
            model=_openrouter(model_id),
            backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
            max_iterations=3,
        )
        rubric.before_agent({"messages": [], "rubric": "- x"}, None)
    assert "rubric" not in caplog.text.lower()


def test_main_agent_grades_a_marked_run(tmp_path):
    agent, grader, _other, _rubric = _agent(
        tmp_path, [PASS], ["a"], require_run_marker=True
    )
    results, values = _run(
        agent, {"messages": [HumanMessage("do it")], "rubric": "- c1"}, marked=True
    )
    assert results == ["satisfied"]
    assert grader.calls == 1
    assert values["rubric"] is None


def test_main_agent_ignores_a_rubric_on_a_run_without_the_marker(tmp_path):
    """A client that only sends the text gets no grading, and no rubric is kept."""
    agent, grader, _other, _rubric = _agent(
        tmp_path, [PASS], ["a"], require_run_marker=True
    )
    results, values = _run(
        agent, {"messages": [HumanMessage("do it")], "rubric": "- c1"}
    )
    assert results == []
    assert grader.calls == 0
    assert values["rubric"] is None


async def test_unmarked_turn_drops_a_rubric_left_by_a_run_that_never_finished(tmp_path):
    """Ctrl+C or a crash leaves the rubric in the thread; any later turn drops it."""
    from EvoScientist.stream.events import build_agent_stream_input

    agent, grader, _other, _rubric = _agent(
        tmp_path, [PASS], ["a", "b"], require_run_marker=True
    )
    cfg = {"configurable": {"thread_id": "stale"}}
    agent.update_state(cfg, {"messages": [HumanMessage("old")], "rubric": "- stale"})
    assert agent.get_state(cfg).values["rubric"] == "- stale"

    payload = await build_agent_stream_input("next")
    assert "rubric" not in payload
    results, values = _run(agent, payload, thread_id="stale")

    assert results == []
    assert values["rubric"] is None
    assert grader.calls == 0


async def test_rubric_key_is_ignored_by_a_graph_without_the_grader():
    from EvoScientist.stream.events import build_agent_stream_input

    plain = create_agent(
        model=MainModel(messages=iter([AIMessage("one")])),
        tools=[],
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "plain"}}
    plain.invoke(await build_agent_stream_input("hi", rubric="- a"), cfg)
    assert sorted(plain.get_state(cfg).values) == ["messages"]


def _replies_then_crash(crash_at: int):
    count = 0
    while True:
        count += 1
        if count == crash_at:
            raise RuntimeError("provider died")
        yield AIMessage(f"reply {count}")


def test_rerun_after_a_cut_short_run_gets_all_its_passes(tmp_path):
    """The same rubric text on the same thread starts a new attempt, not the dead one."""
    grader = ScriptedGrader(verdicts=[FAIL])
    rubric = create_rubric_middleware(
        model=grader,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        max_iterations=3,
    )
    saver = InMemorySaver()
    cfg = {"configurable": {"thread_id": "cut"}}

    first = create_agent(
        model=MainModel(messages=_replies_then_crash(3)),
        tools=[],
        middleware=[rubric],
        checkpointer=saver,
    )
    with pytest.raises(RuntimeError, match="provider died"):
        first.invoke({"messages": [HumanMessage("do it")], "rubric": "- c1"}, cfg)
    assert first.get_state(cfg).values["_rubric_iterations"] == 2

    second = create_agent(
        model=MainModel(messages=_replies_then_crash(99)),
        tools=[],
        middleware=[rubric],
        checkpointer=saver,
    )
    results, _values = _run(
        second, {"messages": [HumanMessage("again")], "rubric": "- c1"}, thread_id="cut"
    )
    assert results == ["needs_revision", "needs_revision", "max_iterations_reached"]


def test_grader_failure_keeps_its_traceback_in_the_log(tmp_path, caplog):
    """The server's log is the only diagnostic of a scheduled run's grader."""
    from EvoScientist.cli.commands import _grader_failure_on_one_line

    agent, _grader, _other, _rubric = _agent(tmp_path, [RuntimeError("boom")], ["a"])
    with caplog.at_level(logging.ERROR, logger="deepagents.middleware.rubric"):
        _run(agent, {"messages": [HumanMessage("do it")], "rubric": "- c1"})
    failures = [r for r in caplog.records if "grader failed" in r.getMessage()]
    assert len(failures) == 1
    assert failures[0].exc_info is not None
    assert not logging.getLogger("deepagents.middleware.rubric").filters
    # Upstream's wording is what the console's one-line form goes by.
    short = _grader_failure_on_one_line(failures[0])
    assert short.exc_info is None
    assert short.getMessage().endswith("RuntimeError: boom")


def test_failed_second_attempt_keeps_its_traceback_too(tmp_path, caplog):
    """An incomplete verdict is retried once; that retry failing is expected as well."""
    from EvoScientist.cli.commands import _grader_failure_on_one_line

    incomplete = {"result": "satisfied", "explanation": "x", "criteria": []}
    agent, _grader, _other, _rubric = _agent(
        tmp_path, [incomplete, RuntimeError("boom")], ["a", "b", "c", "d"]
    )
    with caplog.at_level(logging.ERROR, logger="deepagents.middleware.rubric"):
        _run(agent, {"messages": [HumanMessage("do it")], "rubric": "- c1"})
    retries = [r for r in caplog.records if "coverage retry raised" in r.getMessage()]
    assert retries
    assert all(r.exc_info is not None for r in retries)
    assert (
        _grader_failure_on_one_line(retries[0])
        .getMessage()
        .endswith("RuntimeError: boom")
    )


def test_a_dropped_rubric_is_noted_in_the_debug_log(tmp_path, caplog):
    agent, grader, _other, _rubric = _agent(
        tmp_path, [PASS], ["a", "b"], require_run_marker=True
    )
    with caplog.at_level(logging.DEBUG, logger="EvoScientist.middleware.rubric"):
        _run(agent, {"messages": [HumanMessage("do it")], "rubric": "- c1"})
        _run(agent, {"messages": [HumanMessage("again")]}, thread_id="other")
    dropped = [r for r in caplog.records if "dropped" in r.getMessage()]
    assert len(dropped) == 1
    assert "rubric_run" in dropped[0].getMessage()
    assert grader.calls == 0


def _parked_before_grading(tmp_path):
    """A marked run on the main-agent grader, stopped just before it grades."""
    grader = ScriptedGrader(verdicts=[PASS])
    rubric = create_rubric_middleware(
        model=grader,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        max_iterations=3,
        require_run_marker=True,
    )
    agent = create_agent(
        model=MainModel(messages=iter([AIMessage("a")])),
        tools=[],
        middleware=[rubric],
        checkpointer=InMemorySaver(),
        interrupt_before=["EvoRubricMiddleware.after_agent"],
    )
    marked = {"configurable": {"thread_id": "parked", "rubric_run": {"mode": "input"}}}
    agent.invoke({"messages": [HumanMessage("do it")], "rubric": "- c1"}, marked)
    assert agent.get_state(marked).next == ("EvoRubricMiddleware.after_agent",)
    return agent, grader, marked


def test_continuing_a_parked_run_without_the_marker_is_not_graded(tmp_path):
    """Another client resuming the run did not ask for grading."""
    agent, grader, _marked = _parked_before_grading(tmp_path)
    unmarked = {"configurable": {"thread_id": "parked"}}

    agent.invoke(None, unmarked)

    assert grader.calls == 0
    assert agent.get_state(unmarked).values["rubric"] is None


def test_continuing_a_parked_run_with_the_marker_is_graded(tmp_path):
    agent, grader, marked = _parked_before_grading(tmp_path)

    agent.invoke(None, marked)

    assert grader.calls == 1
    assert agent.get_state(marked).values["_rubric_status"] == "satisfied"
