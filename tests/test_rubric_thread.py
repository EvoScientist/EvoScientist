"""Checklists saved on the thread: command operations, selection by run mode, ownership."""

from __future__ import annotations

from deepagents.backends import FilesystemBackend
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from EvoScientist.message_meta import this_run_messages
from EvoScientist.middleware.rubric import create_rubric_middleware
from tests.test_rubric_middleware import FAIL, PASS, _Main, _ScriptedGrader

GRADER_NODE = "EvoRubricMiddleware.after_agent"


def _agent(tmp_path, verdicts, replies, *, max_iterations=3):
    grader = _ScriptedGrader(verdicts=verdicts)
    rubric = create_rubric_middleware(
        model=grader,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        max_iterations=max_iterations,
        prepare_messages=this_run_messages,
        require_run_marker=True,
    )
    agent = create_agent(
        model=_Main(messages=iter([AIMessage(r) for r in replies])),
        tools=[],
        middleware=[rubric],
        checkpointer=InMemorySaver(),
    )
    return agent, grader


def _op(seq, session="s1", **fields):
    return {"session": session, "seq": seq, **fields}


def _run(agent, text, *, mode=None, changes=(), thread="t", rubric=None):
    cfg = {"configurable": {"thread_id": thread}}
    if mode:
        cfg["configurable"]["rubric_run"] = {"mode": mode, "changes": list(changes)}
    payload = {"messages": [HumanMessage(text)]}
    if rubric is not None:
        payload["rubric"] = rubric
    events = [
        data
        for _mode, data in agent.stream(payload, cfg, stream_mode=["custom"])
        if isinstance(data, dict)
    ]
    results = [e.get("result") for e in events if e["type"] == "rubric_evaluation_end"]
    return results, agent.get_state(cfg).values


def _values(agent, thread="t"):
    return agent.get_state({"configurable": {"thread_id": thread}}).values


def _leave(agent, update, thread="t"):
    """Write state the way a cut-short run leaves it, then settle the thread."""
    cfg = {"configurable": {"thread_id": thread}}
    agent.update_state(cfg, update, as_node=GRADER_NODE)
    agent.update_state(cfg, None, as_node="__end__")


def test_set_operation_stores_the_sticky_checklist_and_grades_the_turn(tmp_path):
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"])
    results, values = _run(agent, "task", mode="thread", changes=[_op(1, sticky="- a")])
    assert results == ["satisfied"]
    assert values["_sticky_rubric"] == {"text": "- a", "id": "s1:1"}
    assert values["_rubric_applied"] == {"s1": 1}
    assert values["rubric"] is None
    assert values["_rubric_grading"] is None


def test_next_wins_over_sticky_and_is_cleared_by_a_final_verdict(tmp_path):
    agent, _ = _agent(tmp_path, [PASS, PASS], ["a", "b"])
    _run(agent, "x", mode="thread", changes=[_op(1, sticky="- s"), _op(2, next="- n")])
    values = _values(agent)
    assert values["_next_rubric"] is None
    assert values["_sticky_rubric"]["text"] == "- s"
    results, _ = _run(agent, "y", mode="thread")
    assert results == ["satisfied"]


def test_operations_are_applied_once_per_session_sequence(tmp_path):
    # Session 1 sets A, session 2 applies 33 operations, session 1 re-sends A.
    agent, _ = _agent(tmp_path, [PASS] * 40, ["r"] * 40)
    _run(agent, "1", mode="thread", changes=[_op(1, session="s1", sticky="- A")])
    for seq in range(1, 34):
        _run(
            agent,
            f"b{seq}",
            mode="thread",
            changes=[_op(seq, session="s2", sticky=f"- B{seq}")],
        )
    _, values = _run(
        agent, "again", mode="thread", changes=[_op(1, session="s1", sticky="- A")]
    )
    assert values["_sticky_rubric"]["text"] == "- B33"
    assert values["_rubric_applied"] == {"s1": 1, "s2": 33}


def test_forty_operations_in_one_run_leave_nothing_pending(tmp_path):
    agent, grader = _agent(tmp_path, [PASS], ["r", "r"])
    changes = [_op(i, sticky=f"- v{i}") for i in range(1, 40)]
    changes.append(_op(40, sticky=None, next=None))
    results, values = _run(agent, "go", mode="thread", changes=changes)
    assert results == []
    assert grader.calls == 0
    assert values["_rubric_applied"] == {"s1": 40}
    assert values["_sticky_rubric"] is None
    assert values["_next_rubric"] is None
    assert values.get("rubric") is None


def test_unmarked_run_drops_rubric_and_grading_but_keeps_saved_checklists(tmp_path):
    agent, grader = _agent(tmp_path, [PASS], ["a", "b"])
    _leave(
        agent,
        {
            "_sticky_rubric": {"text": "- s", "id": "s1:1"},
            "_rubric_applied": {"s1": 1},
            "rubric": "- s",
            "_rubric_grading": {
                "source": "sticky",
                "id": "s1:1",
                "start": None,
                "deferred": False,
                "skip": [],
            },
        },
    )
    results, values = _run(agent, "webui turn")
    assert results == []
    assert grader.calls == 0
    assert values["rubric"] is None
    assert values["_rubric_grading"] is None
    assert values["_sticky_rubric"] == {"text": "- s", "id": "s1:1"}


def test_input_mode_never_touches_saved_checklists_even_with_equal_text(tmp_path):
    agent, _ = _agent(tmp_path, [PASS, PASS], ["a", "b", "c"])
    # Operations are applied before selection, so this thread run grades with
    # next at once and the verdict clears it.
    _run(agent, "store", mode="thread", changes=[_op(1, next="- n")])
    assert _values(agent)["_next_rubric"] is None
    # A followup never selects: the operation is stored, not consumed.
    _run(agent, "noted", mode="followup", changes=[_op(2, next="- n")])
    assert _values(agent)["_next_rubric"] == {"text": "- n", "id": "s1:2"}
    # A -p --rubric with the same text must leave the saved next alone.
    results, values = _run(agent, "single shot", mode="input", rubric="- n")
    assert results == ["satisfied"]
    assert values["_next_rubric"] == {"text": "- n", "id": "s1:2"}


def test_leftover_rubric_is_ignored_under_thread_mode(tmp_path):
    agent, grader = _agent(tmp_path, [PASS], ["a", "b"])
    _leave(agent, {"rubric": "- leftover"})
    results, values = _run(agent, "hello", mode="thread")
    assert results == []
    assert grader.calls == 0
    assert values["rubric"] is None


def test_second_sticky_turn_starts_at_iteration_zero(tmp_path):
    agent, _ = _agent(tmp_path, [PASS, FAIL, PASS], ["a", "b", "c", "d"])
    _run(agent, "one", mode="thread", changes=[_op(1, sticky="- s")])
    results, values = _run(agent, "two", mode="thread")
    assert results == ["needs_revision", "satisfied"]
    assert [e["iteration"] for e in values["_rubric_evaluations"]] == [0, 1]


def test_next_survives_an_interrupted_run(tmp_path):
    agent, _ = _agent(tmp_path, [PASS], ["a", "b"])
    # A cut-short run: the start hook stored next and picked it, no verdict came.
    _leave(
        agent,
        {
            "_next_rubric": {"text": "- n", "id": "s1:9"},
            "rubric": "- n",
            "_rubric_grading": {
                "source": "next",
                "id": "s1:9",
                "start": None,
                "deferred": False,
                "skip": [],
            },
        },
    )
    results, values = _run(agent, "continue", mode="thread")
    assert results == ["satisfied"]
    assert values["_next_rubric"] is None


def test_next_with_the_sticky_text_is_consumed_by_its_own_grading(tmp_path):
    agent, _ = _agent(tmp_path, [PASS, PASS], ["a", "b", "c"])
    _run(agent, "task", mode="thread", changes=[_op(1, sticky="- s")])
    _, values = _run(agent, "again", mode="thread", changes=[_op(2, next="- s")])
    assert values["_next_rubric"] is None
    assert values["_sticky_rubric"] == {"text": "- s", "id": "s1:1"}


def test_continue_after_cut_short_keeps_the_start(tmp_path):
    agent, grader = _agent(tmp_path, [PASS], ["a", "b"])
    _leave(
        agent,
        {
            "messages": [HumanMessage("the task", id="m-task"), AIMessage("partial")],
            "_sticky_rubric": {"text": "- s", "id": "s1:1"},
            "_rubric_applied": {"s1": 1},
            "rubric": "- s",
            "_rubric_grading": {
                "source": "sticky",
                "id": "s1:1",
                "start": "m-task",
                "deferred": False,
                "skip": [],
            },
        },
    )
    results, _ = _run(agent, "continue", mode="thread")
    assert results == ["satisfied"]
    assert "the task" in grader.last_payload


def test_a_missing_start_is_replaced_by_the_current_message(tmp_path):
    agent, grader = _agent(tmp_path, [PASS], ["a", "b"])
    _leave(
        agent,
        {
            "messages": [HumanMessage("old task", id="m-old")],
            "_sticky_rubric": {"text": "- s", "id": "s1:1"},
            "_rubric_applied": {"s1": 1},
            "rubric": "- s",
            "_rubric_grading": {
                "source": "sticky",
                "id": "s1:1",
                "start": "m-gone",
                "deferred": False,
                "skip": [],
            },
        },
    )
    _run(agent, "new task", mode="thread")
    assert "new task" in grader.last_payload
    assert "old task" not in grader.last_payload


def test_evaluations_keep_only_the_current_grading(tmp_path):
    agent, _ = _agent(tmp_path, [FAIL, PASS, PASS], ["a", "b", "c", "d"])
    _run(agent, "one", mode="thread", changes=[_op(1, sticky="- s")])
    _, values = _run(agent, "two", mode="thread")
    assert len(values["_rubric_evaluations"]) == 1
    assert len({e["grading_run_id"] for e in values["_rubric_evaluations"]}) == 1


def test_scheduler_without_marker_requirement_grades_its_input_as_before(tmp_path):
    grader = _ScriptedGrader(verdicts=[PASS])
    rubric = create_rubric_middleware(
        model=grader,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        max_iterations=2,
    )
    agent = create_agent(
        model=_Main(messages=iter([AIMessage("a")])),
        tools=[],
        middleware=[rubric],
        checkpointer=InMemorySaver(),
    )
    results, values = _run(agent, "task", rubric="- s")
    assert results == ["satisfied"]
    assert values["rubric"] is None
