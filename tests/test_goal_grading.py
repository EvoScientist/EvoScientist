"""Grading while a thread has a goal: completion, what is not graded, what the grader reads."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from langchain_core.messages import AIMessage

from EvoScientist.middleware.goal import new_goal_request
from tests.fakes.goal import (
    ACCEPT,
    goal_agent,
    message_input,
    request_input,
    resume_input,
    run,
    seed_goal,
    tool_call,
)
from tests.fakes.rubric import FAIL, PASS

BLOCKED = tool_call("update_goal", "b1", status="blocked", note="need an API key")
EVIDENCE = tool_call("update_goal", "c1", status="complete", note="all tests pass")
NEXT_ONCE = {"session": "s1", "seq": 1, "next": "- one shot"}
FEEDBACK_MARK = "A grader reviewed your work"


@contextmanager
def _marked_run(mode):
    """The run config a hook would see, for calling one rule directly."""
    from langchain_core.runnables.config import var_child_runnable_config

    configurable = {"rubric_run": {"mode": mode, "changes": []}} if mode else {}
    token = var_child_runnable_config.set({"configurable": configurable})
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


_SAME = object()


def _graded(status, *, grading=_SAME, evidence="e"):
    """Thread state in which the goal's own criteria are being graded."""
    if grading is _SAME:
        grading = {"source": "sticky", "id": "goal:x"}
    return {
        "_goal": {"status": status, "rubric_id": "goal:x", "evidence": evidence},
        "_sticky_rubric": {"text": "- a", "id": "goal:x"},
        "_rubric_grading": grading,
    }


def _verdict(state, result="satisfied"):
    from EvoScientist.middleware.rubric import EvoRubricMiddleware

    middleware = EvoRubricMiddleware.__new__(EvoRubricMiddleware)
    middleware._goal_event = lambda *_args, **_fields: None  # no stream here
    return middleware._goal_verdict_update(state, result)


def _calls(*calls) -> AIMessage:
    return AIMessage("", tool_calls=list(calls))


class TestCompletion:
    async def test_a_satisfied_grading_completes_the_goal_in_the_same_write(
        self, tmp_path
    ):
        agent, models = goal_agent(tmp_path, replies=["done"], verdicts=[PASS])
        await seed_goal(agent)
        result = await run(agent, message_input("go on"))
        goal = result.values["_goal"]
        assert goal["status"] == "complete"
        assert goal["note"] == "Acceptance criteria satisfied."
        assert goal["evidence"] is None
        assert result.verdicts == ["satisfied"]
        complete = next(e for e in result.events if e.get("change") == "complete")
        assert complete["status"] == "complete"
        # The verdict line says it already; the event carries only evidence.
        assert not complete.get("note")
        assert result.values.get("rubric") is None
        assert result.values["_sticky_rubric"]["id"] == goal["rubric_id"]
        assert models.grader.calls == 1

    async def test_the_agents_evidence_becomes_the_note(self, tmp_path):
        agent, _ = goal_agent(
            tmp_path, replies=[_calls(EVIDENCE), "done"], verdicts=[PASS]
        )
        await seed_goal(agent)
        result = await run(agent, message_input("go on"))
        assert result.values["_goal"]["status"] == "complete"
        assert result.values["_goal"]["note"] == "all tests pass"
        complete = next(e for e in result.events if e.get("change") == "complete")
        assert complete["note"] == "all tests pass"

    async def test_the_pass_limit_leaves_the_goal_active_and_drops_the_evidence(
        self, tmp_path
    ):
        agent, models = goal_agent(
            tmp_path, replies=[_calls(EVIDENCE), "try"], verdicts=[FAIL]
        )
        await seed_goal(agent)
        result = await run(agent, message_input("go on"))
        assert result.verdicts[-1] == "max_iterations_reached"
        assert models.grader.calls == 3
        assert result.values["_goal"]["status"] == "active"
        assert result.values["_goal"]["evidence"] is None
        assert result.changes[-1] == "unmet"

    async def test_a_grader_error_keeps_the_goal_and_the_evidence(self, tmp_path):
        agent, _ = goal_agent(
            tmp_path,
            replies=[_calls(EVIDENCE), "done"],
            verdicts=[RuntimeError("boom")],
        )
        await seed_goal(agent)
        result = await run(agent, message_input("go on"))
        assert result.verdicts == ["grader_error"]
        assert result.values["_goal"]["status"] == "active"
        assert result.values["_goal"]["evidence"] == "all tests pass"
        assert "unmet" not in result.changes

    async def test_a_goal_made_in_this_run_completes_in_it(self, tmp_path):
        agent, _ = goal_agent(
            tmp_path, drafts=["- a"], replies=["done"], verdicts=[PASS]
        )
        await run(agent, request_input(new_goal_request("create", objective="ship it")))
        result = await run(agent, resume_input(ACCEPT))
        assert result.changes == ["accepted", "complete"]
        assert result.values["_goal"]["status"] == "complete"


class TestNotGraded:
    @pytest.mark.parametrize("status", ["paused", "complete"])
    async def test_a_goal_that_is_not_being_worked_on(self, tmp_path, status):
        agent, models = goal_agent(tmp_path, replies=["sure"], verdicts=[PASS])
        await seed_goal(agent, status=status)
        result = await run(agent, message_input("an unrelated question"))
        assert models.grader.calls == 0
        assert result.verdicts == []
        assert result.values["_goal"]["status"] == status
        assert result.values.get("rubric") is None

    async def test_the_turn_in_which_the_agent_reports_blocked(self, tmp_path):
        agent, models = goal_agent(
            tmp_path, replies=[_calls(BLOCKED), "I need you"], verdicts=[FAIL]
        )
        await seed_goal(agent)
        result = await run(agent, message_input("go on"))
        assert models.grader.calls == 0
        assert result.values["_goal"]["status"] == "blocked"
        assert result.values.get("_rubric_grading") is None
        assert result.values.get("rubric") is None
        # One model call for the tool, one for the reply: nobody pushed it back.
        assert len(models.main.seen) == 2

    async def test_the_message_after_a_blocked_turn_is_graded_again(self, tmp_path):
        agent, models = goal_agent(
            tmp_path, replies=[_calls(BLOCKED), "I need you", "done"], verdicts=[PASS]
        )
        await seed_goal(agent)
        await run(agent, message_input("go on"))
        result = await run(agent, message_input("here you are"))
        assert models.grader.calls == 1
        assert result.values["_goal"]["status"] == "complete"

    async def test_pausing_drops_a_deferred_grading_for_good(self, tmp_path):
        """The grading waited for background work; after the pause the turn
        that work's completion starts is not graded."""
        agent, models = goal_agent(tmp_path, replies=["noted"], verdicts=[PASS])
        target = await seed_goal(agent)
        cfg = {"configurable": {"thread_id": "t"}}
        await run(agent, message_input("start the long job"), mode=None)
        first = (await agent.aget_state(cfg)).values["messages"][0].id
        deferred = {"source": "sticky", "id": target, "start": first, "deferred": True}
        await agent.aupdate_state(
            cfg,
            {"_rubric_grading": deferred, "rubric": "- the report exists"},
            as_node="EvoRubricMiddleware.after_agent",
        )
        await agent.aupdate_state(cfg, None, as_node="__end__")
        await run(agent, request_input(new_goal_request("pause", target=target)))
        result = await run(agent, message_input("the job finished"), mode="followup")
        assert models.grader.calls == 0
        assert result.verdicts == []
        assert result.values.get("_rubric_grading") is None
        assert result.values.get("rubric") is None

    def test_a_verdict_changes_only_an_active_goal(self):
        """The hook leaves a blocked turn before grading; the rule holds by itself too."""
        with _marked_run("thread"):
            assert _verdict(_graded("active")) == {
                "_goal": {
                    "status": "complete",
                    "note": "e",
                    "evidence": None,
                }
            }
            for status in ("blocked", "paused", "complete"):
                assert _verdict(_graded(status)) == {}
                assert _verdict(_graded(status), "max_iterations_reached") == {}

    def test_a_verdict_of_another_grading_never_completes_the_goal(self):
        with _marked_run("thread"):
            for grading in (
                {"source": "sticky", "id": "goal:old"},
                {"source": "next", "id": "goal:x"},
                None,
            ):
                # Only the evidence staged in that turn goes.
                assert _verdict(_graded("active", grading=grading)) == {
                    "_goal": {"evidence": None}
                }
                state = _graded("active", grading=grading, evidence=None)
                assert _verdict(state) == {}
                assert _verdict(_graded("blocked", grading=grading)) == {}

    @pytest.mark.parametrize("mode", ["input", None])
    def test_a_verdict_on_a_run_where_goals_do_not_apply_leaves_the_goal(self, mode):
        with _marked_run(mode):
            assert _verdict(_graded("active")) == {}


class TestOtherChecklists:
    async def test_a_one_shot_verdict_does_not_complete_the_goal(self, tmp_path):
        agent, models = goal_agent(tmp_path, replies=["done"], verdicts=[PASS])
        await seed_goal(agent)
        result = await run(agent, message_input("go on"), changes=[NEXT_ONCE])
        assert result.verdicts == ["satisfied"]
        assert "- one shot" in models.grader.last_payload
        assert result.values["_goal"]["status"] == "active"
        assert result.values.get("_next_rubric") is None
        assert "complete" not in result.changes

    async def test_evidence_written_under_a_one_shot_does_not_outlive_its_verdict(
        self, tmp_path
    ):
        """The agent reported completion in a turn another checklist graded;
        that note says nothing about the turn that later meets the goal."""
        agent, _ = goal_agent(
            tmp_path, replies=[_calls(EVIDENCE), "done", "really done"], verdicts=[PASS]
        )
        await seed_goal(agent)
        first = await run(agent, message_input("go on"), changes=[NEXT_ONCE])
        assert first.values["_goal"]["status"] == "active"
        assert first.values["_goal"]["evidence"] is None
        assert "unmet" not in first.changes
        second = await run(agent, message_input("and now the goal"))
        assert second.values["_goal"]["status"] == "complete"
        assert second.values["_goal"]["note"] == "Acceptance criteria satisfied."

    async def test_a_blocked_turn_under_a_one_shot_is_graded_and_uses_it_up(
        self, tmp_path
    ):
        agent, models = goal_agent(
            tmp_path, replies=[_calls(BLOCKED), "I need you"], verdicts=[PASS]
        )
        await seed_goal(agent)
        result = await run(agent, message_input("go on"), changes=[NEXT_ONCE])
        assert models.grader.calls == 1
        assert result.values.get("_next_rubric") is None
        assert result.values["_goal"]["status"] == "blocked"

    async def test_an_input_rubric_is_graded_on_a_thread_with_a_blocked_goal(
        self, tmp_path
    ):
        agent, models = goal_agent(tmp_path, replies=["done"], verdicts=[PASS])
        await seed_goal(agent, status="blocked", note="need a key")
        payload = {**message_input("one-off"), "rubric": "- the answer is short"}
        result = await run(agent, payload, mode="input")
        assert models.grader.calls == 1
        assert "- the answer is short" in models.grader.last_payload
        assert result.values["_goal"]["status"] == "blocked"
        assert result.values["_goal"]["note"] == "need a key"

    @pytest.mark.parametrize("sticky", ["- another checklist", None])
    async def test_a_checklist_command_replaces_the_goal(self, tmp_path, sticky):
        agent, _ = goal_agent(tmp_path, replies=["done"], verdicts=[PASS])
        await seed_goal(agent)
        operation = {"session": "s1", "seq": 1, "sticky": sticky}
        result = await run(agent, message_input("go on"), changes=[operation])
        assert result.values.get("_goal") is None
        assert result.changes == ["cleared"]
        if sticky:
            assert result.values["_sticky_rubric"]["text"] == sticky
            assert result.verdicts == ["satisfied"]


class TestWhatTheGraderReads:
    async def _two_turns(self, tmp_path, second_turn_verdicts):
        agent, models = goal_agent(
            tmp_path,
            drafts=["- the report exists"],
            replies=[
                "first attempt",
                "second attempt",
                "third attempt",
                "final attempt",
            ],
            verdicts=[FAIL, FAIL, FAIL, *second_turn_verdicts],
        )
        await run(
            agent,
            request_input(new_goal_request("create", objective="ship the report")),
        )
        first = await run(agent, resume_input(ACCEPT))
        assert first.verdicts[-1] == "max_iterations_reached"
        result = await run(agent, message_input("keep going"))
        return result, models

    async def test_a_later_turn_is_graded_from_the_goals_first_message(self, tmp_path):
        result, models = await self._two_turns(tmp_path, [PASS])
        payload = models.grader.last_payload
        assert "ship the report" in payload
        assert "first attempt" in payload
        assert "final attempt" in payload
        assert result.values["_goal"]["status"] == "complete"

    async def test_feedback_of_earlier_gradings_is_left_out(self, tmp_path):
        _, models = await self._two_turns(tmp_path, [PASS])
        assert FEEDBACK_MARK not in models.grader.last_payload

    async def test_feedback_of_the_current_grading_stays(self, tmp_path):
        _, models = await self._two_turns(tmp_path, [FAIL, PASS])
        assert models.grader.last_payload.count(FEEDBACK_MARK) == 1

    async def test_a_goal_whose_first_message_is_gone_is_graded_from_this_turn(
        self, tmp_path
    ):
        agent, models = goal_agent(tmp_path, replies=["done"], verdicts=[PASS])
        await seed_goal(agent, start="no-such-message")
        result = await run(agent, message_input("go on"))
        assert "go on" in models.grader.last_payload
        assert result.values["_goal"]["status"] == "complete"

    async def test_an_amendment_keeps_what_the_grader_reads(self, tmp_path):
        agent, models = goal_agent(
            tmp_path,
            drafts=[
                "- the report exists",
                "OBJECTIVE:\nship the report\n\nCRITERIA:\n- report\n- title",
            ],
            replies=[
                "first attempt",
                "second attempt",
                "third attempt",
                "final attempt",
            ],
            verdicts=[FAIL, FAIL, FAIL, PASS],
        )
        await run(
            agent,
            request_input(new_goal_request("create", objective="ship the report")),
        )
        accepted = await run(agent, resume_input(ACCEPT))
        target = accepted.values["_goal"]["rubric_id"]
        request = new_goal_request("amend", feedback="add a title", target=target)
        await run(agent, request_input(request))
        result = await run(agent, resume_input(ACCEPT))
        assert result.values["_goal"]["start"] == accepted.values["_goal"]["start"]
        assert "first attempt" in models.grader.last_payload
        assert "- title" in models.grader.last_payload


class TestAParkedTurnContinuedElsewhere:
    """A goal's turn parks at a question; the client that answers marks its
    run for another kind of grading. The start hooks do not run again, so the
    thread still holds the goal's grading record."""

    @staticmethod
    def _asking_agent(tmp_path, verdicts):
        from langchain.tools import tool
        from langgraph.types import interrupt

        @tool
        def ask(question: str) -> str:
            """Ask the user."""
            return str(
                interrupt({"type": "ask_user", "questions": [{"question": question}]})
            )

        return goal_agent(
            tmp_path,
            replies=[_calls(tool_call("ask", "q1", question="which one?")), "done"],
            verdicts=verdicts,
            tools=[ask],
        )

    @pytest.mark.parametrize("mode", ["input", None])
    @pytest.mark.parametrize("verdict", [PASS, FAIL])
    async def test_the_goals_criteria_are_not_graded_there(
        self, tmp_path, mode, verdict
    ):
        """On that run the goal does not apply: no verdict, no revisions."""
        agent, models = self._asking_agent(tmp_path, [verdict])
        await seed_goal(agent)
        parked = await run(agent, message_input("go on"))
        assert parked.question is not None
        assert parked.values["_rubric_grading"]["source"] == "sticky"
        result = await run(
            agent, resume_input({"answers": ["a"], "status": "answered"}), mode=mode
        )
        assert result.next == ()
        assert models.grader.calls == 0
        assert result.verdicts == []
        assert result.values["_goal"]["status"] == "active"
        assert result.values.get("_rubric_grading") is None
        assert result.values.get("rubric") is None
        # One model call for the question, one for the reply: nobody pushed it back.
        assert len(models.main.seen) == 2

    async def test_a_plain_checklist_parked_and_continued_elsewhere_is_unchanged(
        self, tmp_path
    ):
        """Only a goal's grading is dropped: what an input-marked continuation
        does with an ordinary ``/rubric set`` checklist is not this layer's."""
        agent, models = self._asking_agent(tmp_path, [PASS])
        operation = {"session": "s1", "seq": 1, "sticky": "- plain checklist"}
        await run(agent, message_input("go on"), changes=[operation])
        result = await run(
            agent, resume_input({"answers": ["a"], "status": "answered"}), mode="input"
        )
        assert models.grader.calls == 1
        assert result.verdicts == ["satisfied"]

    async def test_the_same_client_still_completes_it(self, tmp_path):
        agent, _models = self._asking_agent(tmp_path, [PASS])
        await seed_goal(agent)
        await run(agent, message_input("go on"))
        result = await run(
            agent, resume_input({"answers": ["a"], "status": "answered"})
        )
        assert result.values["_goal"]["status"] == "complete"
