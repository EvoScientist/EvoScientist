"""``/goal`` requests through the graph: drafting, the review pause, status changes."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from EvoScientist.middleware.goal import CONTINUE_MESSAGE, new_goal_request
from tests.goal_fakes import (
    ACCEPT,
    CANCEL,
    ESCAPED,
    feedback,
    goal_agent,
    history,
    message_input,
    request_input,
    resume_input,
    run,
    seed_goal,
)
from tests.rubric_fakes import PASS

OBJECTIVE = "ship the report"
DRAFT_1 = "- the report exists"
DRAFT_2 = "- the report exists\n- it has a title"


def _create(**kwargs):
    return new_goal_request("create", objective=OBJECTIVE, **kwargs)


def _question(result) -> str:
    assert result.question is not None, "no review question was shown"
    assert result.question["type"] == "ask_user"
    (question,) = result.question["questions"]
    assert [c["value"] for c in question["choices"]] == ["Accept", "Cancel"]
    return question["question"]


class TestCreate:
    async def test_the_draft_is_shown_and_nothing_else_happens_yet(self, tmp_path):
        agent, models = goal_agent(tmp_path, drafts=[DRAFT_1])
        result = await run(agent, request_input(_create()))
        text = _question(result)
        assert OBJECTIVE in text
        assert DRAFT_1 in text
        assert text.endswith("Accept them, or type what to change.")
        assert result.changes == ["drafting"]
        assert history(result) == []
        assert result.values.get("_goal") is None
        assert models.main.seen == []

    async def test_accept_makes_the_goal_and_the_same_run_starts_the_work(
        self, tmp_path
    ):
        agent, models = goal_agent(tmp_path, drafts=[DRAFT_1], replies=["on it"])
        request = _create()
        await run(agent, request_input(request))
        result = await run(agent, resume_input(ACCEPT))
        goal = result.values["_goal"]
        assert goal["objective"] == OBJECTIVE
        assert goal["rubric_id"] == f"goal:{request['id']}"
        assert goal["note"] is None
        assert result.values["_sticky_rubric"] == {
            "text": DRAFT_1,
            "id": goal["rubric_id"],
        }
        assert result.values.get("goal_request") is None
        assert history(result)[:2] == [("human", OBJECTIVE), ("ai", "on it")]
        assert goal["start"] == result.values["messages"][0].id
        # Nothing is drafted or announced again when the pause resumes.
        assert result.changes[0] == "accepted"
        assert "drafting" not in result.changes
        assert len(models.drafter.seen) == 1
        assert result.next == ()

    async def test_the_accepted_event_carries_what_a_surface_has_to_show(
        self, tmp_path
    ):
        agent, _ = goal_agent(tmp_path, drafts=[DRAFT_1])
        await run(agent, request_input(_create()))
        result = await run(agent, resume_input(ACCEPT))
        accepted = next(e for e in result.events if e.get("change") == "accepted")
        assert accepted["objective"] == OBJECTIVE
        assert accepted["criteria"] == DRAFT_1
        assert accepted["message"] == OBJECTIVE
        assert accepted["status"] == "active"
        assert "unreviewed" not in accepted

    @pytest.mark.parametrize(
        "answer", [CANCEL, ESCAPED, {"status": "error"}, "garbage"]
    )
    async def test_cancel_leaves_no_trace(self, tmp_path, answer):
        agent, models = goal_agent(tmp_path, drafts=[DRAFT_1])
        await run(agent, request_input(_create()))
        result = await run(agent, resume_input(answer))
        assert result.changes == ["cancelled"]
        assert history(result) == []
        assert result.values.get("_goal") is None
        assert result.values.get("goal_request") is None
        assert result.next == ()
        assert models.main.seen == []
        assert models.grader.calls == 0

    async def test_feedback_redrafts_and_the_shown_draft_is_the_one_accepted(
        self, tmp_path
    ):
        agent, models = goal_agent(tmp_path, drafts=[DRAFT_1, DRAFT_2, "- never used"])
        first = await run(agent, request_input(_create()))
        second = await run(agent, resume_input(feedback("add a title")))
        assert first.changes == ["drafting"]
        assert second.changes == ["drafting"]
        assert DRAFT_2 in _question(second)
        redraft = models.drafter.seen[1][-1].content
        assert "<previous_criteria>\n" + DRAFT_1 in redraft
        assert "<user_feedback>\nadd a title" in redraft
        result = await run(agent, resume_input(ACCEPT))
        assert result.values["_sticky_rubric"]["text"] == DRAFT_2
        assert len(models.drafter.seen) == 2

    async def test_a_first_draft_that_fails_ends_the_run(self, tmp_path):
        agent, models = goal_agent(tmp_path, drafts=[RuntimeError("503 upstream")])
        result = await run(agent, request_input(_create()))
        assert result.question is None
        assert result.changes == ["drafting", "failed"]
        failed = next(e for e in result.events if e.get("change") == "failed")
        assert "503 upstream" in failed["note"]
        assert result.values.get("goal_request") is None
        assert result.next == ()
        assert models.main.seen == []

    async def test_after_a_failed_redraft_accept_takes_the_draft_that_was_shown(
        self, tmp_path
    ):
        agent, models = goal_agent(
            tmp_path, drafts=[DRAFT_1, RuntimeError("503 upstream\nretry"), DRAFT_2]
        )
        await run(agent, request_input(_create()))
        second = await run(agent, resume_input(feedback("add a title")))
        text = _question(second)
        assert DRAFT_1 in text
        assert "The redraft failed: 503 upstream retry" in text
        result = await run(agent, resume_input(ACCEPT))
        assert result.values["_sticky_rubric"]["text"] == DRAFT_1
        # The failed call is a saved result: it is not made again.
        assert len(models.drafter.seen) == 2

    async def test_an_objective_over_the_limit_is_refused_without_drafting(
        self, tmp_path
    ):
        agent, models = goal_agent(tmp_path, drafts=[DRAFT_1])
        request = new_goal_request("create", objective="x" * 8_001)
        result = await run(agent, request_input(request))
        assert result.changes == ["failed"]
        assert models.drafter.seen == []

    async def test_without_review_the_draft_is_applied_at_once(self, tmp_path):
        agent, _ = goal_agent(
            tmp_path, drafts=[DRAFT_1], replies=["on it"], unattended=True
        )
        result = await run(agent, request_input(_create()))
        assert result.question is None
        assert result.changes[:2] == ["drafting", "accepted"]
        assert result.values["_sticky_rubric"]["text"] == DRAFT_1
        assert history(result)[:2] == [("human", OBJECTIVE), ("ai", "on it")]

    async def test_a_rebuilt_graph_resumes_with_the_drafts_already_made(self, tmp_path):
        saver = InMemorySaver()
        agent, models = goal_agent(tmp_path, drafts=[DRAFT_1, DRAFT_2], saver=saver)
        await run(agent, request_input(_create()))
        agent, _ = goal_agent(tmp_path, saver=saver, models=models)
        second = await run(agent, resume_input(feedback("add a title")))
        assert DRAFT_2 in _question(second)
        agent, _ = goal_agent(tmp_path, saver=saver, models=models)
        result = await run(agent, resume_input(ACCEPT))
        assert result.values["_sticky_rubric"]["text"] == DRAFT_2
        assert len(models.drafter.seen) == 2

    async def test_the_same_holds_on_the_sqlite_checkpointer(self, tmp_path):
        from EvoScientist.sessions import PruningCheckpointer

        db = str(tmp_path / "sessions.db")
        models = None
        script = [
            request_input(_create()),
            resume_input(feedback("x")),
            resume_input(ACCEPT),
        ]
        for payload in script:
            async with PruningCheckpointer.from_conn_string_with_keep(
                db, keep_per_ns=1
            ) as saver:
                agent, models = goal_agent(
                    tmp_path, drafts=[DRAFT_1, DRAFT_2], saver=saver, models=models
                )
                result = await run(agent, payload)
        assert result.values["_sticky_rubric"]["text"] == DRAFT_2
        assert len(models.drafter.seen) == 2

    async def test_a_new_goal_replaces_the_old_one_only_when_accepted(self, tmp_path):
        agent, _ = goal_agent(tmp_path, drafts=[DRAFT_2])
        old_id = await seed_goal(agent, note="kept")
        parked = await run(agent, request_input(_create()))
        assert parked.values["_goal"]["rubric_id"] == old_id
        result = await run(agent, resume_input(ACCEPT))
        goal = result.values["_goal"]
        assert goal["rubric_id"] != old_id
        assert goal["note"] is None
        assert result.values["_sticky_rubric"]["text"] == DRAFT_2

    async def test_a_second_request_replaces_a_parked_review(self, tmp_path):
        agent, _ = goal_agent(tmp_path, drafts=[DRAFT_1, DRAFT_2])
        await run(agent, request_input(_create()))
        second = new_goal_request("create", objective="another goal")
        result = await run(agent, request_input(second))
        assert "another goal" in _question(result)
        assert DRAFT_2 in _question(result)
        assert result.values["goal_request"]["id"] == second["id"]


AMENDED = "OBJECTIVE:\nship the report and the slides\n\nCRITERIA:\n- report\n- slides"


class TestAmend:
    async def test_an_amendment_keeps_the_start_and_changes_goal_and_criteria(
        self, tmp_path
    ):
        agent, models = goal_agent(tmp_path, drafts=[AMENDED], replies=["continuing"])
        target = await seed_goal(
            agent, start="m-first", note="old note", evidence="old"
        )
        request = new_goal_request("amend", feedback="also the slides", target=target)
        parked = await run(agent, request_input(request))
        assert "ship the report and the slides" in _question(parked)
        prompt = models.drafter.seen[0][-1].content
        assert "<current_goal>\nship the report" in prompt
        assert "<current_criteria>\n- the report exists" in prompt
        result = await run(agent, resume_input(ACCEPT))
        goal = result.values["_goal"]
        assert goal["objective"] == "ship the report and the slides"
        assert goal["rubric_id"] == f"goal:{request['id']}"
        assert goal["start"] == "m-first"
        assert goal["note"] is None
        assert goal["evidence"] is None
        assert result.values["_sticky_rubric"]["text"] == "- report\n- slides"
        assert history(result)[:2] == [
            ("human", "also the slides"),
            ("ai", "continuing"),
        ]
        assert result.changes[0] == "amended"

    async def test_amending_a_blocked_goal_makes_it_active(self, tmp_path):
        agent, _ = goal_agent(tmp_path, drafts=[AMENDED])
        target = await seed_goal(agent, status="blocked", note="need a key")
        request = new_goal_request("amend", feedback="skip the key", target=target)
        await run(agent, request_input(request))
        result = await run(agent, resume_input(ACCEPT))
        assert result.values["_goal"]["note"] is None

    @pytest.mark.parametrize("status", ["paused", "complete"])
    async def test_a_goal_that_is_not_being_worked_on_cannot_be_amended(
        self, tmp_path, status
    ):
        agent, models = goal_agent(tmp_path, drafts=[AMENDED])
        target = await seed_goal(agent, status=status)
        request = new_goal_request("amend", feedback="x", target=target)
        result = await run(agent, request_input(request))
        assert result.changes == ["ignored"]
        assert models.drafter.seen == []
        assert result.values["_goal"]["status"] == status

    @pytest.mark.parametrize("target", [None, "goal:other"])
    async def test_an_amendment_of_another_goal_is_ignored(self, tmp_path, target):
        agent, models = goal_agent(tmp_path, drafts=[AMENDED])
        await seed_goal(agent)
        request = new_goal_request("amend", feedback="x", target=target)
        result = await run(agent, request_input(request))
        assert result.changes == ["ignored"]
        assert models.drafter.seen == []

    async def test_an_amendment_without_a_goal_is_ignored(self, tmp_path):
        agent, _ = goal_agent(tmp_path, drafts=[AMENDED])
        request = new_goal_request("amend", feedback="x", target="goal:seed")
        result = await run(agent, request_input(request))
        assert result.changes == ["ignored"]
        assert result.values.get("goal_request") is None


class TestStatusRequests:
    async def test_pause_takes_effect_at_once_and_calls_no_model(self, tmp_path):
        agent, models = goal_agent(tmp_path)
        target = await seed_goal(agent, evidence="staged")
        result = await run(
            agent, request_input(new_goal_request("pause", target=target))
        )
        assert result.changes == ["paused"]
        assert result.values["_goal"]["status"] == "paused"
        assert result.values["_goal"]["evidence"] is None
        assert result.values["_sticky_rubric"]["text"] == "- the report exists"
        assert history(result) == []
        assert models.main.seen == []
        assert result.next == ()

    async def test_clear_removes_the_goal_and_its_criteria(self, tmp_path):
        agent, models = goal_agent(tmp_path)
        target = await seed_goal(agent, status="complete")
        result = await run(
            agent, request_input(new_goal_request("clear", target=target))
        )
        assert result.changes == ["cleared"]
        assert result.values.get("_goal") is None
        assert result.values.get("_sticky_rubric") is None
        assert models.main.seen == []

    async def test_resume_adds_the_continuation_and_runs_the_turn(self, tmp_path):
        agent, models = goal_agent(tmp_path, replies=["continuing"])
        target = await seed_goal(agent, status="paused")
        result = await run(
            agent, request_input(new_goal_request("resume", target=target))
        )
        assert result.changes[0] == "resumed"
        assert history(result)[:2] == [
            ("human", CONTINUE_MESSAGE),
            ("ai", "continuing"),
        ]
        assert result.values.get("goal_request") is None
        assert len(models.main.seen) >= 1

    @pytest.mark.parametrize(
        ("kind", "status"),
        [
            ("pause", "paused"),
            ("pause", "blocked"),
            ("pause", "complete"),
            ("resume", "active"),
            ("resume", "blocked"),
            ("resume", "complete"),
        ],
    )
    async def test_a_change_the_status_does_not_allow_is_ignored(
        self, tmp_path, kind, status
    ):
        agent, models = goal_agent(tmp_path)
        target = await seed_goal(agent, status=status)
        result = await run(agent, request_input(new_goal_request(kind, target=target)))
        assert result.changes == ["ignored"]
        assert result.values["_goal"]["status"] == status
        assert history(result) == []
        assert models.main.seen == []

    @pytest.mark.parametrize("kind", ["pause", "resume", "clear"])
    async def test_a_change_aimed_at_another_goal_is_ignored(self, tmp_path, kind):
        agent, _ = goal_agent(tmp_path)
        await seed_goal(agent, status="paused" if kind == "resume" else "active")
        request = new_goal_request(kind, target="goal:replaced-meanwhile")
        result = await run(agent, request_input(request))
        assert result.changes == ["ignored"]
        assert result.values["_goal"]["rubric_id"] == "goal:seed"
        assert history(result) == []

    @pytest.mark.parametrize("kind", ["pause", "resume", "clear"])
    async def test_a_change_without_a_goal_is_ignored(self, tmp_path, kind):
        agent, _ = goal_agent(tmp_path)
        request = new_goal_request(kind, target="goal:seed")
        result = await run(agent, request_input(request))
        assert result.changes == ["ignored"]
        assert result.values.get("goal_request") is None
        assert history(result) == []

    @pytest.mark.parametrize("kind", ["pause", "clear"])
    async def test_pause_and_clear_drop_an_open_grading_of_the_goal(
        self, tmp_path, kind
    ):
        agent, _ = goal_agent(tmp_path)
        target = await seed_goal(agent)
        cfg = {"configurable": {"thread_id": "t"}}
        open_grading = {
            "source": "sticky",
            "id": target,
            "start": "m1",
            "deferred": True,
        }
        await agent.aupdate_state(
            cfg,
            {"_rubric_grading": open_grading},
            as_node="EvoRubricMiddleware.after_agent",
        )
        await agent.aupdate_state(cfg, None, as_node="__end__")
        result = await run(agent, request_input(new_goal_request(kind, target=target)))
        assert result.values.get("_rubric_grading") is None


SET_OTHER = {"session": "s1", "seq": 4, "sticky": "- another checklist"}


class TestPendingChecklistOperations:
    """``/rubric`` operations typed before a goal command ride on its run."""

    async def test_accept_supersedes_them(self, tmp_path):
        agent, _ = goal_agent(tmp_path, drafts=[DRAFT_1])
        await run(agent, request_input(_create()), changes=[SET_OTHER])
        result = await run(agent, resume_input(ACCEPT), changes=[SET_OTHER])
        assert result.values["_rubric_applied"] == {"s1": 4}
        assert result.values["_sticky_rubric"]["text"] == DRAFT_1
        assert (
            result.values["_goal"]["rubric_id"] == result.values["_sticky_rubric"]["id"]
        )

    async def test_resume_supersedes_them(self, tmp_path):
        agent, _ = goal_agent(tmp_path)
        target = await seed_goal(agent, status="paused")
        request = new_goal_request("resume", target=target)
        result = await run(agent, request_input(request), changes=[SET_OTHER])
        assert result.values["_rubric_applied"] == {"s1": 4}
        assert result.values["_sticky_rubric"]["id"] == target

    @pytest.mark.parametrize("kind", ["pause", "clear"])
    async def test_a_run_that_ends_in_the_hook_leaves_them_pending(
        self, tmp_path, kind
    ):
        agent, _ = goal_agent(tmp_path)
        target = await seed_goal(agent)
        request = new_goal_request(kind, target=target)
        result = await run(agent, request_input(request), changes=[SET_OTHER])
        assert not result.values.get("_rubric_applied")

    async def test_cancel_leaves_them_pending(self, tmp_path):
        agent, _ = goal_agent(tmp_path, drafts=[DRAFT_1])
        await run(agent, request_input(_create()), changes=[SET_OTHER])
        result = await run(agent, resume_input(CANCEL), changes=[SET_OTHER])
        assert not result.values.get("_rubric_applied")
        assert result.values.get("_sticky_rubric") is None


class TestBlockedGoal:
    async def test_the_users_next_message_unblocks_it(self, tmp_path):
        agent, _ = goal_agent(tmp_path, replies=["thanks"])
        await seed_goal(agent, status="blocked", note="need a key")
        result = await run(agent, message_input("here is the key"))
        assert result.changes[0] == "unblocked"
        assert result.values["_goal"]["note"] is None

    async def test_a_notifier_turn_does_not_unblock_it(self, tmp_path):
        agent, _ = goal_agent(tmp_path)
        await seed_goal(agent, status="blocked", note="need a key")
        result = await run(agent, message_input("a task finished"), mode="followup")
        assert "unblocked" not in result.changes
        assert result.values["_goal"]["status"] == "blocked"
        assert result.values["_goal"]["note"] == "need a key"

    async def test_a_request_run_does_not_unblock_it(self, tmp_path):
        agent, _ = goal_agent(tmp_path)
        await seed_goal(agent, status="blocked", note="need a key")
        request = new_goal_request("pause", target="goal:seed")
        result = await run(agent, request_input(request))
        assert result.changes == ["ignored"]
        assert result.values["_goal"]["status"] == "blocked"


class TestRunsWithoutTheMarker:
    @pytest.mark.parametrize("mode", [None, "followup", "input"])
    async def test_a_request_is_dropped_and_the_run_ends(self, tmp_path, mode):
        agent, models = goal_agent(tmp_path, drafts=[DRAFT_1])
        result = await run(agent, request_input(_create()), mode=mode)
        assert result.question is None
        assert result.changes == []
        assert result.values.get("goal_request") is None
        assert models.drafter.seen == []
        assert models.main.seen == []
        assert result.next == ()

    async def test_a_message_from_another_client_is_answered_despite_a_leftover(
        self, tmp_path
    ):
        """The TUI left a review behind; a client that writes neither the
        request key nor the marker continues the thread."""
        from langchain_core.messages import HumanMessage

        agent, models = goal_agent(tmp_path, drafts=[DRAFT_1], replies=["hello"])
        await run(agent, request_input(_create()))
        result = await run(agent, {"messages": [HumanMessage("hi")]}, mode=None)
        assert history(result) == [("human", "hi"), ("ai", "hello")]
        assert result.values.get("goal_request") is None
        assert len(models.drafter.seen) == 1


class TestLeftoverRequest:
    """A request whose run never finished must not take over the next message."""

    async def test_the_next_message_after_a_review_left_open_is_answered(
        self, tmp_path
    ):
        agent, models = goal_agent(
            tmp_path, drafts=[DRAFT_1, DRAFT_2], replies=["hello"]
        )
        await run(agent, request_input(_create()))
        result = await run(agent, message_input("hi"))
        assert result.question is None
        assert history(result) == [("human", "hi"), ("ai", "hello")]
        assert result.values.get("goal_request") is None
        assert len(models.drafter.seen) == 1

    async def test_the_next_message_after_a_cut_draft_is_answered(self, tmp_path):
        import asyncio
        import threading

        held = threading.Event()
        agent, models = goal_agent(tmp_path, drafts=[held, DRAFT_2], replies=["hello"])
        drafting = asyncio.ensure_future(run(agent, request_input(_create())))
        try:
            for _ in range(300):
                await asyncio.sleep(0.01)
                if models.drafter.seen:
                    break
            drafting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await drafting
        finally:
            held.set()
        result = await run(agent, message_input("hi"))
        assert history(result) == [("human", "hi"), ("ai", "hello")]
        assert result.values.get("goal_request") is None
        assert len(models.drafter.seen) == 1


class TestUnattendedAgent:
    """An agent built for auto mode has nobody to answer: no surface has to say so."""

    async def test_a_new_goal_is_adopted_without_a_question(self, tmp_path):
        agent, models = goal_agent(
            tmp_path, drafts=["- a"], replies=["done"], verdicts=[PASS], unattended=True
        )
        result = await run(
            agent, request_input(new_goal_request("create", objective="ship it"))
        )
        assert result.question is None
        assert result.changes[:2] == ["drafting", "accepted"]
        assert result.values["_goal"]["objective"] == "ship it"
        assert len(models.main.seen) == 1
        # Nobody saw the draft: the event says so, for the surface to warn.
        accepted = next(e for e in result.events if e.get("change") == "accepted")
        assert accepted["unreviewed"] is True

    async def test_an_amendment_is_adopted_without_a_question(self, tmp_path):
        agent, _ = goal_agent(
            tmp_path,
            drafts=["OBJECTIVE:\nship it\n\nCRITERIA:\n- a\n- b"],
            replies=["done"],
            verdicts=[PASS],
            unattended=True,
        )
        target = await seed_goal(agent)
        request = new_goal_request("amend", feedback="add b", target=target)
        result = await run(agent, request_input(request))
        assert result.question is None
        assert "amended" in result.changes
        assert result.values["_sticky_rubric"]["text"] == "- a\n- b"
        amended = next(e for e in result.events if e.get("change") == "amended")
        assert amended["unreviewed"] is True


class TestEdges:
    async def test_an_early_ending_leaves_a_deferred_grading_for_its_turn(
        self, tmp_path
    ):
        """A grading waits for background work; a cancelled ``/goal`` in between
        must neither run it on its own empty run nor lose it."""
        agent, models = goal_agent(
            tmp_path, drafts=["- a"], replies=["first", "more"], verdicts=[PASS]
        )
        cfg = {"configurable": {"thread_id": "t"}}
        await run(agent, message_input("start the long job"), mode=None)
        first = (await agent.aget_state(cfg)).values["messages"][0].id
        deferred = {"source": "sticky", "id": "s1:1", "start": first, "deferred": True}
        await agent.aupdate_state(
            cfg,
            {
                "_sticky_rubric": {"text": "- the job is done", "id": "s1:1"},
                "rubric": "- the job is done",
                "_rubric_grading": deferred,
            },
            as_node="EvoRubricMiddleware.after_agent",
        )
        await agent.aupdate_state(cfg, None, as_node="__end__")

        await run(agent, request_input(new_goal_request("create", objective="ship it")))
        cancelled = await run(agent, resume_input(CANCEL))
        assert cancelled.changes == ["cancelled"]
        assert models.grader.calls == 0
        assert cancelled.verdicts == []
        assert cancelled.values["_rubric_grading"] == deferred

        finished = await run(agent, message_input("the job finished"), mode="followup")
        assert finished.verdicts == ["satisfied"]
        assert "- the job is done" in models.grader.last_payload

    async def test_a_run_without_a_typed_message_does_not_unblock(self, tmp_path):
        agent, _ = goal_agent(tmp_path, replies=["still waiting"])
        await seed_goal(agent, status="blocked", note="need a key")
        result = await run(agent, {"messages": [], "goal_request": None})
        assert result.values["_goal"]["status"] == "blocked"
        assert "unblocked" not in result.changes

    async def test_accepting_drops_a_one_shot_checklist_that_was_not_used(
        self, tmp_path
    ):
        agent, models = goal_agent(
            tmp_path, drafts=["- goal criterion"], replies=["done"], verdicts=[PASS]
        )
        cfg = {"configurable": {"thread_id": "t"}}
        await agent.aupdate_state(
            cfg,
            {"_next_rubric": {"text": "- one shot", "id": "s1:1"}},
            as_node="EvoRubricMiddleware.after_agent",
        )
        await agent.aupdate_state(cfg, None, as_node="__end__")
        await run(agent, request_input(new_goal_request("create", objective="ship it")))
        result = await run(agent, resume_input(ACCEPT))
        assert result.values.get("_next_rubric") is None
        assert "- goal criterion" in models.grader.last_payload
        assert "- one shot" not in models.grader.last_payload
        assert result.values["_goal"]["status"] == "complete"

    async def test_the_drafter_reads_the_conversation_so_far(self, tmp_path):
        agent, models = goal_agent(
            tmp_path, drafts=["- a"], replies=["the parser is in lexer.py"]
        )
        await run(agent, message_input("where is the parser?"), mode=None)
        await run(agent, request_input(new_goal_request("create", objective="fix it")))
        prompt = str(models.drafter.seen[-1][-1].content)
        assert "where is the parser?" in prompt
        assert "the parser is in lexer.py" in prompt

    async def test_an_unknown_request_ends_the_run_without_a_model_call(self, tmp_path):
        agent, models = goal_agent(tmp_path, replies=["must not run"])
        result = await run(agent, request_input({"kind": "rename", "id": "r1"}))
        assert result.changes == ["ignored"]
        assert models.main.seen == []
        assert result.values.get("goal_request") is None

    @pytest.mark.parametrize(
        ("typed", "change"), [("accept", "accepted"), ("CANCEL", "cancelled")]
    )
    async def test_the_two_choices_are_understood_when_typed(
        self, tmp_path, typed, change
    ):
        agent, models = goal_agent(
            tmp_path, drafts=["- a"], replies=["done"], verdicts=[PASS]
        )
        await run(agent, request_input(new_goal_request("create", objective="ship it")))
        result = await run(agent, resume_input(feedback(typed)))
        assert change in result.changes
        assert len(models.drafter.seen) == 1

    async def test_feedback_on_an_amendment_starts_from_the_proposed_goal(
        self, tmp_path
    ):
        agent, models = goal_agent(
            tmp_path,
            drafts=[
                "OBJECTIVE:\nship the report and the slides\n\nCRITERIA:\n- report\n- slides",
                "OBJECTIVE:\nship the report and the slides\n\nCRITERIA:\n- report\n- ten slides",
            ],
            replies=["done"],
            verdicts=[PASS],
        )
        target = await seed_goal(agent)
        request = new_goal_request("amend", feedback="add slides", target=target)
        await run(agent, request_input(request))
        await run(agent, resume_input(feedback("ten of them")))
        redraft = str(models.drafter.seen[-1][-1].content)
        assert "ship the report and the slides" in redraft
        assert "- slides" in redraft
        assert "ten of them" in redraft
        result = await run(agent, resume_input(ACCEPT))
        assert result.values["_goal"]["objective"] == "ship the report and the slides"
        assert result.values["_sticky_rubric"]["text"] == "- report\n- ten slides"
        # The message in the conversation is what the user asked for first.
        assert history(result)[0] == ("human", "add slides")
