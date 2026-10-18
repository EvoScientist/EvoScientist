"""The goal kept on a thread: its merge rule, validity and size limits."""

from __future__ import annotations

import pytest

from EvoScientist.middleware.goal import (
    GOAL_CRITERIA_LIMIT,
    GOAL_NOTE_LIMIT,
    GOAL_OBJECTIVE_LIMIT,
    GOAL_TOTAL_LIMIT,
    GoalTextError,
    check_goal_note,
    check_goal_text,
    merge_goal,
    new_goal_request,
)
from EvoScientist.rubric_session import thread_goal

GOAL = {
    "objective": "ship it",
    "status": "active",
    "note": None,
    "evidence": None,
    "rubric_id": "goal:r1",
    "start": "m1",
}


class TestMergeGoal:
    def test_none_clears(self):
        assert merge_goal(dict(GOAL), None) is None

    def test_a_dict_merges_over_the_current_goal(self):
        merged = merge_goal(dict(GOAL), {"status": "paused"})
        assert merged == {**GOAL, "status": "paused"}

    def test_a_full_goal_replaces_every_field(self):
        other = {**GOAL, "objective": "other", "rubric_id": "goal:r2", "note": "n"}
        assert merge_goal({**GOAL, "evidence": "old"}, other) == other

    def test_a_partial_write_into_nothing_is_dropped(self):
        """After a clear, a late tool write must not create half a goal."""
        assert merge_goal(None, {"status": "blocked", "note": "x"}) is None
        assert merge_goal({}, {"evidence": "x"}) is None

    def test_a_goal_can_be_written_into_nothing(self):
        assert merge_goal(None, dict(GOAL)) == GOAL

    @pytest.mark.parametrize(
        "writes",
        [
            [{"status": "blocked", "note": "stuck"}, {"evidence": "done"}],
            [{"evidence": "done"}, {"status": "blocked", "note": "stuck"}],
        ],
    )
    def test_a_blocked_goal_holds_no_evidence_whatever_the_order(self, writes):
        goal = dict(GOAL)
        for write in writes:
            goal = merge_goal(goal, write)
        assert goal["status"] == "blocked"
        assert goal["note"] == "stuck"
        assert goal["evidence"] is None

    def test_the_current_value_is_not_mutated(self):
        current = dict(GOAL)
        merge_goal(current, {"status": "paused"})
        assert current == GOAL


class TestThreadGoal:
    def test_a_goal_whose_criteria_are_the_sticky_checklist_is_current(self):
        state = {
            "_goal": dict(GOAL),
            "_sticky_rubric": {"text": "- a", "id": "goal:r1"},
        }
        assert thread_goal(state) == GOAL

    @pytest.mark.parametrize(
        "sticky", [None, {"text": "- a", "id": "s1:3"}, {"text": "- a"}, "not a dict"]
    )
    def test_a_goal_without_its_checklist_is_no_goal(self, sticky):
        assert thread_goal({"_goal": dict(GOAL), "_sticky_rubric": sticky}) is None

    @pytest.mark.parametrize("goal", [None, {}, {"status": "active"}, "not a dict"])
    def test_no_goal(self, goal):
        state = {"_goal": goal, "_sticky_rubric": {"text": "- a", "id": "goal:r1"}}
        assert thread_goal(state) is None


class TestLimits:
    def test_an_objective_within_the_limit_passes(self):
        check_goal_text("x" * GOAL_OBJECTIVE_LIMIT)

    def test_an_empty_objective_is_refused(self):
        with pytest.raises(GoalTextError, match="objective is empty"):
            check_goal_text("  \n ")

    def test_an_objective_over_the_limit_names_the_excess(self):
        with pytest.raises(GoalTextError) as raised:
            check_goal_text("x" * (GOAL_OBJECTIVE_LIMIT + 5))
        assert str(raised.value) == (
            "The goal objective is 8,005 characters; maximum is 8,000. "
            "Remove at least 5 characters."
        )

    def test_empty_criteria_are_refused(self):
        with pytest.raises(GoalTextError, match="criteria are empty"):
            check_goal_text("obj", " ")

    def test_criteria_over_their_limit_are_refused(self):
        with pytest.raises(GoalTextError, match="acceptance criteria are 12,001"):
            check_goal_text("o", "x" * (GOAL_CRITERIA_LIMIT + 1))

    def test_objective_and_criteria_share_a_combined_limit(self):
        objective = "o" * 6_000
        check_goal_text(objective, "c" * (GOAL_TOTAL_LIMIT - 6_000))
        with pytest.raises(
            GoalTextError, match="objective and criteria together are 12,001"
        ):
            check_goal_text(objective, "c" * (GOAL_TOTAL_LIMIT - 6_000 + 1))

    def test_a_note_is_required_and_bounded(self):
        check_goal_note("x" * GOAL_NOTE_LIMIT)
        with pytest.raises(GoalTextError, match="note is empty"):
            check_goal_note(" ")
        with pytest.raises(GoalTextError, match="note is 4,001"):
            check_goal_note("x" * (GOAL_NOTE_LIMIT + 1))


class TestNewGoalRequest:
    def test_a_create_request_carries_the_objective_and_an_id(self):
        request = new_goal_request("create", objective="ship it")
        assert request["kind"] == "create"
        assert request["objective"] == "ship it"
        assert isinstance(request["id"], str)
        assert request["id"]
        assert "target" not in request

    def test_requests_get_distinct_ids(self):
        first = new_goal_request("create", objective="a")
        second = new_goal_request("create", objective="a")
        assert first["id"] != second["id"]

    def test_requests_about_an_existing_goal_name_it(self):
        request = new_goal_request("amend", feedback="shorter", target="goal:r1")
        assert request["feedback"] == "shorter"
        assert request["target"] == "goal:r1"

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ValueError, match="unknown goal request"):
            new_goal_request("finish")
