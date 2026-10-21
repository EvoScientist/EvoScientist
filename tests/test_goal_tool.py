"""``update_goal``: what the agent may report about the goal, and when it is refused."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from tests.goal_fakes import goal_agent, message_input, run, seed_goal, tool_call


def _calls(*calls) -> AIMessage:
    return AIMessage("", tool_calls=list(calls))


def _tool_replies(result) -> list[str]:
    return [str(m.content) for m in result.values["messages"] if m.type == "tool"]


BLOCKED = tool_call("update_goal", "b1", status="blocked", note="need an API key")
COMPLETE = tool_call("update_goal", "c1", status="complete", note="all tests pass")


async def test_blocked_records_the_reason_and_tells_the_surface(tmp_path):
    agent, _ = goal_agent(tmp_path, replies=[_calls(BLOCKED), "waiting for you"])
    await seed_goal(agent, evidence="staged earlier")
    result = await run(agent, message_input("go on"))
    goal = result.values["_goal"]
    assert goal["status"] == "blocked"
    assert goal["note"] == "need an API key"
    assert goal["evidence"] is None
    blocked = next(e for e in result.events if e.get("change") == "blocked")
    assert blocked["note"] == "need an API key"
    assert "marked blocked" in _tool_replies(result)[0]


async def test_complete_only_stages_the_evidence(tmp_path):
    agent, _ = goal_agent(tmp_path, replies=[_calls(COMPLETE), "done"])
    await seed_goal(agent)
    result = await run(agent, message_input("go on"))
    assert "graded as satisfied" in _tool_replies(result)[0]
    # The grader (scripted to fail) decides; the report alone completes nothing.
    assert result.values["_goal"]["status"] == "active"


@pytest.mark.parametrize("calls", [(BLOCKED, COMPLETE), (COMPLETE, BLOCKED)])
async def test_two_reports_in_one_step_do_not_fail_the_run(tmp_path, calls):
    agent, _ = goal_agent(tmp_path, replies=[_calls(*calls), "waiting for you"])
    await seed_goal(agent)
    result = await run(agent, message_input("go on"))
    goal = result.values["_goal"]
    assert goal["status"] == "blocked"
    assert goal["note"] == "need an API key"
    assert goal["evidence"] is None
    assert len(_tool_replies(result)) == 2


class TestRefusals:
    async def _report(self, tmp_path, *, seed=None, mode="thread", call=BLOCKED):
        agent, _ = goal_agent(tmp_path, replies=[_calls(call), "ok"])
        if seed is not None:
            await seed_goal(agent, **seed)
        result = await run(agent, message_input("go on"), mode=mode)
        return result, _tool_replies(result)[0]

    async def test_without_a_goal(self, tmp_path):
        result, reply = await self._report(tmp_path)
        assert "no goal" in reply
        assert result.values.get("_goal") is None

    @pytest.mark.parametrize("status", ["paused", "complete"])
    async def test_a_goal_that_is_not_active(self, tmp_path, status):
        result, reply = await self._report(tmp_path, seed={"status": status})
        assert f"The goal is {status}" in reply
        assert result.values["_goal"]["status"] == status
        assert result.values["_goal"]["note"] is None

    async def test_a_goal_that_is_already_blocked(self, tmp_path):
        """A notifier turn does not unblock; a second report must not replace the reason."""
        result, reply = await self._report(
            tmp_path, seed={"status": "blocked", "note": "need a key"}, mode="followup"
        )
        assert "The goal is blocked" in reply
        assert result.values["_goal"]["note"] == "need a key"

    @pytest.mark.parametrize("mode", [None, "input"])
    async def test_a_run_in_which_goals_do_not_apply(self, tmp_path, mode):
        result, reply = await self._report(tmp_path, seed={}, mode=mode)
        assert "not available in this run" in reply
        assert result.values["_goal"]["status"] == "active"

    async def test_an_empty_note(self, tmp_path):
        call = tool_call("update_goal", "b1", status="blocked", note="  ")
        result, reply = await self._report(tmp_path, seed={}, call=call)
        assert "note is empty" in reply
        assert result.values["_goal"]["status"] == "active"

    async def test_a_note_over_the_limit(self, tmp_path):
        call = tool_call("update_goal", "b1", status="blocked", note="x" * 4_001)
        result, reply = await self._report(tmp_path, seed={}, call=call)
        assert "maximum is 4,000" in reply
        assert result.values["_goal"]["status"] == "active"

    async def test_a_goal_whose_checklist_was_replaced(self, tmp_path):
        agent, _ = goal_agent(tmp_path, replies=[_calls(BLOCKED), "ok"])
        await seed_goal(agent)
        cfg = {"configurable": {"thread_id": "t"}}
        await agent.aupdate_state(
            cfg,
            {"_sticky_rubric": {"text": "- other", "id": "s1:9"}},
            as_node="GoalMiddleware.before_agent",
        )
        await agent.aupdate_state(cfg, None, as_node="__end__")
        result = await run(agent, message_input("go on"))
        assert "no goal" in _tool_replies(result)[0]


class TestWhenTheModelIsOfferedTheTool:
    """The tool is on the list only while there is a goal to report on."""

    @staticmethod
    async def _offered(tmp_path, *, mode="thread", **goal) -> list[str]:
        agent, models = goal_agent(tmp_path, replies=["ok"])
        if goal:
            await seed_goal(agent, **goal)
        await run(agent, message_input("hi"), mode=mode)
        return models.main.offered[-1]

    async def test_with_an_active_goal(self, tmp_path):
        assert await self._offered(tmp_path, status="active") == ["update_goal"]

    async def test_with_a_blocked_goal_the_run_does_not_unblock(self, tmp_path):
        """A call left in the conversation still has its tool beside it."""
        offered = await self._offered(tmp_path, mode="followup", status="blocked")
        assert offered == ["update_goal"]

    @pytest.mark.parametrize("status", ["paused", "complete"])
    async def test_not_with_a_goal_that_is_not_worked_on(self, tmp_path, status):
        assert await self._offered(tmp_path, status=status) == []

    async def test_not_without_a_goal(self, tmp_path):
        assert await self._offered(tmp_path) == []

    @pytest.mark.parametrize("mode", [None, "input"])
    async def test_not_on_a_run_where_goals_do_not_apply(self, tmp_path, mode):
        assert await self._offered(tmp_path, mode=mode, status="active") == []

    async def test_other_tools_stay(self, tmp_path):
        from langchain_core.tools import tool

        @tool
        def lookup(query: str) -> str:
            """Look something up."""
            return query

        agent, models = goal_agent(tmp_path, replies=["ok"], tools=[lookup])
        await run(agent, message_input("hi"))
        assert models.main.offered[-1] == ["lookup"]
        await seed_goal(agent)
        await run(agent, message_input("again"))
        assert sorted(models.main.offered[-1]) == ["lookup", "update_goal"]

    def test_a_provider_tool_given_as_a_dict_is_left_alone(self):
        from types import SimpleNamespace

        from EvoScientist.middleware.goal import _without_update_goal, update_goal

        server_tool = {"type": "web_search_20250305", "name": "web_search"}
        request = SimpleNamespace(tools=[server_tool, update_goal])
        request.override = lambda **changes: SimpleNamespace(**changes)
        assert _without_update_goal(request).tools == [server_tool]

    @pytest.mark.parametrize(
        "as_dict",
        [
            {"name": "update_goal", "description": "", "input_schema": {}},
            {"type": "function", "function": {"name": "update_goal"}},
        ],
        ids=["flat", "function schema"],
    )
    def test_the_tool_given_as_a_dict_is_taken_off_as_well(self, as_dict):
        from types import SimpleNamespace

        from EvoScientist.middleware.goal import _without_update_goal

        other = {"type": "function", "function": {"name": "lookup"}}
        request = SimpleNamespace(tools=[other, as_dict])
        request.override = lambda **changes: SimpleNamespace(**changes)
        assert _without_update_goal(request).tools == [other]

    def test_a_request_without_the_tool_is_passed_on_as_it_is(self):
        from types import SimpleNamespace

        from EvoScientist.middleware.goal import _without_update_goal

        request = SimpleNamespace(tools=[{"name": "web_search"}])
        assert _without_update_goal(request) is request

    async def test_a_call_made_without_the_tool_on_offer_is_still_answered(
        self, tmp_path
    ):
        agent, _ = goal_agent(tmp_path, replies=[_calls(BLOCKED), "ok"])
        result = await run(agent, message_input("hi"))
        assert "no goal" in _tool_replies(result)[0]


def test_the_tool_selector_never_drops_the_tool():
    """With many MCP tools the selector trims the list; the goal tool must stay."""
    from EvoScientist.middleware.tool_selector import DEFAULT_ALWAYS_INCLUDE_TOOLS

    assert "update_goal" in DEFAULT_ALWAYS_INCLUDE_TOOLS


def test_the_sync_end_hook_also_leaves_a_blocked_turn_ungraded(tmp_path):
    """Every surface runs the graph async; the sync hooks must not drift."""
    from tests.rubric_fakes import FAIL

    agent, models = goal_agent(
        tmp_path, replies=[_calls(BLOCKED), "I need you"], verdicts=[FAIL]
    )
    cfg = {
        "configurable": {
            "thread_id": "t",
            "rubric_run": {"mode": "thread", "changes": []},
        }
    }
    agent.update_state(
        {"configurable": {"thread_id": "t"}},
        {
            "_goal": {
                "objective": "ship the report",
                "status": "active",
                "note": None,
                "evidence": None,
                "rubric_id": "goal:seed",
                "start": None,
            },
            "_sticky_rubric": {"text": "- the report exists", "id": "goal:seed"},
        },
        as_node="GoalMiddleware.before_agent",
    )
    agent.update_state({"configurable": {"thread_id": "t"}}, None, as_node="__end__")
    values = agent.invoke(message_input("go on"), cfg)
    state = agent.get_state({"configurable": {"thread_id": "t"}}).values
    assert models.grader.calls == 0
    assert state["_goal"]["status"] == "blocked"
    assert state.get("_rubric_grading") is None
    assert len(values["messages"]) == 4
