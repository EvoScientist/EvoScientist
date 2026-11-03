"""Goal requests on the real main graph: the hooks around the goal middleware."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from EvoScientist.config import EvoScientistConfig
from EvoScientist.EvoScientist import create_cli_agent
from EvoScientist.middleware.goal import new_goal_request
from EvoScientist.paths import Workspace
from tests.fakes.goal import (
    ACCEPT,
    CANCEL,
    ESCAPED,
    OBJECTIVE,
    AllRoles,
    cli_goal_agent,
    history,
    message_input,
    request_input,
    resume_input,
    run,
)

pytestmark = pytest.mark.usefixtures("no_mcp")


def _create(**kwargs):
    return new_goal_request("create", objective=OBJECTIVE, **kwargs)


class TestFirstRunOfAThread:
    """Nothing has initialised the other middleware when the goal hook ends the run."""

    @pytest.mark.parametrize("answer", [CANCEL, ESCAPED])
    async def test_a_cancelled_review_ends_cleanly(self, tmp_path, answer):
        agent, model = cli_goal_agent(tmp_path)
        parked = await run(agent, request_input(_create()))
        assert parked.question is not None
        result = await run(agent, resume_input(answer))
        assert result.changes == ["cancelled"]
        assert result.next == ()
        assert history(result) == []
        assert model.calls == {"main": 0, "draft": 1, "grade": 0}

    async def test_a_failed_first_draft_ends_cleanly(self, tmp_path):
        agent, model = cli_goal_agent(tmp_path, drafts=[RuntimeError("503 upstream")])
        result = await run(agent, request_input(_create()))
        assert result.changes == ["drafting", "failed"]
        assert result.next == ()
        assert model.calls["main"] == 0

    async def test_an_ignored_request_ends_cleanly(self, tmp_path):
        agent, model = cli_goal_agent(tmp_path)
        request = new_goal_request("pause", target="goal:none")
        result = await run(agent, request_input(request))
        assert result.changes == ["ignored"]
        assert result.next == ()
        assert model.calls == {"main": 0, "draft": 0, "grade": 0}

    async def test_the_thread_works_afterwards(self, tmp_path):
        agent, _ = cli_goal_agent(tmp_path)
        await run(agent, request_input(_create()))
        await run(agent, resume_input(CANCEL))
        result = await run(agent, message_input("hi"))
        assert history(result) == [("human", "hi"), ("ai", "working on it")]


class TestAcceptedGoal:
    async def test_the_same_run_does_the_work_and_is_graded(self, tmp_path):
        agent, model = cli_goal_agent(tmp_path)
        await run(agent, request_input(_create()))
        result = await run(agent, resume_input(ACCEPT))
        assert history(result) == [("human", OBJECTIVE), ("ai", "working on it")]
        assert result.changes[0] == "accepted"
        assert result.verdicts == ["satisfied"]
        assert model.calls == {"main": 1, "draft": 1, "grade": 1}
        assert result.next == ()

    async def test_the_agent_is_offered_the_goal_tool_only_with_a_goal(self, tmp_path):
        """On the real main graph, with every other middleware in the list."""
        agent, model = cli_goal_agent(tmp_path)
        await run(agent, message_input("hi"))
        assert model.tools_seen  # the other tools were offered
        assert not any("update_goal" in names for names in model.tools_seen)

        await run(agent, request_input(_create()))
        before = len(model.tools_seen)
        await run(agent, resume_input(ACCEPT))
        assert any("update_goal" in names for names in model.tools_seen[before:])


class TestMemoryWorker:
    """With memory workers on, a run that ends in the goal hook launches none."""

    async def _counting_agent(self, tmp_path, monkeypatch):
        import EvoScientist.middleware.memory_lifecycle as lifecycle

        launched: list = []

        async def _alaunch(context, **_kwargs):
            launched.append(context)
            return None

        monkeypatch.setattr(lifecycle, "alaunch_memory_worker", _alaunch)
        model = AllRoles()
        agent = create_cli_agent(
            workspace=Workspace(tmp_path),
            config=EvoScientistConfig(enable_async_subagents=False),
            chat_model=model,
            checkpointer=InMemorySaver(),
        )
        return agent, launched

    async def test_a_cancelled_request_after_a_turn(self, tmp_path, monkeypatch):
        agent, launched = await self._counting_agent(tmp_path, monkeypatch)
        await run(agent, message_input("hi"))
        assert len(launched) == 1
        await run(agent, request_input(_create()))
        await run(agent, resume_input(CANCEL))
        assert len(launched) == 1

    async def test_a_status_change_after_a_turn(self, tmp_path, monkeypatch):
        agent, launched = await self._counting_agent(tmp_path, monkeypatch)
        await run(agent, request_input(_create()))
        accepted = await run(agent, resume_input(ACCEPT))
        assert len(launched) == 1
        target = accepted.values["_goal"]["rubric_id"]
        await run(agent, request_input(new_goal_request("clear", target=target)))
        assert len(launched) == 1
        await run(agent, message_input("thanks"))
        assert len(launched) == 2


async def test_an_agent_built_for_auto_mode_adopts_the_draft(tmp_path):
    """The TUI passes commands no config, so the graph reads its own."""
    model = AllRoles()
    agent = create_cli_agent(
        workspace=Workspace(tmp_path),
        config=EvoScientistConfig(
            enable_async_subagents=False, memory_workers_enabled=False, auto_mode=True
        ),
        chat_model=model,
        checkpointer=InMemorySaver(),
    )
    result = await run(agent, request_input(_create()))
    assert result.question is None
    assert result.changes[:2] == ["drafting", "accepted"]
    assert result.values["_goal"]["status"] == "complete"
