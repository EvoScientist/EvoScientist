"""Goal requests on the real main graph: the hooks around the goal middleware."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from EvoScientist.config import EvoScientistConfig
from EvoScientist.EvoScientist import create_cli_agent
from EvoScientist.middleware.goal import new_goal_request
from EvoScientist.paths import Workspace
from tests.goal_fakes import (
    ACCEPT,
    CANCEL,
    ESCAPED,
    history,
    message_input,
    request_input,
    resume_input,
    run,
)
from tests.test_rubric_middleware import PASS

pytestmark = pytest.mark.usefixtures("no_mcp")

OBJECTIVE = "ship the report"


class _AllRoles(BaseChatModel):
    """Main model, drafter and grader in one, told apart by the system prompt."""

    profile: dict = Field(default_factory=lambda: {"max_input_tokens": 200_000})
    drafts: list = Field(default_factory=list)
    calls: dict = Field(default_factory=lambda: {"main": 0, "draft": 0, "grade": 0})
    tools_seen: list = Field(default_factory=list)
    keep_system: bool = False
    main_systems: list = Field(default_factory=list)
    # ``threading.Event``s: the role named waits for its event before replying.
    holds: dict = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "scripted-all-roles"

    def _wait(self, role: str) -> None:
        hold = self.holds.get(role)
        if hold is not None:
            assert hold.wait(timeout=30), f"the {role} reply was never released"

    def bind_tools(self, tools, **kwargs):
        self.tools_seen.append(
            [getattr(t, "name", None) or t.get("name") for t in tools]
        )
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = (
            str(messages[0].content)
            if messages and messages[0].type == "system"
            else ""
        )
        if system.startswith("You draft minimal acceptance criteria"):
            self._wait("draft")
            self.calls["draft"] += 1
            reply: Any = self.drafts.pop(0) if self.drafts else "- the report exists"
            if isinstance(reply, Exception):
                raise reply
            message = AIMessage(reply)
        elif system.startswith("You are a grader"):
            self.calls["grade"] += 1
            call = {
                "name": "GraderResponse",
                "args": PASS,
                "id": "g1",
                "type": "tool_call",
            }
            message = AIMessage("", tool_calls=[call])
        else:
            self._wait("main")
            self.calls["main"] += 1
            if self.keep_system:
                self.main_systems.append(system)
            message = AIMessage("working on it")
        return ChatResult(generations=[ChatGeneration(message=message)])


def _agent(tmp_path, **model_kwargs):
    model = _AllRoles(**model_kwargs)
    cfg = EvoScientistConfig(enable_async_subagents=False, memory_workers_enabled=False)
    agent = create_cli_agent(
        workspace=Workspace(tmp_path),
        config=cfg,
        chat_model=model,
        checkpointer=InMemorySaver(),
    )
    return agent, model


def _create(**kwargs):
    return new_goal_request("create", objective=OBJECTIVE, **kwargs)


class TestFirstRunOfAThread:
    """Nothing has initialised the other middleware when the goal hook ends the run."""

    @pytest.mark.parametrize("answer", [CANCEL, ESCAPED])
    async def test_a_cancelled_review_ends_cleanly(self, tmp_path, answer):
        agent, model = _agent(tmp_path)
        parked = await run(agent, request_input(_create()))
        assert parked.question is not None
        result = await run(agent, resume_input(answer))
        assert result.changes == ["cancelled"]
        assert result.next == ()
        assert history(result) == []
        assert model.calls == {"main": 0, "draft": 1, "grade": 0}

    async def test_a_failed_first_draft_ends_cleanly(self, tmp_path):
        agent, model = _agent(tmp_path, drafts=[RuntimeError("503 upstream")])
        result = await run(agent, request_input(_create()))
        assert result.changes == ["drafting", "failed"]
        assert result.next == ()
        assert model.calls["main"] == 0

    async def test_an_ignored_request_ends_cleanly(self, tmp_path):
        agent, model = _agent(tmp_path)
        request = new_goal_request("pause", target="goal:none")
        result = await run(agent, request_input(request))
        assert result.changes == ["ignored"]
        assert result.next == ()
        assert model.calls == {"main": 0, "draft": 0, "grade": 0}

    async def test_the_thread_works_afterwards(self, tmp_path):
        agent, _ = _agent(tmp_path)
        await run(agent, request_input(_create()))
        await run(agent, resume_input(CANCEL))
        result = await run(agent, message_input("hi"))
        assert history(result) == [("human", "hi"), ("ai", "working on it")]


class TestAcceptedGoal:
    async def test_the_same_run_does_the_work_and_is_graded(self, tmp_path):
        agent, model = _agent(tmp_path)
        await run(agent, request_input(_create()))
        result = await run(agent, resume_input(ACCEPT))
        assert history(result) == [("human", OBJECTIVE), ("ai", "working on it")]
        assert result.changes[0] == "accepted"
        assert result.verdicts == ["satisfied"]
        assert model.calls == {"main": 1, "draft": 1, "grade": 1}
        assert result.next == ()

    async def test_the_agent_is_offered_the_goal_tool_only_with_a_goal(self, tmp_path):
        """On the real main graph, with every other middleware in the list."""
        agent, model = _agent(tmp_path)
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
        model = _AllRoles()
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
    model = _AllRoles()
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
