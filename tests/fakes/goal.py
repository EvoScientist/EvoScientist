"""Scripted models and a small graph for the goal tests."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

from deepagents.backends import FilesystemBackend
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import Field

from EvoScientist.config import EvoScientistConfig
from EvoScientist.EvoScientist import create_cli_agent
from EvoScientist.message_meta import this_run_messages
from EvoScientist.middleware.goal import GoalContextMiddleware, GoalMiddleware
from EvoScientist.middleware.rubric import create_rubric_middleware
from EvoScientist.paths import Workspace
from tests.fakes.rubric import FAIL, PASS, ScriptedGrader

ACCEPT = {"answers": ["Accept"], "status": "answered"}
CANCEL = {"answers": ["Cancel"], "status": "answered"}
ESCAPED = {"status": "cancelled"}


def feedback(text: str) -> dict:
    return {"answers": [text], "status": "answered"}


class Scripted(BaseChatModel):
    """Gives the scripted replies in order.

    A reply may be an exception to raise, or a ``threading.Event`` the call
    waits on (work that is still running). Once the script is used up it
    answers ``done``. ``seen`` keeps the messages of every call.
    """

    replies: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)
    # Per call: the names of the tools the model was offered.
    offered: list = Field(default_factory=list)
    bound: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound = [
            tool["name"] if isinstance(tool, dict) else tool.name for tool in tools
        ]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        self.offered.append(self.bound)
        self.bound = []
        reply = self.replies.pop(0) if self.replies else "done"
        if isinstance(reply, threading.Event):
            reply.wait(timeout=10)
            reply = "- late"
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, str):
            reply = AIMessage(reply)
        return ChatResult(generations=[ChatGeneration(message=reply)])


def tool_call(name: str, call_id: str = "c1", **args: Any) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def goal_agent(
    tmp_path,
    *,
    drafts=(),
    replies=(),
    verdicts=(),
    saver=None,
    models=None,
    system_prompt=None,
    unattended=False,
    tools=(),
):
    """A graph with the goal and rubric middleware and one scripted model each.

    Pass ``models`` (from an earlier call) to rebuild the graph around the
    same scripts, as a restarted process would.
    """
    models = models or SimpleNamespace(
        main=Scripted(replies=list(replies)),
        drafter=Scripted(replies=list(drafts)),
        grader=ScriptedGrader(verdicts=list(verdicts) or [FAIL]),
    )
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    agent = create_agent(
        model=models.main,
        tools=list(tools),
        system_prompt=system_prompt,
        middleware=[
            GoalMiddleware(
                model=models.drafter, backend=backend, unattended=unattended
            ),
            create_rubric_middleware(
                model=models.grader,
                backend=backend,
                max_iterations=3,
                prepare_messages=this_run_messages,
                require_run_marker=True,
            ),
            GoalContextMiddleware(),
        ],
        checkpointer=saver if saver is not None else InMemorySaver(),
    )
    return agent, models


def request_input(request: dict) -> dict:
    return {"messages": [], "goal_request": dict(request)}


def message_input(text: str) -> dict:
    return {"messages": [HumanMessage(text)], "goal_request": None}


def resume_input(answer: dict) -> Command:
    return Command(resume=answer)


async def run(agent, payload, *, mode="thread", changes=(), thread="t"):
    """One run; returns the goal events, the question shown and the state."""
    cfg: dict = {"configurable": {"thread_id": thread}}
    if mode:
        cfg["configurable"]["rubric_run"] = {"mode": mode, "changes": list(changes)}
    events: list[dict] = []
    question = None
    async for kind, data in agent.astream(
        payload, cfg, stream_mode=["custom", "updates"]
    ):
        if kind == "custom" and isinstance(data, dict):
            events.append(data)
        elif kind == "updates" and "__interrupt__" in data:
            question = data["__interrupt__"][0].value
    state = await agent.aget_state(cfg)
    return SimpleNamespace(
        events=events,
        changes=[e["change"] for e in events if e.get("type") == "goal_state"],
        verdicts=[
            e["result"] for e in events if e.get("type") == "rubric_evaluation_end"
        ],
        question=question,
        values=state.values,
        next=state.next,
    )


def history(result) -> list[tuple[str, str]]:
    return [(m.type, str(m.content)) for m in result.values.get("messages", [])]


GOAL_NODE = "GoalMiddleware.before_agent"


async def seed_goal(agent, *, status="active", thread="t", **fields):
    """Put a goal on the thread without drafting one; returns its ``rubric_id``."""
    cfg = {"configurable": {"thread_id": thread}}
    goal = {
        "objective": "ship the report",
        "status": status,
        "note": None,
        "evidence": None,
        "rubric_id": "goal:seed",
        "start": None,
        **fields,
    }
    values = {
        "_goal": goal,
        "_sticky_rubric": {"text": "- the report exists", "id": goal["rubric_id"]},
    }
    await agent.aupdate_state(cfg, values, as_node=GOAL_NODE)
    await agent.aupdate_state(cfg, None, as_node="__end__")
    return goal["rubric_id"]


OBJECTIVE = "ship the report"


class AllRoles(BaseChatModel):
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


def cli_goal_agent(tmp_path, **model_kwargs):
    model = AllRoles(**model_kwargs)
    cfg = EvoScientistConfig(enable_async_subagents=False, memory_workers_enabled=False)
    agent = create_cli_agent(
        workspace=Workspace(tmp_path),
        config=cfg,
        chat_model=model,
        checkpointer=InMemorySaver(),
    )
    return agent, model
