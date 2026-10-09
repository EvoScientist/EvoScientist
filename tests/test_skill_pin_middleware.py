"""Tests for SkillPinMiddleware."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.skills import SkillsState
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Overwrite

from EvoScientist.message_meta import is_pinned_skill, pinned_skill_name
from EvoScientist.middleware.skill_pin import SkillPinMiddleware, skills_to_pin
from EvoScientist.middleware.skills_reload import SkillsReloadMiddleware
from tests.test_skills_reload_middleware import _default_middleware

_SEEN: list[list[BaseMessage]] = []


class _RecordingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        _SEEN.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _write_skill(root: Path, name: str) -> None:
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: The {name} skill.\n---\n\nBody of {name}.\n",
        encoding="utf-8",
    )


_CRASH = {"on": False}


class _Crash(AgentMiddleware):
    """Raise after SkillPin wrote its pin but before deepagents consumes it."""

    state_schema = SkillsState

    def before_agent(self, state, runtime):
        if _CRASH["on"]:
            raise RuntimeError("cancelled")

    async def abefore_agent(self, state, runtime):
        return self.before_agent(state, runtime)


def _agent(root: Path, middleware: list | None = None):
    return create_deep_agent(
        model=_RecordingModel(responses=[AIMessage(content="ok")]),
        backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
        skills=["/skills/"],
        middleware=middleware if middleware is not None else [SkillPinMiddleware()],
        checkpointer=InMemorySaver(),
    )


def _config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


class TestSkillsToPin:
    def test_newest_user_message(self):
        state = {"messages": [HumanMessage("/alpha /beta go")]}
        assert skills_to_pin(state) == ["alpha", "beta"]

    def test_plain_message(self):
        assert skills_to_pin({"messages": [HumanMessage("go")]}) == []

    def test_last_message_not_from_user(self):
        """A resumed (HITL) run ends on the model's tool call: never re-pin."""
        state = {
            "messages": [
                HumanMessage("/alpha go"),
                AIMessage("", tool_calls=[{"name": "execute", "args": {}, "id": "1"}]),
            ]
        }
        assert skills_to_pin(state) == []

    def test_tagged_user_message_ignored(self):
        msg = HumanMessage("/alpha", additional_kwargs={"lc_source": "summarization"})
        assert skills_to_pin({"messages": [msg]}) == []

    def test_block_content(self):
        msg = HumanMessage(content=[{"type": "text", "text": "/alpha go"}])
        assert skills_to_pin({"messages": [msg]}) == ["alpha"]

    def test_empty_state(self):
        assert skills_to_pin({}) == []
        assert skills_to_pin({"messages": []}) == []


def test_before_agent_return_values():
    mw = SkillPinMiddleware()
    named = mw.before_agent({"messages": [HumanMessage("/alpha go")]}, None)
    assert isinstance(named["pinned_skills"], Overwrite)
    assert named["pinned_skills"].value == ["alpha"]
    plain = mw.before_agent({"messages": [HumanMessage("go")]}, None)
    assert isinstance(plain["pinned_skills"], Overwrite)
    assert plain["pinned_skills"].value == []


async def test_abefore_agent_return_value():
    mw = SkillPinMiddleware()
    result = await mw.abefore_agent({"messages": [HumanMessage("/alpha go")]}, None)
    assert isinstance(result["pinned_skills"], Overwrite)
    assert result["pinned_skills"].value == ["alpha"]


def test_named_skill_reaches_the_first_model_call(tmp_path):
    _SEEN.clear()
    _write_skill(tmp_path, "alpha")
    agent = _agent(tmp_path)

    agent.invoke({"messages": [("user", "/alpha write it")]}, config=_config())

    humans = [m for m in _SEEN[0] if isinstance(m, HumanMessage)]
    assert humans[0].content == "/alpha write it"
    assert is_pinned_skill(humans[1])
    assert pinned_skill_name(humans[1]) == "alpha"
    assert "Body of alpha." in humans[1].content


def test_unknown_name_inserts_nothing(tmp_path):
    _SEEN.clear()
    _write_skill(tmp_path, "alpha")
    agent = _agent(tmp_path)

    agent.invoke({"messages": [("user", "/nope hi")]}, config=_config())

    assert not any(is_pinned_skill(m) for m in _SEEN[0])


def test_later_plain_turn_does_not_pin_again(tmp_path):
    _SEEN.clear()
    _write_skill(tmp_path, "alpha")
    agent = _agent(tmp_path)
    cfg = _config()

    agent.invoke({"messages": [("user", "/alpha write it")]}, config=cfg)
    agent.invoke({"messages": [("user", "thanks")]}, config=cfg)

    assert sum(is_pinned_skill(m) for m in _SEEN[-1]) == 1


def test_pin_from_a_failed_turn_does_not_leak(tmp_path):
    _SEEN.clear()
    _write_skill(tmp_path, "alpha")
    _write_skill(tmp_path, "beta")
    agent = _agent(tmp_path, [SkillPinMiddleware(), _Crash()])
    cfg = _config()

    _CRASH["on"] = True
    try:
        with pytest.raises(RuntimeError):
            agent.invoke({"messages": [("user", "/alpha do x")]}, config=cfg)
    finally:
        _CRASH["on"] = False
    assert agent.get_state(cfg).values.get("pinned_skills") == ["alpha"]

    turn_two_start = len(_SEEN)
    agent.invoke({"messages": [("user", "plain question")]}, config=cfg)
    assert not any(is_pinned_skill(m) for m in _SEEN[turn_two_start])

    turn_three_start = len(_SEEN)
    agent.invoke({"messages": [("user", "/beta go")]}, config=cfg)
    pinned = [m for m in _SEEN[turn_three_start] if is_pinned_skill(m)]
    assert [pinned_skill_name(m) for m in pinned] == ["beta"]


def test_skill_added_between_turns_can_be_pinned_with_reload(tmp_path):
    _SEEN.clear()
    _write_skill(tmp_path, "alpha")
    agent = _agent(tmp_path, [SkillPinMiddleware(), SkillsReloadMiddleware()])
    cfg = _config()

    agent.invoke({"messages": [("user", "hello")]}, config=cfg)
    _write_skill(tmp_path, "beta")
    turn_two_start = len(_SEEN)
    agent.invoke({"messages": [("user", "/beta go")]}, config=cfg)

    pinned = [m for m in _SEEN[turn_two_start] if is_pinned_skill(m)]
    assert [pinned_skill_name(m) for m in pinned] == ["beta"]


def test_main_agent_stack_pins_skills(workspace):
    stack = _default_middleware(workspace, for_async_subagent=False)
    assert sum(isinstance(m, SkillPinMiddleware) for m in stack) == 1


def test_async_subagent_stack_does_not_pin_skills(workspace):
    stack = _default_middleware(workspace, for_async_subagent=True)
    assert not any(isinstance(m, SkillPinMiddleware) for m in stack)
