"""Tests for SkillsReloadMiddleware."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from EvoScientist.middleware.skills_reload import SkillsReloadMiddleware

_SEEN: list[list[BaseMessage]] = []
_FAIL = {"on": False}


class _RecordingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        _SEEN.append(list(messages))
        if _FAIL["on"]:
            raise RuntimeError("model failure")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _write_skill(root: Path, name: str, description: str) -> None:
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n"
        f"# {name}\n\nDo the {name} thing.\n",
        encoding="utf-8",
    )


def _agent(root: Path, middleware: list):
    return create_deep_agent(
        model=_RecordingModel(responses=[AIMessage(content="ok")]),
        backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
        skills=["/skills/"],
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )


def _system_text(messages: list[BaseMessage]) -> str:
    return next(m for m in messages if isinstance(m, SystemMessage)).text


def _config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def test_before_agent_clears_skills_metadata():
    mw = SkillsReloadMiddleware()
    assert mw.before_agent({"messages": []}, None) == {"skills_metadata": None}


async def test_abefore_agent_clears_skills_metadata():
    mw = SkillsReloadMiddleware()
    assert await mw.abefore_agent({"messages": []}, None) == {"skills_metadata": None}


def test_skill_added_mid_thread_is_listed_next_turn(tmp_path):
    _SEEN.clear()
    _write_skill(tmp_path, "alpha", "Alpha skill.")
    agent = _agent(tmp_path, [SkillsReloadMiddleware()])
    cfg = _config()

    agent.invoke({"messages": [("user", "hi")]}, config=cfg)
    assert "Beta skill." not in _system_text(_SEEN[-1])

    _write_skill(tmp_path, "beta", "Beta skill.")
    agent.invoke({"messages": [("user", "again")]}, config=cfg)
    assert "Beta skill." in _system_text(_SEEN[-1])


def test_without_reload_the_thread_keeps_its_first_skill_list(tmp_path):
    """Guards the premise: deepagents alone caches the list per thread."""
    _SEEN.clear()
    _write_skill(tmp_path, "alpha", "Alpha skill.")
    agent = _agent(tmp_path, [])
    cfg = _config()

    agent.invoke({"messages": [("user", "hi")]}, config=cfg)
    _write_skill(tmp_path, "beta", "Beta skill.")
    agent.invoke({"messages": [("user", "again")]}, config=cfg)
    assert "Beta skill." not in _system_text(_SEEN[-1])


def test_unchanged_library_keeps_the_system_prompt_identical(tmp_path):
    """A reload that finds the same skills must not break the prompt cache."""
    _SEEN.clear()
    _write_skill(tmp_path, "alpha", "Alpha skill.")
    _write_skill(tmp_path, "beta", "Beta skill.")
    agent = _agent(tmp_path, [SkillsReloadMiddleware()])
    cfg = _config()

    agent.invoke({"messages": [("user", "hi")]}, config=cfg)
    agent.invoke({"messages": [("user", "again")]}, config=cfg)
    assert "Alpha skill." in _system_text(_SEEN[0])
    assert _system_text(_SEEN[0]) == _system_text(_SEEN[1])


def _mock_config():
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_mode = False
    cfg.auto_approve = False
    cfg.model_fallbacks = None
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    cfg.code_interpreter_timeout = 60
    cfg.code_interpreter_max_result_chars = 6000
    return cfg


def _default_middleware(workspace, *, for_async_subagent: bool):
    from EvoScientist.EvoScientist import _get_default_middleware

    with (
        patch(
            "EvoScientist.middleware.create_tool_selector_middleware",
            return_value=[MagicMock()],
        ),
        patch(
            "EvoScientist.EvoScientist._ensure_chat_model",
            return_value=MagicMock(profile={"max_input_tokens": 200_000}),
        ),
        patch("EvoScientist.EvoScientist._ensure_config", return_value=_mock_config()),
    ):
        return _get_default_middleware(
            workspace=workspace, for_async_subagent=for_async_subagent
        )


def test_main_agent_stack_reloads_skills(workspace):
    stack = _default_middleware(workspace, for_async_subagent=False)
    assert sum(isinstance(m, SkillsReloadMiddleware) for m in stack) == 1


def test_async_subagent_stack_does_not_reload_skills(workspace):
    stack = _default_middleware(workspace, for_async_subagent=True)
    assert not any(isinstance(m, SkillsReloadMiddleware) for m in stack)


async def _failed_turn(agent, cfg):
    await agent.ainvoke({"messages": [("user", "hi")]}, config=cfg)
    _FAIL["on"] = True
    try:
        with pytest.raises(RuntimeError):
            await agent.ainvoke({"messages": [("user", "boom")]}, config=cfg)
    finally:
        _FAIL["on"] = False


async def test_skill_added_after_a_failed_turn_is_listed_next_turn(tmp_path):
    _SEEN.clear()
    _write_skill(tmp_path, "alpha", "Alpha skill.")
    agent = _agent(tmp_path, [SkillsReloadMiddleware()])
    cfg = _config()
    await _failed_turn(agent, cfg)

    _write_skill(tmp_path, "beta", "Beta skill.")
    await agent.ainvoke({"messages": [("user", "again")]}, config=cfg)
    assert "Beta skill." in _system_text(_SEEN[-1])


async def test_skill_removed_after_a_failed_turn_is_gone_next_turn(tmp_path):
    _SEEN.clear()
    _write_skill(tmp_path, "alpha", "Alpha skill.")
    agent = _agent(tmp_path, [SkillsReloadMiddleware()])
    cfg = _config()
    await _failed_turn(agent, cfg)

    shutil.rmtree(tmp_path / "skills" / "alpha")
    await agent.ainvoke({"messages": [("user", "again")]}, config=cfg)
    assert "Alpha skill." not in _system_text(_SEEN[-1])


async def test_skill_added_mid_thread_is_listed_next_turn_async(tmp_path):
    _SEEN.clear()
    _write_skill(tmp_path, "alpha", "Alpha skill.")
    agent = _agent(tmp_path, [SkillsReloadMiddleware()])
    cfg = _config()
    await agent.ainvoke({"messages": [("user", "hi")]}, config=cfg)
    _write_skill(tmp_path, "beta", "Beta skill.")
    await agent.ainvoke({"messages": [("user", "again")]}, config=cfg)
    assert "Beta skill." in _system_text(_SEEN[-1])
