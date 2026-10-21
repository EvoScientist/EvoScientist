"""A ``/goal`` request through the real Rich display loop on the real main graph."""

from __future__ import annotations

import asyncio
from io import StringIO

import pytest
from rich.console import Console

from EvoScientist.gateway.local import LocalGraphGateway
from EvoScientist.middleware.goal import new_goal_request
from EvoScientist.stream import display as display_mod
from tests.goal_fakes import ACCEPT, CANCEL, OBJECTIVE, cli_goal_agent, feedback

pytestmark = pytest.mark.usefixtures("no_mcp")

MARKED = {"rubric_run": {"mode": "thread", "changes": []}}


@pytest.fixture
def screen(monkeypatch):
    out = StringIO()
    monkeypatch.setattr(
        display_mod, "console", Console(file=out, width=100, color_system=None)
    )
    monkeypatch.setattr(display_mod, "_session_auto_approve", False)
    return out


async def _turn(agent, *, answers=(), message="", goal_request=None, extra=MARKED):
    """Run one Rich turn off the loop, answering review questions in order."""
    asked: list[dict] = []
    replies = list(answers)

    def _answer(question: dict) -> dict:
        asked.append(question)
        return replies.pop(0)

    text = await asyncio.to_thread(
        display_mod._run_streaming,
        agent=agent,
        message=message,
        thread_id="t1",
        show_thinking=False,
        interactive=True,
        configurable_extra=extra,
        ask_user_prompt_fn=_answer,
        gateway=LocalGraphGateway(),
        goal_request=goal_request,
    )
    return text, asked


async def _values(agent) -> dict:
    state = await agent.aget_state({"configurable": {"thread_id": "t1"}})
    return state.values


def _create():
    return new_goal_request("create", objective=OBJECTIVE)


async def test_accepting_runs_the_goal_to_completion_in_one_turn(tmp_path, screen):
    agent, model = cli_goal_agent(tmp_path)
    text, asked = await _turn(agent, answers=[ACCEPT], goal_request=_create())
    assert len(asked) == 1
    assert OBJECTIVE in asked[0]["questions"][0]["question"]
    assert text == "working on it"
    values = await _values(agent)
    assert values["_goal"]["status"] == "complete"
    assert [m.content for m in values["messages"] if m.type == "human"] == [OBJECTIVE]
    assert model.calls == {"main": 1, "draft": 1, "grade": 1}
    shown = screen.getvalue()
    assert "Goal accepted." in shown
    assert "- the report exists" in shown
    # What the verdict did to the goal reads after the verdict.
    final_frame = shown.split("Goal accepted.")[-1]
    assert final_frame.count("Acceptance criteria satisfied") == 1
    assert final_frame.index("Acceptance criteria satisfied") < final_frame.index(
        "Goal complete."
    )
    assert "Processing" not in shown.split("Goal accepted.")[-1]


async def test_feedback_is_redrafted_and_asked_again(tmp_path, screen):
    agent, model = cli_goal_agent(tmp_path, drafts=["- first", "- second"])
    _text, asked = await _turn(
        agent, answers=[feedback("make it shorter"), ACCEPT], goal_request=_create()
    )
    assert len(asked) == 2
    assert "- second" in asked[1]["questions"][0]["question"]
    assert model.calls["draft"] == 2
    values = await _values(agent)
    assert values["_sticky_rubric"]["text"] == "- second"
    final_frame = screen.getvalue().split("Goal accepted.")[-1]
    assert "Drafting acceptance criteria" not in final_frame


async def test_a_cancelled_review_leaves_one_line_and_nothing_else(tmp_path, screen):
    agent, model = cli_goal_agent(tmp_path)
    text, _asked = await _turn(agent, answers=[CANCEL], goal_request=_create())
    assert text == ""
    assert model.calls["main"] == 0
    values = await _values(agent)
    assert values.get("_goal") is None
    assert values.get("messages", []) == []
    shown = screen.getvalue()
    assert "Goal proposal cancelled." in shown
    assert "Processing" not in shown.split("Goal proposal cancelled.")[-1]


async def test_a_status_change_is_one_line_without_a_model_call(tmp_path, screen):
    agent, model = cli_goal_agent(tmp_path)
    await _turn(agent, answers=[ACCEPT], goal_request=_create())
    target = (await _values(agent))["_goal"]["rubric_id"]
    before = dict(model.calls)
    screen.truncate(0)
    screen.seek(0)
    text, asked = await _turn(
        agent, goal_request=new_goal_request("clear", target=target)
    )
    assert (text, asked) == ("", [])
    assert model.calls == before
    assert (await _values(agent)).get("_goal") is None
    shown = screen.getvalue()
    assert "Goal cleared." in shown
    assert "Processing" not in shown


async def test_the_next_typed_message_is_an_ordinary_turn(tmp_path, screen):
    agent, model = cli_goal_agent(tmp_path)
    await _turn(agent, answers=[CANCEL], goal_request=_create())
    text, asked = await _turn(agent, message="hello")
    assert (text, asked) == ("working on it", [])
    assert model.calls["draft"] == 1
    assert (await _values(agent)).get("goal_request") is None
