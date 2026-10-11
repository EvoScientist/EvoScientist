"""Which calls of a TUI turn carry the session "approve all" grant to the graph."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("textual")

from EvoScientist.channels.interaction import ApprovalOutcome
from EvoScientist.gateway import RuntimeGateways
from tests.fakes import FakeGraphGateway
from tests.test_tui_banner_position import _capture_app

APPROVE = [{"type": "approve"}]


def _interrupt(n: int) -> dict:
    return {
        "type": "interrupt",
        "interrupt_id": f"approval-{n}",
        "action_requests": [{"name": "execute", "args": {"command": f"echo {n}"}}],
    }


def _stream(script: list[list[dict]]):
    """One list of events per stream call; a call that pauses stays open until closed."""
    calls = 0

    async def _fake_stream(_request):
        nonlocal calls
        events = script[calls]
        calls += 1
        for event in events:
            yield event
        if events and events[-1]["type"] in ("interrupt", "ask_user"):
            await asyncio.Event().wait()

    return _fake_stream


DONE = [
    {"type": "text", "content": "finished"},
    {"type": "done", "response": "finished"},
]


async def _turn(monkeypatch, workspace, script, *, prepare=None, **kwargs):
    """Run one turn through the real TUI loop; return each call's ``configurable_extra``."""
    app = await _capture_app(monkeypatch, workspace)
    gateway = FakeGraphGateway(stream=_stream(script))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._runtime_gateways = RuntimeGateways(
            thread_store=gateway.thread_store, graph_gateway=gateway
        )
        monkeypatch.setattr(
            app, "_await_agent_ready", AsyncMock(return_value=MagicMock())
        )
        if prepare is not None:
            prepare(app)
        response = await app._stream_with_widgets("hello", **kwargs)
    assert response == "finished"
    return [request.configurable_extra or {} for request in gateway.requests]


def _keyed(extras: list[dict]) -> list[bool]:
    """Per call: does it carry the session grant? (Other extras ride along.)"""
    return [extra.get("hitl_approve_all") is True for extra in extras]


def _no_config_rule(monkeypatch):
    import EvoScientist.channels.interaction as interaction_mod

    monkeypatch.setattr(
        interaction_mod, "config_policy_snapshot", lambda reqs: (None, {})
    )


def _widget_answers(monkeypatch, app, *answers):
    monkeypatch.setattr(app, "_wait_for_approval", AsyncMock(side_effect=list(answers)))


GRANTING = SimpleNamespace(decisions=APPROVE, auto_approve_session=True)
APPROVING_ONCE = SimpleNamespace(decisions=APPROVE, auto_approve_session=False)
REJECTING = SimpleNamespace(decisions=None, auto_approve_session=False)


async def test_a_turn_under_the_grant_is_keyed_from_its_first_call(
    monkeypatch, workspace
):
    extras = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1)], DONE],
        prepare=lambda app: setattr(app, "_hitl_auto_approve", True),
    )
    assert _keyed(extras) == [True, True]


async def test_the_granting_answer_is_sent_unkeyed(monkeypatch, workspace):
    """The widget's answer must be applied as given; the key rides from the
    call after it, the first auto-answered resume."""
    _no_config_rule(monkeypatch)
    extras = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1)], [_interrupt(2)], DONE],
        prepare=lambda app: _widget_answers(monkeypatch, app, GRANTING),
    )
    assert _keyed(extras) == [False, False, True]


async def test_a_grant_on_the_second_of_two_parked_approvals_is_sent_unkeyed(
    monkeypatch, workspace
):
    """Two approvals arrive in one stream: the first is declined at the widget
    (no resume), the grant comes with the second. The decision is taken in
    the handler of each approval, not once per stream call."""
    _no_config_rule(monkeypatch)
    extras = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1), _interrupt(2)], [_interrupt(3)], DONE],
        prepare=lambda app: _widget_answers(monkeypatch, app, REJECTING, GRANTING),
    )
    assert _keyed(extras) == [False, False, True]


async def test_a_config_rule_answer_is_not_keyed(monkeypatch, workspace):
    import EvoScientist.channels.interaction as interaction_mod

    monkeypatch.setattr(
        interaction_mod, "config_policy_snapshot", lambda reqs: (list(APPROVE), {})
    )
    extras = await _turn(monkeypatch, workspace, [[_interrupt(1)], DONE])
    assert _keyed(extras) == [False, False]


async def test_a_channel_turn_starts_keyed_when_its_chat_is_granted(
    monkeypatch, workspace
):
    extras = await _turn(monkeypatch, workspace, [DONE], approve_all_at_start=True)
    assert _keyed(extras) == [True]


@pytest.mark.parametrize("granted", [True, False])
async def test_a_bridge_answer_is_keyed_only_when_the_chat_is_granted(
    monkeypatch, workspace, granted
):
    _no_config_rule(monkeypatch)
    extras = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1)], DONE],
        channel_hitl_fn=lambda _reqs, human_budget_exhausted=False: ApprovalOutcome(
            decisions=list(APPROVE), session_granted=granted, prompted=not granted
        ),
    )
    assert _keyed(extras) == [False, granted]


@pytest.mark.parametrize("granted", [True, False])
async def test_an_ask_user_answer_is_keyed_only_under_a_grant(
    monkeypatch, workspace, granted
):
    ask = {
        "type": "ask_user",
        "interrupt_id": "question-1",
        "questions": [{"question": "Continue?"}],
    }
    extras = await _turn(
        monkeypatch,
        workspace,
        [[ask], DONE],
        prepare=lambda app: setattr(app, "_hitl_auto_approve", granted),
        channel_ask_user_fn=lambda _event: {"answers": ["yes"], "status": "answered"},
    )
    assert _keyed(extras) == [granted, granted]


async def test_other_extras_ride_on_every_call(monkeypatch, workspace):
    _no_config_rule(monkeypatch)

    def _prepare(app):
        app._channel_runtime.active_teams.append("reviewer")
        _widget_answers(monkeypatch, app, GRANTING)

    extras = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1)], [_interrupt(2)], DONE],
        prepare=_prepare,
    )
    assert _keyed(extras) == [False, False, True]
    assert [extra["active_teams"] for extra in extras] == [["reviewer"]] * 3
    # The marker the TUI sends on every call of its own thread is untouched.
    assert all("rubric_run" in extra for extra in extras)


ASK = {
    "type": "ask_user",
    "interrupt_id": "question-1",
    "questions": [{"question": "Continue?"}],
}
ANSWER = {"answers": ["yes"], "status": "answered"}


async def test_an_answer_not_from_the_grant_drops_a_key_the_turn_started_with(
    monkeypatch, workspace
):
    """The turn starts keyed; an approval that still arrives and is answered by
    a config rule must go back unkeyed, whatever the previous call carried."""
    import EvoScientist.channels.interaction as interaction_mod

    monkeypatch.setattr(
        interaction_mod, "config_policy_snapshot", lambda reqs: (list(APPROVE), {})
    )
    extras = await _turn(
        monkeypatch, workspace, [[_interrupt(1)], DONE], approve_all_at_start=True
    )
    assert _keyed(extras) == [True, False]


async def test_a_bridge_answer_with_a_non_approve_decision_is_not_keyed(
    monkeypatch, workspace
):
    """Belt and braces: the key is sent only when every decision approves."""
    _no_config_rule(monkeypatch)
    extras = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1)], DONE],
        channel_hitl_fn=lambda _reqs, human_budget_exhausted=False: ApprovalOutcome(
            decisions=[{"type": "reject", "message": "no"}], session_granted=True
        ),
    )
    assert _keyed(extras) == [False, False]


async def test_a_question_after_the_widget_grant_is_answered_keyed(
    monkeypatch, workspace
):
    _no_config_rule(monkeypatch)
    extras = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1)], [ASK], DONE],
        prepare=lambda app: _widget_answers(monkeypatch, app, GRANTING),
        channel_ask_user_fn=lambda _event: dict(ANSWER),
    )
    assert _keyed(extras) == [False, False, True]


async def test_a_question_after_a_bridge_grant_answer_is_answered_keyed(
    monkeypatch, workspace
):
    """A chat grant seen during the turn holds for the rest of it."""
    _no_config_rule(monkeypatch)
    extras = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1)], [ASK], DONE],
        channel_hitl_fn=lambda _reqs, human_budget_exhausted=False: ApprovalOutcome(
            decisions=list(APPROVE), session_granted=True
        ),
        channel_ask_user_fn=lambda _event: dict(ANSWER),
    )
    assert _keyed(extras) == [False, True, True]


async def test_a_question_after_a_plain_widget_approval_is_answered_unkeyed(
    monkeypatch, workspace
):
    """Approving once grants nothing: the answer to a question that follows is
    unkeyed, so the turn's later gated calls still ask."""
    _no_config_rule(monkeypatch)
    extras = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1)], [ASK], DONE],
        prepare=lambda app: _widget_answers(monkeypatch, app, APPROVING_ONCE),
        channel_ask_user_fn=lambda _event: dict(ANSWER),
    )
    assert _keyed(extras) == [False, False, False]


async def test_a_question_after_a_plain_bridge_approval_is_answered_unkeyed(
    monkeypatch, workspace
):
    """The same for a channel user who approved once."""
    _no_config_rule(monkeypatch)
    extras = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1)], [ASK], DONE],
        channel_hitl_fn=lambda _reqs, human_budget_exhausted=False: ApprovalOutcome(
            decisions=list(APPROVE), prompted=True
        ),
        channel_ask_user_fn=lambda _event: dict(ANSWER),
    )
    assert _keyed(extras) == [False, False, False]


async def test_a_grant_given_on_the_channel_keys_the_next_question_answer(
    monkeypatch, workspace
):
    """The reply that grants is a human answer (unkeyed), but the chat's grant
    is in force afterwards: the answer to the question that follows carries it."""
    _no_config_rule(monkeypatch)
    extras = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1)], [ASK], DONE],
        channel_hitl_fn=lambda _reqs, human_budget_exhausted=False: ApprovalOutcome(
            decisions=list(APPROVE), prompted=True, grant_in_force=True
        ),
        channel_ask_user_fn=lambda _event: dict(ANSWER),
    )
    assert _keyed(extras) == [False, False, True]


@pytest.mark.parametrize("granted", [True, False])
async def test_the_channel_handler_passes_its_chats_grant(
    monkeypatch, workspace, granted
):
    """The handler, not the loop, knows the chat: a turn must start keyed for a
    granted chat and never for a chat that only shares the process with one."""
    from EvoScientist.cli import channel as channel_mod
    from EvoScientist.cli.channel import ChannelMessage, _register_channel_request

    app = await _capture_app(monkeypatch, workspace)
    msg = ChannelMessage(
        msg_id="m-1",
        content="hello",
        sender="u1",
        channel_type="imessage",
        metadata={},
        channel_ref=None,
        bus_ref=None,
        chat_id="chat-a",
        message_id="ts-1",
    )
    channel_mod._approval_policy.clear_sessions()
    channel_mod._approval_policy.grant_session(
        "imessage:chat-a" if granted else "imessage:chat-b"
    )
    stream = AsyncMock(return_value="ok")
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            monkeypatch.setattr(
                app, "_await_agent_ready", AsyncMock(return_value=MagicMock())
            )
            monkeypatch.setattr(app, "_stream_with_widgets", stream)
            _register_channel_request(msg)
            await app._process_channel_message(msg)
    finally:
        channel_mod._approval_policy.clear_sessions()
    assert stream.call_args.kwargs["approve_all_at_start"] is granted
