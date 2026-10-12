"""The TUI closes the parked checkpoint when a turn ends at a declined or unanswered approval."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("textual")

from langchain_core.messages import AIMessage

import EvoScientist.cli.tui_interactive as tui_mod
from EvoScientist.channels.hitl_budget import HITL_BUDGET_STOP_NOTICE
from EvoScientist.channels.interaction import ApprovalOutcome
from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget
from EvoScientist.gateway import RuntimeGateways
from EvoScientist.stream import display as display_mod
from tests.fakes import FakeCheckpointAgent, FakeGraphGateway
from tests.test_tui_approve_all import (
    APPROVE,
    ASK,
    REJECTING,
    _interrupt,
    _no_config_rule,
    _widget_answers,
)
from tests.test_tui_banner_position import _capture_app

BEFORE_TOOLS = ("HumanInTheLoopMiddleware.after_model",)
REJECTED = "User rejected the tool call for `execute` with id call-1. The tool was not executed."
UNANSWERED = "The approval request for the tool call `execute` with id call-1 was left unanswered."
DONE = {"type": "done", "response": ""}
TAIL = [{"type": "text", "content": "tail"}, {"type": "done", "response": "tail"}]
FINISHED = [
    {"type": "text", "content": "must not run"},
    {"type": "done", "response": "must not run"},
]


def _parked_agent(*, name: str = "execute", next: tuple = BEFORE_TOOLS, **kwargs):
    call = {
        "name": name,
        "args": {"command": "echo 1"},
        "id": "call-1",
        "type": "tool_call",
    }
    return FakeCheckpointAgent(
        next=next,
        values={"messages": [AIMessage(content="", tool_calls=[call])]},
        **kwargs,
    )


def _stream(script: list[list[dict]], *, hold: asyncio.Event | None = None):
    """One list of events per stream call. Like the real stream, a call ends when
    its run ends; with ``hold`` the first call stays open (work still running)."""
    calls = 0

    async def _fake_stream(_request):
        nonlocal calls
        events = script[calls]
        calls += 1
        for event in events:
            yield event
        if hold is not None and calls == 1:
            await hold.wait()

    return _fake_stream


def _written(agent: FakeCheckpointAgent) -> list[str]:
    return [
        message.content
        for values, _node in agent.updates
        if isinstance(values, dict)
        for message in values["messages"]
    ]


async def _turn(monkeypatch, workspace, script, agent, *, prepare=None, **kwargs):
    """Run one turn through the real TUI loop."""
    app = await _capture_app(monkeypatch, workspace)
    gateway = FakeGraphGateway(stream=_stream(script), checkpoint=agent)
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
        tools = {
            widget._tool_id: (widget._status, bool(widget.display))
            for widget in app.query(ToolCallWidget)
        }
    return SimpleNamespace(gateway=gateway, response=response, tools=tools)


async def test_a_reject_on_the_widget_closes_as_rejected(monkeypatch, workspace):
    _no_config_rule(monkeypatch)
    agent = _parked_agent()
    turn = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1), DONE], FINISHED],
        agent,
        prepare=lambda app: _widget_answers(monkeypatch, app, REJECTING),
    )
    assert len(turn.gateway.requests) == 1
    assert [text.startswith(REJECTED) for text in _written(agent)] == [True]
    assert agent.next == ()


async def test_a_widget_nobody_answered_closes_as_unanswered(monkeypatch, workspace):
    """The real widget wait runs out: nobody refused."""
    _no_config_rule(monkeypatch)
    monkeypatch.setattr(tui_mod, "_APPROVAL_WIDGET_TIMEOUT", 0.05)
    agent = _parked_agent()
    turn = await _turn(monkeypatch, workspace, [[_interrupt(1), DONE], FINISHED], agent)
    assert len(turn.gateway.requests) == 1
    assert [text.startswith(UNANSWERED) for text in _written(agent)] == [True]


@pytest.mark.parametrize(
    ("rejected", "expected"), [(True, REJECTED), (False, UNANSWERED)]
)
async def test_a_decline_on_the_channel_closes_with_its_reason(
    monkeypatch, workspace, rejected, expected
):
    _no_config_rule(monkeypatch)
    agent = _parked_agent()
    turn = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1), DONE], FINISHED],
        agent,
        channel_hitl_fn=lambda _reqs, human_budget_exhausted=False: ApprovalOutcome(
            prompted=True, rejected=rejected
        ),
    )
    assert len(turn.gateway.requests) == 1
    assert [text.startswith(expected) for text in _written(agent)] == [True]
    # The round-limit notice belongs to a spent budget only.
    assert HITL_BUDGET_STOP_NOTICE not in (turn.response or "")


async def test_the_round_limit_exit_records_the_limit_and_sends_its_notice(
    monkeypatch, workspace
):
    _no_config_rule(monkeypatch)
    agent = _parked_agent()
    turn = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1), DONE], FINISHED],
        agent,
        channel_hitl_fn=lambda _reqs, human_budget_exhausted=False: ApprovalOutcome(
            budget_exhausted=True
        ),
    )
    (text,) = _written(agent)
    assert "(approval round limit reached)" in text
    assert HITL_BUDGET_STOP_NOTICE in turn.response


async def test_a_close_that_fails_ends_the_turn_with_one_short_log_line(
    monkeypatch, workspace, caplog
):
    _no_config_rule(monkeypatch)
    agent = _parked_agent(
        update_error=RuntimeError("409 conflict\nrun kwargs:\n" + "x" * 5000)
    )
    with caplog.at_level(logging.DEBUG, logger="EvoScientist"):
        turn = await _turn(
            monkeypatch,
            workspace,
            [[_interrupt(1), DONE], FINISHED],
            agent,
            prepare=lambda app: _widget_answers(monkeypatch, app, REJECTING),
        )
    assert len(turn.gateway.requests) == 1
    failures = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "checkpoint" in r.getMessage().lower()
    ]
    assert len(failures) == 1
    assert len(failures[0].getMessage()) < 400
    assert "\n" not in failures[0].getMessage()
    assert failures[0].exc_info is None


async def test_a_stop_while_a_pause_is_unresolved_closes_nothing(
    monkeypatch, workspace
):
    """A stopped turn keeps its pause when nobody had declined."""
    agent = _parked_agent(name="ask_user", next=("tools",))

    async def _stream_then_stop(_request):
        yield {"type": "ask_user", "interrupt_id": "q-1", "questions": []}
        display_mod.request_stream_cancel("tui-stop-3")
        yield DONE

    app = await _capture_app(monkeypatch, workspace)
    gateway = FakeGraphGateway(stream=_stream_then_stop, checkpoint=agent)
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app._runtime_gateways = RuntimeGateways(
                thread_store=gateway.thread_store, graph_gateway=gateway
            )
            monkeypatch.setattr(
                app, "_await_agent_ready", AsyncMock(return_value=MagicMock())
            )
            await app._stream_with_widgets("hello", cancel_scope="tui-stop-3")
    finally:
        display_mod.discard_stream_cancel("tui-stop-3")
    assert agent.updates == []


async def test_an_unresolved_pause_is_not_called_a_round_limit(monkeypatch, workspace):
    """A question without questions cannot be answered; no limit was hit."""
    agent = _parked_agent(name="ask_user", next=("tools",))
    await _turn(
        monkeypatch,
        workspace,
        [
            [{"type": "ask_user", "interrupt_id": "q-1", "questions": []}, DONE],
            FINISHED,
        ],
        agent,
    )
    (text,) = _written(agent)
    assert "an approval request or question was left unanswered" in text
    assert "round limit" not in text


def _tool(tool_id: str, name: str = "read_file") -> list[dict]:
    return [
        {"type": "tool_call", "id": tool_id, "name": name, "args": {"file_path": "/a"}},
        {
            "type": "tool_result",
            "id": tool_id,
            "name": name,
            "content": "ok",
            "success": True,
        },
    ]


# More finished tools than the transcript keeps open, so the older ones collapse.
FIVE_FINISHED_TOOLS = [event for n in range(5) for event in _tool(f"done-{n}")]


@pytest.mark.parametrize("surface", ["widget", "channel"])
async def test_after_a_decline_the_rest_of_the_stream_is_read_to_its_end(
    monkeypatch, workspace, surface
):
    """Sibling work that is still running when the user declines finishes: its
    result is shown as it is, the answer that follows is kept, and only the
    declined call ends up rejected."""
    _no_config_rule(monkeypatch)
    agent = _parked_agent()
    script = [
        [
            *FIVE_FINISHED_TOOLS,
            {"type": "tool_call", "id": "late", "name": "write_file", "args": {}},
            {"type": "tool_call", "id": "call-1", "name": "execute", "args": {}},
            _interrupt(1),
            {
                "type": "tool_result",
                "id": "late",
                "name": "write_file",
                "content": "ok",
                "success": True,
            },
            {"type": "text", "content": "sibling finished"},
            {"type": "done", "response": "sibling finished"},
        ],
        FINISHED,
    ]
    if surface == "widget":
        turn = await _turn(
            monkeypatch,
            workspace,
            script,
            agent,
            prepare=lambda app: _widget_answers(monkeypatch, app, REJECTING),
        )
    else:
        turn = await _turn(
            monkeypatch,
            workspace,
            script,
            agent,
            channel_hitl_fn=lambda _reqs, human_budget_exhausted=False: ApprovalOutcome(
                prompted=True, rejected=True
            ),
        )
    assert turn.response == "sibling finished"
    assert len(turn.gateway.requests) == 1
    assert turn.tools["late"][0] == "success"
    assert turn.tools["call-1"][0] == "rejected"
    # Every tool is shown again once the turn has ended (none left collapsed).
    assert len(turn.tools) == 7
    assert all(shown for _status, shown in turn.tools.values())
    assert [text.startswith(REJECTED) for text in _written(agent)] == [True]


async def test_a_stream_that_fails_after_a_decline_is_still_closed(
    monkeypatch, workspace
):
    """Sibling work that breaks after the decline: the decline is recorded and
    the tools collapsed during the turn are shown again."""
    _no_config_rule(monkeypatch)
    agent = _parked_agent()

    async def _fails_after_the_pause(_request):
        for event in (
            *FIVE_FINISHED_TOOLS,
            {"type": "tool_call", "id": "call-1", "name": "execute", "args": {}},
            _interrupt(1),
        ):
            yield event
        raise RuntimeError("sibling failed")

    app = await _capture_app(monkeypatch, workspace)
    gateway = FakeGraphGateway(stream=_fails_after_the_pause, checkpoint=agent)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._runtime_gateways = RuntimeGateways(
            thread_store=gateway.thread_store, graph_gateway=gateway
        )
        monkeypatch.setattr(
            app, "_await_agent_ready", AsyncMock(return_value=MagicMock())
        )
        _widget_answers(monkeypatch, app, REJECTING)
        response = await app._stream_with_widgets("hello")
        tools = {
            widget._tool_id: (widget._status, bool(widget.display))
            for widget in app.query(ToolCallWidget)
        }
    assert response == "Error: sibling failed"
    assert len(gateway.requests) == 1
    assert [text.startswith(REJECTED) for text in _written(agent)] == [True]
    assert tools["call-1"][0] == "rejected"
    assert len(tools) == 6
    assert all(shown for _status, shown in tools.values())


async def test_after_a_decline_a_later_allow_listed_command_does_not_resume(
    monkeypatch, workspace
):
    """A sibling's command that config rules approve arrives after the user
    declined: it must not restart the turn (and bring the declined prompt back)."""
    import EvoScientist.channels.interaction as interaction_mod

    snapshots = iter([(None, {}), (list(APPROVE), {})])
    monkeypatch.setattr(
        interaction_mod, "config_policy_snapshot", lambda _reqs: next(snapshots)
    )
    agent = _parked_agent()
    waits: list[int] = []

    def _prepare(app):
        async def _reject(_widget):
            waits.append(1)
            return REJECTING

        monkeypatch.setattr(app, "_wait_for_approval", _reject)

    turn = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1), _interrupt(2), *TAIL], FINISHED],
        agent,
        prepare=_prepare,
    )
    assert len(turn.gateway.requests) == 1
    assert waits == [1]
    # The stream was read past the later approval, not left at it.
    assert turn.response == "tail"
    assert [text.startswith(REJECTED) for text in _written(agent)] == [True]


async def test_after_a_decline_a_session_grant_does_not_resume(monkeypatch, workspace):
    _no_config_rule(monkeypatch)
    agent = _parked_agent()

    def _prepare(app):
        async def _reject_then_grant(_widget):
            app._hitl_auto_approve = True  # e.g. granted from another turn's widget
            return REJECTING

        monkeypatch.setattr(app, "_wait_for_approval", _reject_then_grant)

    turn = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1), _interrupt(2), *TAIL], FINISHED],
        agent,
        prepare=_prepare,
    )
    assert len(turn.gateway.requests) == 1
    assert turn.response == "tail"
    assert [text.startswith(REJECTED) for text in _written(agent)] == [True]


async def test_after_a_decline_the_channel_is_not_asked_again(monkeypatch, workspace):
    _no_config_rule(monkeypatch)
    agent = _parked_agent()
    asked: list[int] = []

    def _bridge(_reqs, human_budget_exhausted=False):
        asked.append(1)
        if len(asked) == 1:
            return ApprovalOutcome(prompted=True, rejected=True)
        return ApprovalOutcome(decisions=list(APPROVE), session_granted=True)

    questions: list[dict] = []

    def _question(event):
        # Answering would resume the turn; an exception here is swallowed by
        # the loop, so the call is recorded instead.
        questions.append(event)
        return {"answers": ["yes"], "status": "answered"}

    turn = await _turn(
        monkeypatch,
        workspace,
        [[_interrupt(1), _interrupt(2), ASK, DONE], FINISHED],
        agent,
        channel_hitl_fn=_bridge,
        channel_ask_user_fn=_question,
    )
    assert questions == []
    assert len(turn.gateway.requests) == 1
    assert asked == [1]
    assert [text.startswith(REJECTED) for text in _written(agent)] == [True]


async def test_a_stop_flag_set_after_a_decline_still_closes(monkeypatch, workspace):
    """A stop requested through the flag only (a channel's stop command)."""
    _no_config_rule(monkeypatch)
    agent = _parked_agent()

    def _prepare(app):
        async def _reject_then_stop(_widget):
            display_mod.request_stream_cancel("tui-stop")
            return REJECTING

        monkeypatch.setattr(app, "_wait_for_approval", _reject_then_stop)

    try:
        turn = await _turn(
            monkeypatch,
            workspace,
            [[_interrupt(1), DONE], FINISHED],
            agent,
            prepare=_prepare,
            cancel_scope="tui-stop",
        )
    finally:
        display_mod.discard_stream_cancel("tui-stop")
    assert len(turn.gateway.requests) == 1
    assert [text.startswith(REJECTED) for text in _written(agent)] == [True]


async def test_a_stop_without_a_decision_closes_nothing(monkeypatch, workspace):
    _no_config_rule(monkeypatch)
    agent = _parked_agent()

    def _stopped(_reqs, human_budget_exhausted=False):
        display_mod.request_stream_cancel("tui-stop-2")
        return ApprovalOutcome(prompted=True)

    try:
        turn = await _turn(
            monkeypatch,
            workspace,
            [[_interrupt(1), DONE], FINISHED],
            agent,
            channel_hitl_fn=_stopped,
            cancel_scope="tui-stop-2",
        )
    finally:
        display_mod.discard_stream_cancel("tui-stop-2")
    assert len(turn.gateway.requests) == 1
    assert agent.updates == []


# -- Ctrl+C, through the real action (it cancels the turn's task) ------------------------------


async def _ctrl_c_turn(
    monkeypatch, workspace, script, agent, *, hold, when, prepare=None, again_when=None
):
    """Run a turn as the app's turn task and press Ctrl+C once ``when(app)``
    holds; with ``again_when``, press it a second time once that holds."""
    app = await _capture_app(monkeypatch, workspace)
    gateway = FakeGraphGateway(stream=_stream(script, hold=hold), checkpoint=agent)
    try:
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
            task = asyncio.create_task(app._stream_with_widgets("hello"))
            app._run_task = task
            app._busy = True
            for _ in range(200):
                await pilot.pause()
                if when(app):
                    break
            else:
                raise AssertionError("the turn never reached the expected point")
            await pilot.pause()
            app.action_request_quit()
            if again_when is not None:
                for _ in range(200):
                    await pilot.pause()
                    if again_when():
                        break
                else:
                    raise AssertionError("the close never started")
                app.action_request_quit()
            try:
                await asyncio.wait_for(task, timeout=10)
            except asyncio.CancelledError:
                pass
            tools = {
                widget._tool_id: (widget._status, bool(widget.display))
                for widget in app.query(ToolCallWidget)
            }
    finally:
        display_mod.discard_stream_cancel()
    return SimpleNamespace(gateway=gateway, tools=tools)


async def test_ctrl_c_on_the_open_widget_is_a_rejection(monkeypatch, workspace):
    _no_config_rule(monkeypatch)
    agent = _parked_agent()
    turn = await _ctrl_c_turn(
        monkeypatch,
        workspace,
        [[_interrupt(1), DONE], FINISHED],
        agent,
        hold=None,
        when=lambda app: app._approval_future is not None,
    )
    assert len(turn.gateway.requests) == 1
    assert [text.startswith(REJECTED) for text in _written(agent)] == [True]


async def test_ctrl_c_after_a_decline_while_sibling_work_runs_still_closes(
    monkeypatch, workspace
):
    """The user declines, sibling work keeps the stream open, then Ctrl+C cancels
    the turn's task: the decline is still recorded."""
    _no_config_rule(monkeypatch)
    agent = _parked_agent()
    declined: list[int] = []

    def _prepare(app):
        async def _reject(_widget):
            declined.append(1)
            return REJECTING

        monkeypatch.setattr(app, "_wait_for_approval", _reject)

    turn = await _ctrl_c_turn(
        monkeypatch,
        workspace,
        [
            [
                *FIVE_FINISHED_TOOLS,
                {"type": "tool_call", "id": "call-1", "name": "execute", "args": {}},
                _interrupt(1),
            ],
            FINISHED,
        ],
        agent,
        hold=asyncio.Event(),  # never set: the sibling is still working
        when=lambda _app: bool(declined),
        prepare=_prepare,
    )
    assert len(turn.gateway.requests) == 1
    assert [text.startswith(REJECTED) for text in _written(agent)] == [True]
    assert agent.next == ()
    assert turn.tools["call-1"][0] == "rejected"
    assert len(turn.tools) == 6
    assert all(shown for _status, shown in turn.tools.values())


async def test_ctrl_c_without_a_decline_closes_nothing(monkeypatch, workspace):
    """A plain stop of a running turn writes nothing to the thread."""
    agent = _parked_agent()
    turn = await _ctrl_c_turn(
        monkeypatch,
        workspace,
        [
            [{"type": "tool_call", "id": "call-1", "name": "execute", "args": {}}],
            FINISHED,
        ],
        agent,
        hold=asyncio.Event(),
        when=lambda app: bool(app.query(ToolCallWidget)),
    )
    assert len(turn.gateway.requests) == 1
    assert agent.updates == []
    assert turn.tools["call-1"][0] == "interrupted"


class _NeverFinishesClosing(FakeCheckpointAgent):
    """The first write lands, then the close hangs (a slow store)."""

    closing = False

    async def aupdate_state(self, _config, values, as_node=None):
        await super().aupdate_state(_config, values, as_node)
        self.closing = True
        await asyncio.Event().wait()


async def test_a_second_ctrl_c_during_the_close_still_marks_the_declined_call(
    monkeypatch, workspace, caplog
):
    """Ctrl+C after a decline starts the close; a second one cancels it. The
    declined call does not keep spinning and the cut close is reported once."""
    _no_config_rule(monkeypatch)
    call = {"name": "execute", "args": {}, "id": "call-1", "type": "tool_call"}
    agent = _NeverFinishesClosing(
        next=BEFORE_TOOLS,
        values={"messages": [AIMessage(content="", tool_calls=[call])]},
    )
    declined: list[int] = []

    def _prepare(app):
        async def _reject(_widget):
            declined.append(1)
            return REJECTING

        monkeypatch.setattr(app, "_wait_for_approval", _reject)

    with caplog.at_level(logging.WARNING, logger="EvoScientist"):
        turn = await _ctrl_c_turn(
            monkeypatch,
            workspace,
            [
                [
                    {
                        "type": "tool_call",
                        "id": "call-1",
                        "name": "execute",
                        "args": {},
                    },
                    _interrupt(1),
                ],
                FINISHED,
            ],
            agent,
            hold=asyncio.Event(),
            when=lambda _app: bool(declined),
            prepare=_prepare,
            again_when=lambda: agent.closing,
        )
    assert turn.tools["call-1"][0] == "rejected"
    assert len(agent.updates) == 1  # the close was cut after its first write
    cut = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "checkpoint" in r.getMessage().lower()
    ]
    assert len(cut) == 1
    assert "cancelled" in cut[0].getMessage()


async def test_ctrl_c_during_the_close_after_the_stream_still_marks_the_declined_call(
    monkeypatch, workspace, caplog
):
    """The stream ends by itself after the decline; one Ctrl+C lands while the
    close is writing."""
    _no_config_rule(monkeypatch)
    call = {"name": "execute", "args": {}, "id": "call-1", "type": "tool_call"}
    agent = _NeverFinishesClosing(
        next=BEFORE_TOOLS,
        values={"messages": [AIMessage(content="", tool_calls=[call])]},
    )
    with caplog.at_level(logging.WARNING, logger="EvoScientist"):
        turn = await _ctrl_c_turn(
            monkeypatch,
            workspace,
            [
                [
                    {
                        "type": "tool_call",
                        "id": "call-1",
                        "name": "execute",
                        "args": {},
                    },
                    _interrupt(1),
                    DONE,
                ],
                FINISHED,
            ],
            agent,
            hold=None,
            when=lambda _app: agent.closing,
            prepare=lambda app: _widget_answers(monkeypatch, app, REJECTING),
        )
    assert turn.tools["call-1"][0] == "rejected"
    assert len(agent.updates) == 1
    cut = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "checkpoint" in r.getMessage().lower()
    ]
    assert len(cut) == 1
    assert "cancelled" in cut[0].getMessage()


async def test_the_timeout_mark_does_not_leak_into_the_next_wait(
    monkeypatch, workspace
):
    """A wait that ran out, then a wait that Ctrl+C ends: the second is not a timeout."""
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(tui_mod, "_APPROVAL_WIDGET_TIMEOUT", 0.05)
        assert await app._wait_for_approval(object()) is None
        assert app._approval_timed_out is True

        monkeypatch.setattr(tui_mod, "_APPROVAL_WIDGET_TIMEOUT", 30.0)
        wait = asyncio.ensure_future(app._wait_for_approval(object()))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if app._approval_future is not None:
                break
        wait.cancel()
        assert await wait is None
        assert app._approval_timed_out is False
