"""The TUI closes the parked checkpoint when a turn ends at a declined or unanswered approval."""

from __future__ import annotations

import asyncio
import logging
import traceback
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
from tests.fakes.common import (
    FakeCheckpointAgent,
    FakeGraphGateway,
    close_result,
    no_config_rule,
)
from tests.fakes.tui import (
    APPROVE,
    APPROVING_ONCE,
    ASK,
    BEFORE_TOOLS,
    DONE,
    FINISHED,
    GRANTING,
    REJECTING,
    TAIL,
    approval_interrupt,
    capture_app,
    ctrl_c_turn,
    parked_agent,
    scripted_stream,
    widget_answers,
)

# What the close wrote for the one call the user was asked about.
REJECTED = ("asked", "rejected", True)
UNANSWERED = ("asked", "unanswered", True)
ROUND_LIMIT = ("asked", "round_limit", True)


def _written(agent: FakeCheckpointAgent) -> list[str]:
    return [
        message.content
        for values, _node in agent.updates
        if isinstance(values, dict)
        for message in values["messages"]
    ]


def _closed(agent: FakeCheckpointAgent) -> list[tuple[str, str, bool]]:
    """What the close wrote, per call, as facts (``close_result``)."""
    return [close_result(text) for text in _written(agent)]


async def _turn(monkeypatch, workspace, script, agent, *, prepare=None, **kwargs):
    """Run one turn through the real TUI loop."""
    app = await capture_app(monkeypatch, workspace)
    gateway = FakeGraphGateway(stream=scripted_stream(script), checkpoint=agent)
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
    no_config_rule(monkeypatch)
    agent = parked_agent()
    turn = await _turn(
        monkeypatch,
        workspace,
        [[approval_interrupt(1), DONE], FINISHED],
        agent,
        prepare=lambda app: widget_answers(monkeypatch, app, REJECTING),
    )
    assert len(turn.gateway.requests) == 1
    assert _closed(agent) == [REJECTED]
    assert agent.next == ()


async def test_a_widget_nobody_answered_closes_as_unanswered(monkeypatch, workspace):
    """The real widget wait runs out: nobody refused."""
    no_config_rule(monkeypatch)
    monkeypatch.setattr(tui_mod, "_APPROVAL_WIDGET_TIMEOUT", 0.05)
    agent = parked_agent()
    shown: list[str] = []

    def _record(app):
        monkeypatch.setattr(
            app, "_append_system", lambda text, **_k: shown.append(text)
        )

    turn = await _turn(
        monkeypatch,
        workspace,
        [[approval_interrupt(1), DONE], FINISHED],
        agent,
        prepare=_record,
    )
    assert len(turn.gateway.requests) == 1
    assert _closed(agent) == [UNANSWERED]
    assert "Approval left unanswered." in shown
    assert "Tool execution rejected." not in shown


@pytest.mark.parametrize(
    ("rejected", "expected", "line"),
    [
        (True, REJECTED, "Tool execution rejected by channel user."),
        (False, UNANSWERED, "Channel approval left unanswered."),
    ],
)
async def test_a_decline_on_the_channel_closes_with_its_reason(
    monkeypatch, workspace, rejected, expected, line
):
    no_config_rule(monkeypatch)
    agent = parked_agent()
    shown: list[str] = []

    def _record(app):
        monkeypatch.setattr(
            app, "_append_system", lambda text, **_k: shown.append(text)
        )

    turn = await _turn(
        monkeypatch,
        workspace,
        [[approval_interrupt(1), DONE], FINISHED],
        agent,
        channel_hitl_fn=lambda _reqs, human_budget_exhausted=False: ApprovalOutcome(
            prompted=True, rejected=rejected
        ),
        prepare=_record,
    )
    assert len(turn.gateway.requests) == 1
    assert _closed(agent) == [expected]
    assert line in shown
    # The round-limit notice belongs to a spent budget only.
    assert HITL_BUDGET_STOP_NOTICE not in (turn.response or "")


async def test_the_round_limit_exit_records_the_limit_and_sends_its_notice(
    monkeypatch, workspace
):
    no_config_rule(monkeypatch)
    agent = parked_agent()
    turn = await _turn(
        monkeypatch,
        workspace,
        [[approval_interrupt(1), DONE], FINISHED],
        agent,
        channel_hitl_fn=lambda _reqs, human_budget_exhausted=False: ApprovalOutcome(
            budget_exhausted=True
        ),
    )
    assert _closed(agent) == [ROUND_LIMIT]
    assert HITL_BUDGET_STOP_NOTICE in turn.response


async def test_a_close_that_fails_ends_the_turn_with_one_short_log_line(
    monkeypatch, workspace, caplog
):
    no_config_rule(monkeypatch)
    agent = parked_agent(
        update_error=RuntimeError("409 conflict\nrun kwargs:\n" + "x" * 5000)
    )
    with caplog.at_level(logging.DEBUG, logger="EvoScientist"):
        turn = await _turn(
            monkeypatch,
            workspace,
            [[approval_interrupt(1), DONE], FINISHED],
            agent,
            prepare=lambda app: widget_answers(monkeypatch, app, REJECTING),
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
    agent = parked_agent(name="ask_user", next=("tools",))

    async def _stream_then_stop(_request):
        yield {"type": "ask_user", "interrupt_id": "q-1", "questions": []}
        display_mod.request_stream_cancel("tui-stop-3")
        yield DONE

    app = await capture_app(monkeypatch, workspace)
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
    agent = parked_agent(name="ask_user", next=("tools",))
    await _turn(
        monkeypatch,
        workspace,
        [
            [{"type": "ask_user", "interrupt_id": "q-1", "questions": []}, DONE],
            FINISHED,
        ],
        agent,
    )
    assert _closed(agent) == [("did not finish", "unanswered", False)]


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
    no_config_rule(monkeypatch)
    agent = parked_agent()
    script = [
        [
            *FIVE_FINISHED_TOOLS,
            {"type": "tool_call", "id": "late", "name": "write_file", "args": {}},
            {"type": "tool_call", "id": "call-1", "name": "execute", "args": {}},
            approval_interrupt(1),
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
            prepare=lambda app: widget_answers(monkeypatch, app, REJECTING),
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
    assert _closed(agent) == [REJECTED]


async def test_a_stream_that_fails_after_a_decline_is_still_closed(
    monkeypatch, workspace
):
    """Sibling work that breaks after the decline: the decline is recorded and
    the tools collapsed during the turn are shown again."""
    no_config_rule(monkeypatch)
    agent = parked_agent()

    async def _fails_after_the_pause(_request):
        for event in (
            *FIVE_FINISHED_TOOLS,
            {"type": "tool_call", "id": "call-1", "name": "execute", "args": {}},
            approval_interrupt(1),
        ):
            yield event
        raise RuntimeError("sibling failed")

    app = await capture_app(monkeypatch, workspace)
    gateway = FakeGraphGateway(stream=_fails_after_the_pause, checkpoint=agent)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._runtime_gateways = RuntimeGateways(
            thread_store=gateway.thread_store, graph_gateway=gateway
        )
        monkeypatch.setattr(
            app, "_await_agent_ready", AsyncMock(return_value=MagicMock())
        )
        widget_answers(monkeypatch, app, REJECTING)
        response = await app._stream_with_widgets("hello")
        tools = {
            widget._tool_id: (widget._status, bool(widget.display))
            for widget in app.query(ToolCallWidget)
        }
    assert response == "Error: sibling failed"
    assert len(gateway.requests) == 1
    assert _closed(agent) == [REJECTED]
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
    agent = parked_agent()
    waits: list[int] = []

    def _prepare(app):
        async def _reject(_widget):
            waits.append(1)
            return REJECTING

        monkeypatch.setattr(app, "_wait_for_approval", _reject)

    turn = await _turn(
        monkeypatch,
        workspace,
        [[approval_interrupt(1), approval_interrupt(2), *TAIL], FINISHED],
        agent,
        prepare=_prepare,
    )
    assert len(turn.gateway.requests) == 1
    assert waits == [1]
    # The stream was read past the later approval, not left at it.
    assert turn.response == "tail"
    assert _closed(agent) == [REJECTED]


async def test_after_a_decline_a_session_grant_does_not_resume(monkeypatch, workspace):
    no_config_rule(monkeypatch)
    agent = parked_agent()

    def _prepare(app):
        async def _reject_then_grant(_widget):
            app._hitl_auto_approve = True  # e.g. granted from another turn's widget
            return REJECTING

        monkeypatch.setattr(app, "_wait_for_approval", _reject_then_grant)

    turn = await _turn(
        monkeypatch,
        workspace,
        [[approval_interrupt(1), approval_interrupt(2), *TAIL], FINISHED],
        agent,
        prepare=_prepare,
    )
    assert len(turn.gateway.requests) == 1
    assert turn.response == "tail"
    assert _closed(agent) == [REJECTED]


async def test_after_a_decline_the_channel_is_not_asked_again(monkeypatch, workspace):
    no_config_rule(monkeypatch)
    agent = parked_agent()
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
        [[approval_interrupt(1), approval_interrupt(2), ASK, DONE], FINISHED],
        agent,
        channel_hitl_fn=_bridge,
        channel_ask_user_fn=_question,
    )
    assert questions == []
    assert len(turn.gateway.requests) == 1
    assert asked == [1]
    assert _closed(agent) == [REJECTED]


async def test_a_stop_flag_set_after_a_decline_still_closes(monkeypatch, workspace):
    """A stop requested through the flag only (a channel's stop command)."""
    no_config_rule(monkeypatch)
    agent = parked_agent()

    def _prepare(app):
        async def _reject_then_stop(_widget):
            display_mod.request_stream_cancel("tui-stop")
            return REJECTING

        monkeypatch.setattr(app, "_wait_for_approval", _reject_then_stop)

    try:
        turn = await _turn(
            monkeypatch,
            workspace,
            [[approval_interrupt(1), DONE], FINISHED],
            agent,
            prepare=_prepare,
            cancel_scope="tui-stop",
        )
    finally:
        display_mod.discard_stream_cancel("tui-stop")
    assert len(turn.gateway.requests) == 1
    assert _closed(agent) == [REJECTED]


@pytest.mark.parametrize(
    "outcome",
    [
        ApprovalOutcome(prompted=True, rejected=True),
        ApprovalOutcome(prompted=True),
        ApprovalOutcome(decisions=APPROVE, prompted=True),
    ],
    ids=["a stop reply", "an unanswered prompt", "an approval that came too late"],
)
async def test_a_stop_while_the_channel_is_asked_closes_as_rejected(
    monkeypatch, workspace, outcome
):
    """A stop while an approval is open is a refusal: whatever the bridge
    brings back, the stopped turn does not resume and leaves nothing parked."""
    no_config_rule(monkeypatch)
    agent = parked_agent()

    def _stopped(_reqs, human_budget_exhausted=False):
        display_mod.request_stream_cancel("tui-stop-2")
        return outcome

    try:
        turn = await _turn(
            monkeypatch,
            workspace,
            [[approval_interrupt(1), DONE], FINISHED],
            agent,
            channel_hitl_fn=_stopped,
            cancel_scope="tui-stop-2",
        )
    finally:
        display_mod.discard_stream_cancel("tui-stop-2")
    assert len(turn.gateway.requests) == 1
    assert _closed(agent) == [REJECTED]
    assert agent.next == ()


# -- Ctrl+C, through the real action (it cancels the turn's task) ------------------------------


async def test_ctrl_c_on_the_open_widget_is_a_rejection(monkeypatch, workspace):
    no_config_rule(monkeypatch)
    agent = parked_agent()
    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [[approval_interrupt(1), DONE], FINISHED],
        agent,
        hold=None,
        when=lambda app: app._approval_future is not None,
    )
    assert len(turn.gateway.requests) == 1
    assert _closed(agent) == [REJECTED]


async def test_ctrl_c_on_the_open_widget_does_not_cancel_the_turn(
    monkeypatch, workspace
):
    """Ctrl+C answers the open approval; the turn's task is not cancelled, so
    the close runs in a task whose cancellation still means what it says."""
    import EvoScientist.backends as backends_mod

    no_config_rule(monkeypatch)
    agent = parked_agent()
    cancelling: list[int] = []
    close = backends_mod.close_parked_checkpoint

    async def _watched(*args, **kwargs):
        cancelling.append(asyncio.current_task().cancelling())
        return await close(*args, **kwargs)

    monkeypatch.setattr(backends_mod, "close_parked_checkpoint", _watched)
    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [[approval_interrupt(1)], FINISHED],
        agent,
        hold=asyncio.Event(),  # never set: Ctrl+C alone has to end the turn
        when=lambda app: app._approval_future is not None,
    )
    assert cancelling == [0]
    assert turn.cancelled is False
    assert len(turn.gateway.requests) == 1
    assert _closed(agent) == [REJECTED]


@pytest.mark.parametrize(
    ("how", "expected"), [("refused", REJECTED), ("unanswered", UNANSWERED)]
)
async def test_ctrl_c_while_the_declined_widget_is_removed_still_closes(
    monkeypatch, workspace, how, expected
):
    """The decline is on record, with its own reason, before anything else is
    awaited: a Ctrl+C that lands while the widget is removed finds it, and
    the input the widget had switched off comes back."""
    from EvoScientist.cli.widgets.approval_widget import ApprovalWidget

    no_config_rule(monkeypatch)
    agent = parked_agent()
    removing: list[int] = []
    remove = ApprovalWidget.remove

    async def _slow_remove(self):
        remove(self)  # the removal itself is under way
        removing.append(1)
        await asyncio.Event().wait()

    monkeypatch.setattr(ApprovalWidget, "remove", _slow_remove)
    prepare = None
    if how == "refused":

        def prepare(app):
            widget_answers(monkeypatch, app, REJECTING)

    else:
        monkeypatch.setattr(tui_mod, "_APPROVAL_WIDGET_TIMEOUT", 0.05)
    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [[approval_interrupt(1), DONE], FINISHED],
        agent,
        hold=None,
        when=lambda _app: bool(removing),
        prepare=prepare,
    )
    assert len(turn.gateway.requests) == 1
    assert _closed(agent) == [expected]
    assert turn.input_off is False


async def test_two_ctrl_c_at_once_on_the_widget_leave_no_widget_behind(
    monkeypatch, workspace
):
    """Both keys are handled before the turn runs again: the first refused
    the approval and the second finds it answered, so it cancels nothing."""
    no_config_rule(monkeypatch)
    agent = parked_agent()
    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [[approval_interrupt(1), DONE], FINISHED],
        agent,
        hold=None,
        when=lambda app: app._approval_future is not None,
        again_when=lambda: True,
    )
    assert _closed(agent) == [REJECTED]
    assert turn.cancel_requests == 0
    assert turn.approvals_left == 0
    assert turn.input_off is False


async def test_approve_all_then_ctrl_c_at_once_grants_nothing(monkeypatch, workspace):
    """The choice and the key are handled before the turn runs again. The
    stop refuses the approval, so the session grant chosen with it is not kept."""
    no_config_rule(monkeypatch)
    agent = parked_agent()

    def _choose_then_press(app):
        if app._approval_future is None:
            return False
        app._approval_future.set_result(GRANTING)
        app.action_request_quit()
        return True

    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [[approval_interrupt(1), DONE], FINISHED],
        agent,
        hold=None,
        when=_choose_then_press,
        press=False,
    )
    assert len(turn.gateway.requests) == 1
    assert _closed(agent) == [REJECTED]
    assert turn.session_grant is False


async def test_ctrl_c_while_the_approval_widget_mounts_leaves_no_widget_behind(
    monkeypatch, workspace
):
    """The key lands before the wait for an answer has started, so it cancels
    the turn. The widget that was being mounted must not stay on screen."""
    from EvoScientist.cli.widgets.approval_widget import ApprovalWidget

    no_config_rule(monkeypatch)
    agent = parked_agent()
    pressed: list[int] = []
    on_mount = ApprovalWidget.on_mount

    def _pressed_at_mount(self):
        on_mount(self)
        pressed.append(1)
        self.app.action_request_quit()

    monkeypatch.setattr(ApprovalWidget, "on_mount", _pressed_at_mount)
    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [[approval_interrupt(1), DONE], FINISHED],
        agent,
        hold=None,
        when=lambda _app: bool(pressed),
        press=False,
    )
    assert turn.cancel_requests == 1
    assert _closed(agent) == [REJECTED]
    assert turn.approvals_left == 0
    assert turn.input_off is False


async def test_a_second_ctrl_c_while_the_stream_shuts_down_still_closes(
    monkeypatch, workspace
):
    """Ctrl+C on the widget ends the turn; the stream is slow to close and a
    second Ctrl+C lands there. The rows are settled and the decline is closed."""
    no_config_rule(monkeypatch)
    agent = parked_agent()
    closing: list[int] = []

    async def _slow_to_close(_request):
        try:
            yield {"type": "tool_call", "id": "call-1", "name": "execute", "args": {}}
            yield approval_interrupt(1)
            yield DONE
        finally:
            closing.append(1)
            await asyncio.Event().wait()

    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [],
        agent,
        hold=None,
        when=lambda app: app._approval_future is not None,
        again_when=lambda: bool(closing),
        stream=_slow_to_close,
    )
    assert _closed(agent) == [REJECTED]
    assert agent.next == ()
    assert turn.tools["call-1"][0] == "rejected"


async def test_ctrl_c_after_an_approval_while_the_stream_shuts_down_still_closes(
    monkeypatch, workspace
):
    """Approved on the widget, then Ctrl+C before the resume is sent: the
    command never ran and the approval the stop leaves is closed."""
    no_config_rule(monkeypatch)
    agent = parked_agent()
    closing: list[int] = []

    async def _slow_to_close(_request):
        try:
            yield approval_interrupt(1)
            yield DONE
        finally:
            closing.append(1)
            await asyncio.Event().wait()

    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [],
        agent,
        hold=None,
        when=lambda _app: bool(closing),
        prepare=lambda app: widget_answers(monkeypatch, app, APPROVING_ONCE),
        stream=_slow_to_close,
    )
    assert len(turn.gateway.requests) == 1
    assert _closed(agent) == [REJECTED]
    assert agent.next == ()


async def test_ctrl_c_while_a_channel_turns_stream_shuts_down_still_closes(
    monkeypatch, workspace
):
    """A cancellation that comes without the stop flag (the turn has its own
    stop scope; the app itself never cancels a channel turn). The channel had
    approved, the resume was not sent yet and the stream is slow to close:
    the approval the cancellation leaves is closed."""
    no_config_rule(monkeypatch)
    agent = parked_agent()
    closing: list[int] = []

    async def _slow_to_close(_request):
        try:
            yield approval_interrupt(1)
            yield DONE
        finally:
            closing.append(1)
            await asyncio.Event().wait()

    def _prepare(app):
        run = app._stream_with_widgets

        def _as_a_channel_turn(text):
            return run(
                text,
                cancel_scope="tui-channel-turn",
                channel_hitl_fn=lambda _reqs, human_budget_exhausted=False: (
                    ApprovalOutcome(decisions=APPROVE, prompted=True)
                ),
            )

        monkeypatch.setattr(app, "_stream_with_widgets", _as_a_channel_turn)

    try:
        turn = await ctrl_c_turn(
            monkeypatch,
            workspace,
            [],
            agent,
            hold=None,
            when=lambda _app: bool(closing),
            prepare=_prepare,
            stream=_slow_to_close,
        )
    finally:
        display_mod.discard_stream_cancel("tui-channel-turn")
    assert len(turn.gateway.requests) == 1
    assert _closed(agent) == [REJECTED]


async def test_ctrl_c_while_the_answer_is_finalized_still_closes(
    monkeypatch, workspace
):
    """The stream has ended after a decline; Ctrl+C lands in the last wait of
    the cleanup, after the rows were settled."""
    from EvoScientist.cli.widgets.assistant_message import AssistantMessage

    no_config_rule(monkeypatch)
    agent = parked_agent()
    finalizing: list[str] = []

    async def _slow_stop(_self):
        # Where the cleanup is when the answer's stream is stopped.
        finalizing.extend(
            frame.name
            for frame in traceback.extract_stack()
            if frame.name == "_preserve_active_narration"
        )
        finalizing.append("stop")
        await asyncio.Event().wait()

    monkeypatch.setattr(AssistantMessage, "stop_stream", _slow_stop)
    await ctrl_c_turn(
        monkeypatch,
        workspace,
        [
            # Text after the decline, and the stream ends without a final
            # event: the answer is still streaming when the cleanup starts.
            [approval_interrupt(1), {"type": "text", "content": "tail"}],
            FINISHED,
        ],
        agent,
        hold=None,
        when=lambda _app: bool(finalizing),
        prepare=lambda app: widget_answers(monkeypatch, app, REJECTING),
    )
    # The wait that was cut is the answer's own, not an earlier one.
    assert finalizing == ["stop"]
    assert _closed(agent) == [REJECTED]


async def test_ctrl_c_while_the_answer_is_finalized_after_an_approval_closes(
    monkeypatch, workspace
):
    """Approved, but Ctrl+C lands in the cleanup's last wait before the resume is
    sent: the stop leaves the approval open, so it is closed as a refusal."""
    from EvoScientist.cli.widgets.assistant_message import AssistantMessage

    no_config_rule(monkeypatch)
    agent = parked_agent()
    finalizing: list[int] = []

    async def _slow_stop(_self):
        finalizing.append(1)
        await asyncio.Event().wait()

    monkeypatch.setattr(AssistantMessage, "stop_stream", _slow_stop)
    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [[{"type": "text", "content": "head"}, approval_interrupt(1), DONE], FINISHED],
        agent,
        hold=None,
        when=lambda _app: bool(finalizing),
        prepare=lambda app: widget_answers(monkeypatch, app, APPROVING_ONCE),
    )
    assert len(turn.gateway.requests) == 1
    assert _closed(agent) == [REJECTED]


async def test_a_cancellation_in_the_last_step_after_a_channel_approval_closes(
    monkeypatch, workspace
):
    """A channel turn has its own stop scope, so only the cancellation itself
    says the turn was stopped while the approved resume was still unsent."""
    from EvoScientist.cli.widgets.assistant_message import AssistantMessage

    no_config_rule(monkeypatch)
    agent = parked_agent()
    finalizing: list[int] = []

    async def _slow_stop(_self):
        finalizing.append(1)
        await asyncio.Event().wait()

    monkeypatch.setattr(AssistantMessage, "stop_stream", _slow_stop)

    def _prepare(app):
        run = app._stream_with_widgets

        def _as_a_channel_turn(text):
            return run(
                text,
                cancel_scope="tui-channel-turn",
                channel_hitl_fn=lambda _reqs, human_budget_exhausted=False: (
                    ApprovalOutcome(decisions=APPROVE, prompted=True)
                ),
            )

        monkeypatch.setattr(app, "_stream_with_widgets", _as_a_channel_turn)

    try:
        turn = await ctrl_c_turn(
            monkeypatch,
            workspace,
            [
                [{"type": "text", "content": "head"}, approval_interrupt(1), DONE],
                FINISHED,
            ],
            agent,
            hold=None,
            when=lambda _app: bool(finalizing),
            prepare=_prepare,
        )
    finally:
        display_mod.discard_stream_cancel("tui-channel-turn")
    assert len(turn.gateway.requests) == 1
    assert _closed(agent) == [REJECTED]


async def test_a_stream_that_fails_to_shut_down_after_a_decline_is_still_closed(
    monkeypatch, workspace
):
    """Not a cancellation: the stream's own shutdown raises. The error goes
    on its way, and the decline is closed before it does."""
    no_config_rule(monkeypatch)
    agent = parked_agent()

    async def _breaks_on_close(_request):
        try:
            yield {"type": "tool_call", "id": "call-1", "name": "execute", "args": {}}
            yield approval_interrupt(1)
            yield DONE
        finally:
            raise RuntimeError("stream shutdown failed")

    with pytest.raises(RuntimeError, match="stream shutdown failed"):
        await ctrl_c_turn(
            monkeypatch,
            workspace,
            [],
            agent,
            hold=None,
            when=lambda app: app._approval_future is not None,
            stream=_breaks_on_close,
        )
    assert _closed(agent) == [REJECTED]
    assert agent.next == ()


async def test_an_answer_that_fails_to_finalize_after_a_decline_is_still_closed(
    monkeypatch, workspace
):
    from EvoScientist.cli.widgets.assistant_message import AssistantMessage

    no_config_rule(monkeypatch)
    agent = parked_agent()

    async def _fails(_self):
        raise RuntimeError("cannot finalize")

    monkeypatch.setattr(AssistantMessage, "stop_stream", _fails)
    with pytest.raises(RuntimeError, match="cannot finalize"):
        await _turn(
            monkeypatch,
            workspace,
            [[approval_interrupt(1), {"type": "text", "content": "tail"}], FINISHED],
            agent,
            prepare=lambda app: widget_answers(monkeypatch, app, REJECTING),
        )
    assert _closed(agent) == [REJECTED]


async def test_ctrl_c_after_a_decline_while_sibling_work_runs_still_closes(
    monkeypatch, workspace
):
    """The user declines, sibling work keeps the stream open, then Ctrl+C cancels
    the turn's task: the decline is still recorded."""
    no_config_rule(monkeypatch)
    agent = parked_agent()
    declined: list[int] = []

    def _prepare(app):
        async def _reject(_widget):
            declined.append(1)
            return REJECTING

        monkeypatch.setattr(app, "_wait_for_approval", _reject)

    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [
            [
                *FIVE_FINISHED_TOOLS,
                {"type": "tool_call", "id": "call-1", "name": "execute", "args": {}},
                approval_interrupt(1),
            ],
            FINISHED,
        ],
        agent,
        hold=asyncio.Event(),  # never set: the sibling is still working
        when=lambda _app: bool(declined),
        prepare=_prepare,
    )
    assert len(turn.gateway.requests) == 1
    assert _closed(agent) == [REJECTED]
    assert agent.next == ()
    assert turn.tools["call-1"][0] == "rejected"
    assert len(turn.tools) == 6
    assert all(shown for _status, shown in turn.tools.values())


async def test_ctrl_c_without_a_decline_closes_nothing(monkeypatch, workspace):
    """A plain stop of a running turn writes nothing to the thread."""
    agent = parked_agent()
    turn = await ctrl_c_turn(
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
    no_config_rule(monkeypatch)
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
        turn = await ctrl_c_turn(
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
                    approval_interrupt(1),
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
    no_config_rule(monkeypatch)
    call = {"name": "execute", "args": {}, "id": "call-1", "type": "tool_call"}
    agent = _NeverFinishesClosing(
        next=BEFORE_TOOLS,
        values={"messages": [AIMessage(content="", tool_calls=[call])]},
    )
    with caplog.at_level(logging.WARNING, logger="EvoScientist"):
        turn = await ctrl_c_turn(
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
                    approval_interrupt(1),
                    DONE,
                ],
                FINISHED,
            ],
            agent,
            hold=None,
            when=lambda _app: agent.closing,
            prepare=lambda app: widget_answers(monkeypatch, app, REJECTING),
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


async def _open_wait(app):
    """A wait for an approval, started as the app's turn task."""
    wait = asyncio.ensure_future(app._wait_for_approval(object()))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if app._approval_future is not None:
            break
    app._run_task = wait
    app._busy = True
    return wait


async def test_the_timeout_mark_does_not_leak_into_the_next_wait(
    monkeypatch, workspace
):
    """A wait that ran out, then a wait that Ctrl+C ends: the second is not a timeout."""
    app = await capture_app(monkeypatch, workspace)
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            monkeypatch.setattr(tui_mod, "_APPROVAL_WIDGET_TIMEOUT", 0.05)
            assert await app._wait_for_approval(object()) is None
            assert app._approval_timed_out is True

            monkeypatch.setattr(tui_mod, "_APPROVAL_WIDGET_TIMEOUT", 30.0)
            wait = await _open_wait(app)
            app.action_request_quit()
            assert await wait is None
            assert app._approval_timed_out is False
            assert wait.cancelled() is False
    finally:
        display_mod.discard_stream_cancel()


async def test_a_cancelled_wait_is_a_cancellation_not_an_answer(monkeypatch, workspace):
    """Only Ctrl+C answers the open approval; cancelling the task still cancels it."""
    app = await capture_app(monkeypatch, workspace)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        wait = await _open_wait(app)
        wait.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wait
        assert app._approval_future is None
        app._run_task = None
        app._busy = False


async def test_ctrl_c_without_an_open_approval_cancels_the_turn(monkeypatch, workspace):
    app = await capture_app(monkeypatch, workspace)
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            turn = asyncio.ensure_future(asyncio.Event().wait())
            app._run_task = turn
            app._busy = True
            app.action_request_quit()
            with pytest.raises(asyncio.CancelledError):
                await turn
            app._run_task = None
            app._busy = False
    finally:
        display_mod.discard_stream_cancel()
