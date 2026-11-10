"""Calls inside a sub-agent's box around an approval: what their rows say."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("textual")

from textual.app import App
from textual.widgets import Static

import EvoScientist.channels.hitl_budget as budget_mod
import EvoScientist.cli.tui_interactive as tui_mod
from EvoScientist.channels.interaction import ApprovalOutcome
from EvoScientist.cli.widgets.subagent_widget import SubAgentWidget
from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget
from EvoScientist.gateway import RuntimeGateways
from EvoScientist.stream import display as display_mod
from tests.fakes.common import FakeGraphGateway, no_config_rule
from tests.fakes.tui import (
    APPROVE,
    APPROVING_ONCE,
    DONE,
    FINISHED,
    REJECTING,
    TAIL,
    approval_interrupt,
    capture_app,
    ctrl_c_turn,
    parked_agent,
    scripted_stream,
    widget_answers,
)

REJECTED = ("rejected", "■ rejected", False)
INTERRUPTED = ("interrupted", "■ interrupted", False)


def _call(n: int, call_id: str | None = None) -> dict:
    return {
        "type": "subagent_tool_call",
        "subagent": "code-agent",
        "name": "execute",
        "args": {"command": f"echo {n}"},
        "id": call_id or f"sub-{n}",
        "instance_id": f"sa-{n}",
    }


def _sub_agent(n: int) -> list[dict]:
    """A sub-agent is announced and asks to run one command. A resumed round
    sends the same two events for a sub-agent that goes on."""
    start = {
        "type": "subagent_start",
        "name": "code-agent",
        "description": "",
        "instance_id": f"sa-{n}",
        "tool_call_id": f"task-{n}",
    }
    return [start, _call(n)]


def _result(n: int, call_id: str | None = None) -> dict:
    return {
        "type": "subagent_tool_result",
        "subagent": "code-agent",
        "name": "execute",
        "content": "ok",
        "success": True,
        "id": call_id or f"sub-{n}",
        "instance_id": f"sa-{n}",
    }


def _end(n: int, *, paused: bool = False) -> dict:
    event = {"type": "subagent_end", "name": "code-agent", "instance_id": f"sa-{n}"}
    if paused:
        event["paused"] = True
    return event


def _two_parked(order: str) -> list[dict]:
    """Two sub-agents each stop at an approval. The stream reports both as
    paused when the run pauses: after the approvals, or before them when the
    decoder's two producers are out of step."""
    asked = [*_sub_agent(1), _call(1, "sub-1b"), *_sub_agent(2)]
    approvals = [approval_interrupt(1), approval_interrupt(2)]
    ends = [_end(1, paused=True), _end(2, paused=True)]
    if order == "ends first":
        return [*asked, *ends, *approvals, DONE]
    return [*asked, *approvals, *ends, DONE]


def _row(widget: ToolCallWidget) -> tuple[str, str, bool]:
    """A row's state, what its status line shows, and whether its timer runs."""
    shown = str(widget.query_one(".tool-status", Static).render())
    return widget._status, shown, widget._timer_handle is not None


def _rows(app) -> dict[str, tuple[str, str, bool]]:
    return {widget._tool_id: _row(widget) for widget in app.query(ToolCallWidget)}


async def _started(monkeypatch, workspace, stream, agent):
    """The app with a scripted gateway; the caller runs the turn inside it."""
    app = await capture_app(monkeypatch, workspace)
    gateway = FakeGraphGateway(stream=stream, checkpoint=agent)
    return app, gateway


async def _turn(monkeypatch, workspace, script, agent, *, prepare=None, **kwargs):
    """One turn through the real TUI loop: the rows, the boxes still open and
    the number of stream calls."""
    app, gateway = await _started(
        monkeypatch, workspace, scripted_stream(script), agent
    )
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
        await app._stream_with_widgets("hello", **kwargs)
        rows = _rows(app)
        still_open = [box for box in app.query(SubAgentWidget) if box._is_active]
    return rows, still_open, len(gateway.requests)


@pytest.mark.parametrize("order", ["ends last", "ends first"])
@pytest.mark.parametrize("how", ["refused", "unanswered"])
async def test_a_declined_turn_leaves_sub_agent_calls_rejected(
    monkeypatch, workspace, how, order
):
    """Two sub-agents each ask and the first approval is declined: neither
    command ran, and both rows say so, as the main agent's own row does."""
    no_config_rule(monkeypatch)
    prepare = None
    if how == "refused":

        def prepare(app):
            widget_answers(monkeypatch, app, REJECTING)

    else:
        monkeypatch.setattr(tui_mod, "_APPROVAL_WIDGET_TIMEOUT", 0.05)
    rows, still_open, calls = await _turn(
        monkeypatch,
        workspace,
        [_two_parked(order), FINISHED],
        parked_agent(name="task"),
        prepare=prepare,
    )
    assert calls == 1
    assert rows == {"sub-1": REJECTED, "sub-1b": REJECTED, "sub-2": REJECTED}
    assert still_open == []


async def test_ends_that_are_not_marked_as_paused_read_rejected_after_a_decline(
    monkeypatch, workspace
):
    """The server gateway announces every open sub-agent as ended when its
    stream stops, without saying which ones only paused."""
    no_config_rule(monkeypatch)
    script = [
        *_sub_agent(1),
        *_sub_agent(2),
        approval_interrupt(1),
        _end(1),
        _end(2),
        DONE,
    ]
    rows, still_open, _calls = await _turn(
        monkeypatch,
        workspace,
        [script, FINISHED],
        parked_agent(name="task"),
        prepare=lambda app: widget_answers(monkeypatch, app, REJECTING),
    )
    assert rows == {"sub-1": REJECTED, "sub-2": REJECTED}
    assert still_open == []


async def test_a_turn_stopped_at_the_round_limit_leaves_them_rejected(
    monkeypatch, workspace
):
    """The limit ends the turn at the approval itself, before any end event."""
    no_config_rule(monkeypatch)
    rows, still_open, calls = await _turn(
        monkeypatch,
        workspace,
        [_two_parked("ends last"), FINISHED],
        parked_agent(name="task"),
        channel_hitl_fn=lambda _reqs, human_budget_exhausted=False: ApprovalOutcome(
            budget_exhausted=True
        ),
    )
    assert calls == 1
    assert set(rows.values()) == {REJECTED}
    assert still_open == []


async def test_the_round_cap_after_an_automatic_answer_settles_the_box(
    monkeypatch, workspace
):
    """A channel's grant answers by itself and the cap on rounds then ends
    the turn without another stream call: nothing may be left spinning."""
    no_config_rule(monkeypatch)
    monkeypatch.setattr(budget_mod, "MAX_HITL_TOTAL_ROUNDS", 1)
    rows, still_open, calls = await _turn(
        monkeypatch,
        workspace,
        [[*_sub_agent(1), approval_interrupt(1)], FINISHED],
        parked_agent(name="task"),
        channel_hitl_fn=lambda _reqs, human_budget_exhausted=False: ApprovalOutcome(
            decisions=list(APPROVE), session_granted=True, grant_in_force=True
        ),
    )
    assert calls == 1
    assert rows == {"sub-1": INTERRUPTED}
    assert still_open == []


async def test_a_stop_after_the_decline_still_reads_rejected(monkeypatch, workspace):
    """The decline is what ended the turn; the stop came later."""
    no_config_rule(monkeypatch)

    def _prepare(app):
        async def _reject_then_stop(_widget):
            display_mod.request_stream_cancel("tui-stop-declined")
            return REJECTING

        monkeypatch.setattr(app, "_wait_for_approval", _reject_then_stop)

    try:
        rows, still_open, _calls = await _turn(
            monkeypatch,
            workspace,
            [[*_sub_agent(1), approval_interrupt(1), DONE], FINISHED],
            parked_agent(name="task"),
            prepare=_prepare,
            cancel_scope="tui-stop-declined",
        )
    finally:
        display_mod.discard_stream_cancel("tui-stop-declined")
    assert rows == {"sub-1": REJECTED}
    assert still_open == []


@pytest.mark.parametrize("how", ["refused", "unanswered"])
async def test_a_decline_without_end_events_is_settled_when_the_turn_ends(
    monkeypatch, workspace, how
):
    """One approval and nothing after it: only the decline says the turn
    ended at a pause."""
    no_config_rule(monkeypatch)
    prepare = None
    if how == "refused":

        def prepare(app):
            widget_answers(monkeypatch, app, REJECTING)

    else:
        monkeypatch.setattr(tui_mod, "_APPROVAL_WIDGET_TIMEOUT", 0.05)
    rows, still_open, _calls = await _turn(
        monkeypatch,
        workspace,
        [[*_sub_agent(1), approval_interrupt(1), DONE], FINISHED],
        parked_agent(name="task"),
        prepare=prepare,
    )
    assert rows == {"sub-1": REJECTED}
    assert still_open == []


async def test_a_pause_that_cannot_be_answered_leaves_them_rejected(
    monkeypatch, workspace
):
    """A question without questions ends the turn at its pause."""
    question = {"type": "ask_user", "interrupt_id": "q-1", "questions": []}
    rows, still_open, calls = await _turn(
        monkeypatch,
        workspace,
        [[*_sub_agent(1), question, DONE], FINISHED],
        parked_agent(name="ask_user", next=("tools",)),
    )
    assert calls == 1
    assert rows == {"sub-1": REJECTED}
    assert still_open == []


async def test_a_sibling_that_finishes_after_the_decline_keeps_its_result(
    monkeypatch, workspace
):
    no_config_rule(monkeypatch)
    script = [
        *_sub_agent(1),
        *_sub_agent(2),
        approval_interrupt(1),
        _result(2),
        _end(2),
        _end(1, paused=True),
        DONE,
    ]
    rows, still_open, _calls = await _turn(
        monkeypatch,
        workspace,
        [script, FINISHED],
        parked_agent(name="task"),
        prepare=lambda app: widget_answers(monkeypatch, app, REJECTING),
    )
    assert rows["sub-1"] == REJECTED
    assert rows["sub-2"][0] == "success"
    assert still_open == []


async def test_a_sub_agent_that_ends_with_an_open_call_reads_interrupted(
    monkeypatch, workspace
):
    """Nothing was declined: a call left without a result was cut, not refused."""
    no_config_rule(monkeypatch)
    rows, still_open, _calls = await _turn(
        monkeypatch,
        workspace,
        [[*_sub_agent(1), _end(1), *TAIL]],
        parked_agent(name="task"),
    )
    assert rows == {"sub-1": INTERRUPTED}
    assert still_open == []


async def test_a_box_with_nothing_open_is_closed_when_the_round_ends(
    monkeypatch, workspace
):
    """No end event came for it (the stream was left at an approval): its
    calls all have their results and the box must not keep spinning."""
    no_config_rule(monkeypatch)
    rows, still_open, _calls = await _turn(
        monkeypatch,
        workspace,
        [[*_sub_agent(1), _result(1), *TAIL]],
        parked_agent(name="task"),
    )
    assert rows["sub-1"][0] == "success"
    assert still_open == []


async def test_ctrl_c_after_a_decline_marks_only_the_declined_work_rejected(
    monkeypatch, workspace
):
    """One sub-agent's command is declined while another still works and the
    main agent has a call of its own running; then Ctrl+C. What the decline
    stopped reads rejected, what Ctrl+C cut reads interrupted."""
    no_config_rule(monkeypatch)
    agent = parked_agent(name="task")
    declined: list[int] = []

    def _prepare(app):
        async def _reject(_widget):
            declined.append(1)
            return REJECTING

        monkeypatch.setattr(app, "_wait_for_approval", _reject)

    own_call = {"type": "tool_call", "id": "own-1", "name": "read_file", "args": {}}
    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [
            [
                *_sub_agent(1),
                *_sub_agent(2),
                own_call,
                _end(1, paused=True),
                approval_interrupt(1),
            ],
            FINISHED,
        ],
        agent,
        hold=asyncio.Event(),  # never set: the second sub-agent is still working
        when=lambda _app: bool(declined),
        prepare=_prepare,
    )
    assert {call: status for call, (status, _shown) in turn.tools.items()} == {
        "sub-1": "rejected",
        "sub-2": "interrupted",
        "own-1": "interrupted",
    }
    assert agent.next == ()


@pytest.mark.parametrize("order", ["end before approval", "approval before end"])
async def test_ctrl_c_on_the_widget_marks_only_the_asking_box_rejected(
    monkeypatch, workspace, order
):
    """Ctrl+C on the open widget refuses and stops in one step. The asking
    sub-agent's paused end usually comes after its approval and is then never
    read: the box is known by the call the approval asks about."""
    no_config_rule(monkeypatch)
    agent = parked_agent(name="task")
    asked = [approval_interrupt(1), _end(1, paused=True)]
    if order == "end before approval":
        asked.reverse()
    own_call = {"type": "tool_call", "id": "own-1", "name": "read_file", "args": {}}
    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [[*_sub_agent(1), *_sub_agent(2), own_call, *asked], FINISHED],
        agent,
        hold=asyncio.Event(),  # never set: the second sub-agent is still working
        when=lambda app: app._approval_future is not None,
    )
    assert {call: status for call, (status, _shown) in turn.tools.items()} == {
        "sub-1": "rejected",
        "sub-2": "interrupted",
        "own-1": "interrupted",
    }
    assert agent.next == ()


@pytest.mark.parametrize(
    ("other", "reads"),
    [
        ([_result(2, "sub-2a")], "success"),
        ([], "interrupted"),
    ],
    ids=["finished with the same command", "another tool with the same arguments"],
)
async def test_only_an_open_call_of_the_asked_tool_marks_its_box(
    monkeypatch, workspace, other, reads
):
    """The second sub-agent works on. A call of its own that already has its
    result, or that is another tool's, is not the one the approval names."""
    no_config_rule(monkeypatch)
    look_alike = {**_call(2, "sub-2a"), "args": {"command": "echo 1"}}
    if not other:
        look_alike["name"] = "run_in_background"
    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [
            [
                *_sub_agent(1),
                _sub_agent(2)[0],
                look_alike,
                *other,
                _call(2),
                approval_interrupt(1),
            ],
            FINISHED,
        ],
        parked_agent(name="task"),
        hold=asyncio.Event(),  # never set: the second sub-agent is still working
        when=lambda app: app._approval_future is not None,
    )
    assert {call: status for call, (status, _shown) in turn.tools.items()} == {
        "sub-1": "rejected",
        "sub-2a": reads,
        "sub-2": "interrupted",
    }


async def test_two_boxes_asking_for_the_same_command_are_both_at_a_pause(
    monkeypatch, workspace
):
    no_config_rule(monkeypatch)
    same = {**_call(2), "args": {"command": "echo 1"}}
    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [[*_sub_agent(1), _sub_agent(2)[0], same, approval_interrupt(1)], FINISHED],
        parked_agent(name="task"),
        hold=asyncio.Event(),
        when=lambda app: app._approval_future is not None,
    )
    assert {call: status for call, (status, _shown) in turn.tools.items()} == {
        "sub-1": "rejected",
        "sub-2": "rejected",
    }


async def test_an_asker_nobody_was_shown_is_known_by_its_paused_end(
    monkeypatch, workspace
):
    """Two sub-agents ask; both paused ends are read before the first
    approval, and Ctrl+C lands on it. The second asker's approval was never
    read, so only its paused end says it is at a pause."""
    no_config_rule(monkeypatch)
    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [
            [
                *_sub_agent(1),
                *_sub_agent(2),
                _end(1, paused=True),
                _end(2, paused=True),
                approval_interrupt(1),
            ],
            FINISHED,
        ],
        parked_agent(name="task"),
        hold=asyncio.Event(),
        when=lambda app: app._approval_future is not None,
    )
    assert {call: status for call, (status, _shown) in turn.tools.items()} == {
        "sub-1": "rejected",
        "sub-2": "rejected",
    }


async def test_an_asker_after_the_decline_is_at_a_pause_too(monkeypatch, workspace):
    """The first approval is declined; the second sub-agent's approval comes
    after it and is not shown, and no paused end follows. Ctrl+C then cuts
    the main agent's own call, not the second asker."""
    no_config_rule(monkeypatch)
    own_call = {"type": "tool_call", "id": "own-1", "name": "read_file", "args": {}}

    def _own_call_is_up(app):
        return any(w._tool_id == "own-1" for w in app.query(ToolCallWidget))

    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [
            [
                *_sub_agent(1),
                *_sub_agent(2),
                approval_interrupt(1),
                approval_interrupt(2),
                own_call,
            ],
            FINISHED,
        ],
        parked_agent(name="task"),
        hold=asyncio.Event(),
        when=_own_call_is_up,
        prepare=lambda app: widget_answers(monkeypatch, app, REJECTING),
    )
    assert {call: status for call, (status, _shown) in turn.tools.items()} == {
        "sub-1": "rejected",
        "sub-2": "rejected",
        "own-1": "interrupted",
    }


@pytest.mark.parametrize("marked_by", ["the approval", "a paused end"])
async def test_an_asker_of_an_earlier_round_does_not_count_in_a_later_one(
    monkeypatch, workspace, marked_by
):
    """A sub-agent's command was approved; in the next round the main agent
    asks about a command of its own and Ctrl+C lands on that approval. Nothing
    ran beside it, so the main agent's call reads rejected."""
    no_config_rule(monkeypatch)
    own_call = {
        "type": "tool_call",
        "id": "own-1",
        "name": "execute",
        "args": {"command": "echo root"},
    }
    own_approval = {
        "type": "interrupt",
        "interrupt_id": "approval-root",
        "action_requests": [{"name": "execute", "args": {"command": "echo root"}}],
    }
    first = [*_sub_agent(1), approval_interrupt(1)]
    if marked_by == "a paused end":
        first.insert(2, _end(1, paused=True))

    def _prepare(app):
        wait, answered = app._wait_for_approval, []

        async def _approve_the_first(widget):
            if not answered:
                answered.append(1)
                return APPROVING_ONCE
            return await wait(widget)

        monkeypatch.setattr(app, "_wait_for_approval", _approve_the_first)

    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [first, [own_call, own_approval], FINISHED],
        parked_agent(),
        hold=None,
        when=lambda app: app._approval_future is not None,
        prepare=_prepare,
    )
    assert turn.tools["own-1"][0] == "rejected"


async def test_a_plain_decline_beside_a_paused_box_rejects_both(monkeypatch, workspace):
    """No stop: the stream ended with the second box still open, and the
    decline is what ended the turn for both."""
    no_config_rule(monkeypatch)
    rows, still_open, _calls = await _turn(
        monkeypatch,
        workspace,
        [
            [
                *_sub_agent(1),
                *_sub_agent(2),
                _end(1, paused=True),
                approval_interrupt(1),
                DONE,
            ],
            FINISHED,
        ],
        parked_agent(name="task"),
        prepare=lambda app: widget_answers(monkeypatch, app, REJECTING),
    )
    assert rows == {"sub-1": REJECTED, "sub-2": REJECTED}
    assert still_open == []


async def test_a_sub_agent_that_went_on_after_its_approval_is_cut_not_declined(
    monkeypatch, workspace
):
    """The first sub-agent's command was approved and it works on; the second
    one's is declined, then Ctrl+C. Only the second was stopped by the decline."""
    no_config_rule(monkeypatch)
    agent = parked_agent(name="task")
    declined: list[int] = []
    calls = 0

    async def _two_rounds(_request):
        nonlocal calls
        calls += 1
        if calls == 1:
            for event in (*_sub_agent(1), _end(1, paused=True), approval_interrupt(1)):
                yield event
            return
        for event in (
            *_sub_agent(1),
            *_sub_agent(2),
            _end(2, paused=True),
            approval_interrupt(2),
        ):
            yield event
        await asyncio.Event().wait()  # the first sub-agent is still working

    def _prepare(app):
        answers = iter([APPROVING_ONCE, REJECTING])

        async def _answer(_widget):
            answer = next(answers)
            if answer is REJECTING:
                declined.append(1)
            return answer

        monkeypatch.setattr(app, "_wait_for_approval", _answer)

    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [],
        agent,
        hold=None,
        when=lambda _app: bool(declined),
        prepare=_prepare,
        stream=_two_rounds,
    )
    assert {call: status for call, (status, _shown) in turn.tools.items()} == {
        "sub-1": "interrupted",
        "sub-2": "rejected",
    }


async def test_a_stopped_turn_leaves_sub_agent_calls_interrupted(
    monkeypatch, workspace
):
    no_config_rule(monkeypatch)
    agent = parked_agent(name="task")
    turn = await ctrl_c_turn(
        monkeypatch,
        workspace,
        [[*_sub_agent(1)], FINISHED],
        agent,
        hold=asyncio.Event(),  # never set: the sub-agent is still working
        when=lambda app: bool(app.query(ToolCallWidget)),
    )
    assert turn.tools == {"sub-1": ("interrupted", True)}
    assert agent.updates == []


# -- An approved command -----------------------------------------------------------------------

# Where a scripted stream call waits: for the command that is running.
RUNNING = object()


def _stream_with_waits(script: list[list], gates: list[asyncio.Event]):
    """One list of events per stream call; at ``RUNNING`` the call waits for
    the next gate, as a stream does while an approved command runs."""
    calls = 0
    waits = iter(gates)

    async def _fake_stream(_request):
        nonlocal calls
        events = script[calls]
        calls += 1
        for event in events:
            if event is RUNNING:
                await next(waits).wait()
            else:
                yield event

    return _fake_stream


async def _until(pilot, condition) -> None:
    for _ in range(200):
        await pilot.pause()
        if condition():
            return
    raise AssertionError("the turn never reached the expected point")


class _Running:
    """A turn that is driven step by step: the test opens the gates."""

    def __init__(self, monkeypatch, workspace, script, answers, gates):
        self.monkeypatch, self.workspace = monkeypatch, workspace
        self.script, self.answers, self.gates = script, answers, gates

    async def __aenter__(self):
        self.app, self.gateway = await _started(
            self.monkeypatch,
            self.workspace,
            _stream_with_waits(self.script, self.gates),
            parked_agent(name="task"),
        )
        self._test = self.app.run_test(size=(80, 24))
        self.pilot = await self._test.__aenter__()
        await self.pilot.pause()
        self.app._runtime_gateways = RuntimeGateways(
            thread_store=self.gateway.thread_store, graph_gateway=self.gateway
        )
        self.monkeypatch.setattr(
            self.app, "_await_agent_ready", AsyncMock(return_value=MagicMock())
        )
        widget_answers(self.monkeypatch, self.app, *self.answers)
        self.turn = asyncio.create_task(self.app._stream_with_widgets("hello"))
        return self

    async def reached_call(self, number: int) -> None:
        """The stream call ``number`` is open and its first events are drawn."""
        await _until(self.pilot, lambda: len(self.gateway.requests) == number)
        for _ in range(3):
            await self.pilot.pause()

    def boxes(self) -> dict[str, bool]:
        """Which boxes are open. An open box has its spinner's timer running,
        a closed one has none."""
        found = {}
        for n, box in enumerate(self.app.query(SubAgentWidget), start=1):
            assert (box._timer_handle is not None) is box._is_active
            found[f"sub-{n}"] = box._is_active
        return found

    async def finish(self) -> None:
        for gate in self.gates:
            gate.set()
        await asyncio.wait_for(self.turn, timeout=10)

    async def __aexit__(self, *exc_info):
        for gate in self.gates:
            gate.set()
        await asyncio.gather(self.turn, return_exceptions=True)
        await self._test.__aexit__(*exc_info)


@pytest.mark.parametrize("order", ["approval first", "end first"])
async def test_an_approved_sub_agent_command_reads_running_until_its_result(
    monkeypatch, workspace, order
):
    """The turn goes on after the approval: the resumed round announces the
    sub-agent again, and its row runs while the command does."""
    no_config_rule(monkeypatch)
    asked = [*_sub_agent(1), approval_interrupt(1)]
    if order == "end first":
        asked = [*_sub_agent(1), _end(1, paused=True), approval_interrupt(1)]
    script = [asked, [*_sub_agent(1), RUNNING, _result(1), _end(1), *TAIL]]
    gate = asyncio.Event()
    async with _Running(
        monkeypatch, workspace, script, [APPROVING_ONCE], [gate]
    ) as run:
        await run.reached_call(2)
        state, shown, ticking = _rows(run.app)["sub-1"]
        assert (state, ticking) == ("running", True)
        assert "Running..." in shown
        assert run.boxes() == {"sub-1": True}
        await run.finish()
        state, _shown, ticking = _rows(run.app)["sub-1"]
        assert (state, ticking) == ("success", False)
        assert run.boxes() == {"sub-1": False}


async def test_re_opening_a_box_leaves_its_finished_calls_alone(monkeypatch, workspace):
    no_config_rule(monkeypatch)
    asked = [*_sub_agent(1), _result(1), _call(1, "second"), approval_interrupt(1)]
    script = [asked, [*_sub_agent(1), _call(1, "second"), RUNNING, *TAIL]]
    gate = asyncio.Event()
    async with _Running(
        monkeypatch, workspace, script, [APPROVING_ONCE], [gate]
    ) as run:
        await run.reached_call(2)
        rows = _rows(run.app)
        assert rows["sub-1"][0] == "success"
        assert rows["second"][0] == "running"
        await run.finish()


class _Host(App):
    """A sub-agent box on its own."""

    def compose(self):
        yield SubAgentWidget("code-agent")


def _box_shows(box: SubAgentWidget) -> tuple[str, str, bool, bool]:
    """Header, footer, the finished look, and whether the spinner's timer runs."""
    header = str(box.query_one(".sa-header", Static).render())
    footer = str(box.query_one(".sa-footer", Static).render())
    return header, footer, box.has_class("--completed"), box._timer_handle is not None


async def test_a_box_is_drawn_closed_then_open_again_then_closed():
    app = _Host()
    async with app.run_test() as pilot:
        box = app.query_one(SubAgentWidget)
        await box.add_tool_call("execute", {"command": "echo 1"}, "first")
        await box.add_tool_call("execute", {"command": "echo 2"}, "second")
        await pilot.pause()

        box.finalize()
        # A second close must not lose what a re-open puts back.
        box.finalize()
        header, footer, finished, ticking = _box_shows(box)
        assert header.startswith("\u2713 Cooking with code-agent")
        assert (footer, finished, ticking) == ("", True, False)
        assert _rows(app) == {"first": INTERRUPTED, "second": INTERRUPTED}

        box.reopen()
        await pilot.pause()
        header, footer, finished, ticking = _box_shows(box)
        assert header.startswith("\u250c \u25b6 Cooking with code-agent")
        assert (footer, finished, ticking) == ("\u2514 running...", False, True)
        assert [state for state, _shown, _ticking in _rows(app).values()] == [
            "running",
            "running",
        ]

        box.finalize(rejected=True)
        header, footer, finished, ticking = _box_shows(box)
        assert header.startswith("\u2713 Cooking with code-agent")
        assert (footer, finished, ticking) == ("", True, False)
        assert _rows(app) == {"first": REJECTED, "second": REJECTED}


async def test_a_discarded_call_is_not_put_back_when_the_box_re_opens():
    """A retried model call can drop a call the box had already cut."""
    app = _Host()
    async with app.run_test() as pilot:
        box = app.query_one(SubAgentWidget)
        await box.add_tool_call("execute", {"command": "echo 1"}, "dropped")
        await box.add_tool_call("execute", {"command": "echo 2"}, "kept")
        await pilot.pause()
        box.finalize()
        await box.discard_tool_calls(["dropped"])
        box.reopen()
        await pilot.pause()
        assert {w._tool_id: w._status for w in app.query(ToolCallWidget)} == {
            "kept": "running"
        }
        assert box._is_active
        assert box._cut == []


async def test_re_opening_a_box_leaves_a_call_that_got_its_result_meanwhile():
    """A result can still arrive for a call after its box was closed."""
    app = _Host()
    async with app.run_test() as pilot:
        box = app.query_one(SubAgentWidget)
        await box.add_tool_call("execute", {"command": "echo 1"}, "first")
        await box.add_tool_call("execute", {"command": "echo 2"}, "second")
        await pilot.pause()
        box.finalize()
        box.complete_tool("execute", "ok", True, "first")
        box.reopen()
        await pilot.pause()
        assert {w._tool_id: w._status for w in app.query(ToolCallWidget)} == {
            "first": "success",
            "second": "running",
        }


async def test_a_sub_agent_that_is_not_announced_again_does_not_keep_spinning(
    monkeypatch, workspace
):
    """Both sub-agents ask and both are approved. The first command's last
    events arrive while the second approval is open and are dropped with the
    rest of that stream call, and the next call says nothing about it."""
    no_config_rule(monkeypatch)
    script = [
        [*_sub_agent(1), *_sub_agent(2), approval_interrupt(1)],
        [*_sub_agent(1), *_sub_agent(2), approval_interrupt(2), _result(1), _end(1)],
        [*_sub_agent(2), RUNNING, _result(2), _end(2), *TAIL],
    ]
    gate = asyncio.Event()
    async with _Running(
        monkeypatch, workspace, script, [APPROVING_ONCE, APPROVING_ONCE], [gate]
    ) as run:
        await run.reached_call(3)
        rows = _rows(run.app)
        assert rows["sub-1"] == INTERRUPTED
        assert rows["sub-2"][0] == "running"
        assert run.boxes() == {"sub-1": False, "sub-2": True}
        await run.finish()
        assert _rows(run.app)["sub-2"][0] == "success"
        assert run.boxes() == {"sub-1": False, "sub-2": False}


async def test_a_stop_right_after_the_approval_still_settles_the_box(
    monkeypatch, workspace
):
    """A stop requested through the flag ends the turn before the next stream
    call, so nothing would re-open or close the box afterwards."""
    no_config_rule(monkeypatch)
    app, gateway = await _started(
        monkeypatch,
        workspace,
        scripted_stream([[*_sub_agent(1), approval_interrupt(1)], FINISHED]),
        parked_agent(name="task"),
    )

    async def _approve_then_stop(_widget):
        display_mod.request_stream_cancel("tui-stop-box")
        return APPROVING_ONCE

    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app._runtime_gateways = RuntimeGateways(
                thread_store=gateway.thread_store, graph_gateway=gateway
            )
            monkeypatch.setattr(
                app, "_await_agent_ready", AsyncMock(return_value=MagicMock())
            )
            monkeypatch.setattr(app, "_wait_for_approval", _approve_then_stop)
            await asyncio.wait_for(
                app._stream_with_widgets("hello", cancel_scope="tui-stop-box"),
                timeout=10,
            )
            assert len(gateway.requests) == 1
            # The stop refused the approval it left open: nothing ran.
            assert _rows(app) == {"sub-1": REJECTED}
            assert not app.query_one(SubAgentWidget)._is_active
    finally:
        display_mod.discard_stream_cancel("tui-stop-box")
