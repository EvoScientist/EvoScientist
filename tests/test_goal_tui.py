"""``/goal`` in the real TUI (Textual pilot) on the real main graph, and the
queue rule that lets a slash command typed mid-turn run as a command."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("textual")

import EvoScientist.cli.tui_interactive as tui_mod
from EvoScientist.cli.widgets.ask_user_widget import AskUserWidget
from EvoScientist.cli.widgets.assistant_message import AssistantMessage
from EvoScientist.cli.widgets.loading_widget import LoadingWidget
from EvoScientist.cli.widgets.subagent_widget import SubAgentWidget
from EvoScientist.cli.widgets.system_message import SystemMessage
from EvoScientist.cli.widgets.user_message import UserMessage
from EvoScientist.gateway import RuntimeGateways
from EvoScientist.gateway.local import LocalGraphGateway
from EvoScientist.paths import SessionDirs
from tests.test_goal_real_graph import OBJECTIVE, _agent
from tests.test_tui_skill_pin import (
    _isolated_skills,  # noqa: F401  (autouse fixture)
    _NoHistory,
    _submit,
)
from tests.tui_fakes import capture_app as _capture_app

pytestmark = pytest.mark.usefixtures("no_mcp")


def _wire(app, workspace, agent) -> None:
    gateway = LocalGraphGateway()
    app._dirs = SessionDirs(workspace)
    app._runtime_gateways = RuntimeGateways(
        thread_store=gateway.thread_store, graph_gateway=gateway
    )
    app._await_agent_ready = AsyncMock(return_value=agent)
    # Commands and the badge read the thread through the loaded agent.
    app._agent_loader.adopt(agent)
    app._history_suggester = _NoHistory()


async def _until(pilot, condition, *, seconds: float = 20.0) -> None:
    for _ in range(int(seconds / 0.05)):
        if condition():
            return
        await pilot.pause(0.05)
    raise AssertionError("the expected point was never reached")


async def _idle(app, pilot) -> None:
    """Every command, turn and queued input has finished."""
    await _until(
        pilot,
        lambda: (
            not app._busy
            and app._run_task is None
            and not app._queued_messages
            and not app._background_tasks
        ),
    )
    await pilot.pause(0.1)


def _chat(app) -> list[tuple[str, str]]:
    kinds = {
        UserMessage: "user",
        AssistantMessage: "assistant",
        SystemMessage: "system",
        AskUserWidget: "question",
        LoadingWidget: "loading",
    }
    out = []
    for widget in app.query_one("#chat").children:
        for cls, kind in kinds.items():
            if isinstance(widget, cls):
                text = getattr(widget, "_content", None)
                text = str(text if text is not None else widget.render())
                # A user message renders behind its prompt mark.
                out.append((kind, text.removeprefix("> ") if kind == "user" else text))
    return out


def _texts(app, kind: str) -> list[str]:
    return [text for k, text in _chat(app) if k == kind]


async def _values(agent, app) -> dict:
    state = await agent.aget_state(
        {"configurable": {"thread_id": app._conversation_tid}}
    )
    return state.values


async def _question(app, pilot) -> AskUserWidget:
    await _until(pilot, lambda: bool(list(app.query(AskUserWidget))))
    await pilot.pause(0.1)
    return app.query_one(AskUserWidget)


async def test_accepting_a_goal_shows_one_user_message_and_runs_to_completion(
    monkeypatch, workspace, tmp_path
):
    agent, model = _agent(tmp_path)
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _wire(app, workspace, agent)
        await _submit(app, f"/goal {OBJECTIVE}")
        await _question(app, pilot)
        # While the question is open: no spinner, no leftover drafting line.
        assert "loading" not in [kind for kind, _ in _chat(app)]
        assert not any("Drafting" in line for line in _texts(app, "system"))
        assert _texts(app, "user") == []
        await pilot.press("enter")  # Accept
        await _idle(app, pilot)

        system = _texts(app, "system")
        assert f"/goal {OBJECTIVE}" in system  # the command echo
        accepted = system.index("Goal accepted.")
        assert system[accepted + 1] == "  - the report exists"
        assert not any("Drafting" in line for line in system)
        # The message the graph added is shown once, as the user's.
        assert _texts(app, "user") == [OBJECTIVE]
        assert _texts(app, "assistant") == ["working on it"]
        order = [text for _kind, text in _chat(app)]
        assert order.index("Goal accepted.") < order.index(OBJECTIVE)
        assert order.index(OBJECTIVE) < order.index("working on it")
        assert order.index("Acceptance criteria satisfied") < order.index(
            "Goal complete."
        )
        assert not list(app.query(AskUserWidget))
        assert not list(app.query(LoadingWidget))
        # The grader ran, and none of its own work was drawn.
        assert model.calls["grade"] == 1
        assert not list(app.query(SubAgentWidget))

        assert (await _values(agent, app))["_goal"]["status"] == "complete"
        assert model.calls == {"main": 1, "draft": 1, "grade": 1}
        assert app._rubric_badges.get(app._conversation_tid) == "goal: done"
        assert not app.query_one("#prompt").disabled


async def test_while_the_draft_is_written_a_line_and_a_spinner_show(
    monkeypatch, workspace, tmp_path
):
    from EvoScientist.middleware.goal import GoalDrafter

    # The graph may run on another loop than the test: a thread-safe flag.
    release = threading.Event()
    real_draft = GoalDrafter.draft

    async def _held(self, spec):
        while not release.is_set():
            await asyncio.sleep(0.02)
        return await real_draft(self, spec)

    monkeypatch.setattr(GoalDrafter, "draft", _held)
    agent, _model = _agent(tmp_path)
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _wire(app, workspace, agent)
        await _submit(app, f"/goal {OBJECTIVE}")
        try:
            await _until(
                pilot,
                lambda: any("Drafting" in line for line in _texts(app, "system")),
            )
            await pilot.pause(0.1)
            assert _chat(app)[-2][1] == "Drafting acceptance criteria..."
            assert _chat(app)[-1][0] == "loading"
        finally:
            release.set()
        await _question(app, pilot)
        await pilot.press("escape")
        await _idle(app, pilot)


async def test_ctrl_c_while_the_draft_is_written_leaves_nothing_behind(
    monkeypatch, workspace, tmp_path
):
    from EvoScientist.middleware.goal import GoalDrafter
    from EvoScientist.stream import display as display_mod

    release = threading.Event()
    real_draft = GoalDrafter.draft

    async def _held(self, spec):
        while not release.is_set():
            await asyncio.sleep(0.02)
        return await real_draft(self, spec)

    monkeypatch.setattr(GoalDrafter, "draft", _held)
    agent, model = _agent(tmp_path)
    app = await _capture_app(monkeypatch, workspace)
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            _wire(app, workspace, agent)
            await _submit(app, f"/goal {OBJECTIVE}")
            try:
                await _until(
                    pilot,
                    lambda: any("Drafting" in line for line in _texts(app, "system")),
                )
                app.action_request_quit()
                await _idle(app, pilot)
            finally:
                release.set()
            assert not any("Drafting" in line for line in _texts(app, "system"))
            assert not list(app.query(LoadingWidget))
            assert not list(app.query(AskUserWidget))
            assert not app.query_one("#prompt").disabled
            # The next message is an ordinary turn; the draft is not made again.
            await _submit(app, "hello")
            await _idle(app, pilot)
            # (The stopped turn keeps its usual "[Stopped.]" mark.)
            assert _texts(app, "assistant")[-1] == "working on it"
            assert not list(app.query(AskUserWidget))
            assert model.calls["draft"] == 0
    finally:
        display_mod.discard_stream_cancel()


async def test_escape_on_the_review_leaves_one_line_and_no_bubbles(
    monkeypatch, workspace, tmp_path
):
    agent, model = _agent(tmp_path)
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _wire(app, workspace, agent)
        await _submit(app, f"/goal {OBJECTIVE}")
        await _question(app, pilot)
        await pilot.press("escape")
        await _idle(app, pilot)

        assert "Goal proposal cancelled." in _texts(app, "system")
        assert _texts(app, "user") == []
        assert _texts(app, "assistant") == []
        assert not list(app.query(LoadingWidget))
        assert model.calls["main"] == 0
        assert (await _values(agent, app)).get("_goal") is None
        assert app._rubric_badges.get(app._conversation_tid) is None
        assert not app.query_one("#prompt").disabled


async def test_feedback_redrafts_and_asks_again(monkeypatch, workspace, tmp_path):
    agent, model = _agent(tmp_path, drafts=["- first", "- second"])
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _wire(app, workspace, agent)
        await _submit(app, f"/goal {OBJECTIVE}")
        first = await _question(app, pilot)
        await pilot.press("down", "down", "enter")  # Other (type your answer)
        await pilot.pause(0.1)
        for char in "shorter":
            await pilot.press(char)
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: (
                model.calls["draft"] == 2
                and list(app.query(AskUserWidget))
                and app.query_one(AskUserWidget) is not first
            ),
        )
        await pilot.pause(0.1)
        await pilot.press("enter")  # Accept the second draft
        await _idle(app, pilot)

        system = _texts(app, "system")
        assert system[system.index("Goal accepted.") + 1] == "  - second"
        assert _texts(app, "user") == [OBJECTIVE]
        assert (await _values(agent, app))["_sticky_rubric"]["text"] == "- second"


async def test_the_app_stays_busy_between_the_command_and_its_run(
    monkeypatch, workspace, tmp_path
):
    """The 100 ms queue poll starts a channel or notifier turn whenever the app
    is not busy; a gap here would let one slip in before the goal's run."""
    agent, _model = _agent(tmp_path)
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _wire(app, workspace, agent)
        seen: list[tuple[bool, bool]] = []
        real_run_turn = app._run_turn

        def _spy(*args, **kwargs):
            # Called synchronously from the command's last step.
            seen.append((app._busy, kwargs.get("goal_request") is not None))
            return real_run_turn(*args, **kwargs)

        monkeypatch.setattr(app, "_run_turn", _spy)
        token = app._active_command = object()
        await app._handle_command(f"/goal {OBJECTIVE}", token)
        assert seen == [(True, True)]
        assert app._busy is True
        assert app._run_task is not None
        await _question(app, pilot)
        await pilot.press("escape")
        await _idle(app, pilot)


async def test_a_status_change_is_one_line_and_updates_the_badge(
    monkeypatch, workspace, tmp_path
):
    agent, model = _agent(tmp_path)
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _wire(app, workspace, agent)
        await _submit(app, f"/goal {OBJECTIVE}")
        await _question(app, pilot)
        await pilot.press("enter")
        await _idle(app, pilot)
        before = dict(model.calls)
        users = _texts(app, "user")
        assistants = _texts(app, "assistant")

        await _submit(app, "/goal clear")
        await _idle(app, pilot)
        assert "Goal cleared." in _texts(app, "system")
        assert model.calls == before
        assert _texts(app, "user") == users
        assert _texts(app, "assistant") == assistants
        assert app._rubric_badges.get(app._conversation_tid) is None
        assert (await _values(agent, app)).get("_goal") is None


async def test_show_prints_the_goal_without_a_run(monkeypatch, workspace, tmp_path):
    agent, model = _agent(tmp_path)
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _wire(app, workspace, agent)
        await _submit(app, f"/goal {OBJECTIVE}")
        await _question(app, pilot)
        await pilot.press("enter")
        await _idle(app, pilot)
        before = dict(model.calls)
        await _submit(app, "/goal")
        await _idle(app, pilot)
        assert "Goal (complete)" in _texts(app, "system")
        assert model.calls == before


async def test_the_review_question_is_brought_on_screen(
    monkeypatch, workspace, tmp_path
):
    """The user scrolled up while the criteria were drafted; the review needs
    an answer, so it must not open below the visible part of the chat."""
    from EvoScientist.middleware.goal import GoalDrafter

    release = threading.Event()
    real_draft = GoalDrafter.draft

    async def _held(self, spec):
        while not release.is_set():
            await asyncio.sleep(0.02)
        return await real_draft(self, spec)

    monkeypatch.setattr(GoalDrafter, "draft", _held)
    agent, _model = _agent(tmp_path)
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        _wire(app, workspace, agent)
        for index in range(80):
            app._append_system(f"earlier line {index}")
        await pilot.pause(0.2)
        chat = app.query_one("#chat")
        await _submit(app, f"/goal {OBJECTIVE}")
        try:
            await _until(
                pilot,
                lambda: any("Drafting" in line for line in _texts(app, "system")),
            )
            chat.scroll_home(animate=False)
            await pilot.pause(0.3)
            assert chat.scroll_y == 0
        finally:
            release.set()
        question = await _question(app, pilot)
        await pilot.pause(0.3)
        choices = [c for c in question.query(".ask-choice") if c.display]
        assert choices
        visible = chat.region
        for choice in choices:
            assert visible.y <= choice.region.y < visible.y + visible.height
        await pilot.press("escape")
        await _idle(app, pilot)


class _HeldGateway:
    """A fake gateway whose first run waits until the test lets it finish."""

    def __init__(self) -> None:
        from tests.fakes import FakeGraphGateway, FakeThreadStore

        self.release = asyncio.Event()
        self.started = asyncio.Event()
        held = self

        class _Gateway(FakeGraphGateway):
            async def stream_events(self, request):
                self.requests.append(request)
                if len(self.requests) == 1:
                    held.started.set()
                    await held.release.wait()
                yield {"type": "text", "content": "ok"}
                yield {"type": "done", "response": "ok"}

        self.gateway = _Gateway(thread_store=FakeThreadStore(messages=[]))


def _fake_wire(app, workspace, gateway) -> None:
    app._dirs = SessionDirs(workspace)
    app._runtime_gateways = SimpleNamespace(graph_gateway=gateway)
    app._await_agent_ready = AsyncMock(return_value=object())
    app._history_suggester = _NoHistory()


async def test_a_slash_command_typed_during_a_turn_runs_as_a_command(
    monkeypatch, workspace
):
    held = _HeldGateway()
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _fake_wire(app, workspace, held.gateway)
        await _submit(app, "first")
        await asyncio.wait_for(held.started.wait(), timeout=10)
        await _submit(app, "/rubric next - once")
        await _submit(app, "second")
        assert app._queued_messages == ["/rubric next - once", "second"]
        held.release.set()
        await _idle(app, pilot)

        # The command ran as a command, and the queue went on after it.
        sent = [request.message for request in held.gateway.requests]
        assert sent == ["first", "second"]
        assert _texts(app, "user") == ["first", "second"]
        tid = app._conversation_tid
        # The message after the command carries the operation it recorded.
        changes = held.gateway.requests[1].configurable_extra["rubric_run"]["changes"]
        assert [op.get("next") for op in changes] == ["- once"]
        assert app._channel_runtime.rubric.changes_for(tid) == changes
        assert "/rubric next - once" in _texts(app, "system")


async def test_queued_commands_in_a_row_all_run(monkeypatch, workspace):
    held = _HeldGateway()
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _fake_wire(app, workspace, held.gateway)
        await _submit(app, "first")
        await asyncio.wait_for(held.started.wait(), timeout=10)
        await _submit(app, "/nosuchcommand")
        await _submit(app, "/rubric next - once")
        held.release.set()
        await _idle(app, pilot)

        assert [request.message for request in held.gateway.requests] == ["first"]
        system = _texts(app, "system")
        assert "Unknown command: /nosuchcommand" in system
        tid = app._conversation_tid
        assert [
            op.get("next") for op in app._channel_runtime.rubric.changes_for(tid)
        ] == ["- once"]
        assert not app.query_one("#prompt").disabled


async def test_ctrl_c_drops_what_was_queued_before_it(monkeypatch, workspace):
    from EvoScientist.stream import display as display_mod

    held = _HeldGateway()
    app = await _capture_app(monkeypatch, workspace)
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            _fake_wire(app, workspace, held.gateway)
            await _submit(app, "first")
            await asyncio.wait_for(held.started.wait(), timeout=10)
            await _submit(app, "/rubric next - once")
            await _submit(app, "second")
            app.action_request_quit()
            await _idle(app, pilot)
            assert app._queued_messages == []
            assert [request.message for request in held.gateway.requests] == ["first"]
            assert app._channel_runtime.rubric.changes_for(app._conversation_tid) == []
    finally:
        display_mod.discard_stream_cancel()


async def test_input_sent_after_ctrl_c_runs_when_the_stopped_turn_has_ended(
    monkeypatch, workspace
):
    """The stopped turn still closes down for a moment and the app is busy
    until it has. What the user sends meanwhile was entered after the
    interrupt, so it is not left sitting in the queue."""
    from EvoScientist.stream import display as display_mod

    steps = _StepGateway()
    steps.gate("second").set()
    app = await _capture_app(monkeypatch, workspace)
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            _fake_wire(app, workspace, steps.gateway)
            cleanup = asyncio.Event()
            in_cleanup = asyncio.Event()
            real_refresh = app._refresh_status_snapshot
            held_once = False

            async def _refresh(*args, **kwargs):
                nonlocal held_once
                if not args and not held_once:  # the end of the stopped turn
                    held_once = True
                    in_cleanup.set()
                    await cleanup.wait()
                return await real_refresh(*args, **kwargs)

            monkeypatch.setattr(app, "_refresh_status_snapshot", _refresh)
            await _submit(app, "first")
            await _until(pilot, lambda: "first" in steps.running)
            app.action_request_quit()
            await asyncio.wait_for(in_cleanup.wait(), timeout=10)
            await _submit(app, "second")
            assert app._queued_messages == ["second"]
            cleanup.set()
            await _idle(app, pilot)
            sent = [request.message for request in steps.gateway.requests]
            assert sent == ["first", "second"]
            assert app._queued_messages == []
    finally:
        display_mod.discard_stream_cancel()


async def test_input_sent_right_after_stopping_an_unstarted_turn_runs(
    monkeypatch, workspace
):
    from EvoScientist.stream import display as display_mod

    steps = _StepGateway()
    steps.gate("again").set()
    app = await _capture_app(monkeypatch, workspace)
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            _fake_wire(app, workspace, steps.gateway)
            app._dispatch_input("hello")
            app.action_request_quit()  # before the turn's task has run a step
            await _submit(app, "again")  # the app is still held: queued
            assert app._queued_messages == ["again"]
            await _idle(app, pilot)
            assert [request.message for request in steps.gateway.requests] == ["again"]
    finally:
        display_mod.discard_stream_cancel()


async def test_the_app_stays_busy_until_a_turn_has_cleaned_up(monkeypatch, workspace):
    """The end of a turn reads the thread for the status bar. The queue poll
    starts a channel or notifier turn whenever the app is not busy, so that
    wait must still belong to the turn, or the queued input is dispatched
    into the other turn."""
    held = _HeldGateway()
    held.release.set()
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _fake_wire(app, workspace, held.gateway)
        seen: list[bool] = []
        real_refresh = app._refresh_status_snapshot

        async def _refresh(*args, **kwargs):
            if not args:  # the call at the end of the turn
                seen.append(app._busy)
                await asyncio.sleep(0.05)
                seen.append(app._busy)
            return await real_refresh(*args, **kwargs)

        monkeypatch.setattr(app, "_refresh_status_snapshot", _refresh)
        await _submit(app, "first")
        await _submit(app, "second")
        await _idle(app, pilot)
        assert seen == [True, True, True, True]
        assert [request.message for request in held.gateway.requests] == [
            "first",
            "second",
        ]


@pytest.mark.parametrize("text", ["/rubric show", "hello"])
async def test_dispatching_reserves_the_app_in_the_same_step(
    monkeypatch, workspace, text
):
    held = _HeldGateway()
    held.release.set()
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _fake_wire(app, workspace, held.gateway)
        assert app._busy is False
        app._dispatch_input(text)
        # Before the command or the turn has run a single step.
        assert app._busy is True
        await _idle(app, pilot)


async def test_ctrl_c_during_the_command_starts_no_goal_run(
    monkeypatch, workspace, tmp_path
):
    """The command waits for the agent to load; the user gives up on it."""
    agent, model = _agent(tmp_path)
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _wire(app, workspace, agent)
        ready = asyncio.Event()

        async def _held_ready():
            await ready.wait()
            return agent

        app._await_agent_ready = _held_ready
        executed: list[str] = []
        execute = tui_mod.cmd_manager.execute

        async def _watched(command, ctx):
            executed.append(command)
            return await execute(command, ctx)

        monkeypatch.setattr(tui_mod.cmd_manager, "execute", _watched)
        await _submit(app, f"/goal {OBJECTIVE}")
        await pilot.pause(0.2)
        assert app._busy is True
        try:
            app.action_request_quit()
            await pilot.pause(0.1)
            assert app._busy is False
            # The load may never end (a broken MCP load); the input comes back
            # now, so a command such as /mcp can still be typed.
            assert not app.query_one("#prompt").disabled
            assert any("Interrupted by user" in line for line in _texts(app, "system"))
        finally:
            ready.set()
        await _until(pilot, lambda: not app._background_tasks)
        await pilot.pause(0.3)
        assert app._run_task is None
        assert app._busy is False
        # The load finished after the interrupt: the command's body never ran.
        assert executed == []
        assert model.calls == {"main": 0, "draft": 0, "grade": 0}
        assert not list(app.query(AskUserWidget))
        assert not app.query_one("#prompt").disabled


async def test_ctrl_c_on_a_command_past_its_agent_load_leaves_the_input_off(
    monkeypatch, workspace, tmp_path
):
    """Past the load a command may hold its own widget (a picker), so the
    input stays off for it; it comes back when the command ends."""
    agent, _model = _agent(tmp_path)
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _wire(app, workspace, agent)
        release = asyncio.Event()

        async def _held(_command, _ctx):
            await release.wait()
            return True

        monkeypatch.setattr(tui_mod.cmd_manager, "execute", _held)
        # A command that waits for the agent first: the load is over at once.
        await _submit(app, "/goal pause")
        await pilot.pause(0.2)
        assert app._busy is True
        try:
            app.action_request_quit()
            await pilot.pause(0.1)
            assert app._busy is False
            assert app.query_one("#prompt").disabled
        finally:
            release.set()
        await _until(pilot, lambda: not app._background_tasks)
        await pilot.pause(0.3)
        assert not app.query_one("#prompt").disabled


async def test_a_command_cut_by_ctrl_c_leaves_a_later_turn_alone(
    monkeypatch, workspace, tmp_path
):
    """After the interrupt another turn takes the app (the queue poll can
    start a notifier turn); the old command, still waiting in the background,
    ends inside that turn. It must not take the app away, and when everything
    is over the input the command had switched off is on again."""
    agent, model = _agent(tmp_path, holds={"main": threading.Event()})
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _wire(app, workspace, agent)
        ready = asyncio.Event()
        first_wait = True

        async def _ready():
            nonlocal first_wait
            if first_wait:  # only the command waits for the load
                first_wait = False
                await ready.wait()
            return agent

        app._await_agent_ready = _ready
        await _submit(app, "/goal pause")
        await pilot.pause(0.2)
        app.action_request_quit()
        await pilot.pause(0.1)
        # The command is still running, and the input is back already.
        assert not app.query_one("#prompt").disabled
        try:
            app._start_turn("hello")  # as the notifier does: a turn with a handle
            await _until(pilot, lambda: model.calls["main"] == 0 and app._run_task)
            ready.set()  # the old command finishes now, mid-turn
            await _until(pilot, lambda: not app._background_tasks)
            await pilot.pause(0.2)
            assert app._busy is True
            assert app._run_task is not None
        finally:
            ready.set()
            model.holds["main"].set()
        await _idle(app, pilot)
        assert _texts(app, "assistant") == ["working on it"]
        assert not app.query_one("#prompt").disabled


async def test_a_command_cut_by_ctrl_c_leaves_a_channel_turns_input_off(
    monkeypatch, workspace, tmp_path
):
    """The same, with a channel turn taking the app: it keeps its input off
    until it ends, and the old command's end must not switch it on."""
    from unittest.mock import MagicMock

    from EvoScientist.cli.channel import ChannelMessage, _register_channel_request

    agent, _model = _agent(tmp_path)
    app = await _capture_app(monkeypatch, workspace)
    msg = ChannelMessage(
        msg_id="m-2",
        content="from the channel",
        sender="u1",
        channel_type="imessage",
        metadata={},
        channel_ref=None,
        bus_ref=None,
        chat_id="chat-a",
        message_id="ts-2",
    )
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _wire(app, workspace, agent)
        ready = asyncio.Event()
        release = asyncio.Event()
        first_wait = True

        async def _ready():
            nonlocal first_wait
            if first_wait:
                first_wait = False
                await ready.wait()
            return MagicMock()

        async def _held_stream(*_args, **_kwargs):
            await release.wait()
            return "ok"

        app._await_agent_ready = _ready
        monkeypatch.setattr(app, "_stream_with_widgets", _held_stream)
        await _submit(app, "/goal pause")
        await pilot.pause(0.2)
        app.action_request_quit()
        await pilot.pause(0.1)
        _register_channel_request(msg)
        turn = asyncio.ensure_future(app._process_channel_message(msg))
        try:
            await _until(pilot, lambda: app._busy)
            ready.set()  # the old command finishes inside the channel turn
            await _until(pilot, lambda: not app._background_tasks)
            await pilot.pause(0.2)
            assert app._busy is True
            assert app.query_one("#prompt").disabled
        finally:
            ready.set()
            release.set()
            await asyncio.wait_for(turn, timeout=10)
        await pilot.pause(0.1)
        assert app._busy is False
        assert not app.query_one("#prompt").disabled


async def test_a_second_ctrl_c_drops_the_handle_of_a_turn_that_never_ran(
    monkeypatch, workspace
):
    """A turn cancelled before its first step is done one loop step later and
    its release callback runs one step after that. Whatever takes the app in
    between must keep it when that callback arrives."""
    from EvoScientist.stream import display as display_mod

    steps = _StepGateway()
    app = await _capture_app(monkeypatch, workspace)
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            _fake_wire(app, workspace, steps.gateway)
            app._dispatch_input("hello")
            stale = app._run_task
            app.action_request_quit()  # cancels the turn before its first step
            for _ in range(5):
                if stale.done():
                    break
                await asyncio.sleep(0)
            assert stale.done()
            app.action_request_quit()  # busy, and no turn is alive
            assert app._run_task is None
            app._dispatch_input("/rubric show")
            assert app._busy is True
            token = app._active_command
            # The late callback of the turn that never ran.
            app._release_unstarted_turn(stale)
            assert app._busy is True
            assert app._active_command is token
            await _idle(app, pilot)
            assert steps.gateway.requests == []
    finally:
        display_mod.discard_stream_cancel()


class _StepGateway:
    """A fake gateway whose runs each wait until the test lets that run finish."""

    def __init__(self) -> None:
        from tests.fakes import FakeGraphGateway, FakeThreadStore

        self.gates: dict[str, asyncio.Event] = {}
        self.running: list[str] = []
        outer = self

        class _Gateway(FakeGraphGateway):
            async def stream_events(self, request):
                self.requests.append(request)
                name = str(request.message)
                outer.running.append(name)
                try:
                    await outer.gate(name).wait()
                    yield {"type": "text", "content": "ok"}
                    yield {"type": "done", "response": "ok"}
                finally:
                    outer.running.remove(name)

        self.gateway = _Gateway(thread_store=FakeThreadStore(messages=[]))

    def gate(self, name: str) -> asyncio.Event:
        return self.gates.setdefault(name, asyncio.Event())


async def test_ctrl_c_during_a_turns_cleanup_does_not_let_two_turns_run(
    monkeypatch, workspace
):
    """The turn has streamed its answer and reads the thread for the status
    bar. Ctrl+C there must stop that turn, not hand the app to the next one
    while this one can still release it and send the queue."""
    from EvoScientist.stream import display as display_mod

    steps = _StepGateway()
    steps.gate("first").set()
    app = await _capture_app(monkeypatch, workspace)
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            _fake_wire(app, workspace, steps.gateway)
            cleanup = asyncio.Event()
            in_cleanup = asyncio.Event()
            real_refresh = app._refresh_status_snapshot
            held_once = False

            async def _refresh(*args, **kwargs):
                nonlocal held_once
                if not args and not held_once:  # the end of the first turn
                    held_once = True
                    in_cleanup.set()
                    await cleanup.wait()
                return await real_refresh(*args, **kwargs)

            monkeypatch.setattr(app, "_refresh_status_snapshot", _refresh)
            await _submit(app, "first")
            await asyncio.wait_for(in_cleanup.wait(), timeout=10)
            app.action_request_quit()
            await pilot.pause(0.1)
            await _submit(app, "second")
            await _until(pilot, lambda: "second" in steps.running)
            await _submit(app, "third")
            cleanup.set()
            await pilot.pause(0.3)
            # The first turn is over for good; "third" waits for "second".
            assert steps.running == ["second"]
            assert app._queued_messages == ["third"]
            assert app._busy is True
            steps.gate("second").set()
            await _until(pilot, lambda: "third" in steps.running)
            assert steps.running == ["third"]
            steps.gate("third").set()
            await _idle(app, pilot)
            sent = [request.message for request in steps.gateway.requests]
            assert sent == ["first", "second", "third"]
    finally:
        display_mod.discard_stream_cancel()


async def test_ctrl_c_before_a_dispatched_turn_starts_frees_the_app(
    monkeypatch, workspace
):
    from EvoScientist.stream import display as display_mod

    steps = _StepGateway()
    steps.gate("again").set()
    app = await _capture_app(monkeypatch, workspace)
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            _fake_wire(app, workspace, steps.gateway)
            app._dispatch_input("hello")
            app.action_request_quit()  # before the turn's task has run a step
            await pilot.pause(0.3)
            assert app._busy is False
            assert app._run_task is None
            assert steps.gateway.requests == []
            await _submit(app, "again")
            await _idle(app, pilot)
            assert [request.message for request in steps.gateway.requests] == ["again"]
    finally:
        display_mod.discard_stream_cancel()


async def test_ctrl_c_before_a_notifier_turn_starts_frees_the_app(
    monkeypatch, workspace
):
    """A background-task notification starts its turn the way typed input does."""
    from EvoScientist.stream import display as display_mod

    steps = _StepGateway()
    steps.gate("again").set()
    app = await _capture_app(monkeypatch, workspace)
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            _fake_wire(app, workspace, steps.gateway)
            await app._inject_notification_tui("a background task finished", [])
            assert app._busy is True
            app.action_request_quit()  # before the turn's task has run a step
            await pilot.pause(0.3)
            assert app._busy is False
            assert app._run_task is None
            assert steps.gateway.requests == []
            await _submit(app, "again")
            await _idle(app, pilot)
            assert [request.message for request in steps.gateway.requests] == ["again"]
    finally:
        display_mod.discard_stream_cancel()


async def test_ctrl_c_before_a_dispatched_command_starts_stops_it(
    monkeypatch, workspace, tmp_path
):
    agent, model = _agent(tmp_path)
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _wire(app, workspace, agent)
        app._dispatch_input(f"/goal {OBJECTIVE}")
        app.action_request_quit()  # before the command's task has run a step
        await _until(pilot, lambda: not app._background_tasks)
        await pilot.pause(0.5)
        assert app._busy is False
        assert app._run_task is None
        assert model.calls == {"main": 0, "draft": 0, "grade": 0}
        assert not list(app.query(AskUserWidget))
        assert not app.query_one("#prompt").disabled


async def test_ctrl_c_during_a_channel_turn_keeps_the_input_off(monkeypatch, workspace):
    """Ctrl+C on a busy app without a foreground turn never switches the
    input on: whoever switched it off (here a channel turn, started directly)
    restores it at its own end."""
    from unittest.mock import MagicMock

    from EvoScientist.cli.channel import ChannelMessage, _register_channel_request

    app = await _capture_app(monkeypatch, workspace)
    msg = ChannelMessage(
        msg_id="m-1",
        content="from the channel",
        sender="u1",
        channel_type="imessage",
        metadata={},
        channel_ref=None,
        bus_ref=None,
        chat_id="chat-a",
        message_id="ts-1",
    )
    release = asyncio.Event()

    async def _held_stream(*_args, **_kwargs):
        await release.wait()
        return "ok"

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        app._await_agent_ready = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr(app, "_stream_with_widgets", _held_stream)
        _register_channel_request(msg)
        turn = asyncio.ensure_future(app._process_channel_message(msg))
        try:
            await _until(pilot, lambda: app._busy)
            await pilot.pause(0.1)
            prompt = app.query_one("#prompt")
            assert prompt.disabled
            app.action_request_quit()
            await pilot.pause(0.1)
            assert prompt.disabled
        finally:
            release.set()
            await asyncio.wait_for(turn, timeout=10)
        await pilot.pause(0.1)
        assert not app.query_one("#prompt").disabled


async def test_ctrl_c_right_after_the_command_handed_over_frees_the_app(
    monkeypatch, workspace, tmp_path
):
    """The goal run is scheduled but has not run a step; stopping it there
    must not leave the app held by a run that never starts."""
    from EvoScientist.stream import display as display_mod

    agent, model = _agent(tmp_path)
    app = await _capture_app(monkeypatch, workspace)
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            _wire(app, workspace, agent)
            token = app._active_command = object()
            await app._handle_command(f"/goal {OBJECTIVE}", token)
            assert app._run_task is not None
            app.action_request_quit()  # in the same step as the hand-over
            await pilot.pause(0.3)
            assert app._busy is False
            assert app._run_task is None
            assert model.calls == {"main": 0, "draft": 0, "grade": 0}
            assert not app.query_one("#prompt").disabled
    finally:
        display_mod.discard_stream_cancel()


async def test_a_command_cut_by_ctrl_c_leaves_an_open_question_alone(
    monkeypatch, workspace
):
    """The turn that took the app is waiting for an answer when the old
    command ends: the question keeps the focus, and the input comes back
    with its answer."""
    from tests.fakes import FakeGraphGateway, FakeThreadStore

    class _Asking(FakeGraphGateway):
        async def stream_events(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                yield {
                    "type": "ask_user",
                    "questions": [{"question": "which one?", "type": "text"}],
                    "tool_call_id": "q1",
                }
                return
            yield {"type": "text", "content": "ok"}
            yield {"type": "done", "response": "ok"}

    gateway = _Asking(thread_store=FakeThreadStore(messages=[]))
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _fake_wire(app, workspace, gateway)
        ready = asyncio.Event()
        first_wait = True

        async def _ready():
            nonlocal first_wait
            if first_wait:  # only the command waits for the load
                first_wait = False
                await ready.wait()
            return object()

        app._await_agent_ready = _ready
        await _submit(app, "/goal pause")
        await pilot.pause(0.2)
        app.action_request_quit()
        await pilot.pause(0.1)
        try:
            app._start_turn("hello")  # as the notifier does: a turn with a handle
            question = await _question(app, pilot)
            await pilot.pause(0.2)
            prompt = app.query_one("#prompt")
            assert prompt.disabled
            focused = app.focused
            assert focused is not None
            assert question in focused.ancestors_with_self
            ready.set()  # the old command finishes while the question is open
            await _until(pilot, lambda: not app._background_tasks)
            await pilot.pause(0.2)
            assert prompt.disabled
            assert app.focused is focused
        finally:
            ready.set()
        for char in "a":
            await pilot.press(char)
        await pilot.press("enter")
        await _idle(app, pilot)
        assert len(gateway.requests) == 2
        assert not app.query_one("#prompt").disabled


async def test_a_failed_agent_load_keeps_what_was_queued(monkeypatch, workspace):
    """The first message waits for the agent; two more are queued; the load
    fails. Sending the queue would start the load again for each entry and
    lose the entry when it fails again."""
    steps = _StepGateway()
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _fake_wire(app, workspace, steps.gateway)
        failing = asyncio.Event()
        loads = 0

        async def _load():
            nonlocal loads
            loads += 1
            await failing.wait()
            raise RuntimeError("MCP server did not start")

        app._await_agent_ready = _load
        await _submit(app, "first")
        await _until(pilot, lambda: loads == 1)
        await _submit(app, "second")
        await _submit(app, "third")
        failing.set()
        await _until(pilot, lambda: not app._busy and app._run_task is None)
        await pilot.pause(0.2)
        assert app._queued_messages == ["second", "third"]
        assert loads == 1
        assert steps.gateway.requests == []


async def test_a_command_whose_agent_load_fails_keeps_what_was_queued(
    monkeypatch, workspace
):
    """The same for a command that waits for the agent."""
    steps = _StepGateway()
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _fake_wire(app, workspace, steps.gateway)
        failing = asyncio.Event()
        loads = 0

        async def _load():
            nonlocal loads
            loads += 1
            await failing.wait()
            raise RuntimeError("MCP server did not start")

        app._await_agent_ready = _load
        await _submit(app, "/goal pause")
        await _until(pilot, lambda: loads == 1)
        await _submit(app, "second")
        await _submit(app, "third")
        failing.set()
        await _until(
            pilot,
            lambda: (
                not app._busy and app._run_task is None and app._active_command is None
            ),
        )
        await pilot.pause(0.2)
        assert app._queued_messages == ["second", "third"]
        assert loads == 1
        assert steps.gateway.requests == []


async def test_a_turn_started_from_the_queue_can_be_stopped(monkeypatch, workspace):
    """It gets its handle like a typed turn: Ctrl+C must find it."""
    from EvoScientist.stream import display as display_mod

    steps = _StepGateway()
    app = await _capture_app(monkeypatch, workspace)
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            _fake_wire(app, workspace, steps.gateway)
            await _submit(app, "first")
            await _until(pilot, lambda: "first" in steps.running)
            await _submit(app, "second")
            steps.gate("first").set()
            await _until(pilot, lambda: steps.running == ["second"])
            assert app._run_task is not None
            assert not app._run_task.done()
            app.action_request_quit()
            await _idle(app, pilot)
            assert steps.running == []
            assert any("Interrupted by user" in line for line in _texts(app, "system"))
    finally:
        display_mod.discard_stream_cancel()


async def test_the_queue_is_sent_one_entry_at_a_time(monkeypatch, workspace):
    steps = _StepGateway()
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _fake_wire(app, workspace, steps.gateway)
        await _submit(app, "first")
        await _until(pilot, lambda: "first" in steps.running)
        await _submit(app, "second")
        await _submit(app, "third")
        steps.gate("first").set()
        await _until(pilot, lambda: "second" in steps.running)
        await pilot.pause(0.2)
        assert steps.running == ["second"]
        assert app._queued_messages == ["third"]
        steps.gate("second").set()
        await _until(pilot, lambda: steps.running == ["third"])
        steps.gate("third").set()
        await _idle(app, pilot)
        sent = [request.message for request in steps.gateway.requests]
        assert sent == ["first", "second", "third"]


async def test_input_queued_behind_a_goal_command_waits_for_the_goal_run(
    monkeypatch, workspace, tmp_path
):
    """The command hands the app to its run; the message typed after it stays
    queued while the review is open and runs when the goal's turn has ended."""
    agent, model = _agent(tmp_path, holds={"main": threading.Event()})
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _wire(app, workspace, agent)
        try:
            await _submit(app, "hello")
            await _until(pilot, lambda: app._run_task is not None)
            await _submit(app, f"/goal {OBJECTIVE}")
            await _submit(app, "and then this")
            model.holds["main"].set()
            await _question(app, pilot)
            assert app._queued_messages == ["and then this"]
            assert model.calls["main"] == 1
            await pilot.press("enter")  # Accept
        finally:
            model.holds["main"].set()
        await _idle(app, pilot)
        assert _texts(app, "user") == ["hello", OBJECTIVE, "and then this"]
        assert model.calls["main"] == 3


async def test_input_sent_after_ctrl_c_in_a_finished_turns_cleanup_runs(
    monkeypatch, workspace
):
    """The turn has answered and reads the thread for the status bar; Ctrl+C
    cuts that read, and the user sends a message before the cut has gone
    through. The message was entered after the interrupt: it must run."""
    from EvoScientist.stream import display as display_mod

    steps = _StepGateway()
    steps.gate("first").set()
    steps.gate("after").set()
    app = await _capture_app(monkeypatch, workspace)
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            _fake_wire(app, workspace, steps.gateway)
            in_cleanup = asyncio.Event()
            real_refresh = app._refresh_status_snapshot
            held_once = False

            async def _refresh(*args, **kwargs):
                nonlocal held_once
                if not args and not held_once:  # the end of the first turn
                    held_once = True
                    in_cleanup.set()
                    await asyncio.Event().wait()  # until Ctrl+C cuts it
                return await real_refresh(*args, **kwargs)

            monkeypatch.setattr(app, "_refresh_status_snapshot", _refresh)
            await _submit(app, "first")
            await asyncio.wait_for(in_cleanup.wait(), timeout=10)
            app.action_request_quit()
            await _submit(app, "after")  # before the cancellation has landed
            assert app._queued_messages == ["after"]
            await _idle(app, pilot)
            sent = [request.message for request in steps.gateway.requests]
            assert sent == ["first", "after"]
    finally:
        display_mod.discard_stream_cancel()


async def test_a_command_cut_by_ctrl_c_leaves_an_open_approval_alone(
    monkeypatch, workspace
):
    """The same with a tool approval open in the turn that took the app."""
    from EvoScientist.channels import interaction as interaction_mod
    from EvoScientist.cli.widgets.approval_widget import ApprovalWidget
    from tests.fakes import FakeGraphGateway, FakeThreadStore
    from tests.fakes import interrupt_event as _interrupt_event

    # No config rule answers the approval: the widget is shown.
    monkeypatch.setattr(
        interaction_mod, "config_policy_snapshot", lambda reqs: (None, {})
    )

    class _Pausing(FakeGraphGateway):
        async def stream_events(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                yield _interrupt_event(1)
                return
            yield {"type": "text", "content": "ok"}
            yield {"type": "done", "response": "ok"}

    gateway = _Pausing(thread_store=FakeThreadStore(messages=[]))
    app = await _capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _fake_wire(app, workspace, gateway)
        app._hitl_auto_approve = False
        ready = asyncio.Event()
        first_wait = True

        async def _ready():
            nonlocal first_wait
            if first_wait:  # only the command waits for the load
                first_wait = False
                await ready.wait()
            return object()

        app._await_agent_ready = _ready
        await _submit(app, "/goal pause")
        await pilot.pause(0.2)
        app.action_request_quit()
        await pilot.pause(0.1)
        try:
            app._start_turn("hello")
            await _until(pilot, lambda: bool(list(app.query(ApprovalWidget))))
            await pilot.pause(0.2)
            approval = app.query_one(ApprovalWidget)
            prompt = app.query_one("#prompt")
            assert prompt.disabled
            focused = app.focused
            assert focused is not None
            assert approval in focused.ancestors_with_self
            ready.set()  # the old command finishes while the approval is open
            await _until(pilot, lambda: not app._background_tasks)
            await pilot.pause(0.2)
            assert prompt.disabled
            assert app.focused is focused
        finally:
            ready.set()
        await pilot.press("y")  # approve
        await _idle(app, pilot)
        assert len(gateway.requests) == 2
        assert not app.query_one("#prompt").disabled
