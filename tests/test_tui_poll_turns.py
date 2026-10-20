"""Turns the TUI's queue poll starts on its own (channel messages, background
task notifications) take the app only if it is still free when they start.

The poll runs on its timer and schedules the turn as a callback on the app's
message pump, so input handled in between can take the app first. These tests
call the poll themselves, as the timer would, and count the runs that are going
at once. They wait with ``asyncio.sleep``: a channel turn started that way holds
the message pump, which ``pilot.pause`` would wait for."""

from __future__ import annotations

import asyncio
import queue
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("textual")

from EvoScientist.cli import async_notifier
from EvoScientist.cli import channel as channel_mod
from EvoScientist.cli.async_notifier import (
    AsyncTaskNotification,
    drain_notifications,
    enqueue_task_notification,
    has_pending_notifications,
)
from EvoScientist.cli.channel import (
    ChannelMessage,
    _message_queue,
    _register_channel_request,
)
from EvoScientist.commands import manager as cmd_manager
from EvoScientist.paths import SessionDirs
from EvoScientist.stream import display as display_mod
from tests.fakes import FakeGraphGateway, FakeThreadStore
from tests.test_tui_skill_pin import _NoHistory, _submit
from tests.tui_fakes import capture_app

NOTICE = "notice"


def _label(message: str) -> str:
    return NOTICE if message.startswith("[Async tasks update]") else message


class _Runs:
    """A fake gateway whose runs each wait until the test finishes them. It
    records the runs in the order they started and the most that were ever
    going at once."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.running: list[tuple[str, asyncio.Event]] = []
        self.peak = 0
        self.free = False  # once set, a run finishes as soon as it starts
        outer = self

        class _Gateway(FakeGraphGateway):
            async def stream_events(self, request):
                self.requests.append(request)
                entry = (_label(str(request.message)), asyncio.Event())
                outer.started.append(entry[0])
                outer.running.append(entry)
                outer.peak = max(outer.peak, len(outer.running))
                try:
                    if not outer.free:
                        await entry[1].wait()
                    yield {"type": "text", "content": "ok"}
                    yield {"type": "done", "response": "ok"}
                finally:
                    outer.running.remove(entry)

        self.gateway = _Gateway(thread_store=FakeThreadStore(messages=[]))

    def finish(self, name: str | None = None) -> None:
        """Let the oldest running run (named *name*, if given) finish."""
        for entry_name, done in self.running:
            if name is None or entry_name == name:
                done.set()
                return

    def finish_all(self) -> None:
        self.free = True
        for _, done in self.running:
            done.set()


def _empty_queues() -> None:
    while True:
        try:
            _message_queue.get_nowait()
        except queue.Empty:
            break
    drain_notifications()


@pytest.fixture(autouse=True)
def _fresh_queues(monkeypatch):
    """The channel and notification queues are module-wide."""
    _empty_queues()
    monkeypatch.setattr(channel_mod, "_thread_channel_origins", {})
    monkeypatch.setattr(channel_mod, "_channel_requests", {})
    monkeypatch.setattr(channel_mod, "_session_requests", {})
    monkeypatch.setattr(channel_mod, "_cancelled_channel_messages", set())
    yield
    _empty_queues()
    display_mod.discard_stream_cancel()


async def _app(monkeypatch, workspace):
    app = await capture_app(monkeypatch, workspace)
    # No channel start, so no poll timer: each test runs the poll itself.
    app._start_channels = AsyncMock()
    return app


def _wire(app, workspace, runs: _Runs) -> None:
    app._dirs = SessionDirs(workspace)
    app._runtime_gateways = SimpleNamespace(graph_gateway=runs.gateway)
    app._await_agent_ready = AsyncMock(return_value=object())
    app._history_suggester = _NoHistory()


def _channel_message(content: str) -> None:
    msg = ChannelMessage(
        msg_id=f"id-{content}",
        content=content,
        sender="u1",
        channel_type="imessage",
        metadata={},
        channel_ref=None,
        bus_ref=None,
        chat_id="chat-a",
        message_id=f"ts-{content}",
    )
    _register_channel_request(msg)
    _message_queue.put(msg)


def _notification(app, task_id: str) -> None:
    enqueue_task_notification(
        AsyncTaskNotification(
            task_id=task_id,
            agent_name="worker",
            status="success",
            received_at="2026-10-18T00:00:00Z",
            origin_cli_thread_id=app._conversation_tid,
        )
    )


def _idle(app, runs: _Runs) -> bool:
    return (
        not app._busy
        and app._run_task is None
        and not app._queued_messages
        and not app._notification_consuming
        and not runs.running
        and _message_queue.empty()
        and app._held_channel_msg is None
        and not app._channel_turn_scheduled
        and not has_pending_notifications(app._conversation_tid)
    )


async def _until(condition, *, seconds: float = 10.0) -> None:
    for _ in range(int(seconds / 0.05)):
        if condition():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("the expected point was never reached")


async def _poll_until(app, condition, *, seconds: float = 10.0) -> None:
    """Run the poll in place of its timer, every 50 ms, until *condition* holds."""
    for _ in range(int(seconds / 0.05)):
        if condition():
            return
        app._poll_channel_queue()
        await asyncio.sleep(0.05)
    raise AssertionError("the expected point was never reached")


async def test_a_channel_turn_waits_for_input_typed_before_it_starts(
    monkeypatch, workspace
):
    runs = _Runs()
    app = await _app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        try:
            await pilot.pause()
            _wire(app, workspace, runs)
            _channel_message("from the channel")
            app._poll_channel_queue()  # schedules the channel turn
            await _submit(app, "typed")  # Enter, handled before that turn starts
            typed = app._run_task
            await _until(lambda: runs.started)
            await asyncio.sleep(0.3)
            assert runs.started == ["typed"]
            assert app._run_task is typed
            # The app holds the message, for the poll after this turn.
            assert app._held_channel_msg.content == "from the channel"
            runs.finish("typed")
            await _poll_until(app, lambda: len(runs.started) == 2)
            assert runs.started == ["typed", "from the channel"]
            runs.finish_all()
            await _poll_until(app, lambda: _idle(app, runs))
            assert runs.peak == 1
        finally:
            runs.finish_all()


async def test_a_channel_turn_waits_for_a_command_typed_before_it_starts(
    monkeypatch, workspace
):
    """Started anyway, the channel turn would also free the app at its end
    while the command still holds it."""
    runs = _Runs()
    runs.free = True
    hold = asyncio.Event()

    async def _held_execute(_command, _ctx):
        await hold.wait()
        return True

    monkeypatch.setattr(cmd_manager, "execute", _held_execute)
    app = await _app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        try:
            await pilot.pause()
            _wire(app, workspace, runs)
            _channel_message("from the channel")
            app._poll_channel_queue()  # schedules the channel turn
            await _submit(app, "/hold")  # a command that runs until released
            await asyncio.sleep(0.5)
            assert runs.started == []
            assert app._busy is True
            assert app._active_command is not None
            hold.set()
            await _poll_until(app, lambda: runs.started)
            assert runs.started == ["from the channel"]
            await _poll_until(app, lambda: _idle(app, runs))
        finally:
            hold.set()
            runs.finish_all()


async def test_a_notifier_turn_waits_for_input_typed_before_it_starts(
    monkeypatch, workspace
):
    """Started anyway, the notifier turn took the typed turn's handle: Ctrl+C
    then stopped the notifier turn and lost the notification, and the typed
    turn ran on with the app marked free."""
    runs = _Runs()
    app = await _app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        try:
            await pilot.pause()
            _wire(app, workspace, runs)
            _notification(app, "task-1")
            app._poll_channel_queue()  # schedules the notifier
            await _submit(app, "typed")  # Enter, handled before the notifier runs
            typed = app._run_task
            await _until(lambda: runs.started)
            # Past the notifier's batch wait.
            await asyncio.sleep(async_notifier.NOTIFICATION_BATCH_GRACE_SECONDS + 0.3)
            assert runs.started == ["typed"]
            assert app._run_task is typed
            assert has_pending_notifications(app._conversation_tid)
            app.action_request_quit()  # stops the typed turn
            await _until(lambda: not app._busy)
            assert typed.done()
            await _poll_until(app, lambda: len(runs.started) == 2)
            assert runs.started == ["typed", NOTICE]
            runs.finish_all()
            await _poll_until(app, lambda: _idle(app, runs))
            assert runs.peak == 1
        finally:
            runs.finish_all()


async def test_a_channel_turn_waits_for_a_notifier_turn_that_took_the_app_first(
    monkeypatch, workspace
):
    """The channel message arrives while the notifier is in its batch wait,
    with the app still free, and its turn is scheduled behind the notifier."""
    # A wide batch wait, so a slow runner still acts inside it.
    monkeypatch.setattr(async_notifier, "NOTIFICATION_BATCH_GRACE_SECONDS", 1.0)
    runs = _Runs()
    app = await _app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        try:
            await pilot.pause()
            _wire(app, workspace, runs)
            tid = app._conversation_tid
            _notification(app, "task-1")
            app._poll_channel_queue()  # schedules the notifier
            # Drained: the notifier is in its batch wait.
            await _until(lambda: not has_pending_notifications(tid))
            assert not app._busy
            _channel_message("from the channel")
            app._poll_channel_queue()  # the app is free: schedules the channel turn
            await _until(lambda: runs.started)
            await asyncio.sleep(0.3)
            assert runs.started == [NOTICE]
            runs.finish(NOTICE)
            await _poll_until(app, lambda: len(runs.started) == 2)
            assert runs.started == [NOTICE, "from the channel"]
            runs.finish_all()
            await _poll_until(app, lambda: _idle(app, runs))
            assert runs.peak == 1
        finally:
            runs.finish_all()


async def test_a_channel_turn_that_waits_stays_ahead_of_a_later_message(
    monkeypatch, workspace
):
    """The poll schedules the first message, a second one arrives, and Enter
    takes the app before the first turn starts. The first message still runs
    first: if it is ``/new``, the second one belongs in the new thread."""
    runs = _Runs()
    app = await _app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        try:
            await pilot.pause()
            _wire(app, workspace, runs)
            _channel_message("first")
            app._poll_channel_queue()  # schedules the turn for "first"
            _channel_message("second")
            await _submit(app, "typed")  # Enter, handled before that turn starts
            await _until(lambda: runs.started)
            await asyncio.sleep(0.3)
            assert runs.started == ["typed"]
            runs.finish("typed")
            await _until(lambda: not app._busy and app._run_task is None)
            await _poll_until(app, lambda: len(runs.started) == 2)
            assert runs.started == ["typed", "first"]
            runs.finish_all()
            await _poll_until(app, lambda: _idle(app, runs))
            assert runs.started == ["typed", "first", "second"]
            assert runs.peak == 1
        finally:
            runs.finish_all()


async def test_channel_messages_keep_their_order_while_the_app_is_busy(
    monkeypatch, workspace
):
    """A poll that finds the app taken leaves the waiting messages in order."""
    runs = _Runs()
    app = await _app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        try:
            await pilot.pause()
            _wire(app, workspace, runs)
            await _submit(app, "typed")
            await _until(lambda: runs.started)
            _channel_message("first")
            _channel_message("second")
            app._poll_channel_queue()  # the app is taken: nothing starts
            await asyncio.sleep(0.1)
            assert runs.started == ["typed"]
            runs.finish("typed")
            await _until(lambda: not app._busy and app._run_task is None)
            await _poll_until(app, lambda: len(runs.started) == 2)
            assert runs.started == ["typed", "first"]
            runs.finish_all()
            await _poll_until(app, lambda: _idle(app, runs))
            assert runs.started == ["typed", "first", "second"]
            assert runs.peak == 1
        finally:
            runs.finish_all()


async def test_polls_before_a_channel_turn_starts_take_no_second_message(
    monkeypatch, workspace
):
    """The poll runs off the message pump, so it can tick again before the turn
    it scheduled starts. It takes nothing more until then, so no message is
    overwritten or overtaken."""
    runs = _Runs()
    app = await _app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        try:
            await pilot.pause()
            _wire(app, workspace, runs)
            for content in ("first", "second", "third"):
                _channel_message(content)
            app._poll_channel_queue()  # schedules the turn for "first"
            app._poll_channel_queue()  # ticks again before that turn starts
            app._poll_channel_queue()
            await _submit(app, "typed")  # Enter, handled before that turn starts
            await _until(lambda: runs.started)
            await asyncio.sleep(0.3)
            assert runs.started == ["typed"]
            runs.finish_all()
            await _poll_until(app, lambda: _idle(app, runs))
            assert runs.started == ["typed", "first", "second", "third"]
            assert runs.peak == 1
        finally:
            runs.finish_all()


# Weighted: the poll runs most often, as its 100 ms timer does.
ACTIONS = ("poll",) * 3 + ("type", "ctrl+c", "channel", "notice") + ("finish",) * 2


def _ctrl_c(app) -> None:
    # Never the second press of a quit on an idle app.
    app._quit_pending = False
    app.action_request_quit()


@pytest.mark.parametrize("seed", range(1, 7))
async def test_random_input_and_queue_arrivals_never_run_two_turns(
    monkeypatch, workspace, seed
):
    """Typed input, Ctrl+C, channel messages, notifications and runs that end,
    in a random order. Input and Ctrl+C go through the message pump, as keys do;
    the poll runs off it, as its timer does."""
    monkeypatch.setattr(async_notifier, "NOTIFICATION_BATCH_GRACE_SECONDS", 0.05)
    rng = random.Random(seed)
    runs = _Runs()
    app = await _app(monkeypatch, workspace)
    sent: list[str] = []
    async with app.run_test(size=(100, 40)) as pilot:
        try:
            await pilot.pause()
            _wire(app, workspace, runs)
            for step in range(60):
                # Up to three things in the same loop step, before the pump or
                # the timer gets to run again.
                for burst in range(rng.randint(1, 3)):
                    name = f"{step}.{burst}"
                    action = rng.choice(ACTIONS)
                    if action == "poll":
                        app._poll_channel_queue()
                    elif action == "type":
                        app.call_later(
                            app.on_chat_text_area_submitted,
                            SimpleNamespace(value=f"typed-{name}"),
                        )
                    elif action == "ctrl+c":
                        app.call_later(_ctrl_c, app)
                    elif action == "channel":
                        sent.append(f"channel-{name}")
                        _channel_message(sent[-1])
                    elif action == "notice":
                        _notification(app, f"task-{name}")
                    else:
                        runs.finish()
                await asyncio.sleep(rng.choice([0, 0.01, 0.03, 0.08]))
            runs.finish_all()
            await _poll_until(app, lambda: _idle(app, runs), seconds=20)
            assert runs.started
            assert runs.peak == 1
            # Nothing stops a channel turn, so every message got one, in the
            # order the messages arrived.
            channel_turns = [
                name for name in runs.started if name.startswith("channel-")
            ]
            assert channel_turns == sent
        finally:
            runs.finish_all()
