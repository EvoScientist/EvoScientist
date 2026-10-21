"""Shared pieces for tests that drive the real TUI loop.

Kept apart from ``tests.fakes`` because everything here needs textual.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("textual")

from langchain_core.messages import AIMessage

from EvoScientist.cli.widgets.approval_widget import ApprovalWidget
from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget
from EvoScientist.gateway import RuntimeGateways
from EvoScientist.stream import display as display_mod
from tests.fakes import FakeCheckpointAgent, FakeGraphGateway

# -- the app ------------------------------------------------------------------------
#
# ``EvoTextualInteractiveApp`` is defined inside ``run_textual_interactive``,
# so it is not reachable as ``tui_interactive.EvoTextualInteractiveApp``. It is
# grabbed by invoking the factory once with a patched ``App.run_async`` that
# captures the freshly-built instance. The synchronous factory owns its
# top-level event loop, so each test boots it via :func:`capture_app`.


async def capture_app(monkeypatch, workspace) -> object:
    """Build an ``EvoTextualInteractiveApp`` without entering its main loop."""
    from pathlib import Path

    from textual.app import App

    from EvoScientist.cli import tui_interactive as tui_mod
    from EvoScientist.paths import SessionDirs, Workspace

    captured: dict = {}

    async def _no_run(self, *args, **kwargs):
        captured["app"] = self

    monkeypatch.setattr(App, "run_async", _no_run)
    # The status bar would build the user's configured model, which needs its key.
    monkeypatch.setattr(
        "EvoScientist.cli.status_bar._get_default_chat_model",
        lambda: SimpleNamespace(model_name="test-model"),
    )

    # ``load_agent`` is synchronous: the loader runs it in a worker thread, and
    # what it returns stands in for the agent.
    fake_load_agent = MagicMock(return_value=object())

    @asynccontextmanager
    async def _fake_checkpointer(*_a, **_k):
        yield None

    monkeypatch.setattr(
        "EvoScientist.cli.tui_interactive.get_checkpointer",
        _fake_checkpointer,
        raising=False,
    )

    class _FakeSuggester:
        def __init__(self, *_a, **_k):
            pass

    monkeypatch.setattr(
        "EvoScientist.cli.history_suggester.HistorySuggester", _FakeSuggester
    )

    monkeypatch.setattr(
        "EvoScientist.cli.tui_interactive._auto_start_channel",
        lambda *_a, **_k: None,
        raising=False,
    )

    monkeypatch.setattr("EvoScientist.cli.tui_interactive.mode", "dev", raising=False)
    # Note: ``create_run_dir`` and ``load_agent`` are passed
    # as parameters to ``run_textual_interactive`` (the factory), so the
    # closure inside ``EvoTextualInteractiveApp`` uses the fakes directly
    # and the module-level ``_create_run_dir`` / ``load_agent``
    # symbols never get a chance to run.

    # The factory is synchronous at the outer level and owns its top-level
    # loop via ``asyncio.run``.
    try:
        await asyncio.to_thread(
            tui_mod.run_textual_interactive,
            show_thinking=False,
            channel_send_thinking=False,
            # The banner shows this folder: keep it short, so it fits one line on
            # the 80-column screen whatever the temp folder is called (it wraps on
            # Windows and pushes the banner out of view).
            dirs=SessionDirs(Workspace(Path.cwd())),
            mode="dev",
            model=None,
            provider=None,
            run_name="test-run",
            thread_id=None,
            load_agent=fake_load_agent,
            create_run_dir=lambda ws, _name: ws.root,
            config=None,
        )
    except SystemExit:
        pass

    app = captured.get("app")
    if app is None:
        raise RuntimeError(
            "EvoTextualInteractiveApp was not captured — _no_run was never "
            "called, meaning the factory crashed before constructing the app."
        )
    return app


# -- approvals -------------------------------------------------------------------------

APPROVE = [{"type": "approve"}]
GRANTING = SimpleNamespace(decisions=APPROVE, auto_approve_session=True)
APPROVING_ONCE = SimpleNamespace(decisions=APPROVE, auto_approve_session=False)
REJECTING = SimpleNamespace(decisions=None, auto_approve_session=False)
ASK = {
    "type": "ask_user",
    "interrupt_id": "question-1",
    "questions": [{"question": "Continue?"}],
}


def approval_interrupt(n: int) -> dict:
    """The stream event for the *n*-th approval the TUI is asked about."""
    return {
        "type": "interrupt",
        "interrupt_id": f"approval-{n}",
        "action_requests": [{"name": "execute", "args": {"command": f"echo {n}"}}],
    }


def widget_answers(monkeypatch, app, *answers) -> None:
    """The approval widget gives *answers*, one per wait."""
    monkeypatch.setattr(app, "_wait_for_approval", AsyncMock(side_effect=list(answers)))


# -- a turn that ends at a pause ---------------------------------------------------------

BEFORE_TOOLS = ("HumanInTheLoopMiddleware.after_model",)
DONE = {"type": "done", "response": ""}
TAIL = [{"type": "text", "content": "tail"}, {"type": "done", "response": "tail"}]
FINISHED = [
    {"type": "text", "content": "must not run"},
    {"type": "done", "response": "must not run"},
]


def parked_agent(*, name: str = "execute", next: tuple = BEFORE_TOOLS, **kwargs):
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


def scripted_stream(script: list[list[dict]], *, hold: asyncio.Event | None = None):
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


async def ctrl_c_turn(
    monkeypatch,
    workspace,
    script,
    agent,
    *,
    hold,
    when,
    prepare=None,
    again_when=None,
    stream=None,
    press=True,
):
    """Run a turn as the app's turn task and press Ctrl+C once ``when(app)``
    holds; with ``again_when``, press it a second time once that holds.
    ``press=False`` only waits: the test presses the key itself."""
    app = await capture_app(monkeypatch, workspace)
    gateway = FakeGraphGateway(
        stream=stream or scripted_stream(script, hold=hold), checkpoint=agent
    )
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
            if press:
                app.action_request_quit()
            if again_when is not None:
                for _ in range(200):
                    if again_when():
                        break
                    await pilot.pause()
                else:
                    raise AssertionError("the close never started")
                app.action_request_quit()
            try:
                await asyncio.wait_for(task, timeout=10)
            except asyncio.CancelledError:
                pass
            await pilot.pause()
            tools = {
                widget._tool_id: (widget._status, bool(widget.display))
                for widget in app.query(ToolCallWidget)
            }
            input_off = app.query_one("#prompt").disabled
            approvals_left = len(list(app.query(ApprovalWidget)))
            session_grant = app._hitl_auto_approve
    finally:
        display_mod.discard_stream_cancel()
    return SimpleNamespace(
        gateway=gateway,
        tools=tools,
        cancelled=task.cancelled(),
        cancel_requests=task.cancelling(),
        input_off=input_off,
        approvals_left=approvals_left,
        session_grant=session_grant,
    )


def pausing_stream(script: list[list[dict]]):
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


FINISHED_REPLY = [
    {"type": "text", "content": "finished"},
    {"type": "done", "response": "finished"},
]


async def run_turn(monkeypatch, workspace, script, *, prepare=None, **kwargs):
    """Run one turn through the real TUI loop; return each call's ``configurable_extra``."""
    app = await capture_app(monkeypatch, workspace)
    gateway = FakeGraphGateway(stream=pausing_stream(script))
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


def rubric_gateway(events=None, state_values=None, post_state=None):
    """A fake gateway; ``post_state`` becomes the thread state once a run finished."""
    from tests.fakes import FakeGraphGateway, FakeThreadStore

    class _Gateway(FakeGraphGateway):
        async def stream_events(self, request):
            async for event in super().stream_events(request):
                yield event
            if post_state is not None:
                self.state_values = dict(post_state)

    return _Gateway(
        events=events
        or [{"type": "text", "content": "ok"}, {"type": "done", "response": "ok"}],
        thread_store=FakeThreadStore(messages=[]),
        state_values=state_values or {},
    )


def assistant_texts(app) -> list[str]:
    from EvoScientist.cli.widgets.assistant_message import AssistantMessage

    out = []
    for w in app.query_one("#chat").children:
        if isinstance(w, AssistantMessage):
            out.append(str(getattr(w, "_content", None) or w.render()))
    return out


def chat_sequence(app) -> list[tuple[str, str]]:
    from EvoScientist.cli.widgets.assistant_message import AssistantMessage
    from EvoScientist.cli.widgets.system_message import SystemMessage

    seq = []
    for w in app.query_one("#chat").children:
        if isinstance(w, AssistantMessage):
            seq.append(("assistant", str(getattr(w, "_content", None) or w.render())))
        elif isinstance(w, SystemMessage):
            seq.append(("system", str(w.render())))
    return seq


class GatedStream:
    """A scripted stream that stops at each named gate until the test lets it go on."""

    def __init__(self, *script):
        import asyncio

        self._script = script
        self._reached = {s: asyncio.Event() for s in script if isinstance(s, str)}
        self._go_on = {s: asyncio.Event() for s in self._reached}

    async def stream(self, _request):
        for step in self._script:
            if isinstance(step, str):
                self._reached[step].set()
                await self._go_on[step].wait()
            else:
                yield step

    async def chat_at(self, app, pilot, gate) -> list:
        """The chat's widgets once the stream stands at *gate*."""
        import asyncio

        await asyncio.wait_for(self._reached[gate].wait(), 10)
        await pilot.pause()
        return list(app.query_one("#chat").children)

    def go_on(self, gate) -> None:
        self._go_on[gate].set()


async def run_gated(app, workspace, gated) -> None:
    gateway = rubric_gateway()
    gateway.stream = gated.stream
    prepare_app(app, workspace, gateway)
    await submit(app, "task")


@pytest.fixture(autouse=True)
def isolated_skills(monkeypatch, tmp_path):
    """Keep the user's real skills and the notices thread out of the app."""
    from EvoScientist import paths
    from EvoScientist.cli import tui_interactive as tui_mod
    from EvoScientist.tools import skills_manager

    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "noglobal")
    monkeypatch.setattr(tui_mod, "_agent_shell_notices", lambda: [])
    skills_manager._skill_index_cache.clear()
    yield
    skills_manager._skill_index_cache.clear()


class NoHistory:
    """Stand-in for the prompt history file (the one ``capture_app`` fakes has no
    ``append_entry``)."""

    def __init__(self):
        self._entries: list[str] = []

    def append_entry(self, *_args):
        pass


def prepare_app(app, workspace, gateway):
    """Point the booted app at *workspace* and a fake gateway."""
    from EvoScientist.paths import SessionDirs

    app._dirs = SessionDirs(workspace)
    app._runtime_gateways = SimpleNamespace(graph_gateway=gateway)
    app._await_agent_ready = AsyncMock(return_value=object())
    app._history_suggester = NoHistory()


def system_lines(app) -> list[str]:
    from EvoScientist.cli.widgets.system_message import SystemMessage

    return [
        str(w.render())
        for w in app.query_one("#chat").children
        if isinstance(w, SystemMessage)
    ]


async def submit(app, text: str) -> None:
    await app.on_chat_text_area_submitted(SimpleNamespace(value=text))


async def finish_turns(app, pilot) -> None:
    """Wait for the running turn and for every queued turn replayed after it."""
    for _ in range(10):
        task = app._run_task
        if task is None:
            break
        await task
        await pilot.pause()
    assert app._run_task is None, "turns did not finish"
