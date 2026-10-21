"""A decline through the real loops on the real main graph (scripted model, local gateway)."""

from __future__ import annotations

import asyncio
import sys
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("textual")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from EvoScientist.cli.widgets.subagent_widget import SubAgentWidget
from EvoScientist.cli.widgets.tool_call_widget import ToolCallWidget
from EvoScientist.config import EvoScientistConfig
from EvoScientist.EvoScientist import create_cli_agent
from EvoScientist.gateway import RuntimeGateways
from EvoScientist.gateway.local import LocalGraphGateway
from EvoScientist.paths import Workspace
from tests.fakes.common import close_result
from tests.fakes.tui import capture_app

pytestmark = pytest.mark.usefixtures("no_mcp")


# A call that had started when the user rejected an approval elsewhere.
NOT_FINISHED = ("did not finish", "rejected", False)


class _Held:
    """A scripted reply the model gives only once the test releases it (work
    that is still running)."""

    def __init__(self, message: AIMessage) -> None:
        self.message = message
        self.entered = threading.Event()
        self.release = threading.Event()
        self.returned = threading.Event()


class _ByTask(BaseChatModel):
    """Answers from the script named by the task's first human message, so the
    main agent and each sub-agent follow their own script whatever the order."""

    scripts: dict[str, list[Any]]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        key, start = None, 0
        for index, message in enumerate(messages):
            if message.type == "human":
                for name in self.scripts:
                    if str(message.content).strip().startswith(name):
                        key, start = name, index
        script = self.scripts.get(key or "", [])
        done = sum(1 for message in messages[start:] if message.type == "ai")
        step = script[done] if done < len(script) else AIMessage("extra")
        if isinstance(step, _Held):
            step.entered.set()
            step.release.wait(timeout=30)
            step.returned.set()
            step = step.message
        return ChatResult(generations=[ChatGeneration(message=step)])


def _call(call_id: str, name: str, **args) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _task(call_id: str, key: str) -> dict:
    return _call(
        call_id, "task", subagent_type="general-purpose", description=f"{key} work"
    )


def _write(call_id: str, name: str) -> dict:
    return _call(call_id, "write_file", file_path=f"/{name}", content="x")


def _two_sub_agents() -> tuple[dict[str, list[Any]], _Held]:
    """Sub-agent A asks to run a command at once; sub-agent B writes a file,
    keeps working until released, writes another and finishes."""
    held = _Held(AIMessage("", tool_calls=[_write("b2", "b2.txt")]))
    scripts = {
        "go": [AIMessage("", tool_calls=[_task("t1", "SUBA"), _task("t2", "SUBB")])],
        "SUBA": [
            AIMessage("", tool_calls=[_call("a1", "execute", command="echo a > a.txt")])
        ],
        "SUBB": [
            AIMessage("", tool_calls=[_write("b1", "b1.txt")]),
            held,
            AIMessage("subb done"),
        ],
    }
    return scripts, held


async def _until(condition, *, seconds: float = 20.0) -> None:
    for _ in range(int(seconds / 0.02)):
        if condition():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("the expected point was never reached")


async def _settle(task: asyncio.Future) -> None:
    """Never leave a turn running behind a failed test."""
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def _agent(tmp_path, scripts):
    cfg = EvoScientistConfig(enable_async_subagents=False, memory_workers_enabled=False)
    return create_cli_agent(
        workspace=Workspace(tmp_path),
        config=cfg,
        chat_model=_ByTask(scripts=scripts),
        checkpointer=InMemorySaver(),
    )


async def _results(agent, thread_id: str) -> tuple[tuple, dict[str, str]]:
    state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    return state.next, {
        message.tool_call_id: str(message.content)
        for message in state.values["messages"]
        if message.type == "tool"
    }


def _files(tmp_path) -> list[str]:
    return sorted(path.name for path in tmp_path.glob("*.txt"))


def _rows(app) -> dict[str, str]:
    """What the row of each tool call says, sub-agents' calls included."""
    return {widget._tool_id: widget._status for widget in app.query(ToolCallWidget)}


async def test_the_tui_lets_a_working_sibling_finish_then_closes_the_decline(
    monkeypatch, workspace, tmp_path
):
    scripts, held = _two_sub_agents()
    agent = _agent(tmp_path, scripts)
    app = await capture_app(monkeypatch, workspace)
    gateway = LocalGraphGateway()
    shown: list[int] = []

    async def _reject(_widget):
        shown.append(1)
        return SimpleNamespace(decisions=None, auto_approve_session=False)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._runtime_gateways = RuntimeGateways(
            thread_store=gateway.thread_store, graph_gateway=gateway
        )
        monkeypatch.setattr(app, "_await_agent_ready", AsyncMock(return_value=agent))
        monkeypatch.setattr(app, "_wait_for_approval", _reject)
        turn = asyncio.ensure_future(app._stream_with_widgets("go"))
        try:
            await _until(lambda: shown and held.entered.is_set())
            # The sibling is held, so a turn that goes on reading cannot end.
            await asyncio.sleep(0.3)
            assert not turn.done()
            held.release.set()
            await asyncio.wait_for(turn, timeout=20)
            position, results = await _results(agent, app._conversation_tid)
            rows = _rows(app)
        finally:
            held.release.set()
            await _settle(turn)

    assert len(shown) == 1
    assert _files(tmp_path) == ["b1.txt", "b2.txt"]
    assert position == ()
    assert results["t2"] == "subb done"
    assert close_result(results["t1"]) == NOT_FINISHED
    # The declined command says so; the sibling's finished calls keep theirs.
    assert (rows["a1"], rows["b1"], rows["b2"]) == ("rejected", "success", "success")


def _slow_protocol_events(monkeypatch, seconds: float) -> None:
    """Hold back the decoder's protocol events (tool calls, approvals), so the
    sub-agent ends, which another producer announces, overtake them."""
    from EvoScientist.stream import events as events_mod

    process = events_mod._V3EventProcessor.process

    async def _late(self, event):
        await asyncio.sleep(seconds)
        return await process(self, event)

    if seconds:
        monkeypatch.setattr(events_mod._V3EventProcessor, "process", _late)


async def _thread_shows_the_pause(agent, thread_id: str) -> None:
    """Wait until the thread records the approval. An answer that leaves
    before that meets another, existing race: the resume finds nothing parked."""
    config = {"configurable": {"thread_id": thread_id}}
    for _ in range(500):
        state = await agent.aget_state(config)
        if any(task.interrupts for task in state.tasks):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the thread never showed the approval")


async def _tui(monkeypatch, workspace, agent, answer):
    """The app on the real local gateway, its approval widget answered by ``answer``."""
    app = await capture_app(monkeypatch, workspace)
    gateway = LocalGraphGateway()
    monkeypatch.setattr(app, "_await_agent_ready", AsyncMock(return_value=agent))
    monkeypatch.setattr(app, "_wait_for_approval", answer)
    return app, gateway


@pytest.mark.parametrize("lag", [0.0, 0.05])
async def test_the_tui_marks_both_sub_agent_commands_when_the_first_is_declined(
    monkeypatch, workspace, tmp_path, lag
):
    """Two sub-agents each ask to run a command and the first approval is
    declined: the second is not asked, nothing runs, both rows read rejected."""
    _slow_protocol_events(monkeypatch, lag)
    scripts = {
        "go": [AIMessage("", tool_calls=[_task("t1", "SUBA"), _task("t2", "SUBB")])],
        "SUBA": [
            AIMessage("", tool_calls=[_call("a1", "execute", command="echo a > a.txt")])
        ],
        "SUBB": [
            AIMessage("", tool_calls=[_call("b1", "execute", command="echo b > b.txt")])
        ],
    }
    agent = _agent(tmp_path, scripts)
    shown: list[int] = []

    async def _reject(_widget):
        shown.append(1)
        return SimpleNamespace(decisions=None, auto_approve_session=False)

    app, gateway = await _tui(monkeypatch, workspace, agent, _reject)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._runtime_gateways = RuntimeGateways(
            thread_store=gateway.thread_store, graph_gateway=gateway
        )
        turn = asyncio.ensure_future(app._stream_with_widgets("go"))
        try:
            await asyncio.wait_for(turn, timeout=20)
            position, _ = await _results(agent, app._conversation_tid)
            rows = _rows(app)
        finally:
            await _settle(turn)

    assert len(shown) == 1
    assert _files(tmp_path) == []
    assert position == ()
    assert rows == {"a1": "rejected", "b1": "rejected"}


@pytest.mark.skipif(sys.platform == "win32", reason="the command is a POSIX loop")
@pytest.mark.parametrize("lag", [0.0, 0.05])
async def test_the_tui_shows_an_approved_sub_agent_command_as_running(
    monkeypatch, workspace, tmp_path, lag
):
    """The command says when it has started and then waits for a file, so the
    row can be read while it runs."""
    _slow_protocol_events(monkeypatch, lag)
    waiting = (
        "echo started > started.txt; "
        "while [ ! -f go.txt ]; do sleep 0.05; done; echo a > a.txt"
    )
    scripts = {
        "go": [AIMessage("", tool_calls=[_task("t1", "SUBA")]), AIMessage("all done")],
        "SUBA": [
            AIMessage("", tool_calls=[_call("a1", "execute", command=waiting)]),
            AIMessage("suba done"),
        ],
    }
    agent = _agent(tmp_path, scripts)
    shown: list[int] = []

    async def _approve(_widget):
        await _thread_shows_the_pause(agent, app._conversation_tid)
        shown.append(1)
        return SimpleNamespace(
            decisions=[{"type": "approve"}], auto_approve_session=False
        )

    app, gateway = await _tui(monkeypatch, workspace, agent, _approve)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._runtime_gateways = RuntimeGateways(
            thread_store=gateway.thread_store, graph_gateway=gateway
        )
        turn = asyncio.ensure_future(app._stream_with_widgets("go"))
        try:
            # The approved command is running: the round that asked has been
            # cleaned up and the resumed one is under way.
            started = tmp_path / "started.txt"
            await _until(lambda: started.exists() or turn.done(), seconds=10)
            assert not turn.done()
            box = app.query_one(SubAgentWidget)

            def _running() -> bool:
                return _rows(app) == {"a1": "running"} and box._is_active

            await _until(_running, seconds=5)
            # And it stays so while the command runs, which waits for the file.
            await asyncio.sleep(0.6)
            assert _running()
            assert not turn.done()
            (tmp_path / "go.txt").write_text("")
            await asyncio.wait_for(turn, timeout=20)
            rows = _rows(app)
        finally:
            # Let the command end by itself before the turn is given up: a turn
            # cancelled under a running command can keep the process alive.
            (tmp_path / "go.txt").write_text("")
            await asyncio.wait({turn}, timeout=5)
            await _settle(turn)

    assert len(shown) == 1
    assert _files(tmp_path) == ["a.txt", "go.txt", "started.txt"]
    assert rows == {"a1": "success"}
    assert not box._is_active


async def test_the_consumer_closes_a_declined_sub_agent_command(tmp_path, monkeypatch):
    import EvoScientist.channels.interaction as interaction_mod
    from EvoScientist.channels.bus.events import InboundMessage
    from EvoScientist.channels.bus.message_bus import MessageBus
    from EvoScientist.channels.channel_manager import ChannelManager
    from EvoScientist.channels.consumer import InboundConsumer
    from tests.fakes.common import StubChannel

    monkeypatch.setattr(
        interaction_mod, "config_policy_snapshot", lambda reqs: (None, {})
    )
    scripts = {
        "go": [AIMessage("", tool_calls=[_task("t1", "SUBA")])],
        "SUBA": [
            AIMessage("", tool_calls=[_write("a0", "a0.txt")]),
            AIMessage(
                "", tool_calls=[_call("a1", "execute", command="echo a > a.txt")]
            ),
        ],
    }
    work = tmp_path / "work"
    work.mkdir()
    agent = _agent(work, scripts)
    bus = MessageBus()
    manager = ChannelManager(bus, media_dir=tmp_path / "media")
    manager.register(StubChannel())
    consumer = InboundConsumer(
        bus=bus,
        manager=manager,
        agent=agent,
        thread_id="",
        graph_gateway=LocalGraphGateway(),
        max_concurrent=2,
        max_pending=10,
        inference_timeout=30.0,
        drain_timeout=1.0,
    )

    async def _send(content: str) -> None:
        await bus.publish_inbound(
            InboundMessage(
                channel="stub", sender_id="u1", chat_id="c1", content=content
            )
        )

    runner = asyncio.create_task(consumer.run())
    await _send("go")
    prompt = await asyncio.wait_for(bus.consume_outbound(), timeout=20.0)
    assert "Approval Required" in prompt.content
    await _send("2")
    feedback = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
    assert feedback.content == "❌ Rejected"
    await consumer.stop()
    await runner

    position, results = await _results(agent, await consumer._get_thread_id("u1"))
    assert position == ()
    assert close_result(results["t1"]) == NOT_FINISHED
    assert _files(work) == ["a0.txt"]


async def test_the_tui_closes_a_decline_when_ctrl_c_stops_the_sibling(
    monkeypatch, workspace, tmp_path
):
    """The turn runs as the app runs it and is stopped with the app's own
    action while the sibling still works: the decline is recorded for both."""
    from EvoScientist.stream.display import discard_stream_cancel

    scripts, held = _two_sub_agents()
    agent = _agent(tmp_path, scripts)
    app = await capture_app(monkeypatch, workspace)
    gateway = LocalGraphGateway()
    shown: list[int] = []

    async def _reject(_widget):
        shown.append(1)
        return SimpleNamespace(decisions=None, auto_approve_session=False)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._runtime_gateways = RuntimeGateways(
            thread_store=gateway.thread_store, graph_gateway=gateway
        )
        monkeypatch.setattr(app, "_await_agent_ready", AsyncMock(return_value=agent))
        monkeypatch.setattr(app, "_wait_for_approval", _reject)
        turn = asyncio.ensure_future(app._run_turn("go", resolve_mentions=False))
        app._run_task = turn
        try:
            await _until(lambda: shown and held.entered.is_set())
            await pilot.pause()
            assert app._busy
            app.action_request_quit()
            await asyncio.wait({turn}, timeout=20)
            assert turn.done()
            position, results = await _results(agent, app._conversation_tid)
            # The model call that was cut returns later; nothing is written then.
            held.release.set()
            await _until(held.returned.is_set)
            await asyncio.sleep(0.2)
            assert await _results(agent, app._conversation_tid) == (position, results)
        finally:
            held.release.set()
            await _settle(turn)
            discard_stream_cancel()

    assert position == ()
    assert close_result(results["t1"]) == NOT_FINISHED
    assert close_result(results["t2"]) == NOT_FINISHED
    assert _files(tmp_path) == ["b1.txt"]
