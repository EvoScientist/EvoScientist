"""Tests for the graph/thread gateway abstraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

from EvoScientist.gateway import (
    GraphGateway,
    LocalGraphGateway,
    RunRequest,
    RuntimeGateways,
)
from EvoScientist.stream import display as display_mod
from tests.conftest import run_async
from tests.fakes import FakeGraphGateway, FakeThreadStore

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def test_local_gateway_streams_from_injected_streamer():
    seen: dict[str, Any] = {}

    async def _streamer(agent, message, thread_id, **kwargs):
        seen.update(
            {
                "agent": agent,
                "message": message,
                "thread_id": thread_id,
                "metadata": kwargs.get("metadata"),
                "media": kwargs.get("media"),
            }
        )
        yield {"type": "text", "content": "hi"}
        yield {"type": "done", "response": "hi"}

    agent = MagicMock()
    gateway = LocalGraphGateway(agent)

    async def _collect():
        request = RunRequest(
            message="hello",
            thread_id="t1",
            metadata={"workspace_dir": "/tmp/ws"},
            media=["plot.png"],
        )
        return [event async for event in gateway.stream_events(request)]

    with patch("EvoScientist.stream.events.stream_agent_events", new=_streamer):
        events = run_async(_collect())

    assert events == [
        {"type": "text", "content": "hi"},
        {"type": "done", "response": "hi"},
    ]
    assert seen == {
        "agent": agent,
        "message": "hello",
        "thread_id": "t1",
        "metadata": {"workspace_dir": "/tmp/ws"},
        "media": ["plot.png"],
    }


def test_local_graph_gateway_delegates_thread_operations():
    thread_store = FakeThreadStore(
        generated_thread_id="new12345",
        threads=[{"thread_id": "abc12345"}],
        resolved_thread_id="abc12345",
        metadata={"workspace_dir": "/tmp/ws"},
        messages=["message"],
        exists=True,
        deleted=True,
    )
    agent = MagicMock()

    async def _run():
        gateway = LocalGraphGateway(agent, thread_store=thread_store)
        resolution = await gateway.resolve_thread("abc")
        return {
            "created": await gateway.create_thread(),
            "threads": await gateway.list_threads(
                limit=3,
                include_message_count=True,
            ),
            "resolution": resolution,
            "metadata": await gateway.get_thread_metadata("abc12345"),
            "messages": await gateway.get_thread_messages("abc12345"),
            "exists": await gateway.thread_exists("abc12345"),
            "deleted": await gateway.delete_thread("abc12345"),
        }

    result = run_async(_run())

    assert result["created"] == "new12345"
    assert result["threads"] == [{"thread_id": "abc12345"}]
    assert result["resolution"].thread_id == "abc12345"
    assert result["resolution"].matches == ()
    assert result["resolution"].found
    assert not result["resolution"].ambiguous
    assert result["metadata"] == {"workspace_dir": "/tmp/ws"}
    assert result["messages"] == ["message"]
    assert result["exists"] is True
    assert result["deleted"] is True
    assert thread_store.calls == [
        ("resolve_thread_id_prefix", "abc"),
        ("generate_thread_id", None),
        (
            "list_threads",
            {
                "limit": 3,
                "include_message_count": True,
                "include_preview": False,
            },
        ),
        ("get_thread_metadata", "abc12345"),
        ("get_thread_messages", "abc12345"),
        ("thread_exists", "abc12345"),
        ("delete_thread", "abc12345"),
    ]


def test_run_streaming_can_consume_injected_gateway():
    gateway = FakeGraphGateway(
        events=[
            {"type": "text", "content": "gateway-ok"},
            {"type": "done", "response": "gateway-ok"},
        ]
    )

    with patch("EvoScientist.stream.display.Live"):
        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            metadata={"workspace_dir": "/tmp/ws"},
            gateway=gateway,
        )

    assert result == "gateway-ok"
    assert gateway.requests == [
        RunRequest(
            message="hello",
            thread_id="t1",
            metadata={"workspace_dir": "/tmp/ws"},
        )
    ]


def test_resume_command_consumes_context_gateway():
    from EvoScientist.commands.base import CommandContext
    from EvoScientist.commands.implementation.session import ResumeCommand

    ui = MagicMock()
    ui.handle_session_resume = AsyncMock()
    thread_store = FakeThreadStore(
        resolved_thread_id="abc12345",
        metadata={"workspace_dir": "/restored"},
    )
    ctx = CommandContext(
        agent=None,
        thread_id="current",
        ui=ui,
        workspace_dir="/old",
        thread_store=thread_store,
    )

    run_async(ResumeCommand().execute(ctx, ["abc"]))

    assert ctx.thread_id == "abc12345"
    assert ctx.workspace_dir == "/restored"
    ui.handle_session_resume.assert_awaited_once_with("abc12345", "/restored")


def test_cmd_run_passes_local_graph_gateway(monkeypatch):
    from EvoScientist.cli import interactive

    thread_store = FakeThreadStore(generated_thread_id="generated-thread")

    def _graph_gateway_factory(agent: CompiledStateGraph) -> GraphGateway:
        return LocalGraphGateway(
            agent,
            thread_store=thread_store,
        )

    runtime_gateways = RuntimeGateways(
        thread_store=thread_store,
        graph_gateway_factory=_graph_gateway_factory,
    )
    seen: dict[str, Any] = {}

    def _run_streaming(**kwargs):
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr(interactive, "run_streaming", _run_streaming)

    agent = MagicMock()
    interactive.cmd_run(
        agent,
        "hello",
        show_thinking=False,
        workspace_dir="/tmp/ws",
        model="test-model",
        runtime_gateways=runtime_gateways,
    )

    assert seen["agent"] is agent
    assert seen["thread_id"] == "generated-thread"
    assert isinstance(seen["gateway"], LocalGraphGateway)
    assert seen["gateway"].agent is agent
    assert seen["gateway"].thread_store is thread_store
