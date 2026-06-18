"""Tests for the graph/thread gateway abstraction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from EvoScientist.gateway import (
    GraphTarget,
    LangGraphServerGateway,
    LangGraphServerThreadStore,
    LocalGraphGateway,
    RunRequest,
    RuntimeGateways,
    create_runtime_gateways,
)
from EvoScientist.stream import display as display_mod
from tests.conftest import run_async
from tests.fakes import (
    FakeGraphGateway,
    FakeLangGraphClient,
    FakeLangGraphThreadsClient,
    FakeLangGraphThreadStream,
    FakeThreadStore,
)


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
    gateway = LocalGraphGateway()

    async def _collect():
        request = RunRequest(
            message="hello",
            thread_id="t1",
            metadata={"workspace_dir": "/tmp/ws"},
            media=["plot.png"],
            target=GraphTarget(local_graph=agent, workspace_dir="/tmp/ws"),
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

    async def _run():
        gateway = LocalGraphGateway(thread_store=thread_store)
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


def test_local_graph_gateway_reads_state_values():
    agent = MagicMock()
    agent.aget_state = AsyncMock(
        return_value=SimpleNamespace(values={"async_tasks": {"task-1": {}}})
    )
    gateway = LocalGraphGateway()

    values = run_async(
        gateway.get_state_values(GraphTarget(local_graph=agent), "abc12345")
    )

    assert values == {"async_tasks": {"task-1": {}}}
    agent.aget_state.assert_awaited_once_with(
        {"configurable": {"thread_id": "abc12345"}}
    )


def test_local_graph_gateway_updates_state_values():
    agent = MagicMock()
    agent.aupdate_state = AsyncMock()
    gateway = LocalGraphGateway()

    run_async(
        gateway.update_state_values(
            GraphTarget(local_graph=agent),
            "abc12345",
            {"_summarization_event": {"cutoff_index": 2}},
        )
    )

    agent.aupdate_state.assert_awaited_once_with(
        {"configurable": {"thread_id": "abc12345"}},
        {"_summarization_event": {"cutoff_index": 2}},
    )


def test_run_streaming_can_consume_injected_gateway():
    agent = MagicMock()
    gateway = FakeGraphGateway(
        events=[
            {"type": "text", "content": "gateway-ok"},
            {"type": "done", "response": "gateway-ok"},
        ]
    )

    with patch("EvoScientist.stream.display.Live"):
        result = display_mod._run_streaming(
            agent=agent,
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
            target=GraphTarget(local_graph=agent, workspace_dir="/tmp/ws"),
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
        graph_gateway=FakeGraphGateway(thread_store=thread_store),
    )

    run_async(ResumeCommand().execute(ctx, ["abc"]))

    assert ctx.thread_id == "abc12345"
    assert ctx.workspace_dir == "/restored"
    ui.handle_session_resume.assert_awaited_once_with("abc12345", "/restored")


def test_cmd_run_passes_local_graph_gateway(monkeypatch):
    from EvoScientist.cli import interactive

    thread_store = FakeThreadStore(generated_thread_id="generated-thread")

    runtime_gateways = RuntimeGateways(
        thread_store=thread_store,
        graph_gateway=LocalGraphGateway(thread_store=thread_store),
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
        thread_id="generated-thread",
        show_thinking=False,
        workspace_dir="/tmp/ws",
        model="test-model",
        runtime_gateways=runtime_gateways,
    )

    assert seen["agent"] is agent
    assert seen["thread_id"] == "generated-thread"
    assert isinstance(seen["gateway"], LocalGraphGateway)
    assert seen["gateway"].thread_store is thread_store


def test_langgraph_server_thread_store_delegates_to_sdk_threads():
    threads = FakeLangGraphThreadsClient(
        threads=[
            {
                "thread_id": "abc12345",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-02T00:00:00Z",
                "metadata": {"graph_id": "EvoScientist", "workspace_dir": "/tmp/ws"},
            },
            {
                "thread_id": "worker123",
                "metadata": {"graph_id": "evomemory-turn-worker"},
            },
        ],
        states={
            "abc12345": {
                "values": {
                    "messages": [
                        {"role": "user", "content": "hello from server"},
                        {"role": "assistant", "content": "hi"},
                    ]
                }
            }
        },
    )
    client = FakeLangGraphClient(threads)

    def _client_factory(_base_url, _headers):
        return client

    store = LangGraphServerThreadStore(
        base_url="http://localhost:2024",
        client_factory=_client_factory,
    )

    async def _run():
        return {
            "created": await store.create_thread(),
            "threads": await store.list_threads(
                include_message_count=True,
                include_preview=True,
            ),
            "resolution": await store.resolve_thread_id_prefix("abc"),
            "metadata": await store.get_thread_metadata("abc12345"),
            "messages": await store.get_thread_messages("abc12345"),
            "exists": await store.thread_exists("abc12345"),
            "deleted": await store.delete_thread("abc12345"),
        }

    result = run_async(_run())

    assert result["created"] == "server-thread"
    assert threads.created == [
        {
            "thread_id": "server-thread",
            "metadata": {"graph_id": "EvoScientist"},
        }
    ]
    assert result["threads"] == [
        {
            "thread_id": "abc12345",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z",
            "metadata": {"graph_id": "EvoScientist", "workspace_dir": "/tmp/ws"},
            "message_count": 2,
            "preview": "hello from server",
        },
        {
            "thread_id": "server-thread",
            "created_at": None,
            "updated_at": None,
            "metadata": {"graph_id": "EvoScientist"},
            "message_count": 0,
            "preview": "",
        },
    ]
    assert result["resolution"] == ("abc12345", [])
    assert result["metadata"] == {
        "graph_id": "EvoScientist",
        "workspace_dir": "/tmp/ws",
    }
    assert [message.type for message in result["messages"]] == ["human", "ai"]
    assert result["exists"] is True
    assert result["deleted"] is True
    assert threads.deleted == ["abc12345"]


def test_langgraph_server_thread_store_prefix_resolution_skips_exact_lookup():
    threads = FakeLangGraphThreadsClient(
        threads=[
            {
                "thread_id": "abc12345",
                "metadata": {"graph_id": "EvoScientist"},
            }
        ]
    )
    store = LangGraphServerThreadStore(
        base_url="http://localhost:2024",
        client_factory=lambda _base_url, _headers: FakeLangGraphClient(threads),
    )

    result = run_async(store.resolve_thread_id_prefix("abc"))

    assert result == ("abc12345", [])
    assert threads.gets == []
    assert len(threads.searches) == 1


def test_langgraph_server_thread_store_uuid_resolution_uses_exact_lookup():
    thread_id = "019ed9e4-4253-7f62-b50f-f0470a4b3c9f"
    threads = FakeLangGraphThreadsClient(
        threads=[
            {
                "thread_id": thread_id,
                "metadata": {"graph_id": "EvoScientist"},
            }
        ]
    )
    store = LangGraphServerThreadStore(
        base_url="http://localhost:2024",
        client_factory=lambda _base_url, _headers: FakeLangGraphClient(threads),
    )

    result = run_async(store.resolve_thread_id_prefix(thread_id))

    assert result == (thread_id, [])
    assert threads.gets == [thread_id]
    assert threads.searches == []


def test_langgraph_server_thread_store_clones_thread_with_metadata():
    clone_metadata = {
        "clone_purpose": "memory_extraction",
        "source_thread_id": "source-thread",
    }
    threads = FakeLangGraphThreadsClient(
        threads=[
            {
                "thread_id": "source-thread",
                "metadata": {"graph_id": "writing-agent", "workspace_dir": "/tmp/ws"},
            }
        ]
    )
    store = LangGraphServerThreadStore(
        base_url="http://localhost:2024",
        client_factory=lambda _base_url, _headers: FakeLangGraphClient(threads),
    )

    cloned_thread_id = run_async(
        store.clone_thread("source-thread", metadata=clone_metadata)
    )

    assert cloned_thread_id == "source-thread-copy"
    assert threads.copied == ["source-thread"]
    assert threads.metadata_updates == [("source-thread-copy", clone_metadata)]
    assert threads.threads[-1] == {
        "thread_id": "source-thread-copy",
        "metadata": {
            "graph_id": "writing-agent",
            "workspace_dir": "/tmp/ws",
            "clone_purpose": "memory_extraction",
            "source_thread_id": "source-thread",
        },
    }


def test_langgraph_server_thread_store_rejects_copy_without_thread_id():
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "source-thread", "metadata": {"graph_id": "agent"}}],
        copy_response=None,
    )
    store = LangGraphServerThreadStore(
        base_url="http://localhost:2024",
        client_factory=lambda _base_url, _headers: FakeLangGraphClient(threads),
    )

    async def _run():
        await store.clone_thread("source-thread")

    with pytest.raises(RuntimeError, match="did not return a cloned thread id"):
        run_async(_run())


def test_langgraph_server_gateway_clones_thread():
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "source-thread", "metadata": {"graph_id": "agent"}}]
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            base_url="http://localhost:2024",
            client_factory=lambda _base_url, _headers: FakeLangGraphClient(threads),
        )
    )

    cloned_thread_id = run_async(
        gateway.clone_thread(
            "source-thread",
            metadata={"clone_purpose": "manual"},
            target=GraphTarget(graph_id="agent"),
        )
    )

    assert cloned_thread_id == "source-thread-copy"
    assert threads.metadata_updates == [
        ("source-thread-copy", {"clone_purpose": "manual"})
    ]


def test_local_graph_gateway_clone_thread_is_explicitly_unsupported():
    async def _run():
        await LocalGraphGateway().clone_thread("source-thread")

    with pytest.raises(NotImplementedError, match="does not support thread cloning"):
        run_async(_run())


def test_runtime_gateways_can_use_langgraph_server_backend():
    threads = FakeLangGraphThreadsClient()
    client = FakeLangGraphClient(threads)

    def _client_factory(_base_url, _headers):
        return client

    runtime_gateways = create_runtime_gateways(
        backend="langgraph_server",
        base_url="http://localhost:2024",
        client_factory=_client_factory,
    )

    gateway = runtime_gateways.graph_gateway

    assert isinstance(runtime_gateways.thread_store, LangGraphServerThreadStore)
    assert isinstance(gateway, LangGraphServerGateway)
    assert gateway.thread_store is runtime_gateways.thread_store


def test_langgraph_server_gateway_reads_state_values():
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={"abc12345": {"values": {"async_tasks": {"task-1": {}}}}},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            base_url="http://localhost:2024",
            client_factory=lambda _base_url, _headers: FakeLangGraphClient(threads),
        )
    )

    values = run_async(gateway.get_state_values(GraphTarget(), "abc12345"))

    assert values == {"async_tasks": {"task-1": {}}}


def test_langgraph_server_gateway_updates_state_values():
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            base_url="http://localhost:2024",
            client_factory=lambda _base_url, _headers: FakeLangGraphClient(threads),
        )
    )

    run_async(
        gateway.update_state_values(
            GraphTarget(),
            "abc12345",
            {"_summarization_event": {"cutoff_index": 2}},
        )
    )

    assert threads.state_updates == [
        ("abc12345", {"_summarization_event": {"cutoff_index": 2}})
    ]


def test_langgraph_server_gateway_streams_root_protocol_events():
    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[
            {
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": "hello"},
                    },
                },
            },
            {
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": {"event": "message-finish"},
                },
            },
        ],
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            base_url="http://localhost:2024",
            client_factory=lambda _base_url, _headers: FakeLangGraphClient(threads),
        )
    )

    async def _collect():
        return [
            event
            async for event in gateway.stream_events(
                RunRequest(
                    message="hi",
                    thread_id="abc12345",
                    metadata={"workspace_dir": "/tmp/ws"},
                    target=GraphTarget(graph_id="writing-agent"),
                )
            )
        ]

    events = run_async(_collect())

    assert threads.created == [
        {
            "thread_id": "abc12345",
            "metadata": {"graph_id": "writing-agent"},
        }
    ]
    assert threads.stream_calls == [("abc12345", "writing-agent")]
    assert stream.run.starts == [
        {
            "input": {"messages": [{"role": "user", "content": "hi"}]},
            "config": {"configurable": {"thread_id": "abc12345"}},
            "metadata": {"workspace_dir": "/tmp/ws"},
        }
    ]
    assert events == [
        {"type": "text", "content": "hello"},
        {"type": "done", "content": "hello", "response": "hello"},
    ]


def test_langgraph_server_gateway_streams_subagent_protocol_events():
    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[
            {
                "method": "lifecycle",
                "params": {
                    "namespace": ["data-analysis-agent:tool-1"],
                    "data": {"event": "started"},
                },
            },
            {
                "method": "messages",
                "params": {
                    "namespace": ["data-analysis-agent:tool-1"],
                    "data": {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": "sub text"},
                    },
                },
            },
            {
                "method": "lifecycle",
                "params": {
                    "namespace": ["data-analysis-agent:tool-1"],
                    "data": {"event": "completed"},
                },
            },
        ],
    )
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            base_url="http://localhost:2024",
            client_factory=lambda _base_url, _headers: FakeLangGraphClient(threads),
        )
    )

    async def _collect():
        return [
            event
            async for event in gateway.stream_events(
                RunRequest(message="hi", thread_id="abc12345")
            )
        ]

    events = run_async(_collect())

    assert events == [
        {
            "type": "subagent_start",
            "name": "data-analysis-agent",
            "description": "",
            "instance_id": "data-analysis-agent:tool-1",
            "tool_call_id": "tool-1",
        },
        {
            "type": "subagent_text",
            "subagent": "data-analysis-agent",
            "content": "sub text",
            "instance_id": "data-analysis-agent:tool-1",
        },
        {
            "type": "subagent_end",
            "name": "data-analysis-agent",
            "instance_id": "data-analysis-agent:tool-1",
        },
        {"type": "done", "content": "", "response": ""},
    ]


def test_langgraph_server_gateway_resumes_interrupt_with_thread_stream():
    from langgraph.types import Command

    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[],
        interrupts=[{"interrupt_id": "interrupt-1"}],
    )
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            base_url="http://localhost:2024",
            client_factory=lambda _base_url, _headers: FakeLangGraphClient(threads),
        )
    )

    async def _collect():
        return [
            event
            async for event in gateway.stream_events(
                RunRequest(
                    message=Command(resume={"decisions": [{"allowed": True}]}),
                    thread_id="abc12345",
                )
            )
        ]

    events = run_async(_collect())

    assert stream.run.starts == []
    assert stream.run.responses == [
        {
            "response": {"decisions": [{"allowed": True}]},
            "interrupt_id": "interrupt-1",
        }
    ]
    assert events == [{"type": "done", "content": "", "response": ""}]
