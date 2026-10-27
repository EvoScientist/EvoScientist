"""Tests for the graph/thread gateway abstraction."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import typer
from langchain_core.messages import AIMessage, HumanMessage

from EvoScientist.gateway import (
    GraphTarget,
    LangGraphServerGateway,
    LangGraphServerThreadStore,
    LocalGraphGateway,
    RunRequest,
    RuntimeGateways,
    create_runtime_gateways,
)
from EvoScientist.gateway.server import _THREAD_SEARCH_LIMIT
from EvoScientist.gateway.types import DEFAULT_GRAPH_ID
from EvoScientist.paths import SessionDirs, Workspace
from EvoScientist.stream import display as display_mod
from tests.fakes.common import (
    TEST_WORKSPACE,
    FakeGraphGateway,
    FakeLangGraphClient,
    FakeLangGraphThreadsClient,
    FakeLangGraphThreadStream,
    FakeThreadStore,
    _not_found_response,
)


async def test_local_gateway_streams_from_injected_streamer():
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
        events = await _collect()

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


async def test_local_graph_gateway_delegates_thread_operations():
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

    result = await _run()

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


async def test_local_graph_gateway_reads_state_values():
    agent = MagicMock()
    agent.aget_state = AsyncMock(
        return_value=SimpleNamespace(values={"async_tasks": {"task-1": {}}})
    )
    gateway = LocalGraphGateway()

    values = await gateway.get_state_values(GraphTarget(local_graph=agent), "abc12345")

    assert values == {"async_tasks": {"task-1": {}}}
    agent.aget_state.assert_awaited_once_with(
        {"configurable": {"thread_id": "abc12345"}}
    )


async def test_local_graph_gateway_reads_state_snapshot():
    agent = MagicMock()
    snapshot = SimpleNamespace(
        next=("tools",),
        tasks=(),
        interrupts=(object(),),
        values={"async_tasks": {"task-1": {}}},
    )
    agent.aget_state = AsyncMock(return_value=snapshot)
    gateway = LocalGraphGateway()

    got = await gateway.get_state_snapshot(GraphTarget(local_graph=agent), "abc12345")

    assert got is snapshot
    agent.aget_state.assert_awaited_once_with(
        {"configurable": {"thread_id": "abc12345"}}
    )


# ---------------------------------------------------------------------------
# get_run_status seam (Slice 2.4a) — both backends read the live run status
# the state-based client reader uses to detect async-task completion.
# ---------------------------------------------------------------------------


async def test_server_gateway_get_run_status_reads_run_status():
    client = MagicMock()
    client.runs.get = AsyncMock(return_value={"status": "success"})
    thread_store = MagicMock()
    thread_store.client = client
    gateway = LangGraphServerGateway(thread_store=thread_store)

    status = await gateway.get_run_status(GraphTarget(), "task-thread", "run-1")

    assert status == "success"
    client.runs.get.assert_awaited_once_with("task-thread", "run-1")


async def test_local_gateway_get_run_status_reads_dev_server_run(monkeypatch):
    # Async tasks run on the dev server even under the local backend, so
    # get_run_status reads through a dev-server SDK client.
    client = MagicMock()
    client.runs.get = AsyncMock(return_value={"status": "error"})
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.sdk.configured_langgraph_dev_url",
        lambda: "http://dev",
    )
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.sdk.cached_langgraph_async_client",
        lambda url: client,
    )
    gateway = LocalGraphGateway()

    status = await gateway.get_run_status(GraphTarget(), "task-thread", "run-1")

    assert status == "error"
    client.runs.get.assert_awaited_once_with("task-thread", "run-1")


# get_process_status seam (Slice 2.6) — the bg-process reader polls background
# process status through the gateway (custom http route server-side, in-process
# registry locally).
# ---------------------------------------------------------------------------


async def test_server_gateway_get_process_status_reads_http_route():
    client = MagicMock()
    client.http.get = AsyncMock(return_value={"status": "success"})
    thread_store = MagicMock()
    thread_store.client = client
    gateway = LangGraphServerGateway(thread_store=thread_store)

    status = await gateway.get_process_status(GraphTarget(), "cli-thread", "proc-1")

    assert status == "success"
    client.http.get.assert_awaited_once_with(
        "/api/bg_process_status", params={"process_id": "proc-1"}
    )


async def test_local_gateway_get_process_status_reads_registry(monkeypatch):
    # Background processes launched by the in-process main graph live in this
    # process's registry, so get_process_status reads it directly.
    monkeypatch.setattr(
        "EvoScientist.background.poll_status", lambda process_id: "interrupted"
    )
    gateway = LocalGraphGateway()

    status = await gateway.get_process_status(GraphTarget(), "cli-thread", "proc-9")

    assert status == "interrupted"


def _query_dev_server(monkeypatch, client=None, *, enabled=True, url="http://dev"):
    needs = MagicMock(return_value=enabled)
    scoped = MagicMock()
    scoped.return_value.__aenter__.return_value = client
    monkeypatch.setattr("EvoScientist.background.poll_status", lambda _: "unknown")
    monkeypatch.setattr(
        "EvoScientist.EvoScientist._ensure_config", lambda: SimpleNamespace()
    )
    monkeypatch.setattr("EvoScientist.gateway.local.needs_langgraph_dev", needs)
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.sdk.configured_langgraph_dev_url", lambda: url
    )
    if client is not None:
        monkeypatch.setattr(
            "EvoScientist.langgraph_dev.sdk.langgraph_async_client_scoped", scoped
        )
    return needs, scoped


@pytest.mark.parametrize("remote_status", ["running", "success", "error", "unknown"])
async def test_local_process_status_falls_back_to_dev_server(
    monkeypatch, remote_status
):
    client = MagicMock()
    client.http.get = AsyncMock(return_value={"status": remote_status})
    needs, scoped = _query_dev_server(monkeypatch, client)
    assert (
        await LocalGraphGateway().get_process_status(
            GraphTarget(), "thread", "remote-process"
        )
        == remote_status
    )
    assert needs.call_args.kwargs == {"backend": "local"}
    scoped.assert_called_once_with("http://dev")
    client.http.get.assert_awaited_once_with(
        "/api/bg_process_status", params={"process_id": "remote-process"}
    )


async def test_local_process_status_no_dev_server_returns_unknown(monkeypatch):
    _needs, scoped = _query_dev_server(monkeypatch, MagicMock(), enabled=False)
    assert (
        await LocalGraphGateway().get_process_status(GraphTarget(), "thread", "process")
        == "unknown"
    )
    scoped.assert_not_called()


async def test_local_process_status_connect_error_returns_unknown(monkeypatch):
    client = MagicMock()
    client.http.get = AsyncMock(side_effect=httpx.ConnectError("offline"))
    _query_dev_server(monkeypatch, client)
    assert (
        await LocalGraphGateway().get_process_status(GraphTarget(), "thread", "process")
        == "unknown"
    )


@pytest.mark.timeout(3)
async def test_local_process_status_refused_connect_returns_unknown_at_once(
    monkeypatch, unused_tcp_port
):
    _query_dev_server(monkeypatch, url=f"http://127.0.0.1:{unused_tcp_port}")
    assert (
        await LocalGraphGateway().get_process_status(GraphTarget(), "thread", "process")
        == "unknown"
    )


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("timed out"),
        httpx.HTTPStatusError(
            "server error",
            request=httpx.Request("GET", "http://dev"),
            response=httpx.Response(503),
        ),
    ],
)
async def test_local_process_status_other_errors_propagate(monkeypatch, error):
    client = MagicMock()
    client.http.get = AsyncMock(side_effect=error)
    _query_dev_server(monkeypatch, client)
    with pytest.raises(type(error)):
        await LocalGraphGateway().get_process_status(GraphTarget(), "thread", "process")


@pytest.mark.parametrize("status", ["running", "success", "error", "interrupted"])
async def test_known_local_process_never_queries_server(monkeypatch, status):
    needs, scoped = _query_dev_server(monkeypatch, MagicMock())
    monkeypatch.setattr("EvoScientist.background.poll_status", lambda _: status)
    assert (
        await LocalGraphGateway().get_process_status(GraphTarget(), "thread", "process")
        == status
    )
    needs.assert_not_called()
    scoped.assert_not_called()


async def test_local_graph_gateway_updates_state_values():
    agent = MagicMock()
    agent.aupdate_state = AsyncMock()
    gateway = LocalGraphGateway()

    await gateway.update_state_values(
        GraphTarget(local_graph=agent),
        "abc12345",
        {"_summarization_event": {"cutoff_index": 2}},
    )

    agent.aupdate_state.assert_awaited_once_with(
        {"configurable": {"thread_id": "abc12345"}},
        {"_summarization_event": {"cutoff_index": 2}},
        as_node="model",
    )


async def test_local_stream_events_delegates_aclose_to_inner():
    cleanup_ran = False

    async def _streamer(_agent, _message, _thread_id, **_kwargs):
        nonlocal cleanup_ran
        try:
            while True:
                yield {"type": "event"}
        finally:
            cleanup_ran = True

    async def _run():
        gateway = LocalGraphGateway()
        stream = gateway.stream_events(
            RunRequest(
                message="hi",
                thread_id="t1",
                target=GraphTarget(local_graph=object()),
            )
        )
        await stream.__anext__()
        await stream.aclose()
        assert cleanup_ran is True

    with patch("EvoScientist.stream.events.stream_agent_events", new=_streamer):
        await _run()


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


async def test_resume_command_consumes_context_gateway():
    from EvoScientist.commands.base import CommandContext
    from EvoScientist.commands.implementation.session import ResumeCommand

    ui = MagicMock()
    ui.handle_session_resume = AsyncMock()
    thread_store = FakeThreadStore(
        resolved_thread_id="abc12345",
        metadata={"workspace_dir": "/restored"},
    )
    ctx = CommandContext(
        dirs=SessionDirs(TEST_WORKSPACE, Path("/old")),
        agent=None,
        thread_id="current",
        ui=ui,
        graph_gateway=FakeGraphGateway(thread_store=thread_store),
    )

    await ResumeCommand().execute(ctx, ["abc"])

    restored = SessionDirs(Workspace("/restored"))
    assert ctx.thread_id == "abc12345"
    assert ctx.dirs == restored
    ui.handle_session_resume.assert_awaited_once_with("abc12345", restored)


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
        dirs=SessionDirs(Workspace("/tmp/ws")),
        model="test-model",
        runtime_gateways=runtime_gateways,
    )

    assert seen["agent"] is agent
    assert seen["thread_id"] == "generated-thread"
    assert isinstance(seen["gateway"], LocalGraphGateway)
    assert seen["gateway"].thread_store is thread_store


def test_cmd_run_converts_stream_failure_to_controlled_exit(monkeypatch):
    from EvoScientist.cli import interactive

    runtime_gateways = RuntimeGateways(
        thread_store=FakeThreadStore(),
        graph_gateway=FakeGraphGateway(),
    )
    provider_error = RuntimeError("provider unavailable")
    monkeypatch.setattr(
        interactive,
        "run_streaming",
        MagicMock(side_effect=provider_error),
    )

    with pytest.raises(typer.Exit) as exc_info:
        interactive.cmd_run(
            MagicMock(),
            "hello",
            thread_id="failed-thread",
            show_thinking=False,
            dirs=SessionDirs(Workspace("/tmp/ws")),
            runtime_gateways=runtime_gateways,
        )

    assert exc_info.value.exit_code == 1
    assert exc_info.value.__cause__ is provider_error


async def test_langgraph_server_thread_store_delegates_to_sdk_threads():
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

    store = LangGraphServerThreadStore(
        client=client,
    )

    async def _run():
        return {
            "created": await store.create_thread(
                metadata={"model": "test-model"},
                workspace_dir="/tmp/new-ws",
            ),
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

    result = await _run()

    assert result["created"] == "server-thread"
    assert len(threads.created) == 1
    assert threads.created[0]["thread_id"] == "server-thread"
    created_metadata = threads.created[0]["metadata"]
    assert created_metadata["graph_id"] == "EvoScientist"
    assert created_metadata["agent_name"] == "EvoScientist"
    assert created_metadata["workspace_dir"] == "/tmp/new-ws"
    assert created_metadata["model"] == "test-model"
    assert isinstance(created_metadata["updated_at"], str)
    assert result["threads"] == [
        {
            "thread_id": "abc12345",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z",
            "workspace_dir": "/tmp/ws",
            "run_dir": None,
            "model": None,
            "metadata": {"graph_id": "EvoScientist", "workspace_dir": "/tmp/ws"},
            "message_count": 2,
            "preview": "hello from server",
        },
        {
            "thread_id": "server-thread",
            "created_at": None,
            "updated_at": None,
            "workspace_dir": "/tmp/new-ws",
            "run_dir": None,
            "model": "test-model",
            "metadata": created_metadata,
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


async def test_langgraph_server_thread_store_limit_zero_pages_all_threads():
    rows = [
        {
            "thread_id": f"thread-{index}",
            "metadata": {"graph_id": "EvoScientist"},
        }
        for index in range(_THREAD_SEARCH_LIMIT + 1)
    ]
    threads = FakeLangGraphThreadsClient(threads=rows)
    store = LangGraphServerThreadStore(
        client=FakeLangGraphClient(threads),
    )

    result = await store.list_threads(limit=0)

    assert [row["thread_id"] for row in result] == [
        f"thread-{index}" for index in range(_THREAD_SEARCH_LIMIT + 1)
    ]
    assert [(search["limit"], search["offset"]) for search in threads.searches] == [
        (_THREAD_SEARCH_LIMIT, 0),
        (_THREAD_SEARCH_LIMIT, _THREAD_SEARCH_LIMIT),
    ]


async def test_langgraph_server_thread_store_positive_limit_uses_single_search():
    threads = FakeLangGraphThreadsClient(
        threads=[
            {
                "thread_id": f"thread-{index}",
                "metadata": {"graph_id": "EvoScientist"},
            }
            for index in range(3)
        ]
    )
    store = LangGraphServerThreadStore(
        client=FakeLangGraphClient(threads),
    )

    result = await store.list_threads(limit=2)

    assert [row["thread_id"] for row in result] == ["thread-0", "thread-1"]
    assert [(search["limit"], search["offset"]) for search in threads.searches] == [
        (2, 0)
    ]


async def test_langgraph_server_thread_store_prefix_resolution_skips_exact_lookup():
    threads = FakeLangGraphThreadsClient(
        threads=[
            {
                "thread_id": "abc12345",
                "metadata": {"graph_id": "EvoScientist"},
            }
        ]
    )
    store = LangGraphServerThreadStore(
        client=FakeLangGraphClient(threads),
    )

    result = await store.resolve_thread_id_prefix("abc")

    assert result == ("abc12345", [])
    assert threads.gets == []
    assert len(threads.searches) == 1


async def test_langgraph_server_thread_store_prefix_resolution_pages_all_threads():
    rows = [
        {
            "thread_id": f"thread-{index}",
            "metadata": {"graph_id": "EvoScientist"},
        }
        for index in range(_THREAD_SEARCH_LIMIT)
    ]
    rows.append(
        {
            "thread_id": "older-thread-match",
            "metadata": {"graph_id": "EvoScientist"},
        }
    )
    threads = FakeLangGraphThreadsClient(threads=rows)
    store = LangGraphServerThreadStore(
        client=FakeLangGraphClient(threads),
    )

    result = await store.resolve_thread_id_prefix("older-thread")

    assert result == ("older-thread-match", [])
    assert [(search["limit"], search["offset"]) for search in threads.searches] == [
        (_THREAD_SEARCH_LIMIT, 0),
        (_THREAD_SEARCH_LIMIT, _THREAD_SEARCH_LIMIT),
    ]


async def test_langgraph_server_thread_store_uuid_resolution_uses_exact_lookup():
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
        client=FakeLangGraphClient(threads),
    )

    result = await store.resolve_thread_id_prefix(thread_id)

    assert result == (thread_id, [])
    assert threads.gets == [thread_id]
    assert threads.searches == []


async def test_langgraph_server_thread_store_uuid_resolution_filters_graph_id():
    thread_id = "019ed9e4-4253-7f62-b50f-f0470a4b3c9f"
    threads = FakeLangGraphThreadsClient(
        threads=[
            {
                "thread_id": thread_id,
                "metadata": {"graph_id": "other-agent"},
            }
        ]
    )
    store = LangGraphServerThreadStore(
        client=FakeLangGraphClient(threads),
    )

    result = await store.resolve_thread_id_prefix(thread_id)

    assert result == (None, [])
    assert threads.gets == [thread_id]
    assert [(search["limit"], search["offset"]) for search in threads.searches] == [
        (_THREAD_SEARCH_LIMIT, 0)
    ]


async def test_langgraph_server_thread_store_explicitly_clears_agent_name():
    """Non-default graphs must set ``agent_name: None`` so a previously
    stored main-graph stamp is overwritten under PATCH merge semantics.

    Without the explicit clear (``pop``), the key is absent from the
    payload and a stale ``agent_name`` from an earlier default-graph run
    survives the merge, leaving the thread visible to the main-thread
    filter.
    """
    threads = FakeLangGraphThreadsClient(
        threads=[
            {
                "thread_id": "thread-1",
                "metadata": {
                    "graph_id": DEFAULT_GRAPH_ID,
                    "agent_name": DEFAULT_GRAPH_ID,
                },
            }
        ]
    )
    store = LangGraphServerThreadStore(
        client=FakeLangGraphClient(threads),
    )

    new_id = await store.create_thread(
        graph_id="writing-agent",
        metadata={"model": "test-model"},
        workspace_dir="/tmp/ws",
    )

    assert new_id == "server-thread"
    created_metadata = threads.created[0]["metadata"]
    assert created_metadata["agent_name"] is None


async def test_langgraph_server_thread_store_clones_thread_with_metadata():
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
        client=FakeLangGraphClient(threads),
    )

    cloned_thread_id = await store.clone_thread(
        "source-thread", metadata=clone_metadata
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


async def test_langgraph_server_thread_store_rejects_copy_without_thread_id():
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "source-thread", "metadata": {"graph_id": "agent"}}],
        copy_response=None,
    )
    store = LangGraphServerThreadStore(
        client=FakeLangGraphClient(threads),
    )

    async def _run():
        await store.clone_thread("source-thread")

    with pytest.raises(RuntimeError, match="did not return a cloned thread id"):
        await _run()


async def test_langgraph_server_gateway_clones_thread():
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "source-thread", "metadata": {"graph_id": "agent"}}]
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        )
    )

    cloned_thread_id = await gateway.clone_thread(
        "source-thread",
        metadata={"clone_purpose": "manual"},
        target=GraphTarget(graph_id="agent"),
    )

    assert cloned_thread_id == "source-thread-copy"
    assert threads.metadata_updates == [
        ("source-thread-copy", {"clone_purpose": "manual"})
    ]


async def test_local_graph_gateway_clone_thread_is_explicitly_unsupported():
    async def _run():
        await LocalGraphGateway().clone_thread("source-thread")

    with pytest.raises(NotImplementedError, match="does not support thread cloning"):
        await _run()


def test_runtime_gateways_can_use_langgraph_server_backend():
    threads = FakeLangGraphThreadsClient()
    client = FakeLangGraphClient(threads)

    runtime_gateways = create_runtime_gateways(
        backend="langgraph_server",
        langgraph_client=client,
    )

    gateway = runtime_gateways.graph_gateway

    assert isinstance(runtime_gateways.thread_store, LangGraphServerThreadStore)
    assert isinstance(gateway, LangGraphServerGateway)
    assert gateway.thread_store is runtime_gateways.thread_store


async def test_langgraph_server_gateway_reads_state_values():
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={"abc12345": {"values": {"async_tasks": {"task-1": {}}}}},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        )
    )

    values = await gateway.get_state_values(GraphTarget(), "abc12345")

    assert values == {"async_tasks": {"task-1": {}}}


async def test_langgraph_server_gateway_reads_state_snapshot():
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={
            "abc12345": {
                "values": {"async_tasks": {"task-1": {}}},
                "next": ["tools"],
                "tasks": [{"name": "tools", "interrupts": [{"id": "i1"}]}],
                "interrupts": [{"id": "i1"}],
            }
        },
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        )
    )

    snap = await gateway.get_state_snapshot(GraphTarget(), "abc12345")

    assert snap.next == ("tools",)
    assert snap.values == {"async_tasks": {"task-1": {}}}
    assert snap.interrupts == ({"id": "i1"},)
    assert snap.tasks[0].name == "tools"
    assert snap.tasks[0].interrupts == ({"id": "i1"},)


async def test_langgraph_server_gateway_messages_apply_summarization_event():
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={
            "abc12345": {
                "values": {
                    "messages": [
                        HumanMessage(content="first"),
                        AIMessage(content="second"),
                        HumanMessage(content="third"),
                    ],
                    "_summarization_event": {
                        "cutoff_index": 2,
                        "summary_message": AIMessage(content="summary"),
                        "file_path": None,
                    },
                }
            }
        },
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        )
    )

    messages = await gateway.get_thread_messages("abc12345")

    assert len(messages) == 2
    assert isinstance(messages[0], AIMessage)
    assert messages[0].content == "summary"
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "third"


async def test_langgraph_server_gateway_updates_state_values():
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        )
    )

    await gateway.update_state_values(
        GraphTarget(),
        "abc12345",
        {"_summarization_event": {"cutoff_index": 2}},
    )

    assert threads.state_updates == [
        ("abc12345", {"_summarization_event": {"cutoff_index": 2}}, "model")
    ]


async def test_langgraph_server_gateway_streams_root_protocol_events():
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
            client=FakeLangGraphClient(threads),
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

    live_cfg = SimpleNamespace(
        model="live-model", provider="live-provider", recursion_limit=4242
    )
    with patch("EvoScientist.EvoScientist._ensure_config", return_value=live_cfg):
        events = await _collect()

    assert len(threads.created) == 1
    assert threads.created[0]["thread_id"] == "abc12345"
    created_metadata = threads.created[0]["metadata"]
    assert created_metadata["graph_id"] == "writing-agent"
    assert created_metadata["agent_name"] is None
    assert created_metadata["workspace_dir"] == "/tmp/ws"
    assert isinstance(created_metadata["updated_at"], str)
    assert threads.metadata_updates == [("abc12345", threads.created[0]["metadata"])]
    assert threads.stream_calls == [("abc12345", "writing-agent")]
    run_request_id = stream.run.starts[0]["metadata"].get(
        "_evoscientist_gateway_request_id"
    )
    assert isinstance(run_request_id, str)
    assert run_request_id
    assert stream.run.starts == [
        {
            "input": {
                "messages": [{"role": "user", "content": "hi"}],
                "goal_request": None,
            },
            "config": {
                "configurable": {
                    "model": "live-model",
                    "model_provider": "live-provider",
                    "hitl_suppressed": False,
                    "model_retry": {"unattended": False, "wait_budget_seconds": 0},
                    "thread_id": "abc12345",
                },
                "recursion_limit": 4242,
            },
            "metadata": {
                "workspace_dir": "/tmp/ws",
                "_evoscientist_gateway_request_id": run_request_id,
            },
        }
    ]
    assert events == [
        {"type": "text", "content": "hello"},
        {"type": "done", "content": "hello", "response": "hello"},
    ]


def _server_lifecycle_event(run_id: str, status: str) -> dict[str, Any]:
    """Build a root lifecycle event the way the LangGraph server mints it."""
    return {
        "method": "lifecycle",
        "event_id": f"synth:{run_id}:lc||{status}",
        "params": {
            "namespace": [],
            "data": {"event": status, "graph_name": "EvoScientist"},
        },
    }


async def _collect(gateway, request) -> list[dict[str, Any]]:
    """Collect the events of one turn."""
    return [event async for event in gateway.stream_events(request)]


def _server_message_events(text: str, *, index: int) -> list[dict[str, Any]]:
    """Build the message events the server forwards for one answer."""
    return [
        {
            "method": "messages",
            "event_id": f"stream-entry:{index}",
            "params": {
                "namespace": [],
                "data": {
                    "event": "content-block-delta",
                    "delta": {"type": "text-delta", "text": text},
                },
            },
        },
        {
            "method": "messages",
            "event_id": f"stream-entry:{index + 1}",
            "params": {
                "namespace": [],
                "data": {"event": "message-finish"},
            },
        },
    ]


async def test_langgraph_server_gateway_second_turn_does_not_replay_first_turn():
    """A second turn answers with its own run, not the replayed first one.

    The LangGraph server replays a thread's whole history onto every new
    thread-stream connection, so the second turn's stream carries the first
    run's events - including its terminal lifecycle marker - before the
    second run's own events. A channel subscription would stop at that
    replayed terminal, so the turn has to read the stream as a whole and
    keep only the events of the run it started.
    """
    first_run_events = [
        _server_lifecycle_event("run-1", "running"),
        *_server_message_events("first answer", index=0),
        _server_lifecycle_event("run-1", "completed"),
    ]
    # The second turn's connection replays the history and then the events of
    # the run the turn started.
    replayed_with_second_run = [
        *first_run_events,
        _server_lifecycle_event("run-2", "running"),
        *_server_message_events("second answer", index=2),
        _server_lifecycle_event("run-2", "completed"),
    ]
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"abc12345": {"values": {}}},
        streams={
            "abc12345": [
                FakeLangGraphThreadStream(
                    "abc12345", events=first_run_events, run_id="run-1"
                ),
                FakeLangGraphThreadStream(
                    "abc12345", events=replayed_with_second_run, run_id="run-2"
                ),
            ]
        },
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        )
    )

    async def _turn(message: str) -> list[dict[str, Any]]:
        return [
            event
            async for event in gateway.stream_events(
                RunRequest(
                    message=message,
                    thread_id="abc12345",
                    metadata={"workspace_dir": "/tmp/ws"},
                    target=GraphTarget(graph_id="writing-agent"),
                )
            )
        ]

    live_cfg = SimpleNamespace(
        model="live-model", provider="live-provider", recursion_limit=4242
    )
    with patch("EvoScientist.EvoScientist._ensure_config", return_value=live_cfg):
        first_turn = await _turn("hey")
        second_turn = await _turn("list the files in your workspace")

    assert first_turn == [
        {"type": "text", "content": "first answer"},
        {"type": "done", "content": "first answer", "response": "first answer"},
    ]
    assert second_turn == [
        {"type": "text", "content": "second answer"},
        {"type": "done", "content": "second answer", "response": "second answer"},
    ]
    # One stream per turn: the second turn reads the whole replay, so it
    # never has to open a second connection to get past the first run's
    # replayed terminal marker.
    assert len(threads.stream_calls) == 2


async def test_langgraph_server_gateway_turn_scopes_events_across_several_runs():
    """A turn only answers with its own run, whatever the history holds.

    The replayed history of a thread that already ran several times ends on
    the last of those runs' terminal markers. The turn has to read past
    every one of them to reach the run it started.
    """
    history = [
        _server_lifecycle_event("run-1", "running"),
        *_server_message_events("first answer", index=0),
        _server_lifecycle_event("run-1", "completed"),
        _server_lifecycle_event("run-2", "running"),
        *_server_message_events("second answer", index=2),
        _server_lifecycle_event("run-2", "completed"),
    ]
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"abc12345": {"values": {}}},
        streams={
            "abc12345": [
                FakeLangGraphThreadStream(
                    "abc12345",
                    events=[
                        *history,
                        _server_lifecycle_event("run-3", "running"),
                        *_server_message_events("third answer", index=4),
                        _server_lifecycle_event("run-3", "completed"),
                    ],
                    run_id="run-3",
                ),
            ]
        },
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        )
    )

    live_cfg = SimpleNamespace(
        model="live-model", provider="live-provider", recursion_limit=4242
    )
    with patch("EvoScientist.EvoScientist._ensure_config", return_value=live_cfg):
        events = [
            event
            async for event in gateway.stream_events(
                RunRequest(
                    message="hey",
                    thread_id="abc12345",
                    metadata={"workspace_dir": "/tmp/ws"},
                    target=GraphTarget(graph_id="writing-agent"),
                )
            )
        ]

    assert events == [
        {"type": "text", "content": "third answer"},
        {"type": "done", "content": "third answer", "response": "third answer"},
    ]
    assert len(threads.stream_calls) == 1


async def test_langgraph_server_gateway_resumed_run_completes():
    """A resumed run still ends the turn.

    A resumed run has no run id to scope by, so every event of it is kept -
    the synthetic terminal and pause markers included. Without them the turn
    would wait for a terminal that was filtered out.
    """
    import asyncio

    from langgraph.types import Command

    stream = FakeLangGraphThreadStream(
        "thread",
        interrupts=[{"interrupt_id": "interrupt-1"}],
        events=[
            _server_lifecycle_event("run-9", "running"),
            {
                "method": "input.requested",
                "event_id": "synth:run-9:input::interrupt-1",
                "params": {
                    "namespace": [],
                    "data": {
                        "interrupt_id": "interrupt-1",
                        "value": {"question": "continue?"},
                    },
                },
            },
            _server_lifecycle_event("run-9", "interrupted"),
            _server_lifecycle_event("run-10", "running"),
            *_server_message_events("resumed answer", index=0),
            _server_lifecycle_event("run-10", "completed"),
        ],
        # The thread stream stays open after the run ends.
        persistent=True,
    )
    threads = FakeLangGraphThreadsClient(
        states={"thread": {"values": {}}}, streams={"thread": stream}
    )
    client = FakeLangGraphClient(threads)
    client.runs = SimpleNamespace(list=AsyncMock(return_value=[]))
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))

    events = await asyncio.wait_for(
        _collect(
            gateway,
            RunRequest(
                message=Command(
                    resume={"interrupt-1": {"decisions": [{"type": "approve"}]}}
                ),
                thread_id="thread",
            ),
        ),
        timeout=5.0,
    )

    assert events[-1] == {
        "type": "done",
        "content": "resumed answer",
        "response": "resumed answer",
    }


async def test_langgraph_server_gateway_reports_a_failed_transport():
    """A failed stream is an error, not a clean end.

    The SDK's transport records an HTTP or read failure in the stream
    handle and ends the iteration normally. A turn that only watched the
    iteration would answer with a partial result.
    """
    import asyncio

    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[
            _server_lifecycle_event("run-1", "running"),
            *_server_message_events("half an answer", index=0),
        ],
        transport_error=RuntimeError("connection lost"),
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    client = FakeLangGraphClient(threads)
    client.runs = SimpleNamespace(
        cancel_many=AsyncMock(), list=AsyncMock(return_value=[])
    )
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))

    with pytest.raises(RuntimeError, match="connection lost"):
        await asyncio.wait_for(
            _collect(
                gateway,
                RunRequest(message="hi", thread_id="abc12345"),
            ),
            timeout=5.0,
        )


async def test_langgraph_server_gateway_finishes_a_stream_without_run_markers():
    """A stream with no run markers still ends the turn.

    The server marks every run, but a stream that carries no marker at all
    cannot be scoped. Its events are then kept unscoped, and the run's own
    terminal marker ends the turn instead of the stream ending.
    """
    import asyncio

    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[
            {
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": "markerless answer"},
                    },
                },
            },
            {
                "method": "lifecycle",
                "params": {"namespace": [], "data": {"event": "completed"}},
            },
        ],
        # The thread stream stays open after the run ends.
        persistent=True,
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    client = FakeLangGraphClient(threads)
    client.runs = SimpleNamespace(
        cancel_many=AsyncMock(), list=AsyncMock(return_value=[])
    )
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))

    events = await asyncio.wait_for(
        _collect(gateway, RunRequest(message="hi", thread_id="abc12345")),
        timeout=5.0,
    )

    assert events[-1] == {
        "type": "done",
        "content": "markerless answer",
        "response": "markerless answer",
    }


def _ask_user_interrupt(interrupt_id: str, question: str) -> dict[str, Any]:
    """A pending ask_user interrupt, as the thread state records it."""
    return {
        "interrupt_id": interrupt_id,
        "value": {
            "type": "ask_user",
            "questions": [{"question": question}],
            "tool_call_id": f"tc-{interrupt_id}",
        },
    }


async def test_langgraph_server_gateway_delivers_a_prompt_the_watcher_has_not_seen():
    """A prompt is delivered even when the lifecycle watcher lags behind.

    The raw event stream and the SDK's lifecycle watcher are separate
    connections. The watcher's view of the run's interrupts can arrive after
    the event that reported them, so the prompt has to come from the thread
    state - the authoritative record of what the run waits for.
    """
    stream = FakeLangGraphThreadStream(
        "thread",
        events=[
            _server_lifecycle_event("run-1", "running"),
            {
                "method": "input.requested",
                "event_id": "synth:run-1:input::interrupt-1",
                "params": {
                    "namespace": [],
                    "data": {
                        "interrupt_id": "interrupt-1",
                        # The server sends the payload under this key.
                        "payload": {
                            "type": "ask_user",
                            "questions": [{"question": "continue?"}],
                            "tool_call_id": "tc-1",
                        },
                    },
                },
            },
        ],
        # The watcher has not caught up: it knows of no interrupt.
        interrupts=[],
        interrupted=False,
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={
            "thread": {
                "values": {},
                "interrupts": [_ask_user_interrupt("interrupt-1", "continue?")],
            }
        },
        streams={"thread": stream},
    )
    client = FakeLangGraphClient(threads)
    client.runs = SimpleNamespace(
        cancel_many=AsyncMock(), list=AsyncMock(return_value=[])
    )
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))

    events = await _collect(gateway, RunRequest(message="hi", thread_id="thread"))

    prompts = [event for event in events if event["type"] in {"interrupt", "ask_user"}]
    assert len(prompts) == 1, events
    assert events[-1]["type"] == "done", events


async def test_langgraph_server_gateway_delivers_every_pending_prompt():
    """A run that waits on several prompts delivers all of them.

    The raw stream reports the first prompt and the turn stops there; the
    thread state holds every prompt the run is waiting for.
    """
    first = _ask_user_interrupt("interrupt-1", "first question")
    second = _ask_user_interrupt("interrupt-2", "second question")
    stream = FakeLangGraphThreadStream(
        "thread",
        events=[
            _server_lifecycle_event("run-1", "running"),
            {
                "method": "input.requested",
                "event_id": "synth:run-1:input::interrupt-1",
                "params": {
                    "namespace": [],
                    "data": {
                        "interrupt_id": "interrupt-1",
                        "value": first["value"],
                    },
                },
            },
        ],
        interrupts=[],
        interrupted=False,
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"thread": {"values": {}, "interrupts": [first, second]}},
        streams={"thread": stream},
    )
    client = FakeLangGraphClient(threads)
    client.runs = SimpleNamespace(
        cancel_many=AsyncMock(), list=AsyncMock(return_value=[])
    )
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))

    events = await _collect(gateway, RunRequest(message="hi", thread_id="thread"))

    prompts = [event for event in events if event["type"] in {"interrupt", "ask_user"}]
    assert len(prompts) == 2, events
    assert events[-1]["type"] == "done", events


async def test_langgraph_server_gateway_waits_for_a_sibling_to_interrupt():
    """A prompt is not reported until the run is fully paused.

    When the first node interrupts, a sibling task may still be running
    and can interrupt later. The thread state is only final once every task
    carries its interrupt, so the turn waits for that instead of reporting
    the one prompt that happened to arrive first.
    """
    import asyncio

    first = _ask_user_interrupt("interrupt-1", "first question")
    second = _ask_user_interrupt("interrupt-2", "second question")
    stream = FakeLangGraphThreadStream(
        "thread",
        events=[
            _server_lifecycle_event("run-1", "running"),
            {
                "method": "input.requested",
                "event_id": "synth:run-1:input::interrupt-1",
                "params": {
                    "namespace": [],
                    "data": {
                        "interrupt_id": "interrupt-1",
                        "value": first["value"],
                    },
                },
            },
        ],
        interrupts=[],
        interrupted=False,
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={
            "thread": [
                {
                    # The first node interrupted; the sibling still runs.
                    # (The turn's pre-run state read sees this one too.)
                    "values": {},
                    "interrupts": [first],
                    "tasks": [
                        {"id": "t1", "name": "a", "interrupts": [first]},
                        {"id": "t2", "name": "b", "interrupts": []},
                    ],
                },
                {
                    # Still incomplete: the sibling has not interrupted yet.
                    "values": {},
                    "interrupts": [first],
                    "tasks": [
                        {"id": "t1", "name": "a", "interrupts": [first]},
                        {"id": "t2", "name": "b", "interrupts": []},
                    ],
                },
                {
                    # The sibling interrupted too: the run is fully paused.
                    "values": {},
                    "interrupts": [first, second],
                    "tasks": [
                        {"id": "t1", "name": "a", "interrupts": [first]},
                        {"id": "t2", "name": "b", "interrupts": [second]},
                    ],
                },
            ]
        },
        streams={"thread": stream},
    )
    client = FakeLangGraphClient(threads)
    client.runs = SimpleNamespace(
        cancel_many=AsyncMock(), list=AsyncMock(return_value=[])
    )
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))

    events = await asyncio.wait_for(
        _collect(gateway, RunRequest(message="hi", thread_id="thread")),
        timeout=10.0,
    )

    prompts = [event for event in events if event["type"] in {"interrupt", "ask_user"}]
    assert len(prompts) == 2, events
    assert events[-1]["type"] == "done", events


async def test_langgraph_server_gateway_pauses_despite_a_finished_sibling():
    """A sibling that finished normally does not keep the run running.

    The state's task records keep the siblings that completed in the same
    superstep, so a run can be paused while a task without an interrupt is
    still listed.
    """
    import asyncio

    first = _ask_user_interrupt("interrupt-1", "first question")
    stream = FakeLangGraphThreadStream(
        "thread",
        events=[
            _server_lifecycle_event("run-1", "running"),
            {
                "method": "input.requested",
                "event_id": "synth:run-1:input::interrupt-1",
                "params": {
                    "namespace": [],
                    "data": {
                        "interrupt_id": "interrupt-1",
                        "value": first["value"],
                    },
                },
            },
        ],
        interrupts=[],
        interrupted=False,
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={
            "thread": {
                "values": {},
                "interrupts": [first],
                "tasks": [
                    {"id": "t1", "name": "a", "interrupts": [first]},
                    {
                        # The sibling finished normally in the same step.
                        "id": "t2",
                        "name": "b",
                        "interrupts": [],
                        "result": {"ok": True},
                    },
                ],
            }
        },
        streams={"thread": stream},
    )
    client = FakeLangGraphClient(threads)
    client.runs = SimpleNamespace(
        cancel_many=AsyncMock(), list=AsyncMock(return_value=[])
    )
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))

    events = await asyncio.wait_for(
        _collect(gateway, RunRequest(message="hi", thread_id="thread")),
        timeout=10.0,
    )

    prompts = [event for event in events if event["type"] in {"interrupt", "ask_user"}]
    assert len(prompts) == 1, events
    assert events[-1]["type"] == "done", events


def _server_state_from_graph_snapshot(snapshot) -> dict[str, Any]:
    """Keep the real task results and scheduler queue in the server wire shape."""
    return {
        "values": snapshot.values,
        "next": list(snapshot.next),
        "interrupts": [
            {"id": item.id, "value": item.value} for item in snapshot.interrupts
        ],
        "tasks": [
            {
                "id": task.id,
                "name": task.name,
                "error": task.error,
                "result": task.result,
                "interrupts": [
                    {"id": item.id, "value": item.value} for item in task.interrupts
                ],
            }
            for task in snapshot.tasks
        ],
    }


@pytest.mark.parametrize("finished_result", [None, {}], ids=["none", "empty-dict"])
async def test_langgraph_server_gateway_pauses_after_a_single_channel_sibling(
    finished_result,
):
    import asyncio

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    prompt = _ask_user_interrupt("unused", "continue?")["value"]

    def paused(state):
        interrupt(prompt)

    def finished(state):
        return finished_result

    builder = StateGraph(dict)
    builder.add_node("paused", paused)
    builder.add_node("finished", finished)
    for name in ("paused", "finished"):
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "thread"}}
    await graph.ainvoke({"messages": []}, config)
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("paused",)
    state = _server_state_from_graph_snapshot(snapshot)
    completed = next(task for task in state["tasks"] if task["name"] == "finished")
    assert completed["result"] == finished_result
    (pending,) = state["interrupts"]

    stream = FakeLangGraphThreadStream(
        "thread",
        events=[
            _server_lifecycle_event("run-1", "running"),
            {
                "method": "input.requested",
                "event_id": f"synth:run-1:input::{pending['id']}",
                "params": {
                    "namespace": [],
                    "data": {
                        "interrupt_id": pending["id"],
                        "payload": pending["value"],
                    },
                },
            },
        ],
        persistent=True,
    )
    threads = FakeLangGraphThreadsClient(
        states={"thread": state}, streams={"thread": stream}
    )
    client = FakeLangGraphClient(threads)
    client.runs = SimpleNamespace(
        cancel_many=AsyncMock(), list=AsyncMock(return_value=[])
    )
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))

    recovered, confirmed = await gateway._await_paused_interrupts(
        "thread", {pending["id"]}, attempts=1, delay=0
    )
    assert confirmed
    assert recovered == state["interrupts"]
    events = await asyncio.wait_for(
        _collect(gateway, RunRequest(message="hi", thread_id="thread")), timeout=2
    )
    assert [event["type"] for event in events] == ["ask_user", "done"]
    client.runs.cancel_many.assert_not_awaited()


async def test_langgraph_server_gateway_pause_counts_same_name_fanout_tasks():
    import asyncio

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send, interrupt

    from EvoScientist.gateway.server import _state_is_paused

    pausing = asyncio.Event()
    finish_sibling = asyncio.Event()

    async def worker(state):
        if state["pause"]:
            pausing.set()
            interrupt(_ask_user_interrupt("unused", "continue?")["value"])
        else:
            await finish_sibling.wait()
        return None

    builder = StateGraph(dict)
    builder.add_node("worker", worker)
    builder.add_conditional_edges(
        START,
        lambda state: [
            Send("worker", {"pause": True}),
            Send("worker", {"pause": False}),
        ],
    )
    builder.add_edge("worker", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "same-name-fanout"}}
    run = asyncio.create_task(graph.ainvoke({}, config))
    try:
        await asyncio.wait_for(pausing.wait(), timeout=2)
        for _ in range(100):
            snapshot = await graph.aget_state(config)
            if snapshot.interrupts:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("the fanout task never persisted its interrupt")
        assert snapshot.next == ("worker", "worker")
        assert not _state_is_paused(_server_state_from_graph_snapshot(snapshot))
    finally:
        finish_sibling.set()
        await asyncio.wait_for(run, timeout=2)

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("worker",)
    assert _state_is_paused(_server_state_from_graph_snapshot(snapshot))


async def test_langgraph_server_gateway_keeps_a_run_it_cannot_confirm(monkeypatch):
    """A run whose prompts cannot be confirmed is left alone.

    The run is waiting for the user. Cancelling it would destroy the resume
    protocol state, so the turn reports that the set is incomplete and leaves
    the run running.
    """
    from EvoScientist.gateway import server as server_mod

    monkeypatch.setattr(server_mod, "_PAUSE_SYNC_ATTEMPTS", 2)
    monkeypatch.setattr(server_mod, "_PAUSE_SYNC_DELAY", 0.01)
    first = _ask_user_interrupt("interrupt-1", "first question")
    stream = FakeLangGraphThreadStream(
        "thread",
        events=[
            _server_lifecycle_event("run-1", "running"),
            {
                "method": "input.requested",
                "event_id": "synth:run-1:input::interrupt-1",
                "params": {
                    "namespace": [],
                    "data": {
                        "interrupt_id": "interrupt-1",
                        "value": first["value"],
                    },
                },
            },
        ],
        interrupts=[],
        interrupted=False,
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={
            "thread": {
                "values": {},
                "interrupts": [first],
                "tasks": [
                    {"id": "t1", "name": "a", "interrupts": [first]},
                    {
                        # A sibling is still running: the run is not paused.
                        "id": "t2",
                        "name": "b",
                        "interrupts": [],
                    },
                ],
            }
        },
        streams={"thread": stream},
    )
    client = FakeLangGraphClient(threads)
    client.runs = SimpleNamespace(
        cancel_many=AsyncMock(), list=AsyncMock(return_value=[])
    )
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))

    events: list[dict[str, Any]] = []

    async def _run() -> None:
        async for event in gateway.stream_events(
            RunRequest(message="hi", thread_id="thread")
        ):
            events.append(event)

    with pytest.raises(RuntimeError):
        await _run()

    # What is known is still delivered, and the gap is reported.
    prompts = [event for event in events if event["type"] in {"interrupt", "ask_user"}]
    assert len(prompts) == 1, events
    assert any(event["type"] == "error" for event in events), events
    assert not any(event["type"] == "done" for event in events), events
    # The run is left alone: cancelling it would destroy the resume state.
    assert client.runs.cancel_many.await_count == 0


async def test_langgraph_server_gateway_does_not_cancel_a_run_with_nothing_persisted(
    monkeypatch,
):
    """A run the turn could not read at all is still left alone.

    Cancelling it would destroy the resume protocol state, and the checkpoint
    would no longer be answerable. The turn reports the gap instead.
    """
    from EvoScientist.gateway import server as server_mod

    monkeypatch.setattr(server_mod, "_PAUSE_SYNC_ATTEMPTS", 2)
    monkeypatch.setattr(server_mod, "_PAUSE_SYNC_DELAY", 0.01)
    stream = FakeLangGraphThreadStream(
        "thread",
        events=[
            _server_lifecycle_event("run-1", "running"),
            {
                "method": "input.requested",
                "event_id": "synth:run-1:input::interrupt-1",
                "params": {
                    "namespace": [],
                    "data": {"interrupt_id": "interrupt-1", "value": {"q": 1}},
                },
            },
        ],
        interrupts=[],
        interrupted=False,
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        # Nothing is persisted yet, so no prompt can be confirmed.
        states={"thread": {"values": {}, "interrupts": []}},
        streams={"thread": stream},
    )
    client = FakeLangGraphClient(threads)
    client.runs = SimpleNamespace(
        cancel_many=AsyncMock(), list=AsyncMock(return_value=[])
    )
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))

    events: list[dict[str, Any]] = []

    async def _run() -> None:
        async for event in gateway.stream_events(
            RunRequest(message="hi", thread_id="thread")
        ):
            events.append(event)

    with pytest.raises(RuntimeError):
        await _run()

    assert any(event["type"] == "error" for event in events), events
    assert not any(event["type"] == "done" for event in events), events
    # The run is left alone: cancelling it would destroy the resume state.
    assert client.runs.cancel_many.await_count == 0


async def test_langgraph_server_gateway_forwards_configurable_extra():
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
            client=FakeLangGraphClient(threads),
        )
    )

    async def _collect():
        return [
            event
            async for event in gateway.stream_events(
                RunRequest(
                    message="hi",
                    thread_id="abc12345",
                    configurable_extra={
                        "active_teams": ["code-agent"],
                        "custom_key": "custom_value",
                    },
                )
            )
        ]

    live_cfg = SimpleNamespace(
        model="live-model", provider="live-provider", recursion_limit=4242
    )
    with patch("EvoScientist.EvoScientist._ensure_config", return_value=live_cfg):
        events = await _collect()

    run_request_id = stream.run.starts[0]["metadata"].get(
        "_evoscientist_gateway_request_id"
    )
    assert isinstance(run_request_id, str)
    assert run_request_id
    assert stream.run.starts == [
        {
            "input": {
                "messages": [{"role": "user", "content": "hi"}],
                "goal_request": None,
            },
            "config": {
                "configurable": {
                    "model": "live-model",
                    "model_provider": "live-provider",
                    "hitl_suppressed": False,
                    "model_retry": {"unattended": False, "wait_budget_seconds": 0},
                    "active_teams": ["code-agent"],
                    "custom_key": "custom_value",
                    "thread_id": "abc12345",
                },
                "recursion_limit": 4242,
            },
            "metadata": {"_evoscientist_gateway_request_id": run_request_id},
        }
    ]
    assert events == [
        {"type": "text", "content": "hello"},
        {"type": "done", "content": "hello", "response": "hello"},
    ]


class _RecordingSessionEvents:
    """SessionEvents double recording middleware-event dispatches."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def on_tool_selection_started(self, total_tools: int) -> None:
        self.calls.append(("started", total_tools))

    def on_tool_selection(self, selected, total_tools: int) -> None:
        self.calls.append(("selection", list(selected), total_tools))

    def on_tool_selection_ended(self) -> None:
        self.calls.append(("ended",))

    def emit_fallback_notice(self, text: str, style: str = "yellow") -> None:
        self.calls.append(("fallback", text, style))


def _custom_stream_event(payload: object) -> dict:
    return {"method": "custom", "params": {"data": payload}}


async def test_langgraph_server_gateway_delivers_custom_middleware_events():
    """Tagged middleware payloads on the ``custom`` channel reach the
    gateway's ``events`` sink — the delivery path for server-side
    tool-selection / fallback narration (StreamBroadcastSink mirror)."""
    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[
            _custom_stream_event(
                {
                    "evoscientist": {
                        "kind": "tool_selection_started",
                        "total_tools": 9,
                    }
                }
            ),
            _custom_stream_event(
                {
                    "evoscientist": {
                        "kind": "tool_selection",
                        "selected": ["read_file"],
                        "total_tools": 9,
                    }
                }
            ),
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
            _custom_stream_event(
                {
                    "evoscientist": {
                        "kind": "fallback_notice",
                        "text": "fb",
                        "style": "red",
                    }
                }
            ),
            _custom_stream_event({"evoscientist": {"kind": "tool_selection_ended"}}),
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
    # Real session sink: fallback renders via the callback; tool-selection
    # writes are only recorded, and the gateway must poll the read side and
    # emit a tool_selection event (frontends render that event type on both
    # backends).
    fallback_lines: list[tuple[str, str]] = []
    from EvoScientist.stream.sink import SessionEventSink

    sink = SessionEventSink(
        fallback_display=lambda text, style: fallback_lines.append((text, style))
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads)),
        events=sink,
    )

    events = [
        event
        async for event in gateway.stream_events(
            RunRequest(message="hi", thread_id="abc12345")
        )
    ]

    # Fallback notice rendered via the sink's display callback...
    assert fallback_lines == [("fb", "red")]
    # ...and the pending tool selection was polled off the read side and
    # yielded as a stream event in the same shape the local path emits.
    assert {
        "type": "tool_selection",
        "tools": ["read_file"],
    } in events
    # The selection is decided before the model produces text, so its
    # event must reach the stream before the first text delta.
    tool_selection_idx = events.index(
        {"type": "tool_selection", "tools": ["read_file"]}
    )
    first_text_idx = next(i for i, e in enumerate(events) if e["type"] == "text")
    assert tool_selection_idx < first_text_idx
    # Normal graph events unaffected.
    assert [e["type"] for e in events if e["type"] != "tool_selection"] == [
        "text",
        "done",
    ]
    # The transport-level iterator reads every channel, so the custom channel
    # is covered without naming it.
    assert stream.raw_events_used


async def test_langgraph_server_gateway_ignores_untagged_custom_events():
    """Custom-channel traffic without the middleware tag is not ours — it
    must be ignored, and malformed tagged payloads degrade to silence
    instead of failing the run."""
    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[
            _custom_stream_event({"someone_else": {"kind": "tool_selection_started"}}),
            _custom_stream_event({"evoscientist": {"kind": "tool_selection_started"}}),
            _custom_stream_event(
                {"evoscientist": {"kind": "tool_selection", "selected": "not-a-list"}}
            ),
            _custom_stream_event({"evoscientist": {"kind": "unknown_kind"}}),
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
    sink = _RecordingSessionEvents()
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads)),
        events=sink,
    )

    events = [
        event
        async for event in gateway.stream_events(
            RunRequest(message="hi", thread_id="abc12345")
        )
    ]

    # Untagged traffic, missing fields (started without total_tools),
    # wrong-typed fields (selected not a list), and unknown kinds all
    # degrade to silence — nothing dispatches, run completes.
    assert sink.calls == []
    assert events[-1]["type"] == "done"


async def test_langgraph_server_gateway_custom_events_dropped_headless():
    """``events=None`` (headless single-shot) must drop custom payloads
    silently — no crash, run completes."""
    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[
            _custom_stream_event(
                {"evoscientist": {"kind": "tool_selection_started", "total_tools": 1}}
            ),
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
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads)),
    )

    events = [
        event
        async for event in gateway.stream_events(
            RunRequest(message="hi", thread_id="abc12345")
        )
    ]

    assert events[-1]["type"] == "done"


def test_runtime_gateways_attach_events_to_server_backend():
    threads = FakeLangGraphThreadsClient()
    client = FakeLangGraphClient(threads)
    sink = _RecordingSessionEvents()

    runtime_gateways = create_runtime_gateways(
        backend="langgraph_server",
        langgraph_client=client,
        events=sink,
    )

    assert runtime_gateways.graph_gateway.events is sink


async def test_resolve_per_run_config_extras_win_over_per_run_overrides():
    """An explicit per-run ``configurable_extra.model`` beats the session
    default — explicit injection is more specific than session state."""
    from EvoScientist.gateway.types import resolve_per_run_config

    run_config = resolve_per_run_config(
        "t1",
        {"model": "explicit-model"},
        per_run_overrides={
            "model": "session-model",
            "model_provider": "session-provider",
        },
        recursion_limit=100,
    )
    assert run_config["configurable"]["model"] == "explicit-model"
    assert run_config["configurable"]["model_provider"] == "session-provider"
    assert run_config["recursion_limit"] == 100


def test_resolve_per_run_config_default_shape_is_extras_only():
    """Without per-run overrides the shape is the always-written suppression
    key + caller extras + thread_id. Pure assembly - there is no config
    parameter to consult at all."""
    from EvoScientist.gateway.types import resolve_per_run_config

    run_config = resolve_per_run_config("t1", {"active_teams": ["a"]})
    assert run_config == {
        "configurable": {
            "hitl_suppressed": False,
            "active_teams": ["a"],
            "thread_id": "t1",
        }
    }


def test_resolve_per_run_config_hitl_suppression_local():
    """An unattended (auto_mode) run injects hitl_suppressed so the
    always-armed graph disarms the interrupt for THIS run."""
    from EvoScientist.gateway.types import resolve_per_run_config

    run_config = resolve_per_run_config("t1", None, hitl_suppressed=True)
    assert run_config["configurable"]["hitl_suppressed"] is True


def test_resolve_per_run_config_hitl_suppression_server():
    from EvoScientist.gateway.types import resolve_per_run_config

    run_config = resolve_per_run_config(
        "t1", None, per_run_overrides={"model": "m"}, hitl_suppressed=True
    )
    assert run_config["configurable"]["hitl_suppressed"] is True
    assert run_config["configurable"]["model"] == "m"


def test_resolve_per_run_config_attended_run_writes_false():
    """Every gateway run writes the key; an attended run writes it False, so a
    gateway run's arming is fixed by its own config and never falls back to the
    serving process's auto_approve on an absent key."""
    from EvoScientist.gateway.types import resolve_per_run_config

    run_config = resolve_per_run_config("t1", None, hitl_suppressed=False)
    assert run_config["configurable"]["hitl_suppressed"] is False


def test_gateway_assembled_config_is_authoritative_for_is_hitl_suppressed(monkeypatch):
    """Composition: the config resolve_per_run_config assembles, read back by
    is_hitl_suppressed, ignores the serving process's auto_approve. A suppressed
    run reads True and an attended run reads False even when the server itself
    was launched auto-approving — the server-path leak this closes."""
    import EvoScientist.EvoScientist as evo_mod
    from EvoScientist.backends import is_hitl_suppressed
    from EvoScientist.gateway.types import resolve_per_run_config

    # Serving process is auto-approving; without the always-written key this
    # would leak into every keyless run via the fallback.
    monkeypatch.setattr(
        evo_mod,
        "_ensure_config",
        lambda config=None: SimpleNamespace(auto_approve=True),
    )

    suppressed = resolve_per_run_config("t1", None, hitl_suppressed=True)
    assert is_hitl_suppressed(suppressed) is True

    attended = resolve_per_run_config("t1", None, hitl_suppressed=False)
    assert is_hitl_suppressed(attended) is False


def test_resolve_per_run_config_caller_extra_overrides_suppression():
    """An explicit per-run configurable_extra beats the derived value."""
    from EvoScientist.gateway.types import resolve_per_run_config

    run_config = resolve_per_run_config(
        "t1", {"hitl_suppressed": False}, hitl_suppressed=True
    )
    assert run_config["configurable"]["hitl_suppressed"] is False


async def test_server_gateway_resolves_per_run_overrides_from_live_session_config():
    """The gateway reads the live session config and forwards model/provider
    per run; an invalid recursion_limit is filtered out."""
    cfg = SimpleNamespace(model="m", provider="p", recursion_limit=0)
    stream = FakeLangGraphThreadStream("abc12345", events=[])
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads))
    )
    with patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg):
        async for _event in gateway.stream_events(
            RunRequest(message="hi", thread_id="abc12345")
        ):
            pass

    (start,) = stream.run.starts
    assert start["config"]["configurable"]["model"] == "m"
    assert start["config"]["configurable"]["model_provider"] == "p"
    assert start["config"]["configurable"]["thread_id"] == "abc12345"
    assert "recursion_limit" not in start["config"]


async def test_server_gateway_forwards_valid_recursion_limit_per_run():
    cfg = SimpleNamespace(model="m", provider="p", recursion_limit=100)
    stream = FakeLangGraphThreadStream("abc12345", events=[])
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads))
    )
    with patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg):
        async for _event in gateway.stream_events(
            RunRequest(message="hi", thread_id="abc12345")
        ):
            pass

    (start,) = stream.run.starts
    assert start["config"]["recursion_limit"] == 100


@pytest.mark.parametrize("resume", [False, True])
async def test_metadata_refresh_failure_does_not_drop_turn(resume, caplog):
    from langgraph.types import Command

    stream = FakeLangGraphThreadStream("abc12345", events=[])
    if resume:
        stream.interrupts = [{"interrupt_id": "first"}]
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    threads.update = AsyncMock(side_effect=RuntimeError("metadata unavailable"))
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads))
    )
    message = Command(resume={"answer": "yes"}) if resume else "hi"
    events = [
        event
        async for event in gateway.stream_events(
            RunRequest(message=message, thread_id="abc12345")
        )
    ]
    assert events[-1]["type"] == "done"
    records = [r for r in caplog.records if r.name == "EvoScientist.gateway.server"]
    assert any(
        r.levelno == logging.WARNING and "abc12345" in r.getMessage() for r in records
    )
    assert len(threads.created) == 1
    if resume:
        assert stream.run.responses == [
            {"response": {"answer": "yes"}, "interrupt_id": "first"}
        ]
        assert stream.run.starts == []
    else:
        assert stream.run.starts[0]["input"] == {
            "messages": [{"role": "user", "content": "hi"}],
            "goal_request": None,
        }


async def test_metadata_best_effort_does_not_swallow_thread_registration_failure():
    """Guard against a future over-wide try that swallows registration errors."""
    stream = FakeLangGraphThreadStream("thread", events=[])
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"thread": {"values": {}}},
        streams={"thread": stream},
    )
    threads.create = AsyncMock(side_effect=RuntimeError("registration failed"))
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads))
    )
    with pytest.raises(RuntimeError, match="registration failed"):
        async for _ in gateway.stream_events(
            RunRequest(message="hi", thread_id="thread")
        ):
            pass
    assert stream.run.starts == []


async def test_server_gateway_suppresses_hitl_for_auto_mode_session():
    """The gateway derives the run's HITL suppression from the live session
    config's auto_mode and forwards it per run."""
    cfg = SimpleNamespace(model="m", provider="p", auto_mode=True)
    stream = FakeLangGraphThreadStream("abc12345", events=[])
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads))
    )
    with patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg):
        async for _event in gateway.stream_events(
            RunRequest(message="hi", thread_id="abc12345")
        ):
            pass

    (start,) = stream.run.starts
    assert start["config"]["configurable"]["hitl_suppressed"] is True


async def test_server_gateway_never_forwards_the_session_grant():
    """A keyed fresh run on a thread parked at a declined approval would run the
    declined batch on the server, so this gateway drops the key from every run."""
    cfg = SimpleNamespace(model="m", provider="p", auto_mode=False, auto_approve=False)
    stream = FakeLangGraphThreadStream("abc12345", events=[])
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads))
    )
    with patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg):
        async for _event in gateway.stream_events(
            RunRequest(
                message="hi",
                thread_id="abc12345",
                configurable_extra={
                    "hitl_approve_all": True,
                    "active_teams": ["reviewer"],
                },
            )
        ):
            pass

    (start,) = stream.run.starts
    configurable = start["config"]["configurable"]
    assert "hitl_approve_all" not in configurable
    assert configurable["active_teams"] == ["reviewer"]
    assert configurable["hitl_suppressed"] is False


def test_local_run_config_carries_the_session_grant():
    from EvoScientist.gateway.types import resolve_per_run_config

    run_config = resolve_per_run_config(
        "t1", {"hitl_approve_all": True}, hitl_suppressed=False
    )
    assert run_config["configurable"]["hitl_approve_all"] is True
    assert run_config["configurable"]["hitl_suppressed"] is False


async def test_langgraph_server_gateway_warns_on_pre_run_state_fetch_failure(
    caplog: pytest.LogCaptureFixture,
):
    class _StateErrorThreadsClient(FakeLangGraphThreadsClient):
        async def get_state(self, thread_id: str) -> dict[str, Any]:
            raise RuntimeError("connection reset")

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
                "params": {"namespace": [], "data": {"event": "message-finish"}},
            },
        ],
    )
    threads = _StateErrorThreadsClient(
        threads=[],
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        )
    )

    with caplog.at_level("WARNING", logger="EvoScientist.gateway.server"):
        events = [
            event
            async for event in gateway.stream_events(
                RunRequest(message="hi", thread_id="abc12345")
            )
        ]

    assert any(
        "value-message processing disabled" in record.message
        and "abc12345" in record.message
        for record in caplog.records
    )
    assert events == [
        {"type": "text", "content": "hello"},
        {"type": "done", "content": "hello", "response": "hello"},
    ]


_OLD_AI = {"type": "ai", "content": "old", "id": "old-ai"}
_HUMAN = {"type": "human", "content": "hi", "id": "human-1"}
_NEW_AI = {"type": "ai", "content": "new", "id": "new-ai"}


def _value_snapshot(
    messages: list[dict[str, object]],
    *,
    namespace: list[str] | None = None,
) -> dict[str, object]:
    return {
        "method": "values",
        "params": {
            "namespace": namespace or [],
            "data": {"messages": messages},
        },
    }


def _root_text_delta(text: str) -> dict[str, object]:
    return {
        "method": "messages",
        "params": {
            "namespace": [],
            "data": {
                "event": "content-block-delta",
                "delta": {"type": "text-delta", "text": text},
            },
        },
    }


def _root_message_finish() -> dict[str, object]:
    return {
        "method": "messages",
        "params": {"namespace": [], "data": {"event": "message-finish"}},
    }


async def _collect_server_gateway_stream(
    events: list[dict[str, object]],
    *,
    state_messages: list[dict[str, object]] | None = None,
) -> list[dict[str, Any]]:
    stream = FakeLangGraphThreadStream("abc12345", events=events)
    state_values: dict[str, object] = {}
    if state_messages is not None:
        state_values["messages"] = state_messages
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"abc12345": {"values": state_values}},
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        )
    )

    async def _collect():
        return [
            event
            async for event in gateway.stream_events(
                RunRequest(message="hi", thread_id="abc12345")
            )
        ]

    return await _collect()


async def test_langgraph_server_gateway_streams_value_message_snapshots():
    events = await _collect_server_gateway_stream(
        [
            _value_snapshot([_OLD_AI, _HUMAN]),
            _value_snapshot([_OLD_AI, _HUMAN, _NEW_AI]),
        ],
        state_messages=[_OLD_AI],
    )

    assert events == [
        {"type": "text", "content": "new"},
        {"type": "done", "content": "new", "response": "new"},
    ]


async def test_langgraph_server_gateway_registers_thread_before_state_read():
    """A thread missing from the registry must not replay prior turns (#490).

    Checkpoints can exist while the server's thread registry lacks the
    thread (local backend + dev server share one checkpointer), so the
    pre-run state read only succeeds after registration. Without the early
    ensure, the read fails with NotFoundError, the suppression baselines
    stay empty, and the first values snapshot re-emits prior messages.
    """

    class _RegistryBackedStateThreadsClient(FakeLangGraphThreadsClient):
        async def get_state(self, thread_id: str) -> dict[str, Any]:
            from langgraph_sdk.errors import NotFoundError

            if all(thread.get("thread_id") != thread_id for thread in self.threads):
                raise NotFoundError(
                    "not found", response=_not_found_response(), body=None
                )
            return await super().get_state(thread_id)

    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[_value_snapshot([_OLD_AI, _HUMAN, _NEW_AI])],
    )
    threads = _RegistryBackedStateThreadsClient(
        threads=[],
        states={"abc12345": {"values": {"messages": [_OLD_AI]}}},
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        )
    )

    create_calls = 0
    original_create = threads.create

    async def _counting_create(**kwargs):
        nonlocal create_calls
        create_calls += 1
        return await original_create(**kwargs)

    threads.create = _counting_create

    events = [
        event
        async for event in gateway.stream_events(
            RunRequest(message="hi", thread_id="abc12345")
        )
    ]

    # One create: the pre-run ensure. The start/resume path must not
    # repeat it once registration succeeded. The state read happens
    # after that create, which is what lets the old turn be suppressed.
    assert create_calls == 1
    assert len(threads.created) == 1
    assert threads.state_gets
    assert events == [
        {"type": "text", "content": "new"},
        {"type": "done", "content": "new", "response": "new"},
    ]


async def test_langgraph_server_gateway_values_do_not_duplicate_message_stream():
    events = await _collect_server_gateway_stream(
        [
            _root_text_delta("new"),
            _root_message_finish(),
            _value_snapshot([_OLD_AI, _HUMAN, _NEW_AI]),
        ],
        state_messages=[_OLD_AI],
    )

    assert events == [
        {"type": "text", "content": "new"},
        {"type": "done", "content": "new", "response": "new"},
    ]


async def test_langgraph_server_gateway_ignores_non_root_value_messages():
    events = await _collect_server_gateway_stream(
        [
            _value_snapshot(
                [{"type": "ai", "content": "subagent text", "id": "subagent-ai"}],
                namespace=["research:task-1"],
            )
        ],
    )

    assert not any(event.get("type") == "text" for event in events)
    assert events[-1] == {"type": "done", "content": "", "response": ""}


async def test_langgraph_server_gateway_emits_state_interrupt_before_done():
    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[],
        interrupts=[{"interrupt_id": "interrupt-1", "value": None}],
        interrupted=True,
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={
            "abc12345": {
                "values": {},
                "interrupts": [
                    {
                        "id": "interrupt-1",
                        "value": {
                            "action_requests": [
                                {
                                    "name": "execute",
                                    "args": {"command": "echo hello"},
                                    "id": "tool-1",
                                }
                            ],
                            "review_configs": [
                                {
                                    "action_name": "execute",
                                    "allowed_decisions": ["approve", "reject"],
                                }
                            ],
                        },
                    }
                ],
            }
        },
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        )
    )

    async def _collect():
        return [
            event
            async for event in gateway.stream_events(
                RunRequest(message="hi", thread_id="abc12345")
            )
        ]

    events = await _collect()

    assert events == [
        {
            "type": "interrupt",
            "interrupt_id": "interrupt-1",
            "action_requests": [
                {
                    "name": "execute",
                    "args": {"command": "echo hello"},
                    "id": "tool-1",
                }
            ],
            "review_configs": [
                {
                    "action_name": "execute",
                    "allowed_decisions": ["approve", "reject"],
                }
            ],
        },
        {"type": "done", "content": "", "response": ""},
    ]


@pytest.mark.parametrize(
    ("interrupts", "paused"), [([{"id": "i1"}], True), ([], False)]
)
def test_the_server_tracker_says_when_a_child_only_paused(interrupts, paused):
    """The result of the task a child run belongs to tells whether the child
    stopped at an interrupt or finished."""
    from EvoScientist.gateway.server import _ServerSubagentTracker
    from EvoScientist.stream.emitter import StreamEventEmitter
    from EvoScientist.stream.events import _SubagentRegistry

    tracker = _ServerSubagentTracker(
        emitter=StreamEventEmitter(), registry=_SubagentRegistry()
    )
    started = {
        "method": "lifecycle",
        "params": {"namespace": ["tools:task-1"], "data": {"event": "started"}},
    }
    result = {
        "method": "tasks",
        "params": {
            "namespace": [],
            "data": {
                "id": "task-1",
                "name": "tools",
                "error": None,
                "interrupts": interrupts,
                "result": {},
            },
        },
    }
    events = [*tracker.process(started), *tracker.process(result)]
    (end,) = [event for event in events if event["type"] == "subagent_end"]
    assert bool(end.get("paused")) is paused
    assert ("paused" in end) is paused


def test_the_server_tracker_does_not_call_a_child_paused_when_the_stream_stops():
    """A child that is still open when the stream ends gets an end event, and
    nothing says that it paused."""
    from EvoScientist.gateway.server import _ServerSubagentTracker
    from EvoScientist.stream.emitter import StreamEventEmitter
    from EvoScientist.stream.events import _SubagentRegistry

    tracker = _ServerSubagentTracker(
        emitter=StreamEventEmitter(), registry=_SubagentRegistry()
    )
    started = {
        "method": "lifecycle",
        "params": {"namespace": ["tools:task-1"], "data": {"event": "started"}},
    }
    events = [*tracker.process(started), *tracker.finish()]
    (end,) = [event for event in events if event["type"] == "subagent_end"]
    assert "paused" not in end


async def test_langgraph_server_gateway_streams_subagent_protocol_events():
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
            client=FakeLangGraphClient(threads),
        )
    )

    async def _collect():
        return [
            event
            async for event in gateway.stream_events(
                RunRequest(message="hi", thread_id="abc12345")
            )
        ]

    events = await _collect()

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


async def test_langgraph_server_gateway_resumes_interrupt_with_thread_stream():
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
            client=FakeLangGraphClient(threads),
        )
    )

    async def _collect():
        return [
            event
            async for event in gateway.stream_events(
                RunRequest(
                    message=Command(
                        resume={"interrupt-1": {"decisions": [{"allowed": True}]}}
                    ),
                    thread_id="abc12345",
                )
            )
        ]

    events = await _collect()

    assert stream.run.starts == []
    assert stream.run.responses == [
        {
            "response": {"decisions": [{"allowed": True}]},
            "interrupt_id": "interrupt-1",
        }
    ]
    assert events == [{"type": "done", "content": "", "response": ""}]


async def test_langgraph_server_gateway_resume_raises_on_poll_timeout_without_state_lookup():
    """Poll timeout raises even when thread state holds a pending interrupt.

    The server replays pending interrupts to a fresh stream within ~1s, so an
    empty stream.interrupts after the wait means nothing was replayed. The
    gateway must NOT fall back to a state-derived interrupt id: run.respond
    validates the id against stream.interrupts, so a state-recovered id would
    always be rejected there anyway.
    """
    from langgraph.types import Command

    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[],
        interrupts=[],
    )
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={
            "abc12345": {
                "values": {},
                "interrupts": [
                    {
                        "id": "state-interrupt-1",
                        "value": {
                            "action_requests": [
                                {
                                    "name": "execute",
                                    "args": {"command": "echo hi"},
                                    "id": "tool-1",
                                }
                            ],
                            "review_configs": [
                                {
                                    "action_name": "execute",
                                    "allowed_decisions": ["approve", "reject"],
                                }
                            ],
                        },
                    }
                ],
            }
        },
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        ),
        interrupt_wait_seconds=0.01,
    )

    with pytest.raises(RuntimeError, match="No interrupt replayed"):
        async for _event in gateway.stream_events(
            RunRequest(
                message=Command(
                    resume={"state-interrupt-1": {"decisions": [{"allowed": True}]}}
                ),
                thread_id="abc12345",
            )
        ):
            pass
    assert stream.run.responses == []


async def test_langgraph_server_gateway_resume_raises_on_id_mismatch():
    from langgraph.types import Command

    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[],
        interrupts=[{"interrupt_id": "interrupt-1"}],
    )
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        )
    )

    with pytest.raises(RuntimeError, match="does not match the pending interrupt"):
        async for _event in gateway.stream_events(
            RunRequest(
                message=Command(
                    resume={"stale-interrupt": {"decisions": [{"allowed": True}]}}
                ),
                thread_id="abc12345",
            )
        ):
            pass
    assert stream.run.responses == []


async def test_langgraph_server_gateway_resume_forwards_non_hitl_single_key_payload():
    """ask_user resumes ({"status": "cancelled"}) are single-key but keyed by
    their own schema, not the interrupt id - they must forward unchanged
    rather than trip the HITL id-mismatch raise."""
    from langgraph.types import Command

    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[],
        interrupts=[{"interrupt_id": "interrupt-1"}],
    )
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        )
    )

    async for _event in gateway.stream_events(
        RunRequest(
            message=Command(resume={"status": "cancelled"}),
            thread_id="abc12345",
        )
    ):
        pass
    assert stream.run.responses == [
        {"response": {"status": "cancelled"}, "interrupt_id": "interrupt-1"}
    ]


async def test_langgraph_server_gateway_resume_matches_one_of_multiple_interrupts():
    from langgraph.types import Command

    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[],
        interrupts=[{"interrupt_id": "interrupt-a"}, {"interrupt_id": "interrupt-b"}],
    )
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        ),
    )

    async for _event in gateway.stream_events(
        RunRequest(
            message=Command(resume={"interrupt-a": {"decisions": [{"allowed": True}]}}),
            thread_id="abc12345",
        )
    ):
        pass

    assert stream.run.responses == [
        {
            "response": {"decisions": [{"allowed": True}]},
            "interrupt_id": "interrupt-a",
        }
    ]


async def test_langgraph_server_gateway_resume_raises_on_unmatched_multiple_interrupts():
    from langgraph.types import Command

    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[],
        interrupts=[{"interrupt_id": "interrupt-a"}, {"interrupt_id": "interrupt-b"}],
    )
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        ),
    )

    with pytest.raises(RuntimeError, match="2 pending interrupts"):
        async for _event in gateway.stream_events(
            RunRequest(
                message=Command(
                    resume={"interrupt-z": {"decisions": [{"allowed": True}]}}
                ),
                thread_id="abc12345",
            )
        ):
            pass


async def test_langgraph_server_gateway_resume_raises_on_no_interrupt():
    from langgraph.types import Command

    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[],
        interrupts=[],
    )
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(
            client=FakeLangGraphClient(threads),
        ),
        interrupt_wait_seconds=0.01,
    )

    with pytest.raises(RuntimeError, match="No interrupt replayed"):
        async for _event in gateway.stream_events(
            RunRequest(
                message=Command(
                    resume={"unknown-id": {"decisions": [{"allowed": True}]}}
                ),
                thread_id="abc12345",
            )
        ):
            pass


async def test_langgraph_server_gateway_abort_during_subagent_does_not_raise():
    """aclose() mid-subagent must not raise RuntimeError from yields in finally."""
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
                        "delta": {"type": "text-delta", "text": "partial"},
                    },
                },
            },
        ],
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    client = FakeLangGraphClient(threads)

    class _FakeRunsClient:
        async def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            return []

        async def cancel_many(self, *, thread_id: str, run_ids):
            pass

    client.runs = _FakeRunsClient()
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=client),
    )

    gen = gateway.stream_events(RunRequest(message="hi", thread_id="abc12345"))
    await gen.__anext__()
    # Consumer aborts while a subagent is still tracked
    await gen.aclose()


async def test_langgraph_server_gateway_clears_stuck_state_after_run_failure():
    class _FailingStream(FakeLangGraphThreadStream):
        async def _iter_raw_events(self):
            for event in self.events:
                yield event
            raise RuntimeError("provider connection lost")

    failing_stream = _FailingStream(
        "abc12345",
        events=[
            {
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": "partial"},
                    },
                },
            },
        ],
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={
            "abc12345": {
                "values": {},
                "next": ("model",),
            }
        },
        streams={"abc12345": failing_stream},
    )
    client = FakeLangGraphClient(threads)

    class _FakeRunsClient:
        async def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            return []

        async def cancel_many(self, *, thread_id: str, run_ids):
            pass

    client.runs = _FakeRunsClient()
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=client),
    )

    with pytest.raises(RuntimeError, match="provider connection lost"):
        async for _event in gateway.stream_events(
            RunRequest(message="hi", thread_id="abc12345")
        ):
            pass

    assert threads.state_updates == [
        ("abc12345", None, "__end__"),
    ]


@pytest.mark.parametrize(
    "failure",
    [
        "registration",
        "input",
        "resume-validation",
        "non-resume-command",
        "unmatched-id-keyed",
        "wrong-hitl-id",
        "start",
        "respond",
        "multi-respond",
    ],
)
async def test_state_repair_requires_this_request_to_attempt_a_run(
    monkeypatch, failure
):
    from langgraph.types import Command

    stream = FakeLangGraphThreadStream("thread", events=[])
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"thread": {"values": {}, "next": ("model",)}},
        streams={"thread": stream},
    )
    client = FakeLangGraphClient(threads)

    class _FakeRunsClient:
        async def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            return []

        async def cancel_many(self, *, thread_id: str, run_ids):
            pass

    client.runs = _FakeRunsClient()
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=client),
        interrupt_wait_seconds=0,
    )
    message = "hi"
    if failure == "registration":
        threads.create = AsyncMock(side_effect=RuntimeError("before run"))
    elif failure == "input":
        monkeypatch.setattr(
            "EvoScientist.gateway.server.build_agent_stream_input",
            AsyncMock(side_effect=RuntimeError("before run")),
        )
    elif failure == "resume-validation":
        message = Command(resume={"answer": "yes"})
    elif failure == "non-resume-command":
        message = Command(update={"configurable": {"model": "test"}})
    elif failure == "unmatched-id-keyed":
        stream.interrupts = [
            {"interrupt_id": "first"},
            {"interrupt_id": "second"},
        ]
        message = Command(resume={"unknown": {"decisions": [{"type": "approve"}]}})
    elif failure == "wrong-hitl-id":
        stream.interrupts = [{"interrupt_id": "first"}]
        message = Command(resume={"wrong": {"decisions": [{"type": "approve"}]}})
    elif failure == "start":
        stream.run.start = AsyncMock(side_effect=RuntimeError("start attempted"))
    else:
        stream.interrupts = [{"interrupt_id": "first"}]
        if failure == "multi-respond":
            stream.interrupts.append({"interrupt_id": "second"})
        message = Command(resume={"first": {"decisions": [{"type": "approve"}]}})
        stream.run.respond = AsyncMock(side_effect=RuntimeError("respond attempted"))
    with pytest.raises(RuntimeError):
        async for _ in gateway.stream_events(
            RunRequest(message=message, thread_id="thread")
        ):
            pass
    expected = (
        [("thread", None, "__end__")]
        if failure in {"start", "respond", "multi-respond"}
        else []
    )
    assert threads.state_updates == expected


async def test_langgraph_server_gateway_repairs_state_when_consumer_closes_after_error_event():
    """A consumer that stops iterating once it has the error event triggers
    GeneratorExit at the yield - the repair must already have run."""

    class _FailingStream(FakeLangGraphThreadStream):
        async def _iter_raw_events(self):
            for event in self.events:
                yield event
            raise RuntimeError("provider connection lost")

    failing_stream = _FailingStream(
        "abc12345",
        events=[],
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={
            "abc12345": {
                "values": {},
                "next": ("model",),
            }
        },
        streams={"abc12345": failing_stream},
    )
    client = FakeLangGraphClient(threads)

    class _FakeRunsClient:
        async def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            return []

        async def cancel_many(self, *, thread_id: str, run_ids):
            pass

    client.runs = _FakeRunsClient()
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=client),
    )

    gen = gateway.stream_events(RunRequest(message="hi", thread_id="abc12345"))
    async for event in gen:
        assert event["type"] == "error"
        break  # consumer abandons the generator after the error event
    await gen.aclose()

    assert threads.state_updates == [
        ("abc12345", None, "__end__"),
    ]


async def test_langgraph_server_gateway_preserves_hitl_interrupt_after_run_failure():
    class _FailingStream(FakeLangGraphThreadStream):
        async def _iter_raw_events(self):
            for event in self.events:
                yield event
            raise RuntimeError("provider connection lost")

    failing_stream = _FailingStream(
        "abc12345",
        events=[
            {
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": "partial"},
                    },
                },
            },
        ],
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={
            "abc12345": {
                "values": {},
                "next": ("model",),
                "interrupts": [
                    {"id": "hitl-interrupt-1", "value": None},
                ],
            }
        },
        streams={"abc12345": failing_stream},
    )
    client = FakeLangGraphClient(threads)

    class _FakeRunsClient:
        async def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            return []

        async def cancel_many(self, *, thread_id: str, run_ids):
            pass

    client.runs = _FakeRunsClient()
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=client),
    )

    with pytest.raises(RuntimeError, match="provider connection lost"):
        async for _event in gateway.stream_events(
            RunRequest(message="hi", thread_id="abc12345")
        ):
            pass

    assert threads.state_gets.count("abc12345") >= 2
    assert threads.state_updates == []


async def test_langgraph_server_gateway_repair_swallows_update_state_failure(
    caplog: pytest.LogCaptureFixture,
):
    class _FailingStream(FakeLangGraphThreadStream):
        async def _iter_raw_events(self):
            for event in self.events:
                yield event
            raise RuntimeError("provider connection lost")

    failing_stream = _FailingStream(
        "abc12345",
        events=[
            {
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": "partial"},
                    },
                },
            },
        ],
    )

    class _FailingUpdateThreadsClient(FakeLangGraphThreadsClient):
        async def update_state(self, thread_id, values, *, as_node=None):
            raise RuntimeError("repair-time connection reset")

    threads = _FailingUpdateThreadsClient(
        threads=[],
        states={"abc12345": {"values": {}, "next": ("model",)}},
        streams={"abc12345": failing_stream},
    )
    client = FakeLangGraphClient(threads)

    class _FakeRunsClient:
        async def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            return []

        async def cancel_many(self, *, thread_id: str, run_ids):
            pass

    client.runs = _FakeRunsClient()
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=client),
    )

    with caplog.at_level("DEBUG", logger="EvoScientist.gateway.server"):
        with pytest.raises(RuntimeError, match="provider connection lost"):
            async for _event in gateway.stream_events(
                RunRequest(message="hi", thread_id="abc12345")
            ):
                pass

    assert any(
        "Could not clear interrupted thread state" in record.message
        for record in caplog.records
    )


async def test_langgraph_server_gateway_cancels_only_its_run_on_consumer_abort():
    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[
            {
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": "partial"},
                    },
                },
            },
        ],
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    client = FakeLangGraphClient(threads)

    cancel_calls: list[tuple[str, list[str]]] = []

    class _FakeRunsClient:
        async def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            if status == "running":
                return [
                    {"run_id": "run-1", "status": "running"},
                    {"run_id": "other-run", "status": "running"},
                ]
            return []

        async def cancel_many(self, *, thread_id: str, run_ids):
            cancel_calls.append((thread_id, list(run_ids)))

    client.runs = _FakeRunsClient()

    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=client),
    )

    gen = gateway.stream_events(RunRequest(message="hi", thread_id="abc12345"))
    await gen.__anext__()
    await gen.aclose()

    assert cancel_calls == [("abc12345", ["run-1"])]


@pytest.mark.parametrize(
    ("active_after_respond", "expected_run_ids"),
    [
        (
            [
                {
                    "run_id": "resumed-run",
                    "status": "running",
                    "kwargs": {
                        "command": {
                            "resume": {
                                "interrupt-1": {"decisions": [{"type": "approve"}]}
                            }
                        }
                    },
                },
                {
                    "run_id": "concurrent-fresh-run",
                    "status": "running",
                    "kwargs": {"command": None},
                },
            ],
            ["resumed-run"],
        ),
        (
            [
                {
                    "run_id": "unrelated-new-run",
                    "status": "running",
                    "kwargs": {"command": None},
                }
            ],
            [],
        ),
        (
            [
                {
                    "run_id": "pending-resumed-run",
                    "status": "pending",
                    "kwargs": {
                        "command": {
                            "resume": {
                                "interrupt-1": {"decisions": [{"type": "approve"}]}
                            }
                        }
                    },
                }
            ],
            ["pending-resumed-run"],
        ),
        (
            [
                {
                    "run_id": "different-resume",
                    "status": "running",
                    "kwargs": {
                        "command": {"resume": {"interrupt-2": {"decisions": []}}}
                    },
                }
            ],
            [],
        ),
        (
            [
                {
                    "run_id": "different-response",
                    "status": "running",
                    "kwargs": {
                        "command": {
                            "resume": {
                                "interrupt-1": {"decisions": [{"type": "reject"}]}
                            }
                        }
                    },
                }
            ],
            [],
        ),
        (
            [
                {
                    "run_id": "matching-resume-a",
                    "status": "running",
                    "kwargs": {
                        "command": {
                            "resume": {
                                "interrupt-1": {"decisions": [{"type": "approve"}]}
                            }
                        }
                    },
                },
                {
                    "run_id": "matching-resume-b",
                    "status": "running",
                    "kwargs": {
                        "command": {
                            "resume": {
                                "interrupt-1": {"decisions": [{"type": "approve"}]}
                            }
                        }
                    },
                },
                {"run_id": "missing-kwargs", "status": "running"},
                {
                    "run_id": "malformed-command",
                    "status": "running",
                    "kwargs": {"command": []},
                },
            ],
            [],
        ),
    ],
    ids=[
        "matching-resume-among-concurrent-runs",
        "resumed-run-finished-before-list",
        "matching-pending-resume",
        "different-interrupt",
        "same-interrupt-different-response",
        "ambiguous-matches-and-malformed-records",
    ],
)
async def test_server_gateway_scopes_resumed_run_on_consumer_abort(
    active_after_respond: list[dict[str, Any]],
    expected_run_ids: list[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch,
):
    from langgraph.types import Command

    stream = FakeLangGraphThreadStream(
        "abc12345",
        events=[
            {
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": "partial"},
                    },
                },
            },
        ],
        interrupts=[{"interrupt_id": "interrupt-1"}],
    )
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    client = FakeLangGraphClient(threads)
    cancel_calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr("EvoScientist.gateway.server._RUN_LOOKUP_PAGE_SIZE", 1)

    class _FakeRunsClient:
        async def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            runs = [{"run_id": "other-run", "status": "running"}]
            if stream.run.responses:
                runs.extend(active_after_respond)
            return [run for run in runs if run["status"] == status][
                offset : offset + limit
            ]

        async def cancel_many(self, *, thread_id: str, run_ids):
            cancel_calls.append((thread_id, list(run_ids)))

    client.runs = _FakeRunsClient()
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))

    gen = gateway.stream_events(
        RunRequest(
            message=Command(resume={"decisions": [{"type": "approve"}]}),
            thread_id="abc12345",
        )
    )
    with caplog.at_level("WARNING", logger="EvoScientist.gateway.server"):
        await gen.__anext__()
        await gen.aclose()

    assert cancel_calls == (
        [("abc12345", expected_run_ids)] if expected_run_ids else []
    )
    if not expected_run_ids:
        assert any(
            "Could not identify resumed run" in record.message
            for record in caplog.records
        )


@pytest.mark.parametrize("multiple_interrupts", [False, True])
async def test_server_gateway_successful_resume_does_not_list_active_runs(
    multiple_interrupts, caplog
):
    """Successful resumes need no ownership lookup or ambiguity warnings."""
    from langgraph.types import Command

    interrupts = [{"interrupt_id": "interrupt-1"}]
    if multiple_interrupts:
        interrupts.append({"interrupt_id": "interrupt-2"})
    stream = FakeLangGraphThreadStream("thread", interrupts=interrupts)
    threads = FakeLangGraphThreadsClient(
        states={"thread": {"values": {}}}, streams={"thread": stream}
    )
    client = FakeLangGraphClient(threads)
    client.runs = SimpleNamespace(list=AsyncMock(return_value=[]))
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))

    with caplog.at_level("WARNING", logger="EvoScientist.gateway.server"):
        async for _ in gateway.stream_events(
            RunRequest(
                message=Command(
                    resume={"interrupt-1": {"decisions": [{"type": "approve"}]}}
                ),
                thread_id="thread",
            )
        ):
            pass

    assert stream.run.responses == [
        {
            "interrupt_id": "interrupt-1",
            "response": {"decisions": [{"type": "approve"}]},
        }
    ]
    client.runs.list.assert_not_awaited()
    assert not any("resumed run" in record.message for record in caplog.records)


@pytest.mark.parametrize("failure", ["timeout", "cancel"])
@pytest.mark.parametrize(
    ("run_status", "multiple_interrupts", "response"),
    [
        ("pending", False, 0),
        ("running", False, None),
        ("pending", True, {"decisions": [{"type": "approve"}]}),
        ("running", True, {"decisions": [{"type": "approve"}]}),
    ],
)
@pytest.mark.parametrize("has_neighbour", [False, True])
async def test_server_gateway_cleans_up_resume_accepted_before_respond_finishes(
    failure, run_status, multiple_interrupts, response, has_neighbour, monkeypatch
):
    """Record resume identity before the awaited respond can fail or cancel."""
    import asyncio

    from langgraph.types import Command

    interrupts = [{"interrupt_id": "interrupt-1"}]
    if multiple_interrupts:
        interrupts.append({"interrupt_id": "interrupt-2"})
    stream = FakeLangGraphThreadStream("thread", interrupts=interrupts)
    accepted = asyncio.Event()
    respond = stream.run.respond

    async def _accepted_then_interrupted(response, *, interrupt_id=None):
        await respond(response, interrupt_id=interrupt_id)
        accepted.set()
        if failure == "timeout":
            raise TimeoutError("run.respond response timed out")
        await asyncio.Event().wait()

    stream.run.respond = _accepted_then_interrupted
    threads = FakeLangGraphThreadsClient(
        states={"thread": {"values": {}, "next": ("model",)}},
        streams={"thread": stream},
    )
    client = FakeLangGraphClient(threads)
    cancelled = False
    cancel_calls = []
    # Exercise pagination as well as both active statuses; the neighbour shares
    # the interrupt id but has a different response and must never be cancelled.
    monkeypatch.setattr("EvoScientist.gateway.server._RUN_LOOKUP_PAGE_SIZE", 1)

    class _FakeRunsClient:
        async def list(self, thread_id, *, limit, offset, status):
            runs = []
            if has_neighbour:
                runs.append(
                    {
                        "run_id": "other-run",
                        "status": "running",
                        "kwargs": {"command": {"resume": {"interrupt-1": "other"}}},
                    }
                )
            if accepted.is_set() and not cancelled:
                runs.append(
                    {
                        "run_id": "accepted-resume",
                        "status": run_status,
                        "kwargs": {"command": {"resume": {"interrupt-1": response}}},
                    }
                )
            return [run for run in runs if run["status"] == status][
                offset : offset + limit
            ]

        async def cancel_many(self, *, thread_id, run_ids):
            nonlocal cancelled
            cancel_calls.append((thread_id, list(run_ids)))
            cancelled = True

    client.runs = _FakeRunsClient()
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))
    gen = gateway.stream_events(
        RunRequest(
            message=Command(resume={"interrupt-1": response}), thread_id="thread"
        )
    )
    if failure == "timeout":
        event = await gen.__anext__()
        assert event["type"] == "error"
    else:
        pending = asyncio.create_task(gen.__anext__())
        await asyncio.wait_for(accepted.wait(), timeout=2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    # On the error path this must already hold before the first yield, since
    # the consumer may abandon the generator as soon as it receives the error.
    assert cancel_calls == [("thread", ["accepted-resume"])]
    assert threads.state_updates == (
        [("thread", None, "__end__")]
        if failure == "timeout" and not has_neighbour
        else []
    )
    await gen.aclose()
    assert cancel_calls == [("thread", ["accepted-resume"])]


async def test_server_gateway_resume_cleanup_skips_incomplete_run_listing(
    monkeypatch, caplog
):
    """A matching first page is not enough if a later page cannot be read."""
    from langgraph.types import Command

    stream = FakeLangGraphThreadStream(
        "thread",
        events=[
            {
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": {
                        "event": "content-block-delta",
                        "delta": {"type": "text-delta", "text": "partial"},
                    },
                },
            }
        ],
        interrupts=[{"interrupt_id": "interrupt-1"}],
    )
    threads = FakeLangGraphThreadsClient(
        states={"thread": {"values": {}}}, streams={"thread": stream}
    )
    client = FakeLangGraphClient(threads)
    cancel = AsyncMock()
    monkeypatch.setattr("EvoScientist.gateway.server._RUN_LOOKUP_PAGE_SIZE", 1)

    async def _list(thread_id, *, limit, offset, status):
        if not stream.run.responses or status != "pending":
            return []
        if offset:
            raise TimeoutError("second page unavailable")
        return [
            {
                "run_id": "matching-first-page",
                "status": "pending",
                "kwargs": {"command": {"resume": {"interrupt-1": "yes"}}},
            }
        ]

    client.runs = SimpleNamespace(list=_list, cancel_many=cancel)
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))
    gen = gateway.stream_events(
        RunRequest(message=Command(resume="yes"), thread_id="thread")
    )
    with caplog.at_level("WARNING", logger="EvoScientist.gateway.server"):
        await gen.__anext__()
        await gen.aclose()

    cancel.assert_not_awaited()
    assert threads.state_updates == []
    assert any("Could not list active runs" in r.message for r in caplog.records)


async def test_server_gateway_failure_does_not_cancel_or_repair_another_run():
    class _FailingStream(FakeLangGraphThreadStream):
        async def _iter_raw_events(self):
            raise RuntimeError("subscription failed")
            yield  # pragma: no cover

    stream = _FailingStream("abc12345")
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"abc12345": {"values": {}, "next": ("slow",)}},
        streams={"abc12345": stream},
    )
    client = FakeLangGraphClient(threads)
    cancel_calls: list[tuple[str, list[str]]] = []

    class _FakeRunsClient:
        async def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            if status == "running":
                return [{"run_id": "other-run", "status": "running"}]
            if status == "pending":
                return [{"run_id": "run-1", "status": "pending"}]
            return []

        async def cancel_many(self, *, thread_id: str, run_ids):
            cancel_calls.append((thread_id, list(run_ids)))

    client.runs = _FakeRunsClient()
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))

    with pytest.raises(RuntimeError, match="subscription failed"):
        async for _event in gateway.stream_events(
            RunRequest(message="hi", thread_id="abc12345")
        ):
            pass

    assert cancel_calls == [("abc12345", ["run-1"])]
    assert threads.state_updates == []


async def test_server_gateway_recovers_run_accepted_before_start_timeout():
    stream = FakeLangGraphThreadStream("abc12345")

    async def _accepted_then_timeout(**kwargs):
        stream.run.starts.append(kwargs)
        raise TimeoutError("run.start response timed out")

    stream.run.start = _accepted_then_timeout
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    client = FakeLangGraphClient(threads)
    cancel_calls: list[tuple[str, list[str]]] = []
    cancelled = False

    class _FakeRunsClient:
        async def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            if cancelled or status != "running":
                return []
            return [
                {
                    "run_id": "accepted-run",
                    "status": "running",
                    "metadata": stream.run.starts[0]["metadata"],
                },
                {
                    "run_id": "other-run",
                    "status": "success",
                    "metadata": {},
                },
            ]

        async def cancel_many(self, *, thread_id: str, run_ids):
            nonlocal cancelled
            cancel_calls.append((thread_id, list(run_ids)))
            cancelled = True

    client.runs = _FakeRunsClient()
    gateway = LangGraphServerGateway(LangGraphServerThreadStore(client=client))

    with pytest.raises(TimeoutError, match=r"run\.start response timed out"):
        async for _event in gateway.stream_events(
            RunRequest(message="hi", thread_id="abc12345")
        ):
            pass

    assert cancel_calls == [("abc12345", ["accepted-run"])]
    metadata = stream.run.starts[0]["metadata"]
    assert metadata["_evoscientist_gateway_request_id"]


async def test_langgraph_server_gateway_does_not_cancel_on_normal_completion():
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
    client = FakeLangGraphClient(threads)

    cancel_calls: list[tuple[str, list[str]]] = []

    class _FakeRunsClient:
        async def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            return []

        async def cancel_many(self, *, thread_id: str, run_ids):
            cancel_calls.append((thread_id, list(run_ids)))

    client.runs = _FakeRunsClient()

    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=client),
    )

    events = [
        event
        async for event in gateway.stream_events(
            RunRequest(message="hi", thread_id="abc12345")
        )
    ]

    assert cancel_calls == []
    assert events == [
        {"type": "text", "content": "hello"},
        {"type": "done", "content": "hello", "response": "hello"},
    ]


async def test_langgraph_server_thread_store_cancels_runs_before_delete():
    events: list[tuple[str, object]] = []

    class _RecordingThreadsClient(FakeLangGraphThreadsClient):
        async def delete(self, thread_id: str) -> None:
            events.append(("delete", thread_id))
            await super().delete(thread_id)

    threads = _RecordingThreadsClient(threads=[{"thread_id": "abc12345"}])
    client = FakeLangGraphClient(threads)

    class _FakeRunsClient:
        async def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            if status == "pending":
                return [{"run_id": "run-pending", "status": "pending"}]
            return []

        async def cancel_many(self, *, thread_id: str, run_ids):
            events.append(("cancel_many", list(run_ids)))

    client.runs = _FakeRunsClient()
    store = LangGraphServerThreadStore(client=client)

    assert await store.delete_thread("abc12345") is True
    assert events == [
        ("cancel_many", ["run-pending"]),
        ("delete", "abc12345"),
    ]
    assert threads.deleted == ["abc12345"]


async def test_langgraph_server_thread_store_delete_survives_missing_runs_client():
    threads = FakeLangGraphThreadsClient(threads=[{"thread_id": "abc12345"}])
    store = LangGraphServerThreadStore(client=FakeLangGraphClient(threads))

    assert await store.delete_thread("abc12345") is True
    assert threads.deleted == ["abc12345"]


# ---------------------------------------------------------------------------
# Stage 1f — parity suite: shared contract across both backends + divergence pins
# ---------------------------------------------------------------------------


@pytest.fixture
def local_gateway() -> LocalGraphGateway:
    return LocalGraphGateway(
        thread_store=FakeThreadStore(
            generated_thread_id="parity-1",
            threads=[{"thread_id": "parity-1"}],
            resolved_thread_id="parity-1",
            metadata={"workspace_dir": "/ws"},
            messages=[HumanMessage(content="hi")],
            exists=True,
            deleted=True,
        )
    )


@pytest.fixture
def server_gateway() -> LangGraphServerGateway:
    threads = FakeLangGraphThreadsClient(
        threads=[
            {
                "thread_id": "parity-1",
                "metadata": {"graph_id": "EvoScientist", "workspace_dir": "/ws"},
            }
        ],
        states={
            "parity-1": {
                "values": {
                    "messages": [
                        {"role": "user", "content": "hi"},
                    ]
                }
            }
        },
    )
    return LangGraphServerGateway(
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads))
    )


@pytest.mark.parametrize(
    "fixture_name",
    ["local_gateway", "server_gateway"],
)
async def test_gateway_contract_create_returns_str(request, fixture_name):
    gateway = request.getfixturevalue(fixture_name)
    thread_id = await gateway.create_thread()
    assert isinstance(thread_id, str)
    assert thread_id


@pytest.mark.parametrize(
    "fixture_name",
    ["local_gateway", "server_gateway"],
)
async def test_gateway_contract_list_returns_list_of_dicts(request, fixture_name):
    gateway = request.getfixturevalue(fixture_name)
    threads = await gateway.list_threads(limit=10)
    assert isinstance(threads, list)
    assert all(isinstance(row, dict) for row in threads)
    assert all("thread_id" in row for row in threads)


@pytest.mark.parametrize(
    "fixture_name",
    ["local_gateway", "server_gateway"],
)
async def test_gateway_contract_resolve_returns_thread_resolution(
    request, fixture_name
):
    gateway = request.getfixturevalue(fixture_name)
    resolution = await gateway.resolve_thread("parity")
    assert resolution.thread_id == "parity-1"
    assert resolution.found
    assert not resolution.ambiguous


@pytest.mark.parametrize(
    "fixture_name",
    ["local_gateway", "server_gateway"],
)
async def test_gateway_contract_get_metadata_returns_dict(request, fixture_name):
    gateway = request.getfixturevalue(fixture_name)
    metadata = await gateway.get_thread_metadata("parity-1")
    assert isinstance(metadata, dict)
    assert metadata.get("workspace_dir") == "/ws"


@pytest.mark.parametrize(
    "fixture_name",
    ["local_gateway", "server_gateway"],
)
async def test_gateway_contract_get_messages_returns_list(request, fixture_name):
    gateway = request.getfixturevalue(fixture_name)
    messages = await gateway.get_thread_messages("parity-1")
    assert isinstance(messages, list)
    assert len(messages) >= 1


@pytest.mark.parametrize(
    "fixture_name",
    ["local_gateway", "server_gateway"],
)
async def test_gateway_contract_thread_exists_returns_true_for_known_thread(
    request, fixture_name
):
    gateway = request.getfixturevalue(fixture_name)
    exists = await gateway.thread_exists("parity-1")
    assert exists is True


@pytest.mark.parametrize(
    "fixture_name",
    ["local_gateway", "server_gateway"],
)
async def test_gateway_contract_delete_returns_bool(request, fixture_name):
    gateway = request.getfixturevalue(fixture_name)
    deleted = await gateway.delete_thread("parity-1")
    assert deleted is True


# Divergence pins — accepted differences between backends


async def test_divergence_local_clone_unsupported():
    """Local gateway does not support thread cloning; server gateway does."""
    gateway = LocalGraphGateway(thread_store=FakeThreadStore())
    with pytest.raises(NotImplementedError, match="does not support thread cloning"):
        await gateway.clone_thread("source")


async def test_divergence_server_delete_cancels_in_flight_runs():
    """Server delete cancels live runs before deleting; local delete does not."""
    threads = FakeLangGraphThreadsClient(threads=[{"thread_id": "abc12345"}])
    client = FakeLangGraphClient(threads)
    cancel_called = False

    class _FakeRunsClient:
        async def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            if status == "running":
                return [{"run_id": "run-1", "status": "running"}]
            return []

        async def cancel_many(self, *, thread_id: str, run_ids):
            nonlocal cancel_called
            cancel_called = True

    client.runs = _FakeRunsClient()
    store = LangGraphServerThreadStore(client=client)
    await store.delete_thread("abc12345")
    assert cancel_called


async def test_divergence_server_thread_exists_returns_false_for_unknown_thread():
    """Server thread_exists hits the SDK (404 on missing); local delegates to
    session_store which may behave differently for unknown ids."""
    threads = FakeLangGraphThreadsClient(threads=[])
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads))
    )
    assert await gateway.thread_exists("nonexistent") is False


async def test_divergence_local_delete_does_not_cancel_runs():
    """Local delete delegates to session_store without run cancellation."""
    thread_store = FakeThreadStore(
        threads=[{"thread_id": "abc12345"}],
        deleted=True,
    )
    gateway = LocalGraphGateway(thread_store=thread_store)
    deleted = await gateway.delete_thread("abc12345")
    assert deleted is True
    assert thread_store.calls == [("delete_thread", "abc12345")]


async def test_divergence_server_enqueue_serializes_concurrent_runs():
    """Server gateway accepts a second run while one is in flight (enqueue);
    local gateway has no cross-process guard. The serialization is a property
    of the langgraph dev server, not the gateway — this test pins the contract:
    both backends accept the second request without rejection."""
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={"abc12345": {"values": {}}},
        streams={
            "abc12345": FakeLangGraphThreadStream(
                "abc12345",
                events=[
                    {
                        "method": "messages",
                        "params": {
                            "namespace": [],
                            "data": {
                                "event": "content-block-delta",
                                "delta": {"type": "text-delta", "text": "ok"},
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
        },
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads))
    )
    events = [
        event
        async for event in gateway.stream_events(
            RunRequest(message="hi", thread_id="abc12345")
        )
    ]
    assert events[-1]["type"] == "done"


def test_run_streaming_puts_the_rubric_on_the_run_request():
    agent = MagicMock()
    gateway = FakeGraphGateway(
        events=[
            {"type": "text", "content": "ok"},
            {"type": "done", "response": "ok"},
        ]
    )

    with patch("EvoScientist.stream.display.Live"):
        display_mod._run_streaming(
            agent=agent,
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            gateway=gateway,
            rubric="- a",
        )

    assert [request.rubric for request in gateway.requests] == ["- a"]


def _cmd_run_with_fake_stream(monkeypatch, *, verdict, **cmd_kwargs):
    from types import SimpleNamespace

    from EvoScientist.cli import interactive

    thread_store = FakeThreadStore(generated_thread_id="generated-thread")
    runtime_gateways = RuntimeGateways(
        thread_store=thread_store,
        graph_gateway=LocalGraphGateway(thread_store=thread_store),
    )
    seen: dict[str, Any] = {}

    def _run_streaming(**kwargs):
        seen.update(kwargs)
        callback = kwargs.get("on_stream_event")
        if callback is not None:
            callback("rubric_evaluation", SimpleNamespace(rubric_result=verdict))
        return "ok"

    monkeypatch.setattr(interactive, "run_streaming", _run_streaming)
    result = interactive.cmd_run(
        MagicMock(),
        "hello",
        thread_id="generated-thread",
        show_thinking=False,
        dirs=SessionDirs(Workspace("/tmp/ws")),
        model="test-model",
        runtime_gateways=runtime_gateways,
        **cmd_kwargs,
    )
    return result, seen


def test_cmd_run_forwards_the_rubric_and_returns_the_verdict(monkeypatch):
    result, seen = _cmd_run_with_fake_stream(
        monkeypatch, verdict="max_iterations_reached", rubric="- a"
    )
    assert seen["rubric"] == "- a"
    assert result == "max_iterations_reached"


def test_cmd_run_without_a_rubric_tracks_nothing(monkeypatch):
    result, seen = _cmd_run_with_fake_stream(monkeypatch, verdict="satisfied")
    assert seen["rubric"] is None
    assert seen["on_stream_event"] is None
    assert result is None


def test_resolve_per_run_config_marks_only_a_graded_run():
    from EvoScientist.gateway.types import resolve_per_run_config

    marked = resolve_per_run_config("t1", None, rubric_run=True)
    assert marked["configurable"]["rubric_run"] == {"mode": "input"}
    assert "rubric_run" not in resolve_per_run_config("t1", None)["configurable"]


def test_input_mode_wins_over_a_client_marker_but_keeps_its_changes():
    from EvoScientist.gateway.types import resolve_per_run_config

    change = {"session": "s", "seq": 1, "sticky": "- a"}
    extra = {"rubric_run": {"mode": "thread", "changes": [change]}}
    marked = resolve_per_run_config("t1", extra, rubric_run=True)
    assert marked["configurable"]["rubric_run"] == {
        "mode": "input",
        "changes": [change],
    }
    unmarked = resolve_per_run_config("t1", extra)
    assert unmarked["configurable"]["rubric_run"]["mode"] == "thread"


async def test_langgraph_server_gateway_marks_a_graded_run():
    stream = FakeLangGraphThreadStream("abc12345", events=[])
    threads = FakeLangGraphThreadsClient(
        threads=[],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads))
    )
    live_cfg = SimpleNamespace(
        model="live-model", provider="live-provider", recursion_limit=4242
    )
    with patch("EvoScientist.EvoScientist._ensure_config", return_value=live_cfg):
        async for _event in gateway.stream_events(
            RunRequest(message="hi", thread_id="abc12345", rubric="- a")
        ):
            pass

    start = stream.run.starts[0]
    assert start["input"]["rubric"] == "- a"
    assert start["config"]["configurable"]["rubric_run"] == {"mode": "input"}


def test_server_gateway_run_carries_the_retry_key(monkeypatch):
    """A -p --auto-mode run on the server gateway must bring its own tier."""
    import EvoScientist.EvoScientist as evo_mod
    from EvoScientist.backends import MODEL_RETRY_RUN_KEY
    from EvoScientist.config import EvoScientistConfig
    from EvoScientist.gateway import server as server_mod

    cfg = EvoScientistConfig(auto_mode=True, auto_mode_retry_wait_seconds=1800)
    monkeypatch.setattr(evo_mod, "_ensure_config", lambda: cfg)
    gateway = server_mod.LangGraphServerGateway.__new__(
        server_mod.LangGraphServerGateway
    )
    config = gateway._resolve_run_config("t1", None)
    assert config["configurable"][MODEL_RETRY_RUN_KEY] == {
        "unattended": True,
        "wait_budget_seconds": 1800,
    }


def _model_call(phase: str, call: str, status: str | None = None) -> dict:
    payload: dict[str, object] = {"type": "model_call", "phase": phase, "call": call}
    if status is not None:
        payload["status"] = status
    return _custom_stream_event(payload)


async def test_langgraph_server_gateway_recovers_a_whole_message_fallback_after_a_discard():
    """The transport drops a non-streamed message, so the values snapshot must deliver it.

    The failed primary's text had switched snapshot recovery off; once that
    text is discarded the fallback's answer has no other way to arrive.
    """
    events = await _collect_server_gateway_stream(
        [
            _model_call("start", "1"),
            _root_text_delta("WRO"),
            _model_call("end", "1", "failed"),
            _model_call("start", "2"),
            _model_call("end", "2", "clean"),
            _value_snapshot([_OLD_AI, _HUMAN, _NEW_AI]),
        ],
        state_messages=[_OLD_AI],
    )

    assert [event["type"] for event in events] == [
        "text",
        "model_output_discarded",
        "text",
        "done",
    ]
    assert events[-1]["response"] == "new"


async def test_langgraph_server_gateway_recovery_after_a_discard_skips_streamed_messages():
    first = {"type": "ai", "content": "first", "id": "ai-1"}
    events = await _collect_server_gateway_stream(
        [
            _model_call("start", "1"),
            _root_text_delta("first"),
            _root_message_finish(),
            _model_call("end", "1", "clean"),
            _value_snapshot([_OLD_AI, _HUMAN, first]),
            _model_call("start", "2"),
            _root_text_delta("WRO"),
            _model_call("end", "2", "failed"),
            _model_call("start", "3"),
            _model_call("end", "3", "clean"),
            _value_snapshot([_OLD_AI, _HUMAN, first, _NEW_AI]),
        ],
        state_messages=[_OLD_AI],
    )

    texts = [event["content"] for event in events if event["type"] == "text"]
    assert texts == ["first", "WRO", "new"]
    assert events[-1]["response"] == "firstnew"


async def test_langgraph_server_gateway_streamed_retry_after_a_discard_is_not_duplicated():
    events = await _collect_server_gateway_stream(
        [
            _model_call("start", "1"),
            _root_text_delta("WRO"),
            _model_call("end", "1", "failed"),
            _custom_stream_event(
                {
                    "type": "model_retry",
                    "attempt": 2,
                    "wait_seconds": 1.0,
                    "reason": "503: starting up",
                    "budget_left_seconds": 9.0,
                    "cause": "exception",
                }
            ),
            _model_call("start", "2"),
            _root_text_delta("new"),
            _root_message_finish(),
            _model_call("end", "2", "clean"),
            _value_snapshot([_OLD_AI, _HUMAN, _NEW_AI]),
        ],
        state_messages=[_OLD_AI],
    )

    texts = [event["content"] for event in events if event["type"] == "text"]
    assert texts == ["WRO", "new"]
    assert events[-1]["response"] == "new"
