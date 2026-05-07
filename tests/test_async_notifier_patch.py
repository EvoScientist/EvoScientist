"""Verify the deepagents start-tool monkey-patch wires in the watcher."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from EvoScientist.cli import async_notifier
from EvoScientist.llm import patches


def _drain_queue(q):
    """Drain all items from a queue."""
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except Exception:
            return items


def _drain_all_notifications():
    """Drain every per-thread queue plus the unrouted bucket (post Fix #3)."""
    if hasattr(async_notifier, "_notifications_by_thread"):
        for q in list(async_notifier._notifications_by_thread.values()):
            _drain_queue(q)
    if hasattr(async_notifier, "_unrouted_queue"):
        _drain_queue(async_notifier._unrouted_queue)
    if hasattr(async_notifier, "_notification_queue"):
        _drain_queue(async_notifier._notification_queue)


def test_patch_wraps_start_tool_to_spawn_watcher(restore_async_watcher_patch):
    """After patching, calling the start tool's coroutine should spawn a watcher."""
    try:
        import deepagents.middleware.async_subagents as ds_mod
    except ImportError:
        pytest.skip("deepagents not installed")

    spawn_calls = []

    def fake_spawn(client, thread_id, run_id, agent_name, prompt="", **kwargs):
        spawn_calls.append((thread_id, run_id, agent_name, prompt))
        return None

    _drain_all_notifications()

    with patch.object(async_notifier, "spawn_watcher", side_effect=fake_spawn):
        patches._patch_deepagents_async_watcher()

        agent_map = {
            "writing-agent": {"name": "writing-agent", "graph_id": "writing-agent"}
        }
        clients = MagicMock()
        fake_client = MagicMock()
        fake_client.threads.create = AsyncMock(return_value={"thread_id": "T1"})
        fake_client.runs.create = AsyncMock(return_value={"run_id": "R1"})
        clients.get_async = MagicMock(return_value=fake_client)

        tool = ds_mod._build_start_tool(agent_map, clients, "fake desc")

        runtime = MagicMock()
        runtime.tool_call_id = "tc-1"

        async def run_tool():
            return await tool.coroutine(
                description="do work",
                subagent_type="writing-agent",
                runtime=runtime,
            )

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(run_tool())
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    from langgraph.types import Command

    assert isinstance(result, Command), f"Expected Command, got {type(result)}"
    assert spawn_calls == [("T1", "R1", "writing-agent", "do work")], (
        f"Got spawn_calls: {spawn_calls}"
    )


def test_patch_is_idempotent(restore_async_watcher_patch):
    """Calling _patch_deepagents_async_watcher multiple times should only patch once."""
    try:
        import deepagents.middleware.async_subagents as ds_mod
    except ImportError:
        pytest.skip("deepagents not installed")

    patches._async_watcher_patched = False
    patches._patch_deepagents_async_watcher()
    first_build = ds_mod._build_start_tool

    patches._patch_deepagents_async_watcher()
    second_build = ds_mod._build_start_tool

    assert first_build is second_build, "Patch was applied twice"


def test_patch_wraps_update_tool_to_spawn_watcher(restore_async_watcher_patch):
    """After patching, calling update_async_task should also spawn a watcher
    (for the new run_id created by the update)."""
    try:
        import deepagents.middleware.async_subagents as ds_mod
    except ImportError:
        pytest.skip("deepagents not installed")

    spawn_calls = []

    def fake_spawn(client, thread_id, run_id, agent_name, prompt="", **kwargs):
        spawn_calls.append((thread_id, run_id, agent_name, prompt))
        return None

    _drain_all_notifications()

    with patch.object(async_notifier, "spawn_watcher", side_effect=fake_spawn):
        patches._patch_deepagents_async_watcher()

        agent_map = {
            "writing-agent": {"name": "writing-agent", "graph_id": "writing-agent"}
        }
        clients = MagicMock()
        fake_client = MagicMock()
        fake_client.runs.create = AsyncMock(return_value={"run_id": "R2"})
        clients.get_async = MagicMock(return_value=fake_client)

        update_tool = ds_mod._build_update_tool(agent_map, clients)

        runtime = MagicMock()
        runtime.tool_call_id = "tc-2"
        runtime.state = {
            "async_tasks": {
                "T1": {
                    "task_id": "T1",
                    "agent_name": "writing-agent",
                    "thread_id": "T1",
                    "run_id": "R1",
                    "status": "running",
                    "created_at": "2026-01-01T00:00:00Z",
                    "last_checked_at": "2026-01-01T00:00:00Z",
                    "last_updated_at": "2026-01-01T00:00:00Z",
                }
            }
        }

        async def run_tool():
            return await update_tool.coroutine(
                task_id="T1",
                message="follow-up message",
                runtime=runtime,
            )

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(run_tool())
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    from langgraph.types import Command

    assert isinstance(result, Command)
    assert spawn_calls == [("T1", "R2", "writing-agent", "follow-up message")], (
        f"Got spawn_calls: {spawn_calls}"
    )


def test_patch_propagates_origin_cli_thread_id(restore_async_watcher_patch):
    """Verify patches.py extracts the main agent's CLI thread_id from
    runtime.config[configurable][thread_id] and forwards it to spawn_watcher
    as origin_cli_thread_id (Fix #3 — per-thread routing)."""
    try:
        import deepagents.middleware.async_subagents as ds_mod
    except ImportError:
        pytest.skip("deepagents not installed")

    captured: dict = {}

    def fake_spawn(client, thread_id, run_id, agent_name, prompt="", **kwargs):
        captured["origin"] = kwargs.get("origin_cli_thread_id")
        captured["thread_id"] = thread_id

    _drain_all_notifications()

    with patch.object(async_notifier, "spawn_watcher", side_effect=fake_spawn):
        patches._patch_deepagents_async_watcher()

        agent_map = {
            "writing-agent": {"name": "writing-agent", "graph_id": "writing-agent"}
        }
        clients = MagicMock()
        fake_client = MagicMock()
        fake_client.threads.create = AsyncMock(return_value={"thread_id": "T1"})
        fake_client.runs.create = AsyncMock(return_value={"run_id": "R1"})
        clients.get_async = MagicMock(return_value=fake_client)

        tool = ds_mod._build_start_tool(agent_map, clients, "fake desc")

        # runtime.config holds the main agent's CLI thread_id
        runtime = MagicMock()
        runtime.tool_call_id = "tc-1"
        runtime.config = {"configurable": {"thread_id": "main-cli-thread-XYZ"}}

        async def run_tool():
            return await tool.coroutine(
                description="do work",
                subagent_type="writing-agent",
                runtime=runtime,
            )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_tool())
        finally:
            pending = asyncio.all_tasks(loop)
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    assert captured["thread_id"] == "T1"  # sub-agent run thread
    assert captured["origin"] == "main-cli-thread-XYZ"  # main CLI thread


def test_patch_origin_thread_id_none_when_runtime_config_missing(
    restore_async_watcher_patch,
):
    """If runtime.config is unexpected (not a dict / no configurable),
    origin_cli_thread_id falls back to None — notification will route to
    the unrouted bucket and drain regardless of current_thread_id."""
    try:
        import deepagents.middleware.async_subagents as ds_mod
    except ImportError:
        pytest.skip("deepagents not installed")

    captured: dict = {}

    def fake_spawn(client, thread_id, run_id, agent_name, prompt="", **kwargs):
        captured["origin"] = kwargs.get("origin_cli_thread_id")

    _drain_all_notifications()

    with patch.object(async_notifier, "spawn_watcher", side_effect=fake_spawn):
        patches._patch_deepagents_async_watcher()

        agent_map = {
            "writing-agent": {"name": "writing-agent", "graph_id": "writing-agent"}
        }
        clients = MagicMock()
        fake_client = MagicMock()
        fake_client.threads.create = AsyncMock(return_value={"thread_id": "T1"})
        fake_client.runs.create = AsyncMock(return_value={"run_id": "R1"})
        clients.get_async = MagicMock(return_value=fake_client)

        tool = ds_mod._build_start_tool(agent_map, clients, "fake desc")

        runtime = MagicMock()
        runtime.tool_call_id = "tc-1"
        runtime.config = "not a dict"  # malformed runtime config

        async def run_tool():
            return await tool.coroutine(
                description="do work",
                subagent_type="writing-agent",
                runtime=runtime,
            )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_tool())
        finally:
            pending = asyncio.all_tasks(loop)
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    assert "origin" in captured
    assert captured["origin"] is None
