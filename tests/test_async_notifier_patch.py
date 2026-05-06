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


def test_patch_wraps_start_tool_to_spawn_watcher():
    """After patching, calling the start tool's coroutine should spawn a watcher."""
    # Import deepagents module
    try:
        import deepagents.middleware.async_subagents as ds_mod
    except ImportError:
        pytest.skip("deepagents not installed")

    # Prepare to track spawn_watcher calls
    spawn_calls = []

    def fake_spawn(client, thread_id, run_id, agent_name, prompt=""):
        spawn_calls.append((thread_id, run_id, agent_name, prompt))
        return None

    # Clear queue before test
    _drain_queue(async_notifier._notification_queue)

    # Save original state so patching in this test doesn't contaminate later tests
    original_patched = patches._async_watcher_patched
    original_build_start = ds_mod._build_start_tool
    original_build_update = ds_mod._build_update_tool

    result = None
    try:
        # Patch spawn_watcher BEFORE calling _patch_deepagents_async_watcher
        with patch.object(async_notifier, "spawn_watcher", side_effect=fake_spawn):
            # Apply the patch
            patches._patch_deepagents_async_watcher()

            # Construct a tool the way deepagents would
            agent_map = {
                "writing-agent": {"name": "writing-agent", "graph_id": "writing-agent"}
            }

            # Create mock clients
            clients = MagicMock()
            fake_client = MagicMock()
            fake_client.threads.create = AsyncMock(return_value={"thread_id": "T1"})
            fake_client.runs.create = AsyncMock(return_value={"run_id": "R1"})
            clients.get_async = MagicMock(return_value=fake_client)

            # Build the start tool using deepagents factory
            tool = ds_mod._build_start_tool(agent_map, clients, "fake desc")

            # Create a mock runtime
            runtime = MagicMock()
            runtime.tool_call_id = "tc-1"

            # Call the tool's coroutine and run it
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
    finally:
        # Restore original state so later tests see a clean ds_mod
        patches._async_watcher_patched = original_patched
        ds_mod._build_start_tool = original_build_start
        ds_mod._build_update_tool = original_build_update

    # Tool result should be a Command (unchanged)
    from langgraph.types import Command

    assert isinstance(result, Command), f"Expected Command, got {type(result)}"

    # Watcher should have been spawned with the correct identifiers AND
    # the original description forwarded as the prompt
    assert spawn_calls == [("T1", "R1", "writing-agent", "do work")], (
        f"Got spawn_calls: {spawn_calls}"
    )


def test_patch_is_idempotent():
    """Calling _patch_deepagents_async_watcher multiple times should only patch once."""
    try:
        import deepagents.middleware.async_subagents as ds_mod
    except ImportError:
        pytest.skip("deepagents not installed")

    # Reset the flag so we can test idempotence
    original_patched = patches._async_watcher_patched
    original_build_start = ds_mod._build_start_tool
    original_build_update = ds_mod._build_update_tool

    try:
        # Clear the flag
        patches._async_watcher_patched = False

        # Patch once
        patches._patch_deepagents_async_watcher()
        first_build = ds_mod._build_start_tool

        # Patch again
        patches._patch_deepagents_async_watcher()
        second_build = ds_mod._build_start_tool

        # Both should be the same (patched) function
        assert first_build is second_build, "Patch was applied twice"

    finally:
        # Restore original state (both start and update tools)
        patches._async_watcher_patched = original_patched
        ds_mod._build_start_tool = original_build_start
        ds_mod._build_update_tool = original_build_update


def test_patch_wraps_update_tool_to_spawn_watcher():
    """After patching, calling update_async_task should also spawn a watcher
    (for the new run_id created by the update)."""
    try:
        import deepagents.middleware.async_subagents as ds_mod
    except ImportError:
        pytest.skip("deepagents not installed")

    spawn_calls = []

    def fake_spawn(client, thread_id, run_id, agent_name, prompt=""):
        spawn_calls.append((thread_id, run_id, agent_name, prompt))
        return None

    _drain_queue(async_notifier._notification_queue)

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

        # Mock the runtime + state so _resolve_tracked_task succeeds
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
    # Watcher spawned with the NEW run_id and the message as prompt
    assert spawn_calls == [("T1", "R2", "writing-agent", "follow-up message")], (
        f"Got spawn_calls: {spawn_calls}"
    )
