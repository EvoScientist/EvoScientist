"""Tests for ``EvoScientist.middleware.async_watcher.AsyncWatcherMiddleware``.

The middleware is the public-API replacement for the old monkey-patch on
deepagents internals. It hooks into ``awrap_tool_call`` and only fires on
``start_async_task`` / ``update_async_task`` tool invocations.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from EvoScientist.cli import async_notifier


def _drain_all_notifications():
    """Drain every routed/unrouted/legacy queue between tests."""
    if hasattr(async_notifier, "_notifications_by_thread"):
        for q in list(async_notifier._notifications_by_thread.values()):
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break
    for attr in ("_unrouted_queue", "_notification_queue"):
        if hasattr(async_notifier, attr):
            q = getattr(async_notifier, attr)
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break


@pytest.fixture(autouse=True)
def _clean_notifier_state():
    """Reset shared module-level notifier state before and after every test.

    Cleared here:
      - All notification queues (per-thread, unrouted, legacy global)
      - ``_watcher_by_thread`` (replacement-on-update registry)
      - ``_active_watchers`` (in-flight watcher → origin-thread index used
        by ``consume_notifications`` to gate the grace-window wait)

    Without this, a test that touches any of these dicts/queues would
    silently leak state into the next test. ``_active_watchers`` is cleared
    even though current tests patch ``spawn_watcher`` (so they never insert
    into it) — kept as a safeguard for future tests that exercise the real
    spawn path.
    """
    _drain_all_notifications()
    async_notifier._watcher_by_thread.clear()
    async_notifier._active_watchers.clear()
    yield
    _drain_all_notifications()
    async_notifier._watcher_by_thread.clear()
    async_notifier._active_watchers.clear()


def _build_request(tool_name: str, args: dict, *, thread_id: str | None = None):
    """Construct a minimal ToolCallRequest stand-in.

    The middleware reads only ``request.tool_call`` and ``request.runtime``.
    """
    runtime = SimpleNamespace(
        config={"configurable": {"thread_id": thread_id}} if thread_id else {}
    )
    return SimpleNamespace(
        tool_call={
            "name": tool_name,
            "args": args,
            "id": "call-1",
            "type": "tool_call",
        },
        runtime=runtime,
        state={},
        tool=None,
    )


def _make_middleware():
    """Build an AsyncWatcherMiddleware with a stubbed ``_ClientCache``."""
    from EvoScientist.middleware.async_watcher import AsyncWatcherMiddleware

    fake_client = MagicMock(name="LangGraphClient")
    fake_cache = MagicMock(name="ClientCache")
    fake_cache.get_async.return_value = fake_client

    with patch(
        "deepagents.middleware.async_subagents._ClientCache",
        return_value=fake_cache,
    ):
        mw = AsyncWatcherMiddleware(
            {
                "writing-agent": {
                    "name": "writing-agent",
                    "url": "http://x",
                    "graph_id": "writing-agent",
                }
            }
        )
    return mw, fake_client


def test_middleware_spawns_watcher_on_start_async_task():
    """A successful start_async_task tool call must spawn one watcher per task."""
    from langgraph.types import Command

    mw, _fake_client = _make_middleware()

    spawn_calls = []

    def fake_spawn(
        client, thread_id, run_id, agent_name, prompt="", origin_cli_thread_id=None
    ):
        spawn_calls.append(
            (thread_id, run_id, agent_name, prompt, origin_cli_thread_id)
        )

    request = _build_request(
        "start_async_task",
        {"description": "do thing", "subagent_type": "writing-agent"},
        thread_id="cli-thread-A",
    )

    async def fake_handler(req):
        return Command(
            update={
                "async_tasks": {
                    "task-1": {
                        "task_id": "task-1",
                        "agent_name": "writing-agent",
                        "run_id": "run-1",
                        "thread_id": "task-1",
                        "status": "running",
                    }
                }
            }
        )

    with patch.object(async_notifier, "spawn_watcher", side_effect=fake_spawn):
        result = asyncio.run(mw.awrap_tool_call(request, fake_handler))

    assert isinstance(result, Command)
    assert spawn_calls == [
        ("task-1", "run-1", "writing-agent", "do thing", "cli-thread-A")
    ]


def test_middleware_spawns_watcher_on_update_async_task():
    """A successful update_async_task call must also spawn a (replacement) watcher."""
    from langgraph.types import Command

    mw, _ = _make_middleware()

    spawn_calls = []

    def fake_spawn(*args, **kwargs):
        spawn_calls.append((args, kwargs))

    request = _build_request(
        "update_async_task",
        {"task_id": "task-1", "message": "do more"},
        thread_id="cli-thread-A",
    )

    async def fake_handler(req):
        return Command(
            update={
                "async_tasks": {
                    "task-1": {
                        "task_id": "task-1",
                        "agent_name": "writing-agent",
                        "run_id": "run-2",
                        "thread_id": "task-1",
                        "status": "running",
                    }
                }
            }
        )

    with patch.object(async_notifier, "spawn_watcher", side_effect=fake_spawn):
        asyncio.run(mw.awrap_tool_call(request, fake_handler))

    assert len(spawn_calls) == 1
    args, kwargs = spawn_calls[0]
    # spawn_watcher(client, task_id, run_id, agent_name, prompt=..., origin_cli_thread_id=...)
    assert args[1] == "task-1"
    assert args[2] == "run-2"
    assert args[3] == "writing-agent"
    assert kwargs["prompt"] == "do more"
    assert kwargs["origin_cli_thread_id"] == "cli-thread-A"


def test_middleware_pre_cancels_old_watcher_on_update():
    """update_async_task must cancel the existing watcher BEFORE invoking the handler.

    Otherwise the new run interrupts the old run's stream, which closes
    cleanly, and the old watcher would enqueue a stale "success" notification.
    """
    from langgraph.types import Command

    mw, _ = _make_middleware()

    old_watcher = MagicMock()
    old_watcher.done.return_value = False
    async_notifier._watcher_by_thread["task-1"] = old_watcher

    cancel_observed_before_handler = {"value": False}

    async def fake_handler(req):
        cancel_observed_before_handler["value"] = old_watcher.cancel.called
        return Command(update={"async_tasks": {}})

    request = _build_request(
        "update_async_task", {"task_id": "task-1", "message": "x"}, thread_id="t"
    )

    try:
        with patch.object(async_notifier, "spawn_watcher"):
            asyncio.run(mw.awrap_tool_call(request, fake_handler))
    finally:
        async_notifier._watcher_by_thread.pop("task-1", None)

    assert cancel_observed_before_handler["value"] is True


def test_middleware_passes_through_unrelated_tools():
    """A non-launch tool call must not spawn any watcher and must return result unchanged."""
    mw, _ = _make_middleware()

    sentinel = object()

    async def fake_handler(req):
        return sentinel

    request = _build_request("ls", {"path": "/"}, thread_id="t")

    with patch.object(async_notifier, "spawn_watcher") as mock_spawn:
        result = asyncio.run(mw.awrap_tool_call(request, fake_handler))

    assert result is sentinel
    assert mock_spawn.call_count == 0


def test_middleware_handles_non_command_results_gracefully():
    """If the launch tool returns a string (validation error), no watcher is spawned."""
    mw, _ = _make_middleware()

    async def fake_handler(req):
        return "Unknown async subagent type `bogus`"

    request = _build_request(
        "start_async_task",
        {"description": "x", "subagent_type": "bogus"},
        thread_id="t",
    )

    with patch.object(async_notifier, "spawn_watcher") as mock_spawn:
        result = asyncio.run(mw.awrap_tool_call(request, fake_handler))

    assert result == "Unknown async subagent type `bogus`"
    assert mock_spawn.call_count == 0


def test_middleware_origin_thread_id_is_none_when_runtime_config_missing():
    """When runtime.config is empty, origin_cli_thread_id must be None (not crash)."""
    from langgraph.types import Command

    mw, _ = _make_middleware()

    captured = {}

    def fake_spawn(*args, **kwargs):
        captured.update(kwargs)

    request = _build_request(
        "start_async_task",
        {"description": "x", "subagent_type": "writing-agent"},
        thread_id=None,
    )

    async def fake_handler(req):
        return Command(
            update={
                "async_tasks": {
                    "t1": {
                        "task_id": "t1",
                        "agent_name": "writing-agent",
                        "run_id": "r1",
                        "thread_id": "t1",
                        "status": "running",
                    }
                }
            }
        )

    with patch.object(async_notifier, "spawn_watcher", side_effect=fake_spawn):
        asyncio.run(mw.awrap_tool_call(request, fake_handler))

    assert captured.get("origin_cli_thread_id") is None


def test_middleware_swallows_spawn_exceptions():
    """spawn_watcher errors must not propagate up — middleware logs and continues."""
    from langgraph.types import Command

    mw, _ = _make_middleware()

    def boom(*a, **kw):
        raise RuntimeError("intentional")

    request = _build_request(
        "start_async_task",
        {"description": "x", "subagent_type": "writing-agent"},
        thread_id="t",
    )

    async def fake_handler(req):
        return Command(
            update={
                "async_tasks": {
                    "t1": {
                        "task_id": "t1",
                        "agent_name": "writing-agent",
                        "run_id": "r1",
                        "thread_id": "t1",
                        "status": "running",
                    }
                }
            }
        )

    with patch.object(async_notifier, "spawn_watcher", side_effect=boom):
        # Should not raise.
        result = asyncio.run(mw.awrap_tool_call(request, fake_handler))

    assert isinstance(result, Command)


@pytest.mark.parametrize(
    ("tool_name", "args", "prompt_field"),
    [
        (
            "start_async_task",
            {"description": "from start", "subagent_type": "writing-agent"},
            "from start",
        ),
        (
            "update_async_task",
            {"task_id": "t1", "message": "from update"},
            "from update",
        ),
    ],
)
def test_middleware_picks_correct_prompt_field_per_tool(tool_name, args, prompt_field):
    """start_async_task uses 'description'; update_async_task uses 'message'."""
    from langgraph.types import Command

    mw, _ = _make_middleware()

    captured_prompt = {}

    def fake_spawn(*a, prompt="", **kw):
        captured_prompt["value"] = prompt

    async def fake_handler(req):
        return Command(
            update={
                "async_tasks": {
                    "t1": {
                        "task_id": "t1",
                        "agent_name": "writing-agent",
                        "run_id": "r1",
                        "thread_id": "t1",
                        "status": "running",
                    }
                }
            }
        )

    request = _build_request(tool_name, args, thread_id="t")

    with patch.object(async_notifier, "spawn_watcher", side_effect=fake_spawn):
        asyncio.run(mw.awrap_tool_call(request, fake_handler))

    assert captured_prompt["value"] == prompt_field


def test_middleware_prompt_field_is_tool_name_gated_not_fallback_chained():
    """update_async_task with extra `description` arg must still use `message`.

    Guards against the previous `args.get('description') or args.get('message')`
    chained-fallback shape, which would have picked `description` for an update
    call that happened to carry both fields.
    """
    from langgraph.types import Command

    mw, _ = _make_middleware()

    captured_prompt = {}

    def fake_spawn(*a, prompt="", **kw):
        captured_prompt["value"] = prompt

    async def fake_handler(req):
        return Command(
            update={
                "async_tasks": {
                    "t1": {
                        "task_id": "t1",
                        "agent_name": "writing-agent",
                        "run_id": "r1",
                        "thread_id": "t1",
                        "status": "running",
                    }
                }
            }
        )

    request = _build_request(
        "update_async_task",
        {
            "task_id": "t1",
            "message": "use this",
            "description": "do NOT use this",
        },
        thread_id="t",
    )

    with patch.object(async_notifier, "spawn_watcher", side_effect=fake_spawn):
        asyncio.run(mw.awrap_tool_call(request, fake_handler))

    assert captured_prompt["value"] == "use this"


def test_middleware_pre_cancel_swallows_unexpected_errors():
    """A faulty old-watcher handle must not block the handler from running."""
    from langgraph.types import Command

    mw, _ = _make_middleware()

    bad_watcher = MagicMock()
    bad_watcher.done.side_effect = RuntimeError("watcher state corrupted")
    async_notifier._watcher_by_thread["t1"] = bad_watcher

    handler_called = {"value": False}

    async def fake_handler(req):
        handler_called["value"] = True
        return Command(update={"async_tasks": {}})

    request = _build_request(
        "update_async_task", {"task_id": "t1", "message": "x"}, thread_id="t"
    )

    try:
        with patch.object(async_notifier, "spawn_watcher"):
            # Should not raise.
            asyncio.run(mw.awrap_tool_call(request, fake_handler))
    finally:
        async_notifier._watcher_by_thread.pop("t1", None)

    assert handler_called["value"] is True
