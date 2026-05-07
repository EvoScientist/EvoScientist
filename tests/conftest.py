"""Shared fixtures for EvoScientist tests."""

import asyncio

import pytest


def run_async(coro):
    """Run an async coroutine safely, cancelling pending tasks before closing.

    This prevents 'Event loop is closed' errors from asyncio.Queue cleanup
    when tasks are still waiting on Queue.get() at teardown time.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        # Cancel all pending tasks so Queue getters don't raise on close
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


@pytest.fixture(name="run_async")
def run_async_fixture():
    """Pytest fixture that exposes run_async as a callable for test functions."""
    return run_async


@pytest.fixture
def sample_tool_call():
    """A minimal tool call dict."""
    return {"id": "tc_001", "name": "execute", "args": {"command": "ls -la"}}


@pytest.fixture
def sample_tool_result():
    """A minimal tool result dict."""
    return {"name": "execute", "content": "[OK] file1.py file2.py", "success": True}


@pytest.fixture
def sample_events():
    """A sequence of stream event dicts covering common types."""
    return [
        {"type": "thinking", "content": "Let me think..."},
        {"type": "text", "content": "Here is the answer."},
        {
            "type": "tool_call",
            "id": "tc_001",
            "name": "execute",
            "args": {"command": "ls"},
        },
        {
            "type": "tool_result",
            "name": "execute",
            "content": "[OK] done",
            "success": True,
        },
        {
            "type": "subagent_start",
            "name": "research-agent",
            "description": "Find papers",
        },
        {
            "type": "subagent_tool_call",
            "subagent": "research-agent",
            "name": "tavily_search",
            "args": {"query": "test"},
            "id": "tc_sa_001",
        },
        {
            "type": "subagent_tool_result",
            "subagent": "research-agent",
            "name": "tavily_search",
            "content": "Results...",
            "success": True,
        },
        {"type": "subagent_end", "name": "research-agent"},
        {"type": "done", "response": "Here is the answer."},
    ]


@pytest.fixture
def tmp_workspace(tmp_path):
    """Provide a temporary workspace directory path."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)
