"""Shared fixtures for EvoScientist tests."""

import pytest


@pytest.fixture(autouse=True)
def _reset_tool_selection_state():
    """Isolate the process-global tool-selection state around every test.

    ``middleware.tool_selector`` records the last selected tools and the
    selector-active flag in module globals that ``stream/tool_selection.py``
    reads to decide whether to suppress selector output. A test that drives the
    selector or tracker would otherwise leave those globals set and silently
    flip unrelated streaming tests later in the same process. Reset on both ends
    so order and worker sharding can't reintroduce the leak.
    """
    from EvoScientist.middleware.tool_selector import (
        reset_tool_selection_state_for_tests,
    )

    reset_tool_selection_state_for_tests()
    yield
    reset_tool_selection_state_for_tests()


@pytest.fixture
def sample_tool_call():
    """A minimal tool call dict."""
    return {"id": "tc_001", "name": "execute", "args": {"command": "ls -la"}}


@pytest.fixture
def sample_tool_result():
    """A minimal tool result dict."""
    return {
        "id": "tc_001",
        "name": "execute",
        "content": "[OK] file1.py file2.py",
        "success": True,
    }


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
            "id": "tc_001",
            "name": "execute",
            "content": "[OK] done",
            "success": True,
        },
        {
            "type": "subagent_start",
            "name": "research-agent",
            "description": "Find papers",
            "instance_id": "task:research",
            "tool_call_id": "tc_task_001",
        },
        {
            "type": "subagent_tool_call",
            "subagent": "research-agent",
            "instance_id": "task:research",
            "name": "tavily_search",
            "args": {"query": "test"},
            "id": "tc_sa_001",
        },
        {
            "type": "subagent_tool_result",
            "subagent": "research-agent",
            "instance_id": "task:research",
            "name": "tavily_search",
            "content": "Results...",
            "success": True,
            "id": "tc_sa_001",
        },
        {
            "type": "subagent_end",
            "name": "research-agent",
            "instance_id": "task:research",
        },
        {"type": "done", "response": "Here is the answer."},
    ]


@pytest.fixture
def tmp_workspace(tmp_path):
    """Provide a temporary workspace directory path."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


@pytest.fixture
def runtime_paths(tmp_path, monkeypatch):
    """Isolate ``langgraph_dev.manager.RUNTIME`` under a temp directory.

    Replaces the module-level ``RUNTIME`` with a fully temp-rooted bundle
    so every path (``pid_dir``, ``pid_file``, ``log_file``,
    ``workspace_sidecar``, ``lock_file``) is contained under ``tmp_path``.

    Tests that need a variant of a single field can still call
    ``dataclasses.replace(runtime_paths, log_file=…)`` etc. — the
    baseline is already isolated, so forgetting a field just keeps it
    under ``tmp_path``, never ``~/.config/evoscientist``.
    """
    from EvoScientist.langgraph_dev import manager

    runtime = manager.LanggraphRuntimePaths.for_directory(tmp_path / "runtime")
    monkeypatch.setattr(manager, "RUNTIME", runtime)
    return runtime


# Capture deepagents tool factories at conftest load time — BEFORE any test
# imports EvoScientist, which can trigger ``_patch_deepagents_model_passthrough``
# during agent construction. Once captured here, the ``restore_model_passthrough_patch``
# fixture has a stable "truly unpatched" baseline to reset to between tests, even
# if upstream code paths apply the patch as a side effect.
try:
    from deepagents.middleware import async_subagents as _ds_async_subagents

    _DEEPAGENTS_ORIGINAL_BUILD_START = _ds_async_subagents._build_start_tool
    _DEEPAGENTS_ORIGINAL_BUILD_UPDATE = _ds_async_subagents._build_update_tool
except Exception:
    _ds_async_subagents = None
    _DEEPAGENTS_ORIGINAL_BUILD_START = None
    _DEEPAGENTS_ORIGINAL_BUILD_UPDATE = None


@pytest.fixture
def restore_model_passthrough_patch():
    """Reset deepagents internals + ``_model_passthrough_patched`` to unpatched.

    The model-passthrough patch wraps ``deepagents.middleware.async_subagents``
    module-level functions in place. The originals are captured at conftest
    load time (above) so this fixture can always start each test from a
    known-unpatched state regardless of what other tests / agent fixtures
    did to the module before.
    """
    from EvoScientist.llm import patches as patches_mod

    if _ds_async_subagents is None:
        # deepagents not importable — fixture is a no-op (the patch fn itself
        # returns early in that case).
        yield
        return

    def _reset() -> None:
        _ds_async_subagents._build_start_tool = _DEEPAGENTS_ORIGINAL_BUILD_START
        _ds_async_subagents._build_update_tool = _DEEPAGENTS_ORIGINAL_BUILD_UPDATE
        patches_mod._model_passthrough_patched = False

    _reset()
    try:
        yield
    finally:
        _reset()
