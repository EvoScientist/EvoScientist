"""Tests for the deepagents model-passthrough patch.

Verifies that ``_patch_deepagents_model_passthrough`` wraps
``_build_start_tool`` / ``_build_update_tool`` so that ``client.runs.create``
calls inside the launched async-task tools carry
``config={"configurable": {"model": ..., "model_provider": ...}}``,
without affecting other client methods (``threads.create``, ``runs.get``,
``runs.cancel``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from EvoScientist.llm import patches as patches_mod
from tests.conftest import run_async as _run

# =============================================================================
# Helpers
# =============================================================================


def _stub_cfg(model: str = "claude-sonnet-4-6", provider: str = "anthropic"):
    """Build a stand-in for ``_ensure_config()`` return value."""
    return SimpleNamespace(model=model, provider=provider)


def _make_client_cache(
    *,
    create_run_id: str = "run-001",
    thread_id: str = "thread-001",
):
    """Build a fake ``_ClientCache`` with sync + async stub clients.

    Returns ``(cache_mock, runs_create_sync_mock, runs_create_async_mock)``
    so tests can both invoke through the patched factory and inspect the
    exact call kwargs passed to ``runs.create``.
    """
    runs_sync = MagicMock()
    runs_sync.create.return_value = {"run_id": create_run_id}
    runs_sync.cancel = MagicMock(return_value=None)
    runs_sync.get = MagicMock(return_value={"status": "success"})

    threads_sync = MagicMock()
    threads_sync.create.return_value = {"thread_id": thread_id}

    sync_client = MagicMock()
    sync_client.runs = runs_sync
    sync_client.threads = threads_sync

    runs_async = MagicMock()
    runs_async.create = AsyncMock(return_value={"run_id": create_run_id})

    threads_async = MagicMock()
    threads_async.create = AsyncMock(return_value={"thread_id": thread_id})

    async_client = MagicMock()
    async_client.runs = runs_async
    async_client.threads = threads_async

    cache = MagicMock()
    cache.get_sync = MagicMock(return_value=sync_client)
    cache.get_async = MagicMock(return_value=async_client)

    return cache, runs_sync, runs_async


def _runtime_stub():
    """Minimal stand-in for ``ToolRuntime`` accepted by the inner tools."""
    return SimpleNamespace(tool_call_id="tc-001", state={})


# =============================================================================
# 1. Idempotence
# =============================================================================


class TestIdempotence:
    """The patch must be a no-op after the first application."""

    def test_double_apply_does_not_re_wrap(self, restore_model_passthrough_patch):
        try:
            from deepagents.middleware import async_subagents as ds_mod
        except ImportError:
            pytest.skip("deepagents not available")

        original = ds_mod._build_start_tool
        patches_mod._patch_deepagents_model_passthrough()
        wrapped_once = ds_mod._build_start_tool
        assert wrapped_once is not original

        patches_mod._patch_deepagents_model_passthrough()
        wrapped_twice = ds_mod._build_start_tool
        assert wrapped_twice is wrapped_once  # not double-wrapped

    def test_flag_set_after_apply(self, restore_model_passthrough_patch):
        try:
            from deepagents.middleware import async_subagents as _  # noqa: F401
        except ImportError:
            pytest.skip("deepagents not available")
        patches_mod._model_passthrough_patched = False
        patches_mod._patch_deepagents_model_passthrough()
        assert patches_mod._model_passthrough_patched is True


# =============================================================================
# 2. start_async_task injects config into runs.create
# =============================================================================


class TestStartAsyncTaskInjection:
    """Sync and async start_async_task must inject configurable.model."""

    def test_sync_start_injects_config(self, restore_model_passthrough_patch):
        try:
            from deepagents.middleware import async_subagents as ds_mod
        except ImportError:
            pytest.skip("deepagents not available")

        patches_mod._patch_deepagents_model_passthrough()

        cache, runs_sync, _ = _make_client_cache()
        agent_map = {
            "writing-agent": {
                "name": "writing-agent",
                "description": "Draft paper",
                "graph_id": "writing-agent",
                "url": "http://localhost:6174",
            }
        }

        tool = ds_mod._build_start_tool(agent_map, cache, "desc")

        with patch(
            "EvoScientist.EvoScientist._ensure_config",
            return_value=_stub_cfg(model="gpt-5", provider="openai"),
        ):
            tool.func(
                description="hello",
                subagent_type="writing-agent",
                runtime=_runtime_stub(),
            )

        runs_sync.create.assert_called_once()
        kwargs = runs_sync.create.call_args.kwargs
        assert kwargs["thread_id"] == "thread-001"
        assert kwargs["assistant_id"] == "writing-agent"
        assert kwargs["config"] == {
            "configurable": {"model": "gpt-5", "model_provider": "openai"}
        }

    def test_async_start_injects_config(self, restore_model_passthrough_patch):
        try:
            from deepagents.middleware import async_subagents as ds_mod
        except ImportError:
            pytest.skip("deepagents not available")

        patches_mod._patch_deepagents_model_passthrough()

        cache, _, runs_async = _make_client_cache()
        agent_map = {
            "writing-agent": {
                "name": "writing-agent",
                "description": "Draft paper",
                "graph_id": "writing-agent",
                "url": "http://localhost:6174",
            }
        }

        tool = ds_mod._build_start_tool(agent_map, cache, "desc")

        with patch(
            "EvoScientist.EvoScientist._ensure_config",
            return_value=_stub_cfg(model="claude-haiku-4-5", provider="anthropic"),
        ):
            _run(
                tool.coroutine(
                    description="hi",
                    subagent_type="writing-agent",
                    runtime=_runtime_stub(),
                )
            )

        runs_async.create.assert_awaited_once()
        kwargs = runs_async.create.call_args.kwargs
        assert kwargs["config"] == {
            "configurable": {
                "model": "claude-haiku-4-5",
                "model_provider": "anthropic",
            }
        }


# =============================================================================
# 3. Live config read at tool-call time (post-/model switch behavior)
# =============================================================================


class TestLiveConfigRead:
    """The patch must read cfg fresh on every tool call, not at patch time."""

    def test_two_calls_reflect_separate_cfg(self, restore_model_passthrough_patch):
        try:
            from deepagents.middleware import async_subagents as ds_mod
        except ImportError:
            pytest.skip("deepagents not available")

        patches_mod._patch_deepagents_model_passthrough()

        cache, runs_sync, _ = _make_client_cache()
        agent_map = {
            "writing-agent": {
                "name": "writing-agent",
                "description": "x",
                "graph_id": "writing-agent",
                "url": "http://localhost:6174",
            }
        }
        tool = ds_mod._build_start_tool(agent_map, cache, "desc")

        with patch(
            "EvoScientist.EvoScientist._ensure_config",
            return_value=_stub_cfg(model="model-a", provider="anthropic"),
        ):
            tool.func(
                description="t1",
                subagent_type="writing-agent",
                runtime=_runtime_stub(),
            )
        first_kwargs = runs_sync.create.call_args.kwargs

        with patch(
            "EvoScientist.EvoScientist._ensure_config",
            return_value=_stub_cfg(model="model-b", provider="openai"),
        ):
            tool.func(
                description="t2",
                subagent_type="writing-agent",
                runtime=_runtime_stub(),
            )
        second_kwargs = runs_sync.create.call_args.kwargs

        assert first_kwargs["config"]["configurable"]["model"] == "model-a"
        assert second_kwargs["config"]["configurable"]["model"] == "model-b"


# =============================================================================
# 4. update_async_task also injects config
# =============================================================================


class TestUpdateAsyncTaskInjection:
    """update_async_task must inject config too — not just start."""

    def _tracked_task(self, agent_name: str = "writing-agent") -> dict:
        return {
            "task_id": "thread-001",
            "agent_name": agent_name,
            "thread_id": "thread-001",
            "run_id": "old-run",
            "status": "running",
            "created_at": "2026-05-07T00:00:00Z",
            "last_checked_at": "2026-05-07T00:00:00Z",
            "last_updated_at": "2026-05-07T00:00:00Z",
        }

    def test_async_update_injects_config(self, restore_model_passthrough_patch):
        """The async coroutine path must inject config too."""
        try:
            from deepagents.middleware import async_subagents as ds_mod
        except ImportError:
            pytest.skip("deepagents not available")

        patches_mod._patch_deepagents_model_passthrough()

        cache, _, runs_async = _make_client_cache()
        agent_map = {
            "writing-agent": {
                "name": "writing-agent",
                "description": "x",
                "graph_id": "writing-agent",
                "url": "http://localhost:6174",
            }
        }
        runtime = SimpleNamespace(
            tool_call_id="tc-002",
            state={"async_tasks": {"thread-001": self._tracked_task()}},
        )

        tool = ds_mod._build_update_tool(agent_map, cache)

        with patch(
            "EvoScientist.EvoScientist._ensure_config",
            return_value=_stub_cfg(model="gpt-5", provider="openai"),
        ):
            _run(
                tool.coroutine(
                    task_id="thread-001",
                    message="follow up async",
                    runtime=runtime,
                )
            )

        runs_async.create.assert_awaited_once()
        kwargs = runs_async.create.call_args.kwargs
        assert kwargs["config"] == {
            "configurable": {"model": "gpt-5", "model_provider": "openai"}
        }
        assert kwargs.get("multitask_strategy") == "interrupt"

    def test_sync_update_injects_config(self, restore_model_passthrough_patch):
        try:
            from deepagents.middleware import async_subagents as ds_mod
        except ImportError:
            pytest.skip("deepagents not available")

        patches_mod._patch_deepagents_model_passthrough()

        cache, runs_sync, _ = _make_client_cache()
        agent_map = {
            "writing-agent": {
                "name": "writing-agent",
                "description": "x",
                "graph_id": "writing-agent",
                "url": "http://localhost:6174",
            }
        }
        # update_async_task reads tracked task from runtime.state
        tracked_task = {
            "task_id": "thread-001",
            "agent_name": "writing-agent",
            "thread_id": "thread-001",
            "run_id": "old-run",
            "status": "running",
            "created_at": "2026-05-07T00:00:00Z",
            "last_checked_at": "2026-05-07T00:00:00Z",
            "last_updated_at": "2026-05-07T00:00:00Z",
        }
        runtime = SimpleNamespace(
            tool_call_id="tc-002",
            state={"async_tasks": {"thread-001": tracked_task}},
        )

        tool = ds_mod._build_update_tool(agent_map, cache)

        with patch(
            "EvoScientist.EvoScientist._ensure_config",
            return_value=_stub_cfg(model="gpt-5", provider="openai"),
        ):
            tool.func(
                task_id="thread-001",
                message="follow up",
                runtime=runtime,
            )

        runs_sync.create.assert_called_once()
        kwargs = runs_sync.create.call_args.kwargs
        assert kwargs["config"]["configurable"]["model"] == "gpt-5"
        assert kwargs["config"]["configurable"]["model_provider"] == "openai"
        # update preserves multitask_strategy="interrupt" — verify no regression
        assert kwargs.get("multitask_strategy") == "interrupt"


# =============================================================================
# 5. Other client methods are unaffected
# =============================================================================


class TestNonInterceptedMethods:
    """``threads.create``, ``runs.get``, ``runs.cancel`` must pass through."""

    def test_threads_create_not_modified(self, restore_model_passthrough_patch):
        """``threads.create()`` is called by start_async_task pre-runs.create.

        The patch should not inject config here, because thread creation
        doesn't take config (and would error out if we did).
        """
        try:
            from deepagents.middleware import async_subagents as ds_mod
        except ImportError:
            pytest.skip("deepagents not available")

        patches_mod._patch_deepagents_model_passthrough()

        cache, _runs_sync, _ = _make_client_cache()
        threads_create = cache.get_sync.return_value.threads.create
        agent_map = {
            "writing-agent": {
                "name": "writing-agent",
                "description": "x",
                "graph_id": "writing-agent",
                "url": "http://localhost:6174",
            }
        }
        tool = ds_mod._build_start_tool(agent_map, cache, "desc")

        with patch(
            "EvoScientist.EvoScientist._ensure_config",
            return_value=_stub_cfg(),
        ):
            tool.func(
                description="hi",
                subagent_type="writing-agent",
                runtime=_runtime_stub(),
            )

        # threads.create() is called with no kwargs (deepagents pattern).
        threads_create.assert_called_once_with()


# =============================================================================
# 6. Empty cfg → no config kwarg added
# =============================================================================


class TestEmptyCfg:
    """If neither model nor provider is set, don't inject anything."""

    def test_empty_cfg_no_config_kwarg(self, restore_model_passthrough_patch):
        try:
            from deepagents.middleware import async_subagents as ds_mod
        except ImportError:
            pytest.skip("deepagents not available")

        patches_mod._patch_deepagents_model_passthrough()

        cache, runs_sync, _ = _make_client_cache()
        agent_map = {
            "writing-agent": {
                "name": "writing-agent",
                "description": "x",
                "graph_id": "writing-agent",
                "url": "http://localhost:6174",
            }
        }
        tool = ds_mod._build_start_tool(agent_map, cache, "desc")

        with patch(
            "EvoScientist.EvoScientist._ensure_config",
            return_value=SimpleNamespace(model=None, provider=None),
        ):
            tool.func(
                description="hi",
                subagent_type="writing-agent",
                runtime=_runtime_stub(),
            )

        runs_sync.create.assert_called_once()
        kwargs = runs_sync.create.call_args.kwargs
        # No config kwarg should be added when there's nothing to override.
        assert "config" not in kwargs


# =============================================================================
# 7. Caller-supplied config keys are preserved
# =============================================================================


class TestPreserveExistingConfig:
    """If a caller already supplied config.configurable.X, our merge keeps it."""

    def test_existing_configurable_preserved(self):
        """Direct unit test of the merge helper (integration covered above)."""
        with patch(
            "EvoScientist.EvoScientist._ensure_config",
            return_value=_stub_cfg(model="gpt-5", provider="openai"),
        ):
            merged = patches_mod._merge_runs_config_kwargs(
                {
                    "thread_id": "t1",
                    "config": {
                        "configurable": {"thread_id": "outer-t", "extra": 42},
                        "tags": ["debug"],
                    },
                }
            )
        assert merged["thread_id"] == "t1"
        assert merged["config"]["tags"] == ["debug"]
        assert merged["config"]["configurable"]["thread_id"] == "outer-t"
        assert merged["config"]["configurable"]["extra"] == 42
        assert merged["config"]["configurable"]["model"] == "gpt-5"
        assert merged["config"]["configurable"]["model_provider"] == "openai"

    def test_non_dict_config_replaced(self):
        """Non-dict ``config`` (e.g. a Pydantic RunnableConfig) is replaced.

        Documents the policy: callers that pass a non-dict ``config`` lose
        any other fields they may have set there. Acceptable today because
        deepagents' built-in ``runs.create`` path doesn't pass a config at
        all, but a future caller passing e.g. a Pydantic model would have
        their non-configurable fields silently dropped. If that becomes a
        real use case, ``_merge_runs_config_kwargs`` should grow a
        ``dict()`` coercion or raise.
        """

        class _Sentinel:
            """Stand-in for any non-dict config-shaped object."""

        with patch(
            "EvoScientist.EvoScientist._ensure_config",
            return_value=_stub_cfg(model="gpt-5", provider="openai"),
        ):
            merged = patches_mod._merge_runs_config_kwargs(
                {"thread_id": "t1", "config": _Sentinel()}
            )
        assert merged["thread_id"] == "t1"
        # Non-dict input was replaced with a fresh dict carrying only our
        # injected keys.
        assert merged["config"] == {
            "configurable": {"model": "gpt-5", "model_provider": "openai"}
        }
