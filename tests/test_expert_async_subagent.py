"""Tests for the skill-name-injecting AsyncSubAgentMiddleware subclass."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from EvoScientist.middleware.expert_async_subagent import (
    EvoAsyncSubAgentMiddleware,
    _build_run_input,
)


class _TestPayloadValidationRemoved:
    """Placeholder — the ``_payload_validation_error`` helper was deleted
    when ``payload`` was dropped from the tool schema (PR #391 review, X-4).
    The seven tests that lived here (``TestPayloadValidation``) no longer
    apply: subagent_type is validated by ``_validate_agent_type``,
    ``skill_name`` is injected by construction, and no other user-supplied
    fields reach ``client.runs.create(input=...)``. See
    ``TestBuildRunInput`` below and ``TestStartToolInvocation`` for the
    replacement coverage.
    """


# =============================================================================
# _build_run_input — the shared input-dict factory
# =============================================================================


class TestBuildRunInput:
    """``skill_name`` is injected for expert specs, absent for standard specs.
    The description always lands in ``messages`` verbatim — no LLM-authored
    key can overwrite it (was the pre-fix bug when ``payload`` was in scope).
    """

    def test_expert_spec_injects_skill_name(self):
        spec = {"name": "e", "graph_id": "g", "is_expert": True}
        result = _build_run_input(spec, "literature-review", "write a survey")
        assert result == {
            "messages": [{"role": "user", "content": "write a survey"}],
            "skill_name": "literature-review",
        }

    def test_standard_spec_matches_upstream_shape(self):
        """Standard specs (writing-agent, scheduler, ...) reach ``runs.create``
        with the upstream single-key shape — no ``skill_name`` injected."""
        spec = {"name": "writing-agent", "graph_id": "writing_agent"}
        result = _build_run_input(spec, "writing-agent", "hi")
        assert result == {"messages": [{"role": "user", "content": "hi"}]}

    def test_is_expert_false_treated_as_standard(self):
        """Explicit ``is_expert=False`` matches the default (absent) behaviour."""
        spec = {"name": "std", "graph_id": "writing_agent", "is_expert": False}
        result = _build_run_input(spec, "std", "hi")
        assert result == {"messages": [{"role": "user", "content": "hi"}]}

    def test_description_lands_verbatim(self):
        """Regression guard against the pre-fix bug where an LLM-authored
        ``payload`` could overwrite ``messages`` — description now travels
        through a channel the LLM cannot corrupt."""
        spec = {"name": "e", "graph_id": "g", "is_expert": True}
        result = _build_run_input(
            spec, "e", "write to ./artifacts/e/foo.md a summary of X"
        )
        assert result["messages"][0]["content"] == (
            "write to ./artifacts/e/foo.md a summary of X"
        )


# =============================================================================
# EvoAsyncSubAgentMiddleware — end-to-end tool invocation
# =============================================================================


def _standard_spec():
    return {
        "name": "writing-agent",
        "description": "std writer",
        "graph_id": "writing_agent",
    }


def _expert_spec():
    return {
        "name": "literature-review",
        "description": "expert lit review",
        "graph_id": "expert_container",
        "is_expert": True,
    }


class TestMiddlewareConstruction:
    def test_middleware_has_five_tools(self):
        mw = EvoAsyncSubAgentMiddleware(async_subagents=[_standard_spec()])
        names = [t.name for t in mw.tools]
        assert set(names) == {
            "start_async_task",
            "check_async_task",
            "update_async_task",
            "cancel_async_task",
            "list_async_tasks",
        }

    def test_start_tool_schema_matches_upstream(self):
        """The tool signature returned to upstream's exact shape when
        ``payload`` was dropped — schema is now ``deepagents``'s
        ``StartAsyncTaskSchema``."""
        from deepagents.middleware.async_subagents import StartAsyncTaskSchema

        mw = EvoAsyncSubAgentMiddleware(async_subagents=[_standard_spec()])
        start = next(t for t in mw.tools if t.name == "start_async_task")
        assert start.args_schema is StartAsyncTaskSchema

    def test_construction_rejects_empty_subagents(self):
        with pytest.raises(ValueError, match="At least one async subagent"):
            EvoAsyncSubAgentMiddleware(async_subagents=[])

    def test_construction_rejects_duplicate_names(self):
        with pytest.raises(ValueError, match="Duplicate"):
            EvoAsyncSubAgentMiddleware(
                async_subagents=[_standard_spec(), _standard_spec()]
            )


def _fake_sync_client():
    client = MagicMock()
    client.threads.create.return_value = {"thread_id": "task-abc"}
    client.runs.create.return_value = {"run_id": "run-xyz"}
    return client


def _fake_async_client():
    client = MagicMock()
    client.threads.create = AsyncMock(return_value={"thread_id": "task-abc"})
    client.runs.create = AsyncMock(return_value={"run_id": "run-xyz"})
    return client


class TestStartToolInvocation:
    """Direct invocation of the start tool's sync function.

    Mocks ``_ClientCache.get_sync`` so we can assert on the ``input`` dict
    handed to ``runs.create`` without any real network round-trip.
    """

    def test_start_injects_skill_name_for_expert_spec(self):
        """The middleware sets ``input_dict['skill_name'] = subagent_type``
        by construction — the shared container graph resolves the right
        persona without a payload dict crossing the LLM channel."""
        mw = EvoAsyncSubAgentMiddleware(async_subagents=[_expert_spec()])
        start = next(t for t in mw.tools if t.name == "start_async_task")

        client = _fake_sync_client()
        with patch(
            "EvoScientist.middleware.expert_async_subagent._ClientCache.get_sync",
            return_value=client,
        ):
            result = start.func(
                description="write to ./artifacts/literature-review/attn.md a survey on X",
                subagent_type="literature-review",
                runtime=SimpleNamespace(tool_call_id="tc1"),
            )

        client.runs.create.assert_called_once()
        kwargs = client.runs.create.call_args.kwargs
        assert kwargs["assistant_id"] == "expert_container"
        assert kwargs["input"]["messages"] == [
            {
                "role": "user",
                "content": (
                    "write to ./artifacts/literature-review/attn.md a survey on X"
                ),
            }
        ]
        assert kwargs["input"]["skill_name"] == "literature-review"
        assert "payload" not in kwargs["input"]
        assert "output_path" not in kwargs["input"]
        # Return value stamps the task into async_tasks state.
        assert "async_tasks" in result.update
        assert "task-abc" in result.update["async_tasks"]

    def test_start_injects_cfg_model_into_configurable(self):
        """cfg.model / cfg.provider land in ``config.configurable`` on every
        ``runs.create`` so the deployed graph re-resolves its chat model per
        run instead of using whatever was baked at container-build time.
        Without this the ``/model`` CLI switch silently doesn't propagate to
        expert launches.
        """
        from EvoScientist.config.settings import EvoScientistConfig

        mw = EvoAsyncSubAgentMiddleware(async_subagents=[_expert_spec()])
        start = next(t for t in mw.tools if t.name == "start_async_task")

        client = _fake_sync_client()
        fake_cfg = EvoScientistConfig(model="test-model-abc", provider="test-provider")
        with (
            patch(
                "EvoScientist.middleware.expert_async_subagent._ClientCache.get_sync",
                return_value=client,
            ),
            patch("EvoScientist.EvoScientist._ensure_config", return_value=fake_cfg),
        ):
            start.func(
                description="w",
                subagent_type="literature-review",
                runtime=SimpleNamespace(tool_call_id="tc1"),
            )

        kwargs = client.runs.create.call_args.kwargs
        assert "config" in kwargs
        configurable = kwargs["config"]["configurable"]
        assert configurable["model"] == "test-model-abc"
        assert configurable["model_provider"] == "test-provider"

    def test_start_standard_spec_matches_upstream_input_shape(self):
        """Standard subagents (writing-agent, scheduler, ...) reach
        ``runs.create`` with the upstream single-key ``messages`` shape."""
        mw = EvoAsyncSubAgentMiddleware(async_subagents=[_standard_spec()])
        start = next(t for t in mw.tools if t.name == "start_async_task")

        client = _fake_sync_client()
        with patch(
            "EvoScientist.middleware.expert_async_subagent._ClientCache.get_sync",
            return_value=client,
        ):
            start.func(
                description="hi",
                subagent_type="writing-agent",
                runtime=SimpleNamespace(tool_call_id="tc1"),
            )
        kwargs = client.runs.create.call_args.kwargs
        assert kwargs["input"] == {"messages": [{"role": "user", "content": "hi"}]}

    def test_start_unknown_subagent_returns_error(self):
        mw = EvoAsyncSubAgentMiddleware(async_subagents=[_standard_spec()])
        start = next(t for t in mw.tools if t.name == "start_async_task")

        result = start.func(
            description="hi",
            subagent_type="does-not-exist",
            runtime=SimpleNamespace(tool_call_id="tc1"),
        )
        assert isinstance(result, str)
        assert "Unknown async subagent type" in result


class TestAstartToolInvocation:
    """Mirror ``TestStartToolInvocation`` against ``astart_async_task`` — the
    coroutine langgraph_api actually runs in production. Pre-fix zero
    coverage: X-iZhang flagged that a fix applied only to the sync body
    would leave tests green and production broken."""

    @pytest.mark.asyncio
    async def test_astart_injects_skill_name_for_expert_spec(self):
        mw = EvoAsyncSubAgentMiddleware(async_subagents=[_expert_spec()])
        start = next(t for t in mw.tools if t.name == "start_async_task")

        client = _fake_async_client()
        with patch(
            "EvoScientist.middleware.expert_async_subagent._ClientCache.get_async",
            return_value=client,
        ):
            result = await start.coroutine(
                description="write to ./artifacts/literature-review/attn.md a survey on X",
                subagent_type="literature-review",
                runtime=SimpleNamespace(tool_call_id="tc1"),
            )

        client.runs.create.assert_awaited_once()
        kwargs = client.runs.create.await_args.kwargs
        assert kwargs["assistant_id"] == "expert_container"
        assert kwargs["input"]["skill_name"] == "literature-review"
        assert kwargs["input"]["messages"][0]["content"].startswith(
            "write to ./artifacts/literature-review/attn.md"
        )
        assert "payload" not in kwargs["input"]
        assert "async_tasks" in result.update
        assert "task-abc" in result.update["async_tasks"]

    @pytest.mark.asyncio
    async def test_astart_injects_cfg_model_into_configurable(self):
        from EvoScientist.config.settings import EvoScientistConfig

        mw = EvoAsyncSubAgentMiddleware(async_subagents=[_expert_spec()])
        start = next(t for t in mw.tools if t.name == "start_async_task")

        client = _fake_async_client()
        fake_cfg = EvoScientistConfig(model="test-model-abc", provider="test-provider")
        with (
            patch(
                "EvoScientist.middleware.expert_async_subagent._ClientCache.get_async",
                return_value=client,
            ),
            patch("EvoScientist.EvoScientist._ensure_config", return_value=fake_cfg),
        ):
            await start.coroutine(
                description="w",
                subagent_type="literature-review",
                runtime=SimpleNamespace(tool_call_id="tc1"),
            )

        kwargs = client.runs.create.await_args.kwargs
        assert "config" in kwargs
        configurable = kwargs["config"]["configurable"]
        assert configurable["model"] == "test-model-abc"
        assert configurable["model_provider"] == "test-provider"

    @pytest.mark.asyncio
    async def test_astart_standard_spec_matches_upstream_input_shape(self):
        mw = EvoAsyncSubAgentMiddleware(async_subagents=[_standard_spec()])
        start = next(t for t in mw.tools if t.name == "start_async_task")

        client = _fake_async_client()
        with patch(
            "EvoScientist.middleware.expert_async_subagent._ClientCache.get_async",
            return_value=client,
        ):
            await start.coroutine(
                description="hi",
                subagent_type="writing-agent",
                runtime=SimpleNamespace(tool_call_id="tc1"),
            )
        kwargs = client.runs.create.await_args.kwargs
        assert kwargs["input"] == {"messages": [{"role": "user", "content": "hi"}]}

    @pytest.mark.asyncio
    async def test_astart_unknown_subagent_returns_error(self):
        mw = EvoAsyncSubAgentMiddleware(async_subagents=[_standard_spec()])
        start = next(t for t in mw.tools if t.name == "start_async_task")

        result = await start.coroutine(
            description="hi",
            subagent_type="does-not-exist",
            runtime=SimpleNamespace(tool_call_id="tc1"),
        )
        assert isinstance(result, str)
        assert "Unknown async subagent type" in result
