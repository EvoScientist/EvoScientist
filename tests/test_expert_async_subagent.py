"""Tests for the payload-aware AsyncSubAgentMiddleware subclass."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from EvoScientist.middleware.expert_async_subagent import (
    EvoAsyncSubAgentMiddleware,
    ExpertStartAsyncTaskSchema,
    _payload_validation_error,
)

# =============================================================================
# _payload_validation_error — the presence-rules helper
# =============================================================================


class TestPayloadValidation:
    """Rules: expert specs require payload with skill_name; standard specs reject payload."""

    def test_expert_spec_requires_payload(self):
        spec = {
            "name": "e",
            "description": "d",
            "graph_id": "expert_container",
            "is_expert": True,
        }
        assert _payload_validation_error(spec, "e", None) is not None
        assert "required" in _payload_validation_error(spec, "e", None)

    def test_expert_spec_requires_dict_payload(self):
        spec = {
            "name": "e",
            "description": "d",
            "graph_id": "expert_container",
            "is_expert": True,
        }
        # `not isinstance(payload, dict)` catches a truthy-but-wrong-type payload
        # like a non-empty string. An empty string is falsy and hits the earlier
        # "payload required" branch — same rejection, different message.
        assert _payload_validation_error(spec, "e", "not a dict") is not None
        assert "must be a dict" in _payload_validation_error(spec, "e", "not a dict")

    def test_expert_spec_requires_skill_name(self):
        spec = {
            "name": "e",
            "description": "d",
            "graph_id": "expert_container",
            "is_expert": True,
        }
        assert (
            _payload_validation_error(spec, "e", {"output_path": "./x.md"}) is not None
        )
        assert "skill_name" in _payload_validation_error(
            spec, "e", {"output_path": "./x.md"}
        )

    def test_expert_spec_accepts_valid_payload(self):
        spec = {
            "name": "e",
            "description": "d",
            "graph_id": "expert_container",
            "is_expert": True,
        }
        assert (
            _payload_validation_error(
                spec, "e", {"skill_name": "lit-review", "output_path": "./x.md"}
            )
            is None
        )

    def test_standard_spec_accepts_no_payload(self):
        spec = {"name": "std", "description": "d", "graph_id": "writing_agent"}
        assert _payload_validation_error(spec, "std", None) is None

    def test_standard_spec_rejects_payload(self):
        spec = {"name": "std", "description": "d", "graph_id": "writing_agent"}
        error = _payload_validation_error(spec, "std", {"skill_name": "anything"})
        assert error is not None
        assert "not accepted for standard subagent" in error

    def test_is_expert_false_treated_as_standard(self):
        """Explicit ``is_expert=False`` matches the default (absent) behaviour."""
        spec = {
            "name": "std",
            "description": "d",
            "graph_id": "writing_agent",
            "is_expert": False,
        }
        assert _payload_validation_error(spec, "std", None) is None
        assert _payload_validation_error(spec, "std", {"skill_name": "x"}) is not None


# =============================================================================
# ExpertStartAsyncTaskSchema — the tool's input schema
# =============================================================================


class TestExpertStartAsyncTaskSchema:
    """The tool schema documents ``payload`` as optional so old callers still work."""

    def test_schema_accepts_no_payload(self):
        parsed = ExpertStartAsyncTaskSchema(description="d", subagent_type="std")
        assert parsed.payload is None

    def test_schema_accepts_payload(self):
        parsed = ExpertStartAsyncTaskSchema(
            description="d",
            subagent_type="e",
            payload={"skill_name": "lit-review", "output_path": "./x.md"},
        )
        assert parsed.payload == {"skill_name": "lit-review", "output_path": "./x.md"}

    def test_schema_field_descriptions_mention_expert_dispatch(self):
        """LLM-visible schema descriptions must document the payload contract."""
        fields = ExpertStartAsyncTaskSchema.model_fields
        assert fields["payload"].default is None
        # Description should mention when payload is required.
        assert "expert" in fields["payload"].description.lower()


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

    def test_start_tool_schema_is_payload_aware(self):
        mw = EvoAsyncSubAgentMiddleware(async_subagents=[_standard_spec()])
        start = next(t for t in mw.tools if t.name == "start_async_task")
        # Args schema should be OUR schema, not upstream's — payload accepted.
        assert start.args_schema is ExpertStartAsyncTaskSchema

    def test_construction_rejects_empty_subagents(self):
        with pytest.raises(ValueError, match="At least one async subagent"):
            EvoAsyncSubAgentMiddleware(async_subagents=[])

    def test_construction_rejects_duplicate_names(self):
        with pytest.raises(ValueError, match="Duplicate"):
            EvoAsyncSubAgentMiddleware(
                async_subagents=[_standard_spec(), _standard_spec()]
            )


class TestStartToolInvocation:
    """Direct invocation of the start tool's sync function.

    Mocks the LangGraph SDK client so we can assert on the input= merged
    into runs.create without any real network round-trip.
    """

    def _fake_client(self):
        client = MagicMock()
        client.threads.create.return_value = {"thread_id": "task-abc"}
        client.runs.create.return_value = {"run_id": "run-xyz"}
        return client

    def _invoke(self, mw, **kwargs):
        """Invoke the start tool's sync function, patching the client cache."""
        start = next(t for t in mw.tools if t.name == "start_async_task")
        client = self._fake_client()
        # Patch the client cache captured in the tool's closure.
        with patch.object(
            mw.tools[0]
            .func.__closure__[0]
            .cell_contents,  # too deep; use module attr instead
            "get_sync",
            return_value=client,
        ):
            pass
        # Simpler: patch at the module level — the tool's closure holds a
        # reference to a _ClientCache instance whose .get_sync we intercept.
        return client, start

    def test_start_merges_payload_into_input(self):
        mw = EvoAsyncSubAgentMiddleware(async_subagents=[_expert_spec()])
        start = next(t for t in mw.tools if t.name == "start_async_task")

        # Build a fake client and patch _ClientCache.get_sync on the module.
        client = self._fake_client()
        with patch(
            "EvoScientist.middleware.expert_async_subagent._ClientCache.get_sync",
            return_value=client,
        ):
            result = start.func(
                description="write a survey on X",
                subagent_type="literature-review",
                payload={"skill_name": "literature-review", "output_path": "./x.md"},
                runtime=SimpleNamespace(tool_call_id="tc1"),
            )

        # runs.create should have been called with input containing both the
        # user message AND the payload keys.
        client.runs.create.assert_called_once()
        kwargs = client.runs.create.call_args.kwargs
        assert kwargs["assistant_id"] == "expert_container"
        assert kwargs["input"]["messages"] == [
            {"role": "user", "content": "write a survey on X"}
        ]
        assert kwargs["input"]["skill_name"] == "literature-review"
        assert kwargs["input"]["output_path"] == "./x.md"
        # Return value is a Command that stamps the task into async_tasks state.
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

        client = self._fake_client()
        fake_cfg = EvoScientistConfig(model="test-model-abc", provider="test-provider")
        with (
            patch(
                "EvoScientist.middleware.expert_async_subagent._ClientCache.get_sync",
                return_value=client,
            ),
            # ``_read_cfg_configurable`` in llm/patches.py imports
            # ``_ensure_config`` from ``EvoScientist.EvoScientist`` at call
            # time; patch there so the proxy sees our fake config.
            patch("EvoScientist.EvoScientist._ensure_config", return_value=fake_cfg),
        ):
            start.func(
                description="w",
                subagent_type="literature-review",
                payload={"skill_name": "literature-review", "output_path": "./x.md"},
                runtime=SimpleNamespace(tool_call_id="tc1"),
            )

        kwargs = client.runs.create.call_args.kwargs
        assert "config" in kwargs, (
            "runs.create must receive config kwarg for model passthrough"
        )
        configurable = kwargs["config"]["configurable"]
        assert configurable["model"] == "test-model-abc"
        assert configurable["model_provider"] == "test-provider"

    def test_start_rejects_expert_without_payload(self):
        mw = EvoAsyncSubAgentMiddleware(async_subagents=[_expert_spec()])
        start = next(t for t in mw.tools if t.name == "start_async_task")

        with patch(
            "EvoScientist.middleware.expert_async_subagent._ClientCache.get_sync"
        ) as get_sync:
            result = start.func(
                description="anything",
                subagent_type="literature-review",
                payload=None,
                runtime=SimpleNamespace(tool_call_id="tc1"),
            )
        # Error surfaced as a string tool-result; client never contacted.
        assert isinstance(result, str)
        assert "payload required" in result
        get_sync.assert_not_called()

    def test_start_rejects_standard_with_payload(self):
        mw = EvoAsyncSubAgentMiddleware(async_subagents=[_standard_spec()])
        start = next(t for t in mw.tools if t.name == "start_async_task")

        with patch(
            "EvoScientist.middleware.expert_async_subagent._ClientCache.get_sync"
        ) as get_sync:
            result = start.func(
                description="anything",
                subagent_type="writing-agent",
                payload={"skill_name": "would-not-make-sense-here"},
                runtime=SimpleNamespace(tool_call_id="tc1"),
            )
        assert isinstance(result, str)
        assert "not accepted for standard subagent" in result
        get_sync.assert_not_called()

    def test_start_standard_without_payload_matches_upstream_shape(self):
        """A standard subagent invoked without payload should pass exactly the
        upstream input shape — just messages, no extra keys."""
        mw = EvoAsyncSubAgentMiddleware(async_subagents=[_standard_spec()])
        start = next(t for t in mw.tools if t.name == "start_async_task")

        client = self._fake_client()
        with patch(
            "EvoScientist.middleware.expert_async_subagent._ClientCache.get_sync",
            return_value=client,
        ):
            start.func(
                description="hi",
                subagent_type="writing-agent",
                payload=None,
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
            payload=None,
            runtime=SimpleNamespace(tool_call_id="tc1"),
        )
        assert isinstance(result, str)
        assert "Unknown async subagent type" in result
