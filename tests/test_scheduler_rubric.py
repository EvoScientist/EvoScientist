"""Scheduler graph mounts ``RubricMiddleware`` last, with a read-only grader.

The middleware must be the final entry so its ``after_agent`` runs first in
the reverse-ordered chain and a ``needs_revision`` verdict jumps back to the
model *before* ``EvoMemoryLifecycleMiddleware`` launches a memory worker.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Imported before any patch is active: the module binds ``get_effective_config``
# at import, and a first import under the patch would freeze the mock in place.
import EvoScientist.EvoScientist  # noqa: F401
from EvoScientist.config import MemoryObservationWriter


def _build(name: str, workspace, aux_model=None):
    """Build ``name`` through the real factory with heavy deps mocked.

    The backend is a real ``FilesystemBackend`` on ``workspace`` because
    ``FilesystemMiddleware`` rejects callable stand-ins (a ``MagicMock`` looks
    like a removed backend factory).

    Returns ``(create_deep_agent kwargs, backend, aux_model, lifecycle_stub)``.
    """
    from deepagents.backends import FilesystemBackend

    backend = FilesystemBackend(root_dir=workspace)
    cfg = MagicMock()
    cfg.recursion_limit = 1_000_000
    cfg.memory_profile_enabled = True
    cfg.memory_observations_enabled = True
    cfg.memory_observation_writer = MemoryObservationWriter.ALL
    cfg.memory_workers_enabled = True
    lifecycle_stub = MagicMock(name="EvoMemoryLifecycleMiddleware")

    with ExitStack() as stack:
        stack.enter_context(
            patch("EvoScientist.config.get_effective_config", return_value=cfg)
        )
        stack.enter_context(patch("EvoScientist.config.apply_config_to_env"))
        stack.enter_context(
            patch(
                "EvoScientist.utils.load_subagents",
                return_value=[
                    {"name": name, "system_prompt": "", "tools": [], "skills": None}
                ],
            )
        )
        stack.enter_context(patch("EvoScientist.EvoScientist._ensure_chat_model"))
        aux = stack.enter_context(
            patch(
                "EvoScientist.EvoScientist._ensure_auxiliary_chat_model",
                **({"return_value": aux_model} if aux_model is not None else {}),
            )
        )
        stack.enter_context(
            patch(
                "EvoScientist.EvoScientist._get_default_backend", return_value=backend
            )
        )
        stack.enter_context(
            patch(
                "EvoScientist.EvoScientist._get_default_middleware",
                side_effect=lambda **_: [lifecycle_stub],
            )
        )
        stack.enter_context(
            patch("EvoScientist.EvoScientist._load_mcp_tools_cached", return_value={})
        )
        create = stack.enter_context(patch("deepagents.create_deep_agent"))
        create.return_value.with_config.return_value = MagicMock()

        from EvoScientist.subagents._factory import build_async_subagent_graph

        build_async_subagent_graph(name)
        return create.call_args.kwargs, backend, aux.return_value, lifecycle_stub


def test_scheduler_graph_mounts_rubric_middleware_last(tmp_path):
    from deepagents import RubricMiddleware

    kwargs, _backend, _aux, lifecycle_stub = _build("scheduler", tmp_path)
    middleware = kwargs["middleware"]
    assert middleware[0] is lifecycle_stub
    assert isinstance(middleware[-1], RubricMiddleware)


def test_other_async_graphs_do_not_mount_rubric(tmp_path):
    from deepagents import RubricMiddleware

    kwargs, _backend, _aux, lifecycle_stub = _build("writing-agent", tmp_path)
    assert kwargs["middleware"] == [lifecycle_stub]
    assert not any(isinstance(m, RubricMiddleware) for m in kwargs["middleware"])


def test_scheduler_grader_gets_read_only_tools_on_the_agent_backend(tmp_path):
    from deepagents import FilesystemMiddleware

    kwargs, backend, _aux, _stub = _build("scheduler", tmp_path)
    rubric = kwargs["middleware"][-1]
    grader_fs = rubric._grader_middleware[0]
    assert isinstance(grader_fs, FilesystemMiddleware)
    assert [t.name for t in grader_fs.tools] == ["ls", "read_file"]
    assert grader_fs.backend is backend
    assert kwargs["backend"] is backend


def test_scheduler_rubric_uses_scheduler_model_and_allows_one_retry(tmp_path):
    kwargs, _backend, aux, _stub = _build("scheduler", tmp_path)
    rubric = kwargs["middleware"][-1]
    assert rubric._model is aux
    assert rubric.max_iterations == 2


def test_scheduler_graph_build_emits_no_beta_warning(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build("scheduler", tmp_path)
    assert not [w for w in caught if w.category.__name__ == "LangChainBetaWarning"]


def test_rubric_evaluation_is_logged_at_info(caplog):
    from EvoScientist.subagents._factory import _log_rubric_evaluation

    with caplog.at_level(logging.INFO, logger="EvoScientist.subagents._factory"):
        _log_rubric_evaluation(
            {
                "grading_run_id": "g-1",
                "iteration": 0,
                "result": "needs_revision",
                "explanation": "digest.md is missing today's date",
                "criteria": [],
            }
        )
    assert "needs_revision" in caplog.text
    assert "missing today's date" in caplog.text


def test_scheduler_grader_never_evicts_to_the_shared_workspace(tmp_path):
    """Both eviction paths write files through the backend; the grader must
    stay read-only even on an oversized rubric or transcript."""
    kwargs, _backend, _aux, _stub = _build("scheduler", tmp_path)
    grader_fs = kwargs["middleware"][-1]._grader_middleware[0]
    assert grader_fs._tool_token_limit_before_evict is None
    assert grader_fs._human_message_token_limit_before_evict is None


# ---------------------------------------------------------------------------
# Grader structured-output strategy is explicit per OpenRouter model family
# ---------------------------------------------------------------------------


def _openrouter(model_id: str):
    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(model=model_id, api_key="test-key")


def test_grader_strategy_is_json_mode_for_gemini_on_openrouter():
    """OpenRouter drops the criteria oneOf from Gemini tool schemas; JSON mode works."""
    from langchain.agents.structured_output import ProviderStrategy

    from EvoScientist.subagents._factory import _grader_strategy

    strategy = _grader_strategy(_openrouter("google/gemini-3.8-flash"))
    assert isinstance(strategy, ProviderStrategy)


def test_grader_strategy_is_tool_calling_for_anthropic_on_openrouter():
    """OpenRouter's Anthropic JSON mode returns non-JSON; tool calling works.

    ``claude-fable-5`` matters: its id matches langchain's name-regex fallback,
    which would force JSON mode if we only pinned the profile.
    """
    from langchain.agents.structured_output import ToolStrategy

    from EvoScientist.subagents._factory import _grader_strategy

    for model_id in ("anthropic/claude-fable-5", "anthropic/claude-sonnet-4.6"):
        assert isinstance(_grader_strategy(_openrouter(model_id)), ToolStrategy)


def test_grader_strategy_defers_to_langchain_elsewhere():
    from langchain_anthropic import ChatAnthropic

    from EvoScientist.subagents._factory import _grader_strategy

    assert (
        _grader_strategy(ChatAnthropic(model="claude-haiku-4-5", api_key="k")) is None
    )
    assert _grader_strategy(_openrouter("qwen/qwen3.8-flash")) is None


def test_scheduler_grader_is_built_with_the_explicit_strategy(tmp_path):
    from deepagents.middleware.rubric import GraderResponse
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy

    kwargs, _backend, aux, _stub = _build(
        "scheduler", tmp_path, aux_model=_openrouter("anthropic/claude-fable-5")
    )
    rubric = kwargs["middleware"][-1]
    assert rubric._model is aux  # no model copy; the strategy is passed explicitly
    with patch(
        "EvoScientist.subagents._factory.create_agent", wraps=create_agent
    ) as spy:
        rubric._ensure_grader()
    response_format = spy.call_args.kwargs["response_format"]
    assert isinstance(response_format, ToolStrategy)
    assert response_format.schema is GraderResponse


def test_scheduler_grader_builds_against_current_upstream_attributes(tmp_path):
    """Unpatched build: the private deepagents names we mirror still exist."""
    kwargs, _backend, _aux, _stub = _build(
        "scheduler", tmp_path, aux_model=_openrouter("google/gemini-3.8-flash")
    )
    rubric = kwargs["middleware"][-1]
    grader = rubric._ensure_grader()
    assert grader is rubric._ensure_grader()  # memoised like upstream


# ---------------------------------------------------------------------------
# Grader call budget: a parse-error ping-pong must fail closed, not spin
# ---------------------------------------------------------------------------


class _BrokenGrader(BaseChatModel):
    """Always answers with a GraderResponse whose criteria are null."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "broken-grader"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "GraderResponse",
                    "args": {
                        "result": "satisfied",
                        "explanation": "x",
                        "criteria": [None],
                    },
                    "id": f"call-{self.calls}",
                    "type": "tool_call",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_grader_call_budget_stops_a_parse_error_loop():
    from deepagents.middleware.rubric import GraderResponse
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy

    from EvoScientist.subagents._factory import _GraderCallBudget

    fake = _BrokenGrader()
    grader = create_agent(
        model=fake,
        middleware=[_GraderCallBudget(max_calls=3)],
        response_format=ToolStrategy(GraderResponse),
    )
    with pytest.raises(RuntimeError, match="rubric grader"):
        grader.invoke(
            {"messages": [HumanMessage("grade this")]},
            config={"recursion_limit": 60},
        )
    assert fake.calls == 3


def test_scheduler_grader_carries_a_call_budget(tmp_path):
    from EvoScientist.subagents._factory import _GraderCallBudget

    kwargs, _backend, _aux, _stub = _build("scheduler", tmp_path)
    grader_mw = kwargs["middleware"][-1]._grader_middleware
    assert [type(m).__name__ for m in grader_mw] == [
        "FilesystemMiddleware",
        "_GraderCallBudget",
    ]
    assert isinstance(grader_mw[1], _GraderCallBudget)
    assert grader_mw[1].max_calls == 12


async def test_grader_call_budget_also_guards_the_async_path():
    """langgraph dev grades through ``aafter_agent`` → ``ainvoke``."""
    from deepagents.middleware.rubric import GraderResponse
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy

    from EvoScientist.subagents._factory import _GraderCallBudget

    fake = _BrokenGrader()
    grader = create_agent(
        model=fake,
        middleware=[_GraderCallBudget(max_calls=2)],
        response_format=ToolStrategy(GraderResponse),
    )
    with pytest.raises(RuntimeError, match="rubric grader"):
        await grader.ainvoke(
            {"messages": [HumanMessage("grade this")]},
            config={"recursion_limit": 60},
        )
    assert fake.calls == 2


def test_factory_warns_when_openrouter_fable_cannot_grade(tmp_path, caplog):
    """Fable/Mythos via OpenRouter rejects forced tool_choice and returns JSON
    missing required fields, so no grader strategy works; say so at build."""
    with caplog.at_level(logging.WARNING, logger="EvoScientist.subagents._factory"):
        _build(
            "scheduler", tmp_path, aux_model=_openrouter("anthropic/claude-fable-5.1")
        )
    assert "claude-fable-5.1" in caplog.text
    assert "auxiliary_model" in caplog.text


def test_factory_stays_quiet_for_supported_openrouter_graders(tmp_path, caplog):
    """Fable 5 (not 5.1) grades fine through OpenRouter, probed 2026-09-04."""
    with caplog.at_level(logging.WARNING, logger="EvoScientist.subagents._factory"):
        _build(
            "scheduler", tmp_path, aux_model=_openrouter("anthropic/claude-sonnet-5")
        )
        _build("scheduler", tmp_path, aux_model=_openrouter("anthropic/claude-fable-5"))
        _build("scheduler", tmp_path, aux_model=_openrouter("google/gemini-3.8-flash"))
    assert "rubric" not in caplog.text.lower()
