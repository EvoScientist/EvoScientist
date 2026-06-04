"""Tests for ``EvoScientist.subagents._factory.build_async_subagent_graph``.

Pins the integration contract that the factory must request middleware
in async-safe mode (``for_async_subagent=True``). Without this, a future
refactor that drops the keyword argument would silently re-introduce
``AskUserMiddleware`` into the deployed graph and reproduce the
``interrupt()``-based deadlock the flag was added to prevent.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from EvoScientist.config import MemoryObservationWriter


def _single_middleware(subagent: dict, class_name: str):
    matches = [m for m in subagent["middleware"] if type(m).__name__ == class_name]
    assert len(matches) == 1
    return matches[0]


def _assert_subagent_memory_middleware(subagent: dict, *, source_agent: str) -> None:
    from EvoScientist.middleware.memory_lifecycle import MemoryLifecycleRole

    memory_middleware = _single_middleware(subagent, "EvoMemoryMiddleware")
    lifecycle_middleware = _single_middleware(
        subagent,
        "EvoMemoryLifecycleMiddleware",
    )

    assert [tool.name for tool in memory_middleware.tools] == ["record_observation"]
    assert lifecycle_middleware._role == MemoryLifecycleRole.SUBAGENT
    assert lifecycle_middleware._source_agent == source_agent
    assert lifecycle_middleware._project_id == memory_middleware.project_id


@patch("deepagents.create_deep_agent")
@patch("EvoScientist.EvoScientist._load_mcp_tools_cached", return_value={})
@patch("EvoScientist.EvoScientist._get_default_middleware", return_value=[])
@patch("EvoScientist.EvoScientist._get_default_backend")
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.utils.load_subagents")
@patch("EvoScientist.config.apply_config_to_env")
@patch("EvoScientist.config.get_effective_config")
def test_factory_requests_async_safe_middleware(
    mock_get_cfg,
    mock_apply_env,
    mock_load_subs,
    mock_chat,
    mock_backend,
    mock_get_mw,
    mock_mcp,
    mock_create,
):
    """``build_async_subagent_graph`` must call ``_get_default_middleware``
    with ``for_async_subagent=True``.

    The bare argument call would silently include ``AskUserMiddleware`` in
    the deployed graph, which deadlocks via ``interrupt()`` (no UI in the
    langgraph dev subprocess to resume the interrupt).
    """
    # Minimal config stub so factory's `cfg.recursion_limit` access works.
    cfg = MagicMock()
    cfg.recursion_limit = 1_000_000
    cfg.memory_profile_enabled = True
    cfg.memory_observations_enabled = True
    cfg.memory_observation_writer = MemoryObservationWriter.ALL
    cfg.memory_workers_enabled = True
    mock_get_cfg.return_value = cfg
    # Factory looks up the requested name in the loaded subagent specs;
    # any matching name is fine.
    mock_load_subs.return_value = [
        {
            "name": "writing-agent",
            "system_prompt": "",
            "tools": [],
            "skills": None,
        }
    ]
    # ``create_deep_agent(...).with_config({...})`` chain — return something
    # chainable so the factory's terminal ``.with_config(...)`` doesn't blow up.
    mock_create.return_value.with_config.return_value = MagicMock()

    from EvoScientist.subagents._factory import build_async_subagent_graph

    build_async_subagent_graph("writing-agent")

    # The contract: factory MUST pass async-safe mode and the source agent name.
    mock_get_mw.assert_called_once_with(
        for_async_subagent=True,
        memory_source_agent="writing-agent",
    )
    subagents = mock_create.call_args.kwargs["subagents"]
    assert subagents[0]["name"] == "general-purpose"
    _assert_subagent_memory_middleware(
        subagents[0],
        source_agent="general-purpose",
    )


@patch("EvoScientist.EvoScientist._ensure_chat_model")
def test_inject_subagent_adds_memory_middleware(mock_model, tmp_path):
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})

    from EvoScientist.EvoScientist import _inject_subagent_middleware

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subs = [{"name": "test-agent"}]

    _inject_subagent_middleware(subs, workspace_dir=workspace)

    _assert_subagent_memory_middleware(subs[0], source_agent="test-agent")


@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_inject_subagent_omits_memory_middleware_when_memory_disabled(
    mock_config, mock_model, tmp_path
):
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})
    cfg = MagicMock()
    cfg.memory_profile_enabled = False
    cfg.memory_observations_enabled = False
    cfg.memory_observation_writer = MemoryObservationWriter.ALL
    cfg.memory_workers_enabled = True
    mock_config.return_value = cfg

    from EvoScientist.EvoScientist import _inject_subagent_middleware

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subs = [{"name": "test-agent"}]

    _inject_subagent_middleware(subs, workspace_dir=workspace)

    assert not [
        m
        for m in subs[0]["middleware"]
        if type(m).__name__ in {"EvoMemoryMiddleware", "EvoMemoryLifecycleMiddleware"}
    ]


@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_inject_subagent_worker_only_observation_writer_keeps_live_tool_off(
    mock_config, mock_model, tmp_path
):
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})
    cfg = MagicMock()
    cfg.memory_profile_enabled = False
    cfg.memory_observations_enabled = True
    cfg.memory_observation_writer = MemoryObservationWriter.WORKER
    cfg.memory_workers_enabled = True
    mock_config.return_value = cfg

    from EvoScientist.EvoScientist import _inject_subagent_middleware
    from EvoScientist.middleware.memory_lifecycle import MemoryLifecycleRole

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subs = [{"name": "test-agent"}]

    _inject_subagent_middleware(subs, workspace_dir=workspace)

    memory_middleware = _single_middleware(subs[0], "EvoMemoryMiddleware")
    lifecycle_middleware = _single_middleware(
        subs[0],
        "EvoMemoryLifecycleMiddleware",
    )
    assert memory_middleware.tools == []
    assert lifecycle_middleware._role == MemoryLifecycleRole.SUBAGENT


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    return_value=[MagicMock()],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_all_observation_writer_skips_turn_worker_without_profile_memory(
    mock_config, mock_chat, mock_tool_selector
):
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_mode = False
    cfg.auto_approve = False
    cfg.model_fallbacks = None
    cfg.memory_profile_enabled = False
    cfg.memory_observations_enabled = True
    cfg.memory_observation_writer = MemoryObservationWriter.ALL
    cfg.memory_workers_enabled = True
    mock_config.return_value = cfg
    mock_chat.return_value = MagicMock(profile={"max_input_tokens": 200_000})

    from EvoScientist.EvoScientist import _get_default_middleware

    middleware = _get_default_middleware()
    memory_middleware = next(
        m for m in middleware if type(m).__name__ == "EvoMemoryMiddleware"
    )

    assert [tool.name for tool in memory_middleware.tools] == ["record_observation"]
    assert not any(
        type(m).__name__ == "EvoMemoryLifecycleMiddleware" for m in middleware
    )


def test_configured_system_prompt_matches_live_observation_tool():
    cfg = MagicMock()
    cfg.memory_profile_enabled = True
    cfg.memory_observations_enabled = True
    cfg.memory_observation_writer = MemoryObservationWriter.WORKER
    cfg.memory_workers_enabled = True

    from EvoScientist.EvoScientist import _configured_system_prompt

    prompt = _configured_system_prompt(cfg)

    assert "/memories/observations/" in prompt
    assert "record_observation" not in prompt


# ---------------------------------------------------------------------------
# Direct behavior test for ``_get_default_middleware`` filter
# ---------------------------------------------------------------------------
#
# The factory test above pins the *contract* (factory passes the flag).
# This test pins the *behavior* (the flag actually excludes
# AskUserMiddleware), so a future refactor that renames the flag or
# restructures the middleware list cannot silently re-introduce the
# interrupt-based deadlock.


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    return_value=[MagicMock()],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_async_subagent_mode_filters_ask_user(
    mock_config, mock_chat, mock_tool_selector
):
    """``_get_default_middleware(for_async_subagent=True)`` must drop
    ``AskUserMiddleware`` even when ``enable_ask_user`` is on.

    Without mocking the middleware list itself: we let the real list be
    constructed and assert ``AskUserMiddleware`` is absent. Mocks here
    cover only the heavy dependencies (chat model, tool-selector) that
    the middleware list builder pulls in transitively.
    """
    cfg = MagicMock()
    cfg.enable_ask_user = True  # would normally include AskUserMiddleware
    cfg.auto_mode = False
    cfg.auto_approve = False
    cfg.model_fallbacks = None
    cfg.memory_profile_enabled = True
    cfg.memory_observations_enabled = True
    cfg.memory_observation_writer = MemoryObservationWriter.ALL
    cfg.memory_workers_enabled = True
    mock_config.return_value = cfg
    mock_chat.return_value = MagicMock(profile={"max_input_tokens": 200_000})

    from EvoScientist.EvoScientist import _get_default_middleware
    from EvoScientist.middleware.ask_user import AskUserMiddleware

    # CLI / in-process path includes AskUserMiddleware …
    cli_mw = _get_default_middleware()
    assert any(isinstance(m, AskUserMiddleware) for m in cli_mw), (
        "Sanity check: with enable_ask_user=True and CLI mode, "
        "AskUserMiddleware should be present."
    )

    # … but the async-subagent path filters it out.
    async_mw = _get_default_middleware(for_async_subagent=True)
    assert not any(isinstance(m, AskUserMiddleware) for m in async_mw), (
        "AskUserMiddleware leaked into async sub-agent middleware — its "
        "interrupt() call deadlocks the deployed graph (no UI to resume)."
    )
