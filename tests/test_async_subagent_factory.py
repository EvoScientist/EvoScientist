"""Tests for ``EvoScientist.subagents._factory.build_async_subagent_graph``.

Pins the integration contract that the factory must request middleware
in async-safe mode (``for_async_subagent=True``). Without this, a future
refactor that drops the keyword argument would silently re-introduce
``AskUserMiddleware`` into the deployed graph and reproduce the
``interrupt()``-based deadlock the flag was added to prevent.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


@patch("deepagents.create_deep_agent")
@patch("EvoScientist.EvoScientist._load_mcp_tools_cached", return_value={})
@patch("EvoScientist.EvoScientist._get_default_middleware", return_value=[])
@patch("EvoScientist.EvoScientist._get_default_backend")
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._build_prompt_refs", return_value={})
@patch("EvoScientist.utils.load_subagents")
@patch("EvoScientist.config.apply_config_to_env")
@patch("EvoScientist.config.get_effective_config")
def test_factory_requests_async_safe_middleware(
    mock_get_cfg,
    mock_apply_env,
    mock_load_subs,
    mock_prompt_refs,
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

    # The contract: factory MUST pass ``for_async_subagent=True``.
    mock_get_mw.assert_called_once_with(for_async_subagent=True)
