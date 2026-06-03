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
