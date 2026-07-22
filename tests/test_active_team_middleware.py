"""Tests for EvoScientist.middleware.active_team."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import SystemMessage

from EvoScientist.middleware.active_team import (
    ActiveTeamMiddleware,
    _read_active_teams,
    create_active_team_middleware,
)


def _request():
    """A minimal ModelRequest stand-in supporting the fields the middleware
    reads (`system_message`) and the `.override(**kwargs)` mutator."""
    request = SimpleNamespace(
        state={},
        runtime=object(),
        system_message=SystemMessage(content="base system"),
    )
    request.override = lambda **kwargs: SimpleNamespace(
        **{
            "state": request.state,
            "runtime": request.runtime,
            "system_message": kwargs.get("system_message", request.system_message),
        }
    )
    return request


def _system_text(modified) -> str:
    system_message = modified.system_message
    assert system_message is not None
    return str(system_message.content)


def _mock_config():
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_mode = False
    cfg.auto_approve = False
    cfg.model_fallbacks = None
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    cfg.code_interpreter_timeout = 60
    cfg.code_interpreter_max_result_chars = 6000
    return cfg


# ---- unit tests: _read_active_teams behavior --------------------------------


@patch("langgraph.config.get_config")
def test_read_active_teams_returns_list_when_present(mock_get_config):
    mock_get_config.return_value = {
        "configurable": {"active_teams": ["idea-brainstorm"]},
    }
    assert _read_active_teams() == ["idea-brainstorm"]


@patch("langgraph.config.get_config")
def test_read_active_teams_returns_empty_when_configurable_missing(mock_get_config):
    mock_get_config.return_value = {}
    assert _read_active_teams() == []


@patch("langgraph.config.get_config")
def test_read_active_teams_returns_empty_when_active_teams_missing(mock_get_config):
    mock_get_config.return_value = {"configurable": {"other_field": "x"}}
    assert _read_active_teams() == []


@patch("langgraph.config.get_config")
def test_read_active_teams_returns_empty_when_value_not_list(mock_get_config):
    """WebUI mistakenly sends a scalar instead of a list; must not crash."""
    mock_get_config.return_value = {
        "configurable": {"active_teams": "idea-brainstorm"},
    }
    assert _read_active_teams() == []


@patch("langgraph.config.get_config")
def test_read_active_teams_filters_non_string_entries(mock_get_config):
    mock_get_config.return_value = {
        "configurable": {
            "active_teams": ["idea-brainstorm", None, 42, "", "lit-review"]
        },
    }
    assert _read_active_teams() == ["idea-brainstorm", "lit-review"]


@patch("langgraph.config.get_config", side_effect=RuntimeError("outside context"))
def test_read_active_teams_returns_empty_outside_runnable_context(mock_get_config):
    assert _read_active_teams() == []


# ---- unit tests: middleware behavior ---------------------------------------


@patch("langgraph.config.get_config")
def test_middleware_no_op_when_active_teams_absent(mock_get_config):
    mock_get_config.return_value = {"configurable": {}}
    middleware = ActiveTeamMiddleware()
    request = _request()
    modified = middleware.modify_request(request)
    # No override applied: original request returned as-is.
    assert modified is request


@patch("langgraph.config.get_config")
def test_middleware_no_op_when_active_teams_empty_list(mock_get_config):
    mock_get_config.return_value = {"configurable": {"active_teams": []}}
    middleware = ActiveTeamMiddleware()
    request = _request()
    modified = middleware.modify_request(request)
    assert modified is request


@patch("langgraph.config.get_config")
def test_middleware_appends_single_expert_cue(mock_get_config):
    mock_get_config.return_value = {
        "configurable": {"active_teams": ["idea-brainstorm"]},
    }
    middleware = ActiveTeamMiddleware()
    modified = middleware.modify_request(_request())
    text = _system_text(modified)
    assert "<active_expert>" in text
    assert "`idea-brainstorm`" in text
    assert "Consult it via `task(" in text
    assert "base system" in text  # original preserved


@patch("langgraph.config.get_config")
def test_middleware_appends_multi_expert_cue(mock_get_config):
    mock_get_config.return_value = {
        "configurable": {"active_teams": ["idea-brainstorm", "literature-review"]},
    }
    middleware = ActiveTeamMiddleware()
    modified = middleware.modify_request(_request())
    text = _system_text(modified)
    assert "<active_experts>" in text
    assert "`idea-brainstorm`" in text
    assert "`literature-review`" in text
    # Multi-cue: header names both experts, then per-expert dispatch lines follow.
    assert "The user has invited the following experts" in text
    assert "Per-expert dispatch" in text
    assert "base system" in text


@patch("langgraph.config.get_config")
def test_middleware_appends_cue_for_unknown_expert_names(mock_get_config):
    """Middleware doesn't validate names against the registry; main decides."""
    mock_get_config.return_value = {
        "configurable": {"active_teams": ["nonexistent-expert"]},
    }
    middleware = ActiveTeamMiddleware()
    modified = middleware.modify_request(_request())
    text = _system_text(modified)
    assert "`nonexistent-expert`" in text
    # Unknown name defaults to sync cue.
    assert "task(" in text


@patch("EvoScientist.tools.skills_manager.list_expert_skills")
@patch("langgraph.config.get_config")
def test_middleware_uses_start_async_task_cue_for_async_dispatch(
    mock_get_config, mock_list_experts
):
    """An expert declared ``default_dispatch: async`` gets the async cue."""
    from unittest.mock import MagicMock

    mock_get_config.return_value = {
        "configurable": {"active_teams": ["literature-review"]},
    }
    info = MagicMock(default_dispatch="async")
    # ``name`` on MagicMock has to be set via attribute assignment; passing
    # ``name=`` to the constructor names the mock instance itself.
    info.name = "literature-review"
    mock_list_experts.return_value = [info]

    middleware = ActiveTeamMiddleware()
    modified = middleware.modify_request(_request())
    text = _system_text(modified)
    assert "<active_expert>" in text
    assert "start_async_task(" in text
    assert "subagent_type: 'literature-review'" in text
    assert "payload" in text
    assert "skill_name: 'literature-review'" in text
    assert "output_path" in text
    assert "check_async_task" in text
    # Sync cue must NOT be advertised for async experts.
    assert "Consult it via `task(" not in text


@patch("EvoScientist.tools.skills_manager.list_expert_skills")
@patch("langgraph.config.get_config")
def test_middleware_uses_task_cue_for_sync_dispatch(mock_get_config, mock_list_experts):
    """Sync-dispatched experts get the ``task()`` cue."""
    from unittest.mock import MagicMock

    mock_get_config.return_value = {
        "configurable": {"active_teams": ["idea-brainstorm"]},
    }
    info = MagicMock(default_dispatch="sync")
    info.name = "idea-brainstorm"
    mock_list_experts.return_value = [info]

    middleware = ActiveTeamMiddleware()
    modified = middleware.modify_request(_request())
    text = _system_text(modified)
    assert "Consult it via `task(" in text
    assert "runs synchronously" in text
    # No async-specific fragments for a sync expert.
    assert "start_async_task(" not in text
    assert "output_path" not in text


@patch("EvoScientist.tools.skills_manager.list_expert_skills")
@patch("langgraph.config.get_config")
def test_middleware_multi_mixed_dispatch(mock_get_config, mock_list_experts):
    """When both sync and async experts are active, each gets its own cue."""
    from unittest.mock import MagicMock

    mock_get_config.return_value = {
        "configurable": {"active_teams": ["idea-brainstorm", "literature-review"]},
    }
    sync = MagicMock(default_dispatch="sync")
    sync.name = "idea-brainstorm"
    async_ = MagicMock(default_dispatch="async")
    async_.name = "literature-review"
    mock_list_experts.return_value = [sync, async_]

    middleware = ActiveTeamMiddleware()
    modified = middleware.modify_request(_request())
    text = _system_text(modified)
    # Both cue shapes appear once each in the per-expert block.
    assert text.count("`task(") == 1
    assert text.count("start_async_task(") == 1
    assert "`idea-brainstorm`:" in text
    assert "`literature-review`:" in text


@patch("langgraph.config.get_config", side_effect=RuntimeError("outside context"))
def test_middleware_no_op_outside_runnable_context(mock_get_config):
    middleware = ActiveTeamMiddleware()
    request = _request()
    modified = middleware.modify_request(request)
    assert modified is request


# ---- composition tests: _get_default_middleware ----------------------------


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    return_value=[MagicMock(), MagicMock()],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_default_middleware_includes_active_team_for_main_agent(
    mock_config, mock_model, mock_tool_selector
):
    mock_config.return_value = _mock_config()
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})

    from EvoScientist.EvoScientist import _get_default_middleware

    middleware = _get_default_middleware()

    assert any(isinstance(m, ActiveTeamMiddleware) for m in middleware)


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    return_value=[MagicMock(), MagicMock()],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_default_middleware_excludes_active_team_for_async_subagent(
    mock_config, mock_model, mock_tool_selector
):
    mock_config.return_value = _mock_config()
    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})

    from EvoScientist.EvoScientist import _get_default_middleware

    middleware = _get_default_middleware(for_async_subagent=True)

    assert not any(isinstance(m, ActiveTeamMiddleware) for m in middleware)


# ---- factory --------------------------------------------------------------


def test_factory_returns_middleware_instance():
    assert isinstance(create_active_team_middleware(), ActiveTeamMiddleware)
