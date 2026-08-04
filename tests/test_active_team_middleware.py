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


def _mock_expert(name: str) -> MagicMock:
    """Build a MagicMock ``SkillInfo`` for a dispatchable expert.

    ``name`` on ``MagicMock`` must be set via attribute assignment; passing
    ``name=`` to the constructor names the mock instance itself.
    """
    info = MagicMock()
    info.name = name
    return info


@patch("EvoScientist.subagents.expert_container.list_dispatchable_experts")
@patch("langgraph.config.get_config")
def test_middleware_appends_single_expert_cue(mock_get_config, mock_dispatchable):
    mock_get_config.return_value = {
        "configurable": {"active_teams": ["idea-brainstorm"]},
    }
    mock_dispatchable.return_value = [_mock_expert("idea-brainstorm")]
    middleware = ActiveTeamMiddleware()
    modified = middleware.modify_request(_request())
    text = _system_text(modified)
    assert "<active_expert>" in text
    assert "`idea-brainstorm`" in text
    # The cue names the expert and stops; choosing a tool is the
    # orchestrator's job, and the tool shapes live in DELEGATION_STRATEGY.
    assert "task(" not in text
    assert "start_async_task" not in text
    assert "base system" in text  # original preserved


@patch("EvoScientist.subagents.expert_container.list_dispatchable_experts")
@patch("langgraph.config.get_config")
def test_middleware_appends_multi_expert_cue(mock_get_config, mock_dispatchable):
    mock_get_config.return_value = {
        "configurable": {"active_teams": ["idea-brainstorm", "literature-review"]},
    }
    mock_dispatchable.return_value = [
        _mock_expert("idea-brainstorm"),
        _mock_expert("literature-review"),
    ]
    middleware = ActiveTeamMiddleware()
    modified = middleware.modify_request(_request())
    text = _system_text(modified)
    assert "<active_experts>" in text
    assert "`idea-brainstorm`" in text
    assert "`literature-review`" in text
    assert "The user has invited the following experts" in text
    assert "task(" not in text
    assert "start_async_task" not in text
    assert "base system" in text


@patch("EvoScientist.subagents.expert_container.list_dispatchable_experts")
@patch("langgraph.config.get_config")
def test_middleware_omits_cue_for_undispatchable_names(
    mock_get_config, mock_dispatchable
):
    """Names not in ``list_dispatchable_experts`` are dropped from the cue.

    Covers uninstalled experts, empty actor definitions, and name
    collisions — anything the model would find missing at dispatch time.
    """
    mock_get_config.return_value = {
        "configurable": {"active_teams": ["nonexistent-expert"]},
    }
    mock_dispatchable.return_value = []  # nothing dispatchable
    request = _request()
    middleware = ActiveTeamMiddleware()
    modified = middleware.modify_request(request)
    # No cue appended — modify_request returns the original request untouched.
    assert modified is request


@patch("EvoScientist.subagents.expert_container.list_dispatchable_experts")
@patch("langgraph.config.get_config")
def test_middleware_drops_invited_expert_that_is_not_dispatchable(
    mock_get_config, mock_dispatchable
):
    """An invited expert that stops being dispatchable — uninstalled, or its
    actor definition emptied — must drop out of the cue. Naming an expert
    the model cannot reach is worse than saying nothing."""
    mock_get_config.return_value = {
        "configurable": {
            "active_teams": ["idea-brainstorm", "literature-review"],
        },
    }
    # literature-review invited but not dispatchable this turn.
    mock_dispatchable.return_value = [_mock_expert("idea-brainstorm")]

    middleware = ActiveTeamMiddleware()
    modified = middleware.modify_request(_request())
    text = _system_text(modified)
    # Single-cue shape (only one expert survived the filter).
    assert "<active_expert>" in text
    assert "`idea-brainstorm`" in text
    assert "literature-review" not in text


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
