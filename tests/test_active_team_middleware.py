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
def test_middleware_injects_concept_when_no_experts_invited(mock_get_config):
    """The ## Experts concept is injected every turn, even with no invites.

    Gating the whole block on invitation would make the expert mechanism
    vanish when nothing is invited — the trap the design avoids.
    """
    mock_get_config.return_value = {"configurable": {}}
    middleware = ActiveTeamMiddleware()
    modified = middleware.modify_request(_request())
    text = _system_text(modified)
    assert "## Experts" in text
    assert "The user has invited" not in text  # no invite block without invitees
    assert "base system" in text


@patch("langgraph.config.get_config")
def test_middleware_injects_concept_when_active_teams_empty_list(mock_get_config):
    mock_get_config.return_value = {"configurable": {"active_teams": []}}
    middleware = ActiveTeamMiddleware()
    modified = middleware.modify_request(_request())
    text = _system_text(modified)
    assert "## Experts" in text
    assert "The user has invited" not in text


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
def test_middleware_appends_invite_for_single_expert(
    mock_get_config, mock_dispatchable
):
    mock_get_config.return_value = {
        "configurable": {"active_teams": ["idea-brainstorm"]},
    }
    mock_dispatchable.return_value = [_mock_expert("idea-brainstorm")]
    middleware = ActiveTeamMiddleware()
    modified = middleware.modify_request(_request())
    text = _system_text(modified)
    assert "## Experts" in text  # concept always present
    assert "The user has invited" in text  # plus the invite block
    assert "`idea-brainstorm`" in text
    assert "base system" in text  # original preserved


@patch("EvoScientist.subagents.expert_container.list_dispatchable_experts")
@patch("langgraph.config.get_config")
def test_middleware_appends_invite_for_multiple_experts(
    mock_get_config, mock_dispatchable
):
    """One <active_expert> tag names one or many invited experts."""
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
    assert "## Experts" in text
    assert "The user has invited" in text
    # One tag for any number of names — no separate plural tag.
    assert "<active_experts>" not in text
    assert "`idea-brainstorm`" in text
    assert "`literature-review`" in text
    assert "base system" in text


@patch("EvoScientist.subagents.expert_container.list_dispatchable_experts")
@patch("langgraph.config.get_config")
def test_middleware_omits_invite_for_undispatchable_names(
    mock_get_config, mock_dispatchable
):
    """Names not in ``list_dispatchable_experts`` are dropped from the invite.

    Covers uninstalled experts, empty actor definitions, and name collisions
    — anything the model would find missing at dispatch time. The concept
    still shows; only the invite block is suppressed.
    """
    mock_get_config.return_value = {
        "configurable": {"active_teams": ["nonexistent-expert"]},
    }
    mock_dispatchable.return_value = []  # nothing dispatchable
    middleware = ActiveTeamMiddleware()
    modified = middleware.modify_request(_request())
    text = _system_text(modified)
    assert "## Experts" in text
    assert "The user has invited" not in text


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
def test_middleware_injects_concept_outside_runnable_context(mock_get_config):
    """Outside a runnable context there are no invites, but the concept still
    injects — ``_read_active_teams`` degrades to an empty list, not a raise."""
    middleware = ActiveTeamMiddleware()
    modified = middleware.modify_request(_request())
    text = _system_text(modified)
    assert "## Experts" in text
    assert "The user has invited" not in text


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
