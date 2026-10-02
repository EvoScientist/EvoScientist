"""Unit tests for helpers in ``EvoScientist.commands.base``."""

from __future__ import annotations

from EvoScientist.commands.base import ChannelRuntime, active_teams_configurable_extra


class TestActiveTeamsConfigurableExtra:
    """``active_teams_configurable_extra`` is used at every stream-call site
    that needs to forward /expert invites into ``RunRequest.configurable_extra``.
    """

    def test_none_runtime_returns_none(self):
        assert active_teams_configurable_extra(None) is None

    def test_runtime_without_invites_returns_none(self):
        # Empty list must produce ``None`` so callers can pass the result
        # unconditionally without polluting ``configurable`` with an empty
        # ``active_teams: []`` (which ``ActiveTeamMiddleware`` would treat
        # as no-op anyway, but the wire stays cleaner without it).
        runtime = ChannelRuntime()
        assert active_teams_configurable_extra(runtime) is None

    def test_runtime_with_invites_returns_dict_copy(self):
        runtime = ChannelRuntime()
        runtime.active_teams = ["idea-brainstorm", "paper-review"]
        result = active_teams_configurable_extra(runtime)
        assert result == {"active_teams": ["idea-brainstorm", "paper-review"]}
        # Must be a *copy* — mutating the returned list may not leak back
        # to the runtime's session-scoped invite list.
        result["active_teams"].append("mutated")
        assert runtime.active_teams == ["idea-brainstorm", "paper-review"]


class TestChannelRuntimeClear:
    """``ChannelRuntime.clear`` runs on channel shutdown; it must leave the
    session-scoped ``active_teams`` list intact so stopping a channel does
    not silently dismiss the user's invited experts. ``/new`` and
    ``/expert clear`` handle invite reset explicitly.
    """

    def test_clear_preserves_active_teams(self):
        runtime = ChannelRuntime()
        runtime.agent = object()
        runtime.thread_id = "t-42"
        runtime.active_teams = ["idea-brainstorm"]
        runtime.clear()
        assert runtime.agent is None
        assert runtime.thread_id is None
        assert runtime.active_teams == ["idea-brainstorm"]
