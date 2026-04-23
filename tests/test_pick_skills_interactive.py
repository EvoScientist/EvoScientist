"""Tests for _pick_skills_interactive (Phase C extraction).

Verifies empty-result vs cancel semantics so InstallSkills.execute
can distinguish "picker handled the no-op case" from "user cancelled".
"""

from unittest.mock import MagicMock, patch

_INDEX = [
    {
        "name": "paper-writing",
        "description": "author papers",
        "install_source": "repo@paper-writing",
        "tags": ["writing"],
    },
    {
        "name": "research-ideation",
        "description": "brainstorm ideas",
        "install_source": "repo@research-ideation",
        "tags": ["core"],
    },
]


class TestPickSkillsInteractive:
    def test_no_tag_match_returns_empty_list(self):
        """Pre-filter with no matches → [] (not None). Caller should
        suppress its own "cancelled" message since the picker already
        printed a specific one."""
        from EvoScientist.cli.skills_cmd import _pick_skills_interactive

        result = _pick_skills_interactive(_INDEX, set(), "nonexistent-tag")
        assert result == []

    def test_all_installed_returns_empty_list(self):
        """If every skill matching the tag is already installed → []."""
        from EvoScientist.cli.skills_cmd import _pick_skills_interactive

        installed = {"paper-writing", "research-ideation"}
        # Pre-filter skips tag picker; directly hits all-installed guard
        result = _pick_skills_interactive(_INDEX, installed, "writing")
        assert result == []

    def test_tag_picker_cancel_returns_none(self, monkeypatch):
        """User cancels tag picker (Esc) → None."""
        from EvoScientist.cli import skills_cmd

        select_prompt = MagicMock()
        select_prompt.ask.return_value = None
        monkeypatch.setattr("questionary.select", lambda *a, **k: select_prompt)

        result = skills_cmd._pick_skills_interactive(_INDEX, set(), "")
        assert result is None

    def test_checkbox_cancel_returns_none(self, monkeypatch):
        """User cancels checkbox (Esc) → None."""
        from EvoScientist.cli import skills_cmd

        # Skip tag picker by pre-filtering
        checkbox_prompt = MagicMock()
        checkbox_prompt.ask.return_value = None
        monkeypatch.setattr("questionary.checkbox", lambda *a, **k: checkbox_prompt)

        result = skills_cmd._pick_skills_interactive(_INDEX, set(), "writing")
        assert result is None

    def test_checkbox_confirmed_with_selection(self, monkeypatch):
        """User confirms with selections → list of install sources."""
        from EvoScientist.cli import skills_cmd

        checkbox_prompt = MagicMock()
        checkbox_prompt.ask.return_value = ["repo@paper-writing"]
        monkeypatch.setattr("questionary.checkbox", lambda *a, **k: checkbox_prompt)

        result = skills_cmd._pick_skills_interactive(_INDEX, set(), "writing")
        assert result == ["repo@paper-writing"]


class TestInstallSkillsHandlesEmpty:
    """InstallSkills.execute must distinguish None vs [] from the picker."""

    def test_empty_list_suppresses_cancel_message(self):
        """When picker returns [], user should NOT see "Browse cancelled"
        (the picker already printed its own message)."""
        from unittest.mock import AsyncMock

        from EvoScientist.commands.base import CommandContext
        from EvoScientist.commands.implementation.skills import InstallSkills
        from tests.conftest import run_async as _run

        ui = MagicMock()
        ui.supports_interactive = True
        ui.wait_for_skill_browse = AsyncMock(return_value=[])
        ctx = CommandContext(agent=None, thread_id="tid", ui=ui)

        with patch(
            "EvoScientist.tools.skills_manager.fetch_remote_skill_index",
            return_value=_INDEX,
        ):
            _run(InstallSkills().execute(ctx, []))

        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert not any("Browse cancelled" in m for m in msgs)

    def test_none_shows_cancel_message(self):
        """When picker returns None (actual cancel), user sees the message."""
        from unittest.mock import AsyncMock

        from EvoScientist.commands.base import CommandContext
        from EvoScientist.commands.implementation.skills import InstallSkills
        from tests.conftest import run_async as _run

        ui = MagicMock()
        ui.supports_interactive = True
        ui.wait_for_skill_browse = AsyncMock(return_value=None)
        ctx = CommandContext(agent=None, thread_id="tid", ui=ui)

        with patch(
            "EvoScientist.tools.skills_manager.fetch_remote_skill_index",
            return_value=_INDEX,
        ):
            _run(InstallSkills().execute(ctx, []))

        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Browse cancelled" in m for m in msgs)
