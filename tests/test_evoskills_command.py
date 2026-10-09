"""Tests for the /evoskills command (InstallSkills)."""

from unittest.mock import AsyncMock, MagicMock, patch

from EvoScientist.paths import SessionDirs


def _ctx(workspace, supports_interactive=True):
    from EvoScientist.commands.base import CommandContext

    ui = MagicMock()
    ui.supports_interactive = supports_interactive
    ui.wait_for_skill_browse = AsyncMock()
    return CommandContext(
        agent=None, thread_id="tid", ui=ui, dirs=SessionDirs(workspace)
    ), ui


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


class TestInstallSkills:
    async def test_picker_cancel_no_install(self, workspace):
        from EvoScientist.commands.implementation.skills import InstallSkills

        ctx, ui = _ctx(workspace)
        ui.wait_for_skill_browse.return_value = None  # cancelled
        with (
            patch(
                "EvoScientist.tools.skills_manager.fetch_remote_skill_index",
                return_value=_INDEX,
            ),
            patch(
                "EvoScientist.tools.skills_manager.install_skill",
            ) as install_mock,
        ):
            await InstallSkills().execute(ctx, [])
        install_mock.assert_not_called()
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Browse cancelled" in m for m in msgs)

    async def test_picker_returns_selections_installs_each(self, workspace):
        from EvoScientist.commands.implementation.skills import InstallSkills

        ctx, ui = _ctx(workspace)
        ui.wait_for_skill_browse.return_value = [
            "repo@paper-writing",
            "repo@research-ideation",
        ]
        with (
            patch(
                "EvoScientist.tools.skills_manager.fetch_remote_skill_index",
                return_value=_INDEX,
            ),
            patch(
                "EvoScientist.tools.skills_manager.install_skill",
                return_value={"success": True, "name": "x"},
            ) as install_mock,
        ):
            await InstallSkills().execute(ctx, [])
        assert install_mock.call_count == 2
        assert all(
            c.kwargs["workspace"] is workspace for c in install_mock.call_args_list
        )

    async def test_channel_auto_install_on_tag(self, workspace):
        """Non-interactive UI + tag arg → auto-installs matching skills."""
        from EvoScientist.commands.implementation.skills import InstallSkills

        ctx, ui = _ctx(workspace, supports_interactive=False)
        with (
            patch(
                "EvoScientist.tools.skills_manager.fetch_remote_skill_index",
                return_value=_INDEX,
            ),
            patch(
                "EvoScientist.tools.skills_manager.install_skill",
                return_value={"success": True, "name": "x"},
            ) as install_mock,
        ):
            await InstallSkills().execute(ctx, ["core"])
        # "core" matches research-ideation only → 1 install, no picker call
        assert install_mock.call_count == 1
        ui.wait_for_skill_browse.assert_not_called()

    async def test_fetch_failure_prints_error(self, workspace):
        from EvoScientist.commands.implementation.skills import InstallSkills

        ctx, ui = _ctx(workspace)
        with patch(
            "EvoScientist.tools.skills_manager.fetch_remote_skill_index",
            side_effect=RuntimeError("network fail"),
        ):
            await InstallSkills().execute(ctx, [])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Failed to fetch" in m for m in msgs)
        assert any("Try: /install-skill" in m for m in msgs)

    async def test_missing_git_prints_the_git_message(self, workspace, no_git):
        from EvoScientist.commands.implementation.skills import InstallSkills
        from EvoScientist.tools.skills_manager import _REMOTE_INDEX_CACHE

        _REMOTE_INDEX_CACHE.clear()
        ctx, ui = _ctx(workspace)
        await InstallSkills().execute(ctx, [])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any(
            m.startswith("Failed to fetch skill index: git was not found on PATH.")
            for m in msgs
        )
        # The /install-skill hint needs git too, so it is left out.
        assert not any("Try: /install-skill" in m for m in msgs)

    async def test_install_loop_reports_missing_git(self, tmp_path, workspace, no_git):
        """A git that disappears after the index fetch fails each install cleanly."""
        from EvoScientist.commands.implementation.skills import InstallSkills

        ctx, ui = _ctx(workspace)
        ui.wait_for_skill_browse.return_value = ["owner/repo@paper-writing"]
        with (
            patch(
                "EvoScientist.tools.skills_manager.fetch_remote_skill_index",
                return_value=_INDEX,
            ),
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", tmp_path / "skills"),
        ):
            await InstallSkills().execute(ctx, [])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any(m.startswith("Failed: git was not found on PATH.") for m in msgs)
