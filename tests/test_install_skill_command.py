"""Tests for /install-skill and /uninstall-skill commands."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from EvoScientist.paths import SessionDirs


def _ctx(workspace):
    from EvoScientist.commands.base import CommandContext

    ui = MagicMock()
    ui.supports_interactive = True
    return CommandContext(
        agent=None, thread_id="tid", ui=ui, dirs=SessionDirs(workspace)
    ), ui


class TestInstallSkill:
    async def test_usage_message_when_no_args(self, workspace):
        from EvoScientist.commands.implementation.skills import InstallSkill

        ctx, ui = _ctx(workspace)
        await InstallSkill().execute(ctx, [])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Usage:" in m for m in msgs)

    async def test_happy_path(self, workspace):
        from EvoScientist.commands.implementation.skills import InstallSkill

        ctx, ui = _ctx(workspace)
        with patch(
            "EvoScientist.tools.skills_manager.install_skill",
            return_value={
                "success": True,
                "name": "demo-skill",
                "description": "demo",
                "path": "/tmp/demo",
            },
        ) as install_mock:
            await InstallSkill().execute(ctx, ["./some-path"])
        assert install_mock.call_args.kwargs["workspace"] is workspace
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Installed: demo-skill" in m for m in msgs)

    async def test_missing_git_prints_the_git_message(
        self, tmp_path, workspace, no_git
    ):
        from EvoScientist.commands.implementation.skills import InstallSkill

        ctx, ui = _ctx(workspace)
        with patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", tmp_path / "skills"):
            await InstallSkill().execute(ctx, ["owner/repo@skill"])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any(m.startswith("Failed: git was not found on PATH.") for m in msgs)


class TestUninstallSkill:
    async def test_usage_message_when_no_args(self, workspace):
        from EvoScientist.commands.implementation.skills import UninstallSkill

        ctx, ui = _ctx(workspace)
        await UninstallSkill().execute(ctx, [])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Usage:" in m for m in msgs)

    async def test_uninstall_success(self, workspace):
        from EvoScientist.commands.implementation.skills import UninstallSkill

        ctx, ui = _ctx(workspace)
        with patch(
            "EvoScientist.tools.skills_manager.uninstall_skill",
            return_value={"success": True},
        ) as uninstall_mock:
            await UninstallSkill().execute(ctx, ["demo-skill"])
        uninstall_mock.assert_called_once_with("demo-skill", workspace=workspace)
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Uninstalled: demo-skill" in m for m in msgs)

    async def test_uninstall_failure(self, workspace):
        from EvoScientist.commands.implementation.skills import UninstallSkill

        ctx, ui = _ctx(workspace)
        with patch(
            "EvoScientist.tools.skills_manager.uninstall_skill",
            return_value={"success": False, "error": "not found"},
        ):
            await UninstallSkill().execute(ctx, ["missing"])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Failed: not found" in m for m in msgs)


class TestNoReloadHint:
    async def test_install_prints_no_reload_hint(self, workspace):
        from EvoScientist.commands.implementation.skills import InstallSkill

        ctx, ui = _ctx(workspace)
        with patch(
            "EvoScientist.tools.skills_manager.install_skill",
            return_value={
                "success": True,
                "name": "demo-skill",
                "description": "demo",
                "path": "/tmp/demo",
            },
        ):
            await InstallSkill().execute(ctx, ["./some-path"])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Installed: demo-skill" in m for m in msgs)
        assert not any("/new" in m for m in msgs)

    async def test_uninstall_prints_no_reload_hint(self, workspace):
        from EvoScientist.commands.implementation.skills import UninstallSkill

        ctx, ui = _ctx(workspace)
        with patch(
            "EvoScientist.tools.skills_manager.uninstall_skill",
            return_value={"success": True},
        ):
            await UninstallSkill().execute(ctx, ["demo-skill"])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert msgs == ["Uninstalled: demo-skill"]


def _make_skill(parent, name, *, expert=False):
    d = parent / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: demo\n---\n\n# {name}\n"
    )
    if expert:
        (d / "EXPERT.md").write_text("You are a demo expert.\n")
    return d


class TestExpertDispatchHint:
    @pytest.fixture(autouse=True)
    def _global_tier(self, tmp_path):
        with patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", tmp_path / "global"):
            yield

    async def test_single_expert_install_prints_hint_once(self, tmp_path, workspace):
        from EvoScientist.commands.implementation.experts import (
            NEW_EXPERT_DISPATCH_HINT,
        )
        from EvoScientist.commands.implementation.skills import InstallSkill

        src = _make_skill(tmp_path / "src", "idea-expert", expert=True)
        ctx, ui = _ctx(workspace)
        await InstallSkill().execute(ctx, [str(src)])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert msgs.count(NEW_EXPERT_DISPATCH_HINT) == 1

    async def test_plain_install_shadowed_by_workspace_expert_prints_no_hint(
        self, tmp_path, workspace
    ):
        from EvoScientist.commands.implementation.skills import InstallSkill

        _make_skill(workspace.skills_dir, "foo", expert=True)
        src = _make_skill(tmp_path / "src", "foo")
        ctx, ui = _ctx(workspace)
        await InstallSkill().execute(ctx, [str(src)])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Installed: foo" in m for m in msgs)
        assert not any("/new" in m for m in msgs)

    async def test_batch_install_with_one_expert_prints_hint_once(
        self, tmp_path, workspace
    ):
        from EvoScientist.commands.implementation.experts import (
            NEW_EXPERT_DISPATCH_HINT,
        )
        from EvoScientist.commands.implementation.skills import InstallSkill

        pack = tmp_path / "pack"
        _make_skill(pack, "plain-skill")
        _make_skill(pack, "idea-expert", expert=True)
        ctx, ui = _ctx(workspace)
        await InstallSkill().execute(ctx, [str(pack)])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert msgs.count(NEW_EXPERT_DISPATCH_HINT) == 1

    async def test_evoskills_install_of_expert_prints_hint_once(
        self, tmp_path, workspace
    ):
        from EvoScientist.commands.implementation.experts import (
            NEW_EXPERT_DISPATCH_HINT,
        )
        from EvoScientist.commands.implementation.skills import InstallSkills

        installed = _make_skill(tmp_path / "global", "idea-expert", expert=True)
        ctx, ui = _ctx(workspace)
        ui.wait_for_skill_browse = AsyncMock(return_value=["repo@idea-expert"])
        index = [
            {
                "name": "idea-expert",
                "description": "d",
                "install_source": "repo@idea-expert",
                "tags": [],
            }
        ]
        with (
            patch(
                "EvoScientist.tools.skills_manager.fetch_remote_skill_index",
                return_value=index,
            ),
            patch(
                "EvoScientist.tools.skills_manager.install_skill",
                return_value={
                    "success": True,
                    "name": "idea-expert",
                    "path": str(installed),
                },
            ),
        ):
            await InstallSkills().execute(ctx, [])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert msgs.count(NEW_EXPERT_DISPATCH_HINT) == 1

    async def test_evoskills_result_without_path_prints_no_hint(self, workspace):
        from EvoScientist.commands.implementation.skills import InstallSkills

        ctx, ui = _ctx(workspace)
        ui.wait_for_skill_browse = AsyncMock(return_value=["repo@x"])
        index = [
            {"name": "x", "description": "d", "install_source": "repo@x", "tags": []}
        ]
        with (
            patch(
                "EvoScientist.tools.skills_manager.fetch_remote_skill_index",
                return_value=index,
            ),
            patch(
                "EvoScientist.tools.skills_manager.install_skill",
                return_value={"success": True, "name": "x"},
            ),
        ):
            await InstallSkills().execute(ctx, [])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert "Installed: x" in msgs
        assert not any("/new" in m for m in msgs)

    async def test_expert_uninstall_prints_removed_hint(self, workspace):
        from EvoScientist.commands.implementation.experts import (
            REMOVED_EXPERT_DISPATCH_HINT,
        )
        from EvoScientist.commands.implementation.skills import UninstallSkill

        target = _make_skill(workspace.skills_dir, "foo", expert=True)
        ctx, ui = _ctx(workspace)
        await UninstallSkill().execute(ctx, ["foo"])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert msgs == ["Uninstalled: foo", REMOVED_EXPERT_DISPATCH_HINT]
        assert not target.exists()

    async def test_plain_uninstall_prints_only_confirmation(self, workspace):
        from EvoScientist.commands.implementation.skills import UninstallSkill

        _make_skill(workspace.skills_dir, "foo")
        ctx, ui = _ctx(workspace)
        await UninstallSkill().execute(ctx, ["foo"])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert msgs == ["Uninstalled: foo"]

    async def test_uninstall_succeeds_when_skill_listing_fails(self, workspace):
        from EvoScientist.commands.implementation.skills import UninstallSkill

        target = _make_skill(workspace.skills_dir, "foo")
        ctx, ui = _ctx(workspace)
        with patch(
            "EvoScientist.tools.skills_manager.list_skills",
            side_effect=OSError("global tier unreadable"),
        ):
            await UninstallSkill().execute(ctx, ["foo"])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert msgs == ["Uninstalled: foo"]
        assert not target.exists()
