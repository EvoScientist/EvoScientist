"""Tests for /install-skill and /uninstall-skill commands."""

from unittest.mock import MagicMock, patch


def _ctx(workspace):
    from EvoScientist.commands.base import CommandContext

    ui = MagicMock()
    ui.supports_interactive = True
    return CommandContext(agent=None, thread_id="tid", ui=ui, workspace=workspace), ui


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
