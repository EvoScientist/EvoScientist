"""Tests for /install-skill and /uninstall-skill commands."""

from unittest.mock import MagicMock, patch

from tests.conftest import run_async as _run


def _ctx():
    from EvoScientist.commands.base import CommandContext

    ui = MagicMock()
    ui.supports_interactive = True
    return CommandContext(agent=None, thread_id="tid", ui=ui), ui


class TestInstallSkill:
    def test_usage_message_when_no_args(self):
        from EvoScientist.commands.implementation.skills import InstallSkill

        ctx, ui = _ctx()
        _run(InstallSkill().execute(ctx, []))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Usage:" in m for m in msgs)

    def test_happy_path(self):
        from EvoScientist.commands.implementation.skills import InstallSkill

        ctx, ui = _ctx()
        with patch(
            "EvoScientist.tools.skills_manager.install_skill",
            return_value={
                "success": True,
                "name": "demo-skill",
                "description": "demo",
                "path": "/tmp/demo",
            },
        ):
            _run(InstallSkill().execute(ctx, ["./some-path"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Installed: demo-skill" in m for m in msgs)


class TestUninstallSkill:
    def test_usage_message_when_no_args(self):
        from EvoScientist.commands.implementation.skills import UninstallSkill

        ctx, ui = _ctx()
        _run(UninstallSkill().execute(ctx, []))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Usage:" in m for m in msgs)

    def test_uninstall_success(self):
        from EvoScientist.commands.implementation.skills import UninstallSkill

        ctx, ui = _ctx()
        with patch(
            "EvoScientist.tools.skills_manager.uninstall_skill",
            return_value={"success": True},
        ):
            _run(UninstallSkill().execute(ctx, ["demo-skill"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Uninstalled: demo-skill" in m for m in msgs)

    def test_uninstall_failure(self):
        from EvoScientist.commands.implementation.skills import UninstallSkill

        ctx, ui = _ctx()
        with patch(
            "EvoScientist.tools.skills_manager.uninstall_skill",
            return_value={"success": False, "error": "not found"},
        ):
            _run(UninstallSkill().execute(ctx, ["missing"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Failed: not found" in m for m in msgs)
