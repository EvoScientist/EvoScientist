"""Tests for the /mcp command (MCPCommand subcommand dispatch)."""

from unittest.mock import MagicMock, patch

from tests.conftest import run_async as _run


def _ctx():
    from EvoScientist.commands.base import CommandContext

    ui = MagicMock()
    ui.supports_interactive = True
    return CommandContext(agent=None, thread_id="tid", ui=ui), ui


class TestMCPCommandDispatch:
    def test_no_args_lists(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        ctx, ui = _ctx()
        with patch("EvoScientist.mcp.load_mcp_config", return_value={}):
            _run(MCPCommand().execute(ctx, []))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("No MCP servers configured" in m for m in msgs)

    def test_list_subcommand(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        ctx, ui = _ctx()
        cfg = {
            "srv1": {"transport": "stdio", "tools": ["foo"], "expose_to": ["main"]},
        }
        with patch("EvoScientist.mcp.load_mcp_config", return_value=cfg):
            _run(MCPCommand().execute(ctx, ["list"]))
        ui.mount_renderable.assert_called_once()

    def test_add_subcommand_dispatches(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        ctx, _ui = _ctx()
        with (
            patch(
                "EvoScientist.mcp.parse_mcp_add_args",
                return_value={"name": "srv1"},
            ),
            patch(
                "EvoScientist.mcp.add_mcp_server",
                return_value={"transport": "stdio"},
            ) as add_mock,
        ):
            _run(MCPCommand().execute(ctx, ["add", "srv1", "python"]))
        add_mock.assert_called_once()

    def test_edit_subcommand_dispatches(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        ctx, _ui = _ctx()
        with (
            patch(
                "EvoScientist.mcp.parse_mcp_edit_args",
                return_value=("srv1", {"tools": ["bar"]}),
            ),
            patch(
                "EvoScientist.mcp.edit_mcp_server",
            ) as edit_mock,
        ):
            _run(MCPCommand().execute(ctx, ["edit", "srv1", "--tools", "bar"]))
        edit_mock.assert_called_once_with("srv1", tools=["bar"])

    def test_remove_subcommand_success(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        ctx, ui = _ctx()
        with patch("EvoScientist.mcp.remove_mcp_server", return_value=True):
            _run(MCPCommand().execute(ctx, ["remove", "srv1"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Removed MCP server: srv1" in m for m in msgs)

    def test_remove_subcommand_not_found(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        ctx, ui = _ctx()
        with patch("EvoScientist.mcp.remove_mcp_server", return_value=False):
            _run(MCPCommand().execute(ctx, ["remove", "missing"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Server not found" in m for m in msgs)

    def test_install_delegates_to_install_mcp_command(self):
        """/mcp install should instantiate InstallMCPCommand and execute it."""
        from EvoScientist.commands.implementation.mcp import MCPCommand

        ctx, _ui = _ctx()
        with patch(
            "EvoScientist.commands.implementation.mcp_install.InstallMCPCommand"
        ) as klass:
            instance = MagicMock()
            instance.execute = MagicMock(return_value=None)

            async def fake_execute(ctx, args):
                return None

            instance.execute = fake_execute
            klass.return_value = instance
            _run(MCPCommand().execute(ctx, ["install", "foo"]))
        klass.assert_called_once()

    def test_unknown_subcommand_prints_help(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        ctx, ui = _ctx()
        _run(MCPCommand().execute(ctx, ["bogus"]))
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("MCP commands:" in m for m in msgs)
