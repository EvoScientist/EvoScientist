"""Tests for the /mcp command (MCPCommand subcommand dispatch)."""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


def _ctx():
    from EvoScientist.commands.base import CommandContext

    ui = MagicMock()
    ui.supports_interactive = True
    return CommandContext(agent=None, thread_id="tid", ui=ui), ui


class TestMCPCommandDispatch:
    async def test_no_args_lists(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        ctx, ui = _ctx()
        with patch("EvoScientist.mcp.load_mcp_config", return_value={}):
            await MCPCommand().execute(ctx, [])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("No MCP servers configured" in m for m in msgs)

    async def test_list_subcommand(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        ctx, ui = _ctx()
        cfg = {
            "srv1": {"transport": "stdio", "tools": ["foo"], "expose_to": ["main"]},
        }
        with patch("EvoScientist.mcp.load_mcp_config", return_value=cfg):
            await MCPCommand().execute(ctx, ["list"])
        ui.mount_renderable.assert_called_once()

    async def test_add_subcommand_dispatches(self):
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
            await MCPCommand().execute(ctx, ["add", "srv1", "python"])
        add_mock.assert_called_once()

    async def test_edit_subcommand_dispatches(self):
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
            await MCPCommand().execute(ctx, ["edit", "srv1", "--tools", "bar"])
        edit_mock.assert_called_once_with("srv1", tools=["bar"])

    async def test_remove_subcommand_success(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        ctx, ui = _ctx()
        with patch("EvoScientist.mcp.remove_mcp_server", return_value=True):
            await MCPCommand().execute(ctx, ["remove", "srv1"])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Removed MCP server: srv1" in m for m in msgs)

    async def test_remove_subcommand_not_found(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        ctx, ui = _ctx()
        with patch("EvoScientist.mcp.remove_mcp_server", return_value=False):
            await MCPCommand().execute(ctx, ["remove", "missing"])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Server not found" in m for m in msgs)

    async def test_install_delegates_to_install_mcp_command(self):
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
            await MCPCommand().execute(ctx, ["install", "foo"])
        klass.assert_called_once()

    async def test_unknown_subcommand_prints_help(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        ctx, ui = _ctx()
        await MCPCommand().execute(ctx, ["bogus"])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("MCP commands:" in m for m in msgs)


async def test_install_mcp_runs_package_install_off_event_loop_thread():
    from EvoScientist.commands.base import CommandContext
    from EvoScientist.commands.implementation.mcp_install import InstallMCPCommand

    ui = MagicMock()
    ui.flush = AsyncMock()
    entry = SimpleNamespace(name="slow-server")
    install_threads: list[str] = []

    def _install(_entry, *, print_fn, cancel_event=None, commit_gate=None):
        install_threads.append(threading.current_thread().name)
        print_fn("Installing dependency...", "dim")
        if commit_gate is not None and not commit_gate(_entry.name, lambda: True):
            return False
        return True

    ctx = CommandContext(agent=None, thread_id="tid", ui=ui)
    event_loop_thread = threading.current_thread().name

    with (
        patch(
            "EvoScientist.mcp.registry.fetch_marketplace_index",
            return_value=[entry],
        ),
        patch(
            "EvoScientist.mcp.registry.find_server_by_name",
            return_value=entry,
        ),
        patch("EvoScientist.mcp.registry.get_installed_names", return_value=set()),
        patch("EvoScientist.mcp.registry.install_mcp_server", side_effect=_install),
    ):
        await InstallMCPCommand().execute(ctx, ["slow-server"])

    assert len(install_threads) == 1
    assert install_threads[0] != event_loop_thread
    assert any(
        call.args[0] == "Installing dependency..."
        for call in ui.append_system.call_args_list
    )
