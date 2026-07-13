"""Tests for CommandManager subcommand metadata."""

from EvoScientist.commands.base import SubCommand
from EvoScientist.commands.manager import CommandManager


class TestSubCommand:
    def test_creation_with_defaults(self):
        sc = SubCommand("list", "List servers")
        assert sc.name == "list"
        assert sc.description == "List servers"
        assert sc.arguments == []

    def test_creation_with_arguments(self):
        from EvoScientist.commands.base import Argument

        sc = SubCommand(
            "add",
            "Add a server",
            arguments=[Argument("name", str, "Server name", required=True)],
        )
        assert len(sc.arguments) == 1
        assert sc.arguments[0].name == "name"


class TestCommandManagerSubcommands:
    def test_mcp_has_six_subcommands(self):
        """``/mcp`` must expose all 6 subcommands via the manager."""
        manager = CommandManager()
        from EvoScientist.commands.implementation.mcp import MCPCommand

        manager.register(MCPCommand())
        scs = manager.list_subcommands("/mcp")
        names = {name for name, _desc in scs}
        assert names == {"list", "config", "add", "edit", "remove", "install"}

    def test_model_fallback_has_subcommands(self):
        """``/model-fallback`` must expose its subcommands."""
        manager = CommandManager()
        from EvoScientist.commands.implementation.model_fallback import (
            ModelFallbackCommand,
        )

        manager.register(ModelFallbackCommand())
        scs = manager.list_subcommands("/model-fallback")
        names = {name for name, _desc in scs}
        assert {"list", "add", "remove", "clear", "save", "help"} <= names

    def test_channel_has_subcommands(self):
        """``/channel`` must expose status, stop, and channel type subcommands."""
        manager = CommandManager()
        from EvoScientist.commands.implementation.channel import ChannelCommand

        manager.register(ChannelCommand())
        scs = manager.list_subcommands("/channel")
        names = {name for name, _desc in scs}
        assert {"status", "stop", "telegram", "discord"}.issubset(names)

    def test_command_without_subcommands_returns_empty(self):
        """A command with no subcommands must return an empty list."""
        manager = CommandManager()
        from EvoScientist.commands.implementation.general import HelpCommand

        manager.register(HelpCommand())
        assert manager.list_subcommands("/help") == []
        assert manager.get_subcommands("/help") == []

    def test_unknown_command_returns_empty(self):
        """get_subcommands for a nonexistent command must return empty list."""
        manager = CommandManager()
        assert manager.list_subcommands("/nonexistent") == []
        assert manager.get_subcommands("/nonexistent") == []

    def test_subcommand_via_alias(self):
        """Registry by alias should still expose subcommands."""
        manager = CommandManager()
        from EvoScientist.commands.implementation.model_fallback import (
            ModelFallbackCommand,
        )

        manager.register(ModelFallbackCommand())
        scs = manager.list_subcommands("/fallback")
        names = {name for name, _desc in scs}
        assert {"list", "add", "remove"} <= names
