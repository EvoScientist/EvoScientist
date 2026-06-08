"""Tests for multi-stage command completions and category grouping."""

from unittest.mock import patch

from EvoScientist.commands.base import Command, SubCommand
from EvoScientist.commands.manager import CommandManager


def _make_manager() -> CommandManager:
    """Build a manager with a representative set of commands."""
    from EvoScientist.commands import implementation  # noqa: F401  trigger registration
    from EvoScientist.commands.manager import manager

    return manager


class TestGetCompletionsForInput:
    def setup_method(self):
        self.mgr = _make_manager()

    def test_slash_returns_all_commands(self):
        results = self.mgr.get_completions_for_input("/")
        names = [r[0] for r in results]
        assert "/mcp" in names
        assert "/help" in names
        assert "/new" in names

    def test_prefix_filters(self):
        results = self.mgr.get_completions_for_input("/mc")
        names = [r[0] for r in results]
        assert "/mcp" in names
        assert "/help" not in names

    def test_exact_match_leaf_hides(self):
        """Exact match on a command with no subcommands returns empty."""
        results = self.mgr.get_completions_for_input("/new")
        assert results == []

    def test_exact_match_with_subcommands_shows(self):
        """Exact match on /mcp should still return it (has subcommands)."""
        results = self.mgr.get_completions_for_input("/mcp")
        names = [r[0] for r in results]
        assert "/mcp" in names

    def test_space_after_command_shows_subcommands(self):
        results = self.mgr.get_completions_for_input("/mcp ")
        names = [r[0] for r in results]
        assert "list" in names
        assert "config" in names
        assert "add" in names
        assert "install" in names

    def test_subcommand_prefix_filters(self):
        results = self.mgr.get_completions_for_input("/mcp c")
        names = [r[0] for r in results]
        assert "config" in names
        assert "list" not in names

    def test_leaf_command_space_returns_empty(self):
        """/new has no subcommands, so /new + space returns nothing."""
        results = self.mgr.get_completions_for_input("/new ")
        assert results == []

    def test_nonexistent_returns_empty(self):
        results = self.mgr.get_completions_for_input("/nonexistent")
        assert results == []

    def test_empty_input_returns_empty(self):
        results = self.mgr.get_completions_for_input("")
        assert results == []

    def test_non_slash_returns_empty(self):
        results = self.mgr.get_completions_for_input("hello")
        assert results == []

    def test_results_have_categories(self):
        results = self.mgr.get_completions_for_input("/")
        cats = {r[2] for r in results}
        assert "Session" in cats
        assert "MCP" in cats

    def test_subcommand_results_have_empty_category(self):
        results = self.mgr.get_completions_for_input("/mcp ")
        for _, _, cat in results:
            assert cat == ""

    def test_category_ordering(self):
        results = self.mgr.get_completions_for_input("/")
        cats_in_order = []
        for _, _, cat in results:
            if not cats_in_order or cats_in_order[-1] != cat:
                cats_in_order.append(cat)
        # Session should come before General
        assert cats_in_order.index("Session") < cats_in_order.index("General")

    def test_channel_subcommands(self):
        results = self.mgr.get_completions_for_input("/channel ")
        names = [r[0] for r in results]
        assert "status" in names
        assert "telegram" in names
        assert "stop" in names

    def test_model_fallback_subcommands(self):
        results = self.mgr.get_completions_for_input("/model-fallback ")
        names = [r[0] for r in results]
        assert "add" in names
        assert "list" in names
        assert "clear" in names

    def test_subcommand_selected_stops_completing(self):
        """Once a subcommand is fully typed, don't loop back to subcommand list."""
        results = self.mgr.get_completions_for_input("/model-fallback help ")
        assert results == []
        results = self.mgr.get_completions_for_input("/channel status ")
        assert results == []

    def test_alias_matches_canonical_command(self):
        """/fallback should surface /model-fallback."""
        results = self.mgr.get_completions_for_input("/fallback")
        names = [r[0] for r in results]
        assert "/model-fallback" in names

    def test_alias_prefix_matches(self):
        """/qu should surface /exit via the /quit alias."""
        results = self.mgr.get_completions_for_input("/qu")
        names = [r[0] for r in results]
        assert "/exit" in names

    def test_alias_exact_match_hides_leaf(self):
        """/quit and /q are exact aliases for /exit (leaf), hide popup."""
        assert self.mgr.get_completions_for_input("/quit") == []
        assert self.mgr.get_completions_for_input("/q") == []

    def test_alias_exact_match_with_subcommands_shows(self):
        """/fallback is an alias for /model-fallback (has subcommands)."""
        results = self.mgr.get_completions_for_input("/fallback")
        names = [r[0] for r in results]
        assert "/model-fallback" in names

    def test_alias_delegates_subcommands(self):
        """/fallback + space should show model-fallback subcommands."""
        results = self.mgr.get_completions_for_input("/fallback ")
        names = [r[0] for r in results]
        assert "add" in names
        assert "list" in names


class TestMCPDynamicCompletions:
    def test_config_shows_server_names(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        cmd = MCPCommand()
        fake_config = {"myserver": {}, "other": {}}
        with patch("EvoScientist.mcp.load_mcp_config", return_value=fake_config):
            results = cmd.get_completions(["config", ""])
        names = [r[0] for r in results]
        assert "myserver" in names
        assert "other" in names

    def test_config_filters_by_prefix(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        cmd = MCPCommand()
        fake_config = {"myserver": {}, "other": {}}
        with patch("EvoScientist.mcp.load_mcp_config", return_value=fake_config):
            results = cmd.get_completions(["config", "my"])
        names = [r[0] for r in results]
        assert names == ["myserver"]

    def test_remove_shows_server_names(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        cmd = MCPCommand()
        fake_config = {"srv1": {}}
        with patch("EvoScientist.mcp.load_mcp_config", return_value=fake_config):
            results = cmd.get_completions(["remove", ""])
        assert [r[0] for r in results] == ["srv1"]


class TestCommandGetCompletions:
    def test_default_no_subcommands(self):
        """Command with no subcommands returns empty."""
        from EvoScientist.commands.implementation.session import NewCommand

        cmd = NewCommand()
        assert cmd.get_completions([""]) == []

    def test_default_walks_subcommands(self):
        from EvoScientist.commands.implementation.channel import ChannelCommand

        cmd = ChannelCommand()
        results = cmd.get_completions([""])
        names = [r[0] for r in results]
        assert "status" in names
        assert "telegram" in names

    def test_default_prefix_filter(self):
        from EvoScientist.commands.implementation.channel import ChannelCommand

        cmd = ChannelCommand()
        results = cmd.get_completions(["st"])
        names = [r[0] for r in results]
        assert "status" in names
        assert "stop" in names
        assert "telegram" not in names


class TestBuildHelpText:
    def test_includes_description(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        text = MCPCommand().build_help_text()
        assert "Manage MCP servers" in text

    def test_includes_subcommands(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        text = MCPCommand().build_help_text()
        assert "config" in text
        assert "install" in text

    def test_includes_aliases(self):
        from EvoScientist.commands.implementation.model_fallback import (
            ModelFallbackCommand,
        )

        text = ModelFallbackCommand().build_help_text()
        assert "/fallback" in text

    def test_includes_examples(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand

        text = MCPCommand().build_help_text()
        assert "/mcp add myserver" in text
