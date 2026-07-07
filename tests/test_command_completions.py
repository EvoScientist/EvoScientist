"""Tests for multi-stage command completions, categories, and dynamic completions."""

from unittest.mock import patch

from EvoScientist.commands._completion_engine import compute_completions


class TestTopLevelCompletions:
    def test_slash_returns_all_commands(self):
        r = compute_completions("/", 1)
        names = [c.text for c in r.candidates]
        assert "/mcp" in names
        assert "/help" in names
        assert "/new" in names

    def test_prefix_filters(self):
        r = compute_completions("/mc", 3)
        names = [c.text for c in r.candidates]
        assert "/mcp" in names
        assert "/help" not in names

    def test_exact_leaf_hides(self):
        r = compute_completions("/new", 4)
        assert r.kind == "empty"

    def test_exact_with_subcommands_hides_without_space(self):
        r = compute_completions("/mcp", 4)
        assert r.kind == "empty"

    def test_results_have_categories(self):
        r = compute_completions("/", 1)
        cats = {c.category for c in r.candidates}
        assert "Session" in cats
        assert "MCP" in cats
        assert "General" in cats

    def test_category_ordering(self):
        r = compute_completions("/", 1)
        cats = []
        for c in r.candidates:
            if not cats or cats[-1] != c.category:
                cats.append(c.category)
        assert cats.index("Session") < cats.index("General")


class TestAliasVisibility:
    def test_alias_prefix_matches(self):
        r = compute_completions("/fa", 3)
        names = [c.text for c in r.candidates]
        assert "/model-fallback" in names

    def test_alias_exact_hides_leaf(self):
        r = compute_completions("/quit", 5)
        assert r.kind == "empty"

    def test_alias_exact_hides_without_space(self):
        r = compute_completions("/fallback", 9)
        assert r.kind == "empty"

    def test_alias_space_shows_subcommands(self):
        r = compute_completions("/fallback ", 10)
        names = [c.text for c in r.candidates]
        assert "add" in names
        assert "list" in names


class TestSubcommandCompletions:
    def test_space_shows_subcommands(self):
        r = compute_completions("/mcp ", 5)
        names = [c.text for c in r.candidates]
        assert "list" in names
        assert "config" in names
        assert "install" in names

    def test_prefix_filters_subcommands(self):
        r = compute_completions("/mcp c", 6)
        names = [c.text for c in r.candidates]
        assert "config" in names
        assert "list" not in names

    def test_leaf_subcommand_stops(self):
        r = compute_completions("/model-fallback help ", 21)
        assert r.kind == "empty"

    def test_channel_all_types(self):
        r = compute_completions("/channel ", 9)
        names = [c.text for c in r.candidates]
        assert "status" in names
        assert "telegram" in names
        assert "discord" in names

    def test_model_fallback_subcommands(self):
        r = compute_completions("/model-fallback ", 16)
        names = [c.text for c in r.candidates]
        assert "add" in names
        assert "clear" in names


class TestDynamicCompletions:
    def _invalidate_mcp_cache(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand
        from EvoScientist.commands.manager import manager

        cmd = manager.get_command("/mcp")
        if isinstance(cmd, MCPCommand):
            cmd._invalidate_server_cache()

    def test_mcp_config_server_names(self):
        self._invalidate_mcp_cache()
        fake_config = {"myserver": {}, "other": {}}
        with patch("EvoScientist.mcp.load_mcp_config", return_value=fake_config):
            r = compute_completions("/mcp config ", 12)
        names = [c.text for c in r.candidates]
        assert "myserver" in names
        assert "other" in names

    def test_mcp_config_prefix_filters(self):
        self._invalidate_mcp_cache()
        fake_config = {"myserver": {}, "other": {}}
        with patch("EvoScientist.mcp.load_mcp_config", return_value=fake_config):
            r = compute_completions("/mcp config my", 14)
        names = [c.text for c in r.candidates]
        assert names == ["myserver"]

    def test_mcp_remove_shows_servers(self):
        self._invalidate_mcp_cache()
        fake_config = {"srv1": {}}
        with patch("EvoScientist.mcp.load_mcp_config", return_value=fake_config):
            r = compute_completions("/mcp remove ", 12)
        assert [c.text for c in r.candidates] == ["srv1"]
