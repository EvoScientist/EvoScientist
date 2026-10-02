"""Tests for CLI SlashCommandCompleter (prompt_toolkit adapter)."""

from unittest.mock import MagicMock

from EvoScientist.cli.interactive import SlashCommandCompleter


def _doc(text: str):
    """Create a minimal prompt_toolkit Document stub."""
    doc = MagicMock()
    doc.text_before_cursor = text
    return doc


class TestSlashCommandCompleter:
    """Verify that ``SlashCommandCompleter.get_completions`` correctly
    delegates to the shared ``compute_completions`` engine and translates
    candidates into prompt_toolkit ``Completion`` objects.
    """

    def test_top_level_slash_shows_commands(self):
        completer = SlashCommandCompleter()
        completions = list(completer.get_completions(_doc("/he"), None))
        texts = {c.text for c in completions}
        assert "/help" in texts

    def test_exact_command_no_space_hides(self):
        completer = SlashCommandCompleter()
        completions = list(completer.get_completions(_doc("/help"), None))
        assert completions == []

    def test_exact_command_hides_even_when_prefix_of_another(self):
        """Regression for #293: ``/model`` must hide on exact match so Enter
        submits it, even though ``/model-fallback`` shares the prefix. Before
        the fix the popup stayed visible (two prefix matches) and the TUI's
        Enter handler completed the text instead of executing the command.
        """
        completer = SlashCommandCompleter()
        # Sanity: both commands share the prefix, so a partial prefix lists both.
        partial = {c.text for c in completer.get_completions(_doc("/mode"), None)}
        assert {"/model", "/model-fallback"} <= partial
        # Exact ``/model`` with no trailing space → hide.
        completions = list(completer.get_completions(_doc("/model"), None))
        assert completions == []

    def test_non_slash_returns_empty(self):
        completer = SlashCommandCompleter()
        completions = list(completer.get_completions(_doc("hello"), None))
        assert completions == []

    def test_trailing_space_shows_subcommands(self):
        completer = SlashCommandCompleter()
        completions = list(completer.get_completions(_doc("/mcp "), None))
        texts = {c.text for c in completions}
        assert "list" in texts
        assert "add" in texts

    def test_subcommand_prefix_filters(self):
        completer = SlashCommandCompleter()
        completions = list(completer.get_completions(_doc("/mcp lis"), None))
        texts = {c.text for c in completions}
        assert texts == {"list"}

    def test_exact_subcommand_hides(self):
        completer = SlashCommandCompleter()
        completions = list(completer.get_completions(_doc("/mcp list"), None))
        assert completions == []

    def test_results_match_engine_order(self):
        """Completer preserves the category-based order from compute_completions."""
        from EvoScientist.commands._completion_engine import compute_completions

        completer = SlashCommandCompleter()
        completions = list(completer.get_completions(_doc("/"), None))
        engine_result = compute_completions("/", 1)
        assert [c.text for c in completions] == [
            c.text for c in engine_result.candidates
        ]

    def test_display_meta_is_description(self):
        completer = SlashCommandCompleter()
        completions = list(completer.get_completions(_doc("/he"), None))
        for c in completions:
            if c.text == "/help":
                assert c.display_meta is not None

    def test_subcommand_completions_match_engine_order(self):
        """Subcommand completions preserve the order from compute_completions."""
        from EvoScientist.commands._completion_engine import compute_completions

        completer = SlashCommandCompleter()
        completions = list(completer.get_completions(_doc("/mcp "), None))
        engine_result = compute_completions("/mcp ", 5)
        assert [c.text for c in completions] == [
            c.text for c in engine_result.candidates
        ]
