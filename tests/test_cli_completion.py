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

    def test_results_sorted_alphabetically(self):
        completer = SlashCommandCompleter()
        completions = list(completer.get_completions(_doc("/"), None))
        texts = [c.text for c in completions]
        assert texts == sorted(texts)

    def test_display_meta_is_description(self):
        completer = SlashCommandCompleter()
        completions = list(completer.get_completions(_doc("/he"), None))
        for c in completions:
            if c.text == "/help":
                assert c.display_meta is not None

    def test_subcommand_completions_sorted(self):
        completer = SlashCommandCompleter()
        completions = list(completer.get_completions(_doc("/mcp "), None))
        texts = [c.text for c in completions]
        assert texts == sorted(texts)
