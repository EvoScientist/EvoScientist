"""Tests for Rich markup escape safety in formatter."""

from EvoScientist.stream.formatter import ToolResultFormatter


class TestRichMarkupEscape:
    def test_panel_title_with_brackets(self):
        """Tool name with brackets shouldn't crash formatter."""
        formatter = ToolResultFormatter()
        result = formatter.format("arr[0]", "[OK]\nDone", max_length=800)
        # Should not raise — content is safely rendered
        assert result.elements

    def test_error_format_with_brackets(self):
        """Error content with markup-like text shouldn't crash."""
        formatter = ToolResultFormatter()
        result = formatter.format(
            "test", "Error: expected [int] got [str]", max_length=800
        )
        assert result.elements
