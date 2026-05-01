"""Tests for Rich streaming display helpers."""

from EvoScientist.stream.display import (
    _fix_markdown_heading_spacing,
    resolve_final_status_footer,
)


def test_resolve_final_status_footer_hides_footer_for_interactive_cli():
    """Interactive CLI hides the final status footer (the prompt redraws it)."""
    assert resolve_final_status_footer(True, lambda: "footer") is None


def test_resolve_final_status_footer_keeps_footer_for_noninteractive():
    """Non-interactive output keeps the footer so callers see the trailing status."""
    assert resolve_final_status_footer(False, lambda: "footer") == "footer"


class TestFixMarkdownHeadingSpacing:
    """Pure-helper tests: heading levels, idempotence, EOS / CRLF / fenced
    code. The display-copy-only contract at call sites is covered by
    ``TestAssistantMessageBufferContract``.
    """

    def test_inserts_missing_space(self):
        """Inserts a space after `#`-marker for all 6 ATX heading levels."""
        assert _fix_markdown_heading_spacing("#Bar") == "# Bar"
        assert _fix_markdown_heading_spacing("##Bar") == "## Bar"
        assert _fix_markdown_heading_spacing("###Bar") == "### Bar"
        assert _fix_markdown_heading_spacing("####Bar") == "#### Bar"
        assert _fix_markdown_heading_spacing("#####Bar") == "##### Bar"
        assert _fix_markdown_heading_spacing("######Bar") == "###### Bar"

    def test_idempotent_on_valid_headings(self):
        """Already-spaced markers are unchanged; ``f(f(x)) == f(x)``."""
        assert _fix_markdown_heading_spacing("### Foo") == "### Foo"
        assert _fix_markdown_heading_spacing("# Bar\n## Baz") == "# Bar\n## Baz"
        # Running twice gives the same result as running once.
        once = _fix_markdown_heading_spacing("###Foo\n##Bar")
        twice = _fix_markdown_heading_spacing(once)
        assert once == twice == "### Foo\n## Bar"

    def test_multiline_mixed(self):
        """Each line of a multiline string is normalised independently."""
        src = "###A\n## B\n#C\n#### D"
        assert _fix_markdown_heading_spacing(src) == "### A\n## B\n# C\n#### D"

    def test_indented_and_blockquote_unchanged(self):
        """Lines whose `#` is not at column 0 are left alone (`^` requires col 0)."""
        # Indented lines (treated as code by CommonMark) — `^` only matches
        # column 0, so the helper leaves them alone.
        assert _fix_markdown_heading_spacing("   ###Indented") == "   ###Indented"
        # Blockquote-prefixed lines — the `>` shifts the heading away from
        # column 0; helper does not touch them. Documents accepted edge case.
        assert _fix_markdown_heading_spacing("> ###Quoted") == "> ###Quoted"
        # Empty string and whitespace-only inputs are unchanged.
        assert _fix_markdown_heading_spacing("") == ""
        assert _fix_markdown_heading_spacing("\n\n") == "\n\n"

    def test_bare_hash_at_end_of_string_unchanged(self):
        """A trailing `#` (e.g. mid-stream chunk) must not gain a spurious
        space. Positive lookahead requires a real non-excluded char to
        follow, so EOS naturally fails the match.
        """
        assert _fix_markdown_heading_spacing("#") == "#"
        assert _fix_markdown_heading_spacing("##") == "##"
        assert _fix_markdown_heading_spacing("######") == "######"
        # Trailing hash on a non-trailing line is also untouched (the line
        # has no follow-up content yet).
        assert _fix_markdown_heading_spacing("Foo\n###") == "Foo\n###"

    def test_crlf_line_endings(self):
        """CRLF (`\\r\\n`) line endings: `\\r` is in the exclusion set so
        empty CRLF heading lines are unchanged, and a real CRLF heading
        gets a space inserted in front of the carriage-return-free part.
        """
        # Empty CRLF heading — must not become `# \r\n`.
        assert _fix_markdown_heading_spacing("#\r\n") == "#\r\n"
        assert _fix_markdown_heading_spacing("###\r\n") == "###\r\n"
        # Multi-line mixed CRLF — both lines fixed.
        assert _fix_markdown_heading_spacing("###A\r\n##B") == "### A\r\n## B"
        # Trailing CRLF after content — fix applies, line ending preserved.
        assert _fix_markdown_heading_spacing("###Foo\r\n") == "### Foo\r\n"

    def test_fenced_code_block_known_limitation(self):
        """The regex is context-free, so `###define` at column 0 inside a
        backtick fence WILL get a space in the display copy. This test
        documents (and locks in) the accepted trade-off — flip these
        assertions if a future fix gates on fence parsing.
        """
        src = "```c\n###define X 1\n```"
        # Currently DOES alter the line inside the fence.
        assert _fix_markdown_heading_spacing(src) == "```c\n### define X 1\n```"


class TestAssistantMessageBufferContract:
    """Regression guard: ``AssistantMessage`` flush/mount must apply the
    heading fix to a display copy and leave ``self._content`` untouched.
    """

    def _make_widget(self, initial: str = ""):
        from unittest.mock import MagicMock

        from EvoScientist.cli.widgets.assistant_message import AssistantMessage

        msg = AssistantMessage(initial_content=initial)
        fake_md = MagicMock()
        msg.query_one = MagicMock(return_value=fake_md)
        return msg, fake_md

    def test_flush_markdown_does_not_mutate_buffer(self):
        msg, fake_md = self._make_widget()
        msg._content = "###Foo\n##Bar"

        msg._flush_markdown()

        # Raw streaming buffer is preserved verbatim.
        assert msg._content == "###Foo\n##Bar"
        # The Textual Markdown widget receives the fixed display copy.
        fake_md.update.assert_called_once_with("### Foo\n## Bar")
        # Flush latch is cleared.
        assert msg._flush_pending is False

    def test_on_mount_does_not_mutate_initial_content(self):
        msg, fake_md = self._make_widget(initial="###Hello")

        msg.on_mount()

        assert msg._content == "###Hello"
        fake_md.update.assert_called_once_with("### Hello")

    def test_on_mount_no_op_when_initial_empty(self):
        msg, fake_md = self._make_widget(initial="")

        msg.on_mount()

        assert msg._content == ""
        fake_md.update.assert_not_called()
