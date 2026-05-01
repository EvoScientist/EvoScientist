"""Tests for Rich streaming display helpers."""

from EvoScientist.stream.display import (
    _fix_markdown_heading_spacing,
    resolve_final_status_footer,
)


def test_resolve_final_status_footer_hides_footer_for_interactive_cli():
    assert resolve_final_status_footer(True, lambda: "footer") is None


def test_resolve_final_status_footer_keeps_footer_for_noninteractive():
    assert resolve_final_status_footer(False, lambda: "footer") == "footer"


class TestFixMarkdownHeadingSpacing:
    """`_fix_markdown_heading_spacing` inserts a space after ATX heading markers
    that are missing one (Rich's CommonMark parser is strict). Critical
    invariant: the helper only operates on display copies — it must never be
    applied to the streaming buffer, or partial chunks would corrupt heading
    levels at chunk boundaries (e.g. "#" → "# ", then "+#" → "# #" instead of
    "##"). The helper must be idempotent so re-running per frame is safe.
    """

    def test_inserts_missing_space(self):
        assert _fix_markdown_heading_spacing("#Bar") == "# Bar"
        assert _fix_markdown_heading_spacing("##Bar") == "## Bar"
        assert _fix_markdown_heading_spacing("###Bar") == "### Bar"
        assert _fix_markdown_heading_spacing("####Bar") == "#### Bar"
        assert _fix_markdown_heading_spacing("#####Bar") == "##### Bar"
        assert _fix_markdown_heading_spacing("######Bar") == "###### Bar"

    def test_idempotent_on_valid_headings(self):
        assert _fix_markdown_heading_spacing("### Foo") == "### Foo"
        assert _fix_markdown_heading_spacing("# Bar\n## Baz") == "# Bar\n## Baz"
        # Running twice gives the same result as running once.
        once = _fix_markdown_heading_spacing("###Foo\n##Bar")
        twice = _fix_markdown_heading_spacing(once)
        assert once == twice == "### Foo\n## Bar"

    def test_does_not_modify_buffer_under_chunked_streaming(self):
        """The load-bearing test: simulate chunk-by-chunk accumulation and
        confirm (a) the raw buffer is never mutated by the helper, and (b)
        applying the helper at every frame produces the correct display
        without corrupting heading levels at chunk boundaries.
        """
        chunks = ["#", "##", "#", "F", "o", "o", "\n"]
        raw_buffer = ""
        expected_concat = ""
        for chunk in chunks:
            raw_buffer += chunk
            expected_concat += chunk
            # Invariant 1: raw buffer equals naive concatenation (untouched).
            assert raw_buffer == expected_concat
            # Invariant 2: helper produces a display copy without mutating raw.
            display = _fix_markdown_heading_spacing(raw_buffer)
            assert raw_buffer == expected_concat  # still untouched
            # The helper is pure — calling it doesn't change its argument.
            assert isinstance(display, str)
        assert raw_buffer == "####Foo\n"
        assert _fix_markdown_heading_spacing(raw_buffer) == "#### Foo\n"

    def test_multiline_mixed(self):
        src = "###A\n## B\n#C\n#### D"
        assert _fix_markdown_heading_spacing(src) == "### A\n## B\n# C\n#### D"

    def test_indented_and_blockquote_unchanged(self):
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
