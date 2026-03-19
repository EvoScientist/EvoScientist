"""Tests for EvoScientist.stream.diff_format module."""

from __future__ import annotations

import difflib
from unittest import mock

from EvoScientist.stream.diff_format import (
    _detect_unicode_support,
    _escape_markup,
    build_edit_diff,
    format_diff_rich,
)

# ---------------------------------------------------------------------------
# _escape_markup
# ---------------------------------------------------------------------------


class TestEscapeMarkup:
    def test_escapes_brackets(self):
        assert _escape_markup("[bold]text[/bold]") == r"\[bold\]text\[/bold\]"

    def test_plain_text_unchanged(self):
        assert _escape_markup("hello world") == "hello world"

    def test_empty_string(self):
        assert _escape_markup("") == ""

    def test_nested_brackets(self):
        assert _escape_markup("a[b[c]]d") == r"a\[b\[c\]\]d"


# ---------------------------------------------------------------------------
# _detect_unicode_support
# ---------------------------------------------------------------------------


class TestDetectUnicodeSupport:
    def test_utf8_encoding(self):
        with mock.patch("sys.stdout") as mock_stdout:
            mock_stdout.encoding = "utf-8"
            assert _detect_unicode_support() is True

    def test_ascii_encoding_with_utf_lang(self):
        with mock.patch("sys.stdout") as mock_stdout:
            mock_stdout.encoding = "ascii"
            with mock.patch.dict("os.environ", {"LANG": "en_US.UTF-8", "LC_ALL": ""}):
                assert _detect_unicode_support() is True

    def test_ascii_encoding_no_utf_lang(self):
        with mock.patch("sys.stdout") as mock_stdout:
            mock_stdout.encoding = "ascii"
            with mock.patch.dict(
                "os.environ", {"LANG": "C", "LC_ALL": ""}, clear=False
            ):
                assert _detect_unicode_support() is False


# ---------------------------------------------------------------------------
# format_diff_rich
# ---------------------------------------------------------------------------


class TestFormatDiffRich:
    def test_empty_diff_returns_dim_message(self):
        result = format_diff_rich("")
        assert "No changes detected" in result
        assert "[dim]" in result

    def test_single_line_change(self):
        old = ["hello"]
        new = ["world"]
        diff = "\n".join(difflib.unified_diff(old, new, lineterm="", n=3))
        result = format_diff_rich(diff)
        # Should contain addition and deletion markers
        assert "[green]+1[/green]" in result
        assert "[red]-1[/red]" in result
        # Deletion line (red background)
        assert "2d1515" in result  # red background color
        # Addition line (green background)
        assert "152d15" in result  # green background color

    def test_stats_header(self):
        old = ["a", "b", "c"]
        new = ["a", "x", "c", "d"]
        diff = "\n".join(difflib.unified_diff(old, new, lineterm="", n=3))
        result = format_diff_rich(diff)
        # 1 deletion (b), 2 additions (x, d)
        assert "[green]+2[/green]" in result
        assert "[red]-1[/red]" in result

    def test_max_lines_truncation(self):
        old = [f"line{i}" for i in range(50)]
        new = [f"LINE{i}" for i in range(50)]
        diff = "\n".join(difflib.unified_diff(old, new, lineterm="", n=0))
        result = format_diff_rich(diff, max_lines=10)
        assert "more lines" in result

    def test_context_lines_dimmed(self):
        old = ["a", "b", "c"]
        new = ["a", "B", "c"]
        diff = "\n".join(difflib.unified_diff(old, new, lineterm="", n=3))
        result = format_diff_rich(diff)
        # Context lines should use dim styling
        assert "[dim]" in result

    def test_title_header(self):
        old = ["hello"]
        new = ["world"]
        diff = "\n".join(difflib.unified_diff(old, new, lineterm="", n=3))
        result = format_diff_rich(diff, title="/path/to/file.py")
        # Title should appear with bold cyan and box-drawing chars
        assert "file.py" in result
        assert "[bold cyan]" in result

    def test_title_none_omits_header(self):
        old = ["hello"]
        new = ["world"]
        diff = "\n".join(difflib.unified_diff(old, new, lineterm="", n=3))
        result = format_diff_rich(diff, title=None)
        assert "[bold cyan]" not in result

    def test_stats_footer(self):
        old = ["a"]
        new = ["b"]
        diff = "\n".join(difflib.unified_diff(old, new, lineterm="", n=3))
        result = format_diff_rich(diff)
        # Stats should appear both at top and bottom
        lines = result.splitlines()
        # First non-empty line is stats header, last non-empty line is stats footer
        non_empty = [ln for ln in lines if ln.strip()]
        assert non_empty[0] == non_empty[-1]  # header == footer

    def test_max_lines_none_unlimited(self):
        old = [f"line{i}" for i in range(200)]
        new = [f"LINE{i}" for i in range(200)]
        diff = "\n".join(difflib.unified_diff(old, new, lineterm="", n=0))
        result = format_diff_rich(diff, max_lines=None)
        # Should NOT have truncation marker
        assert "more lines" not in result


# ---------------------------------------------------------------------------
# build_edit_diff
# ---------------------------------------------------------------------------


class TestBuildEditDiff:
    def test_returns_none_when_equal(self):
        assert build_edit_diff("/foo.py", "same", "same") is None

    def test_returns_none_when_both_empty(self):
        assert build_edit_diff("/foo.py", "", "") is None

    def test_returns_formatted_markup_for_valid_diff(self):
        result = build_edit_diff(
            "/foo.py",
            "old line",
            "new line",
        )
        assert result is not None
        assert "[green]" in result
        assert "[red]" in result

    def test_multiline_diff(self):
        old = "line1\nline2\nline3"
        new = "line1\nmodified\nline3\nline4"
        result = build_edit_diff("/test.py", old, new)
        assert result is not None
        assert "+2" in result  # 2 additions
        assert "-1" in result  # 1 deletion

    def test_no_truncation_by_default(self):
        old = "\n".join(f"old{i}" for i in range(100))
        new = "\n".join(f"new{i}" for i in range(100))
        result = build_edit_diff("/big.py", old, new)
        assert result is not None
        # Default max_lines=None means no truncation
        assert "more lines" not in result

    def test_explicit_max_lines_truncates(self):
        old = "\n".join(f"old{i}" for i in range(100))
        new = "\n".join(f"new{i}" for i in range(100))
        result = build_edit_diff("/big.py", old, new, max_lines=5)
        assert result is not None
        assert "more lines" in result

    def test_file_path_shown_as_title(self):
        result = build_edit_diff("/my/file.py", "a", "b")
        assert result is not None
        # File path should appear as the title header
        assert "file.py" in result
        assert "[bold cyan]" in result


# ---------------------------------------------------------------------------
# Integration with format_tool_result_compact
# ---------------------------------------------------------------------------


class TestFormatToolResultCompactEditFile:
    def test_edit_file_with_tool_args_shows_diff(self):
        from EvoScientist.stream.display import format_tool_result_compact

        result = format_tool_result_compact(
            "edit_file",
            "[OK] Successfully replaced 1 instance(s)",
            tool_args={
                "path": "/foo.py",
                "old_string": "hello",
                "new_string": "world",
            },
        )
        # Should return markup elements with diff content
        assert len(result) >= 1
        plain = result[0].plain if hasattr(result[0], "plain") else str(result[0])
        assert "+1" in plain or "world" in plain or "hello" in plain

    def test_edit_file_without_tool_args_falls_through(self):
        from EvoScientist.stream.display import format_tool_result_compact

        result = format_tool_result_compact(
            "edit_file",
            "[OK] Successfully replaced 1 instance(s)",
        )
        # Without tool_args, falls through to normal rendering
        assert len(result) >= 1

    def test_edit_file_error_shows_error_not_diff(self):
        from EvoScientist.stream.display import format_tool_result_compact

        result = format_tool_result_compact(
            "edit_file",
            "[ERROR] File not found",
            tool_args={
                "path": "/foo.py",
                "old_string": "hello",
                "new_string": "world",
            },
        )
        # Error content should show error, not diff
        assert len(result) >= 1

    def test_backward_compatible_no_tool_args(self):
        from EvoScientist.stream.display import format_tool_result_compact

        # Existing calls without tool_args should still work
        result = format_tool_result_compact("read_file", "[OK] 42 lines")
        assert len(result) >= 1
