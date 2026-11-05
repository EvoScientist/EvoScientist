"""Tests for EvoScientist/stream/utils.py pure functions."""

from EvoScientist.stream.utils import (
    _shorten_path,
    count_lines,
    cut_middle,
    failure_summary,
    format_tool_compact,
    format_tool_compact_with_result,
    has_args,
    is_success,
    truncate,
    truncate_with_line_hint,
)

# === is_success ===


class TestIsSuccess:
    def test_ok_prefix(self):
        assert is_success("[OK] all good") is True

    def test_failed_prefix(self):
        assert is_success("[FAILED] bad") is False

    def test_traceback(self):
        assert is_success("Traceback (most recent call last)\n  File ...") is False

    def test_exception(self):
        assert is_success("Exception: something went wrong") is False

    def test_error(self):
        assert is_success("Error: file not found") is False

    def test_clean_output(self):
        assert is_success("file1.py\nfile2.py") is True

    def test_whitespace_stripped(self):
        assert is_success("  [OK] with spaces  ") is True

    def test_error_in_code_content_not_false_positive(self):
        # read_file returning code with "Error:" deep inside should be success
        content = '#!/usr/bin/env python3\n"""\nSkill Packager\n"""\n\nprint(f"Error: not found")'
        assert is_success(content) is True

    def test_error_on_line4_not_false_positive(self):
        content = "line1\nline2\nline3\nError: buried deep\nline5"
        assert is_success(content) is True

    def test_error_on_first_line(self):
        assert is_success("Error: file not found\nsome detail") is False

    def test_error_invoking_tool(self):
        assert is_success("Error invoking tool 'write_file'") is False

    def test_failed_to_uninstall(self):
        assert (
            is_success("Failed to uninstall skill: Skill not found: latex-paper-en")
            is False
        )

    def test_failed_to_install(self):
        assert is_success("Failed to install skill: git clone failed: ...") is False

    def test_failed_in_code_content_not_false_positive(self):
        content = '#!/usr/bin/env python3\n"""Helper"""\n\nif x:\n    print("Failed to connect")'
        assert is_success(content) is True


# === format_tool_compact ===


class TestFormatToolCompact:
    def test_no_args(self):
        assert format_tool_compact("execute", None) == "execute()"
        assert format_tool_compact("execute", {}) == "execute()"

    def test_execute(self):
        result = format_tool_compact("execute", {"command": "ls -la"})
        assert result == "execute(ls -la)"

    def test_execute_long_command(self):
        long_cmd = "x" * 60
        result = format_tool_compact("execute", {"command": long_cmd})
        assert len(result) < 70
        assert result.endswith("\u2026)")

    def test_read_file(self):
        result = format_tool_compact("read_file", {"path": "src/main.py"})
        assert result == "read_file(src/main.py)"

    def test_write_file(self):
        result = format_tool_compact("write_file", {"path": "out.txt"})
        assert result == "write_file(out.txt)"

    def test_edit_file(self):
        result = format_tool_compact("edit_file", {"path": "f.py"})
        assert result == "edit_file(f.py)"

    # Global profile memory display (/memories/ = global)
    def test_read_file_global_memory(self):
        result = format_tool_compact(
            "read_file", {"path": "/memories/profile/USER_PROFILE.md"}
        )
        assert result == "Reading memory"

    def test_read_file_global_memory_file_path_alias(self):
        result = format_tool_compact(
            "read_file", {"file_path": "/memories/profile/USER_PROFILE.md"}
        )
        assert result == "Reading memory"

    def test_read_file_any_global_memory_file(self):
        result = format_tool_compact("read_file", {"path": "/memories/history.md"})
        assert result == "Reading memory"

    def test_write_file_global_memory(self):
        result = format_tool_compact(
            "write_file", {"path": "/memories/profile/USER_PROFILE.md"}
        )
        assert result == "Updating memory"

    def test_edit_file_global_memory(self):
        result = format_tool_compact(
            "edit_file", {"path": "/memories/profile/USER_PROFILE.md"}
        )
        assert result == "Updating memory"

    def test_write_edit_any_global_memory_file(self):
        write_result = format_tool_compact("write_file", {"path": "/memories/soul.md"})
        edit_result = format_tool_compact(
            "edit_file", {"path": "/memories/skills-context.md"}
        )
        assert write_result == "Updating memory"
        assert edit_result == "Updating memory"

    # Project-local /memory/ files show normal tool display
    def test_read_file_project_memory(self):
        result = format_tool_compact(
            "read_file", {"path": "/memory/ideation-memory.md"}
        )
        assert result == "read_file(/memory/ideation-memory.md)"

    def test_edit_file_project_memory(self):
        result = format_tool_compact(
            "edit_file", {"path": "/memory/experiment-memory.md"}
        )
        assert result == "edit_file(/memory/experiment-memory.md)"

    def test_memory_display_inferred_from_result_when_args_sparse(self):
        read_result = format_tool_compact_with_result(
            "read_file",
            {},
            "# User profile\n\nFounder: Zachary",
        )
        assert read_result == "Reading memory"

        edit_result = format_tool_compact_with_result(
            "edit_file",
            {},
            "Successfully replaced 1 instance(s) of the string in '/memories/profile/USER_PROFILE.md'",
        )
        assert edit_result == "Updating memory"

        write_result = format_tool_compact_with_result(
            "write_file",
            {},
            "Wrote updated content to '/memories/history.md'",
        )
        assert write_result == "Updating memory"

    def test_profile_memory_inference_uses_profile_template_headings(self, monkeypatch):
        from EvoScientist.middleware import memory
        from EvoScientist.stream import utils

        monkeypatch.setitem(
            memory.PROFILE_TEMPLATES,
            "/profile/CUSTOM.md",
            "# Custom profile\n\n## Notes\n",
        )
        utils._profile_memory_headings.cache_clear()

        try:
            result = format_tool_compact_with_result(
                "read_file",
                {},
                "# Custom profile\n\n- remembered",
            )
        finally:
            utils._profile_memory_headings.cache_clear()

        assert result == "Reading memory"

    def test_profile_memory_headings_skip_user_profile_frontmatter(self):
        from EvoScientist.stream import utils

        utils._profile_memory_headings.cache_clear()
        try:
            headings = utils._profile_memory_headings()
        finally:
            utils._profile_memory_headings.cache_clear()

        assert "# User profile" in headings
        assert not any(heading.startswith("---") for heading in headings)

    def test_project_memory_result_not_special(self):
        result = format_tool_compact_with_result(
            "write_file",
            {},
            "Wrote updated content to '/memory/ideation-memory.md'",
        )
        assert result != "Updating memory"

    def test_glob(self):
        result = format_tool_compact("glob", {"pattern": "*.py"})
        assert result == "glob(*.py)"

    def test_grep(self):
        result = format_tool_compact("grep", {"pattern": "TODO", "path": "src/"})
        assert result == "grep(TODO, src/)"

    def test_ls(self):
        assert format_tool_compact("ls", {"path": "/src"}) == "ls(/src)"

    def test_write_todos_list(self):
        todos = [{"status": "todo", "content": "a"}, {"status": "todo", "content": "b"}]
        result = format_tool_compact("write_todos", {"todos": todos})
        assert result == "write_todos(2 items)"

    def test_write_todos_non_list(self):
        result = format_tool_compact("write_todos", {"todos": "something"})
        assert result == "write_todos(...)"

    def test_read_todos(self):
        assert format_tool_compact("read_todos", {}) == "read_todos()"

    def test_task_with_type_and_desc(self):
        result = format_tool_compact(
            "task", {"subagent_type": "research-agent", "description": "Find papers"}
        )
        assert "Cooking with research-agent" in result
        assert "Find papers" in result

    def test_task_with_type_only(self):
        result = format_tool_compact("task", {"subagent_type": "code-agent"})
        assert result == "Cooking with code-agent"

    def test_task_with_desc_only(self):
        result = format_tool_compact("task", {"description": "do stuff"})
        assert result == "task(description=do stuff)"

    def test_task_no_info(self):
        result = format_tool_compact("task", {"other": "value"})
        assert result == "task(other=value)"

    def test_tavily_search(self):
        result = format_tool_compact("tavily_search", {"query": "python testing"})
        assert result == "tavily_search(python testing)"

    def test_think_tool(self):
        result = format_tool_compact("think_tool", {"reflection": "need more data"})
        assert result == "think_tool(need more data)"

    def test_unknown_tool(self):
        result = format_tool_compact("custom_tool", {"key": "value"})
        assert "custom_tool(" in result
        assert "key=value" in result

    def test_unknown_tool_long_value(self):
        result = format_tool_compact("custom_tool", {"key": "a" * 30})
        assert "\u2026" in result


# === truncate ===


class TestTruncate:
    def test_within_limit(self):
        assert truncate("hello", 10) == "hello"

    def test_at_limit(self):
        assert truncate("hello", 5) == "hello"

    def test_over_limit(self):
        result = truncate("hello world", 5)
        assert result.startswith("hello")
        assert "truncated" in result


class TestFailureSummary:
    STATUS = "\n\nExit code: 1\n[Command failed with exit code 1]"

    def test_longer_command_output_names_the_exit_status(self):
        pip = (
            "Collecting foo\n  Downloading foo-1.0.tar.gz\n"
            "[stderr] ERROR: No matching distribution found for foo" + self.STATUS
        )
        assert failure_summary("execute", pip) == "[Command failed with exit code 1]"

    def test_a_one_line_output_is_its_own_reason(self):
        cat = "[stderr] cat: missing.txt: No such file or directory" + self.STATUS
        assert failure_summary("execute", cat) == (
            "cat: missing.txt: No such file or directory"
        )
        blocked = (
            "Command blocked: contains '..' path traversal.\n"
            "[Command failed with exit code 1]"
        )
        assert failure_summary("execute", blocked).startswith("Command blocked:")

    def test_no_output_names_the_exit_status(self):
        grep = "<no output>" + self.STATUS
        assert failure_summary("execute", grep) == "[Command failed with exit code 1]"

    def test_the_sandbox_timeout_notice_is_named(self):
        timeout = (
            "Error: Command timed out after 300 seconds.\n\n"
            "Recovery \u2014 pick one:\n  1. Needs more time?\n  3. Report it back.\n"
            "[Command failed with exit code 124]"
        )
        assert failure_summary("execute", timeout).startswith(
            "Error: Command timed out"
        )

    def test_truncated_output_names_the_exit_status(self):
        cut = (
            "\n".join(str(i) for i in range(1, 50))
            + "\n\n... Output truncated at 100000 bytes.\n\nExit code: 3\n"
            "[Command failed with exit code 3]\n[Output was truncated due to size limits]"
        )
        assert failure_summary("execute", cut) == "[Command failed with exit code 3]"

    def test_an_execute_call_that_never_ran_names_its_first_line(self):
        invalid = (
            "1 validation error for ExecuteSchema\ncommand\n  Field required\n"
            "    For further information visit https://errors.pydantic.dev/v/missing"
        )
        assert failure_summary("execute", invalid) == (
            "1 validation error for ExecuteSchema"
        )

    def test_an_interpreter_error_names_its_type_and_message(self):
        result = (
            "<stdout>\nloaded 3 files\n</stdout>\n"
            "<error type=\"TypeError\">cannot read property 'score' of &lt;x&gt;\n"
            "    at main (eval:3)</error>"
        )
        assert failure_summary("code_interpreter", result) == (
            "TypeError: cannot read property 'score' of <x>"
        )

    def test_status_lines_inside_the_output_are_not_trusted(self):
        echoed = (
            "[Command failed with exit code 7]\n[stderr] boom"
            "\n\nExit code: 2\n[Command failed with exit code 2]"
        )
        assert failure_summary("execute", echoed) == "[Command failed with exit code 2]"
        two_lines = (
            "starting job\nExit code: 0\n\nExit code: 3\n"
            "[Command failed with exit code 3]"
        )
        assert failure_summary("execute", two_lines) == (
            "[Command failed with exit code 3]"
        )
        for printed in (
            "[Command failed with exit code 7]",
            "Exit code: 0",
            "[stderr] [Command failed with exit code 7]",
            "[stderr] Exit code: 0",
        ):
            only = printed + "\n\nExit code: 2\n[Command failed with exit code 2]"
            assert failure_summary("execute", only) == (
                "[Command failed with exit code 2]"
            )

    def test_every_appended_line_is_stripped(self):
        cut = (
            "fatal: checksum mismatch\n\n... Output truncated at 100000 bytes.\n\n"
            "Exit code: 3\n[Command failed with exit code 3]\n"
            "[Output was truncated due to size limits]"
        )
        assert failure_summary("execute", cut) == "fatal: checksum mismatch"

    def test_an_interpreter_message_stays_on_one_line(self):
        result = '<error type="ValueError">bad&#10;second line</error>'
        assert failure_summary("code_interpreter", result) == "ValueError: bad"

    def test_an_interpreter_error_cut_from_its_tag_names_nothing(self):
        cut = (
            "<stdout>\nrow 0: ok\n... (truncated) ...\n"
            "very long detail\n    at main (eval:3)</error>"
        )
        assert failure_summary("code_interpreter", cut) == ""

    def test_other_tools_name_their_first_line(self):
        error = "[TOOL ERROR] Tool 'lookup' failed:\n\nValueError: x\n\nYou may retry."
        assert failure_summary("lookup", error) == "[TOOL ERROR] Tool 'lookup' failed:"
        assert failure_summary("lookup", "") == ""


class TestCutMiddle:
    def test_within_limit(self):
        assert cut_middle("short", 10, 4) == "short"

    def test_keeps_the_start_and_whole_last_lines(self):
        text = "\n".join(f"line {i}" for i in range(100))
        result = cut_middle(text, 200, 40)
        head, marker, end = result.partition("\n... (truncated) ...\n")
        assert marker
        assert head == text[:160]
        assert end.endswith("line 99")
        assert end.split("\n")[0].startswith("line ")

    def test_an_end_that_starts_on_a_line_keeps_that_line(self):
        reason = "Error: disk full\n"
        for filler in ("z", "\n"):
            text = "x" * 400 + "\n" + reason + filler * (500 - len(reason))
            assert "Error: disk full" in cut_middle(text, 800, 500)

    def test_a_long_last_line_is_kept_in_part(self):
        reason = "[stderr] ValueError: " + "detail " * 95 + "END"
        text = "\n".join(f"line {i}" for i in range(200)) + "\n" + reason
        text += "\nExit code: 1\n[Command failed with exit code 1]"
        end = cut_middle(text, 2000, 500).partition("\n... (truncated) ...\n")[2]
        assert not end.startswith("Exit code")
        assert "[stderr] ValueError" not in end
        assert "detail END\nExit code: 1" in end

    def test_a_short_line_across_the_boundary_is_kept_whole(self):
        reason = "[stderr] OSError: disk full"
        filler = "\n".join(f"log {i:03d}" for i in range(400))
        trailer = "".join(f"step {i}\n" for i in range(60))
        trailer += "\nExit code: 1\n[Command failed with exit code 1]"
        for into in (1, 10, 20):
            text = filler + "\n" + reason + ("\n" + trailer)[: 500 - len(reason) + into]
            assert text[len(text) - 500 :].startswith(reason[into:])
            end = cut_middle(text, 2000, 500).partition("\n... (truncated) ...\n")[2]
            assert end.startswith(reason)

    def test_an_end_never_repeats_the_start(self):
        text = "a" * 1450 + "\n" + "b" * 620
        out = cut_middle(text, 2000, 500)
        assert "... (truncated) ..." in out
        assert out.count("b") <= text.count("b")

    def test_a_line_right_after_the_head_is_kept(self):
        reason = "[stderr] OSError: disk full\n"
        text = "h" * 1499 + "\n" + reason + "t" * (2026 - 1500 - len(reason))
        assert "[stderr] OSError: disk full" in cut_middle(text, 2000, 500)

    def test_a_cut_that_hides_almost_nothing_is_not_made(self):
        text = "h" * 1500 + "\n" + "e" * 599
        assert cut_middle(text, 2000, 500) == text


# === _shorten_path ===


class TestShortenPath:
    def test_short_path(self):
        assert _shorten_path("src/main.py") == "src/main.py"

    def test_long_path(self):
        long_path = "a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r.py"
        result = _shorten_path(long_path, max_len=20)
        assert result.startswith(".../")
        assert result.endswith("r.py")


# === has_args ===


class TestHasArgs:
    def test_none(self):
        assert has_args(None) is False

    def test_empty_dict(self):
        assert has_args({}) is False

    def test_non_empty(self):
        assert has_args({"key": "val"}) is True


# === count_lines ===


class TestCountLines:
    def test_empty(self):
        assert count_lines("") == 0

    def test_single_line(self):
        assert count_lines("hello") == 1

    def test_multi_line(self):
        assert count_lines("a\nb\nc") == 3


# === truncate_with_line_hint ===


class TestTruncateWithLineHint:
    def test_within_limit(self):
        text, remaining = truncate_with_line_hint("a\nb\nc", max_lines=5)
        assert remaining == 0
        assert "a" in text

    def test_over_limit(self):
        text, remaining = truncate_with_line_hint("a\nb\nc\nd\ne\nf", max_lines=3)
        assert remaining == 3
        assert "d" not in text
