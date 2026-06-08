"""Tests for Rich streaming display helpers."""

from typing import Any, cast

from rich.console import Console
from rich.markdown import Markdown

from EvoScientist.stream.display import (
    _fix_markdown_heading_spacing,
    create_streaming_display,
    resolve_final_status_footer,
)
from EvoScientist.stream.state import SubAgentState


def _render_text(renderable) -> str:
    console = Console(record=True, width=100, color_system=None)
    console.print(renderable)
    return console.export_text()


def test_resolve_final_status_footer_hides_footer_for_interactive_cli():
    """Interactive CLI hides the final status footer (the prompt redraws it)."""
    assert resolve_final_status_footer(True, lambda: "footer") is None


def test_resolve_final_status_footer_keeps_footer_for_noninteractive():
    """Non-interactive output keeps the footer so callers see the trailing status."""
    assert resolve_final_status_footer(False, lambda: "footer") == "footer"


def test_streaming_display_keeps_narration_visible_with_pending_memory_tool():
    """Profile-memory reads still block, while lead-in text remains visible."""
    narration = "Here is the answer."
    renderable = create_streaming_display(
        response_text=narration,
        narrated_response_end=len(narration),
        narration_segments=[(0, narration)],
        tool_calls=[
            {
                "id": "tc1",
                "name": "read_file",
                "args": {"path": "/memories/profile/USER_PROFILE.md"},
            }
        ],
        tool_results=[],
    )

    rendered = _render_text(renderable)

    assert "Here is the answer." in rendered
    assert "Reading memory" in rendered
    assert "Running" in rendered
    assert rendered.index("Here is the answer.") < rendered.index("Reading memory")


def test_streaming_display_keeps_narration_visible_while_processing_tool_result():
    """Completed tools still block while their result is being processed."""
    narration = "Here is the answer."
    renderable = create_streaming_display(
        response_text=narration,
        narrated_response_end=len(narration),
        narration_segments=[(0, narration)],
        tool_calls=[
            {
                "id": "tc1",
                "name": "read_file",
                "args": {"path": "/memories/profile/USER_PROFILE.md"},
            }
        ],
        tool_results=[
            {
                "id": "tc1",
                "name": "read_file",
                "content": "# User profile\n\n- Likes concise updates.",
            }
        ],
        is_processing=True,
    )

    rendered = _render_text(renderable)

    assert "Here is the answer." in rendered
    assert "Reading memory" in rendered
    assert "Analyzing results" in rendered
    assert rendered.index("Here is the answer.") < rendered.index("Reading memory")


def test_streaming_display_pairs_root_tool_results_by_id():
    """Out-of-order same-name root tool results should not pair by list index."""
    renderable = create_streaming_display(
        tool_calls=[
            {
                "id": "tc-a",
                "name": "execute",
                "args": {"command": "python a.py"},
            },
            {
                "id": "tc-b",
                "name": "execute",
                "args": {"command": "python b.py"},
            },
        ],
        tool_results=[
            {
                "id": "tc-b",
                "name": "execute",
                "content": "Error: result from b",
            }
        ],
    )

    rendered = _render_text(renderable)

    assert "execute(python a.py)" in rendered
    assert "execute(python b.py)" in rendered
    assert rendered.index("execute(python a.py)") < rendered.index("Running")
    assert rendered.index("execute(python b.py)") < rendered.index("result from b")


def test_streaming_display_keeps_narration_visible_with_pending_normal_tool():
    """Ordinary tools use the same pending-tool behavior as memory reads."""
    narration = "I will inspect the files first."
    renderable = create_streaming_display(
        response_text=narration,
        narrated_response_end=len(narration),
        narration_segments=[(0, narration)],
        tool_calls=[
            {
                "id": "tc1",
                "name": "execute",
                "args": {"command": "rg -n TODO ."},
            }
        ],
        tool_results=[],
    )

    rendered = _render_text(renderable)

    assert "execute(rg -n TODO .)" in rendered
    assert "Running" in rendered
    assert "I will inspect the files first." in rendered
    assert rendered.index("I will inspect the files first.") < rendered.index(
        "execute(rg -n TODO .)"
    )


def test_streaming_display_keeps_narration_separate_when_answer_streams():
    """Post-tool answers should not re-render pre-tool narration as Markdown."""
    narration = "I will inspect the files first.\n"
    answer = "The delayed check completed."
    renderable = create_streaming_display(
        response_text=f"{narration}{answer}",
        latest_text=answer,
        narrated_response_end=len(narration),
        narration_segments=[(0, narration)],
        tool_calls=[
            {
                "id": "tc1",
                "name": "execute",
                "args": {"command": "python check.py"},
            }
        ],
        tool_results=[
            {
                "id": "tc1",
                "name": "execute",
                "content": "check complete",
            }
        ],
        response_markdown=Markdown("SHOULD NOT RENDER"),
    )

    rendered = _render_text(renderable)

    assert "I will inspect the files first." in rendered
    assert "The delayed check completed." in rendered
    assert "SHOULD NOT RENDER" not in rendered
    assert rendered.index("I will inspect the files first.") < rendered.index(
        "execute(python check.py)"
    )
    assert rendered.index("execute(python check.py)") < rendered.index(
        "The delayed check completed."
    )


def test_streaming_display_keeps_narration_separate_in_final_frame():
    """The final Rich frame should preserve narration without one concat block."""
    narration = "I will inspect the files first.\n"
    answer = "The delayed check completed."
    renderable = create_streaming_display(
        response_text=f"{narration}{answer}",
        latest_text=answer,
        narrated_response_end=len(narration),
        narration_segments=[(0, narration)],
        tool_calls=[
            {
                "id": "tc1",
                "name": "execute",
                "args": {"command": "python check.py"},
            }
        ],
        tool_results=[
            {
                "id": "tc1",
                "name": "execute",
                "content": "check complete",
            }
        ],
        is_final=True,
        response_markdown=Markdown("SHOULD NOT RENDER"),
    )

    rendered = _render_text(renderable)

    assert "I will inspect the files first." in rendered
    assert "The delayed check completed." in rendered
    assert "SHOULD NOT RENDER" not in rendered
    assert rendered.index("I will inspect the files first.") < rendered.index(
        "execute(python check.py)"
    )
    assert rendered.index("execute(python check.py)") < rendered.index(
        "The delayed check completed."
    )


def test_streaming_display_preserves_narration_for_pending_final_tool():
    """Stopped/error final frames keep narration attached to pending tools."""
    narration = "I will inspect the files first.\n"
    answer = "[Stopped.]"
    renderable = create_streaming_display(
        response_text=f"{narration}{answer}",
        latest_text=answer,
        narrated_response_end=len(narration),
        narration_segments=[(0, narration)],
        tool_calls=[
            {
                "id": "tc1",
                "name": "execute",
                "args": {"command": "sleep 30"},
            }
        ],
        tool_results=[],
        is_final=True,
    )

    rendered = _render_text(renderable)

    assert "I will inspect the files first." in rendered
    assert "execute(sleep 30)" in rendered
    assert "[Stopped.]" in rendered
    assert rendered.index("I will inspect the files first.") < rendered.index(
        "execute(sleep 30)"
    )
    assert rendered.index("execute(sleep 30)") < rendered.index("[Stopped.]")


def test_streaming_display_interleaves_multiple_narration_segments():
    """Multiple narrated segments should stay attached to their following tools."""
    first = "I will inspect the files first.\n"
    second = "I found one file, now I will run it.\n"
    answer = "The delayed check completed."
    renderable = create_streaming_display(
        response_text=f"{first}{second}{answer}",
        latest_text=answer,
        narrated_response_end=len(first) + len(second),
        narration_segments=[
            (0, first),
            (1, second),
        ],
        tool_calls=[
            {
                "id": "tc1",
                "name": "execute",
                "args": {"command": "rg -n delayed ."},
            },
            {
                "id": "tc2",
                "name": "execute",
                "args": {"command": "python check.py"},
            },
        ],
        tool_results=[
            {
                "id": "tc1",
                "name": "execute",
                "content": "check.py",
            },
            {
                "id": "tc2",
                "name": "execute",
                "content": "check complete",
            },
        ],
        is_final=True,
    )

    rendered = _render_text(renderable)

    assert rendered.index("I will inspect the files first.") < rendered.index(
        "execute(rg -n delayed .)"
    )
    assert rendered.index("execute(rg -n delayed .)") < rendered.index(
        "I found one file, now I will run it."
    )
    assert rendered.index("I found one file, now I will run it.") < rendered.index(
        "execute(python check.py)"
    )
    assert rendered.index("execute(python check.py)") < rendered.index(
        "The delayed check completed."
    )


def test_streaming_display_preserves_narration_for_collapsed_completed_tool():
    """Live collapsed completed summaries keep narration from hidden tools."""
    narration = "I will check the early result.\n"
    tool_calls = [
        {
            "id": f"tc{i}",
            "name": "execute",
            "args": {"command": f"python step_{i}.py"},
        }
        for i in range(5)
    ]
    tool_results = [
        {
            "id": f"tc{i}",
            "name": "execute",
            "content": f"step {i} complete",
        }
        for i in range(5)
    ]

    renderable = create_streaming_display(
        response_text=narration,
        narrated_response_end=len(narration),
        narration_segments=[(0, narration)],
        tool_calls=tool_calls,
        tool_results=tool_results,
    )

    rendered = _render_text(renderable)

    assert "I will check the early result." in rendered
    assert "1 completed" in rendered
    assert "execute(python step_0.py)" not in rendered
    assert rendered.index("I will check the early result.") < rendered.index(
        "1 completed"
    )


def test_streaming_display_preserves_narration_for_collapsed_running_tool():
    """Live collapsed running summaries keep narration from hidden tools."""
    narration = "I will start the long-running check.\n"
    tool_calls = [
        {
            "id": f"tc{i}",
            "name": "execute",
            "args": {"command": f"sleep {i + 1}"},
        }
        for i in range(4)
    ]

    renderable = create_streaming_display(
        response_text=narration,
        narrated_response_end=len(narration),
        narration_segments=[(0, narration)],
        tool_calls=tool_calls,
        tool_results=[],
    )

    rendered = _render_text(renderable)

    assert "I will start the long-running check." in rendered
    assert "1 more running" in rendered
    assert "execute(sleep 1)" not in rendered
    assert rendered.index("I will start the long-running check.") < rendered.index(
        "1 more running"
    )


def test_streaming_display_preserves_task_narration_while_subagent_runs():
    """Narration before a task call should stay attached to the task section."""
    narration = "I'll ask a specialist to inspect this.\n"
    subagent = SubAgentState("code-agent", "inspect this", "task:code", "task1")
    subagent.is_active = True
    subagent.add_tool_call("execute", {"command": "rg -n TODO ."}, "sa1")

    renderable = create_streaming_display(
        response_text=narration,
        narrated_response_end=len(narration),
        narration_segments=[(0, narration)],
        tool_calls=[
            {
                "id": "task1",
                "name": "task",
                "args": {
                    "subagent_type": "code-agent",
                    "description": "inspect this",
                },
            }
        ],
        tool_results=[],
        subagents=[subagent],
    )

    rendered = _render_text(renderable)

    assert "I'll ask a specialist to inspect this." in rendered
    assert "Cooking with code-agent" in rendered
    assert "execute(rg -n TODO .)" in rendered
    assert rendered.index("I'll ask a specialist to inspect this.") < rendered.index(
        "Cooking with code-agent"
    )


def test_streaming_display_orders_final_task_narration_by_tool_index():
    """Task narration should not move after regular-tool narration in final frames."""
    first = "I'll ask a specialist to inspect this.\n"
    second = "Now I will run the result locally.\n"
    answer = "The local run passed."
    subagent = SubAgentState("code-agent", "inspect this", "task:code", "task1")
    subagent.is_active = False
    subagent.add_tool_call("execute", {"command": "rg -n TODO ."}, "sa1")
    subagent.add_tool_result("execute", "todo.py", True, "sa1")

    renderable = create_streaming_display(
        response_text=f"{first}{second}{answer}",
        latest_text=answer,
        narrated_response_end=len(first) + len(second),
        narration_segments=[
            (0, first),
            (1, second),
        ],
        tool_calls=[
            {
                "id": "task1",
                "name": "task",
                "args": {
                    "subagent_type": "code-agent",
                    "description": "inspect this",
                },
            },
            {
                "id": "tc2",
                "name": "execute",
                "args": {"command": "python todo.py"},
            },
        ],
        tool_results=[
            {
                "id": "task1",
                "name": "task",
                "content": "todo.py",
            },
            {
                "id": "tc2",
                "name": "execute",
                "content": "passed",
            },
        ],
        subagents=[subagent],
        is_final=True,
    )

    rendered = _render_text(renderable)

    assert rendered.index("I'll ask a specialist to inspect this.") < rendered.index(
        "Cooking with code-agent"
    )
    assert rendered.index("Cooking with code-agent") < rendered.index(
        "Now I will run the result locally."
    )
    assert rendered.index("Now I will run the result locally.") < rendered.index(
        "execute(python todo.py)"
    )
    assert rendered.index("execute(python todo.py)") < rendered.index(
        "The local run passed."
    )


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
        cast(Any, msg).query_one = MagicMock(return_value=fake_md)
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
