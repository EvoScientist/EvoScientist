"""Tool call widget with status lifecycle and collapsible output."""

from __future__ import annotations

from rich.text import Text
from textual.containers import Vertical
from textual.events import Click
from textual.widgets import Static

from ...stream.diff_format import build_edit_diff
from ...stream.utils import format_tool_compact

_SPINNER_FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"

# Output is collapsed when it exceeds these limits
_COLLAPSE_LINES = 6
_COLLAPSE_CHARS = 400


class ToolCallWidget(Vertical):
    """Displays a single tool call with running spinner → result.

    Lifecycle: ``running`` → ``success`` | ``error`` | ``interrupted`` | ``rejected``

    Usage::

        w = ToolCallWidget("read_file", {"path": "/foo.py"}, tool_id="abc")
        await container.mount(w)
        # ... later ...
        w.set_success("[OK] 42 lines")
    """

    DEFAULT_CSS = """
    ToolCallWidget {
        height: auto;
        padding: 0 0;
    }
    ToolCallWidget .tool-header {
        height: auto;
    }
    ToolCallWidget .tool-status {
        height: auto;
        padding: 0 0 0 2;
    }
    ToolCallWidget .tool-output {
        height: auto;
        padding: 0 0 0 4;
        color: #9ca3af;
        display: none;
    }
    ToolCallWidget .tool-output.--visible {
        display: block;
    }
    """

    def __init__(
        self,
        tool_name: str,
        tool_args: dict | None = None,
        tool_id: str = "",
    ) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._tool_args = tool_args or {}
        self._tool_id = tool_id
        self._status = "running"
        self._result_content = ""
        self._diff_markup: str | None = None  # cached full diff for toggle
        self._frame = 0
        self._elapsed = 0.0
        self._timer_handle = None
        self._collapsed = True

    @property
    def tool_id(self) -> str:
        return self._tool_id

    @property
    def tool_name(self) -> str:
        return self._tool_name

    def compose(self):
        yield Static("", classes="tool-header")
        yield Static("", classes="tool-status")
        yield Static("", classes="tool-output")

    def on_mount(self) -> None:
        self._timer_handle = self.set_interval(0.1, self._tick)
        self._render_header()
        self._render_status()

    def _tick(self) -> None:
        if self._status == "running":
            self._frame = (self._frame + 1) % len(_SPINNER_FRAMES)
            self._elapsed += 0.1
            self._render_status()

    def _render_header(self) -> None:
        compact = format_tool_compact(self._tool_name, self._tool_args)
        header = self.query_one(".tool-header", Static)
        line = Text()
        if self._status == "running":
            line.append("\u25cf ", style="bold yellow")
            line.append(compact, style="bold yellow")
        elif self._status == "success":
            line.append("\u2713 ", style="bold green")
            line.append(compact, style="bold green")
        elif self._status == "interrupted":
            line.append("\u25a0 ", style="bold yellow")
            line.append(compact, style="bold yellow dim")
        elif self._status == "rejected":
            line.append("\u25a0 ", style="bold yellow")
            line.append(compact, style="bold yellow dim")
        else:
            line.append("\u2717 ", style="bold red")
            line.append(compact, style="bold red")
        header.update(line)

    def _render_status(self) -> None:
        status_w = self.query_one(".tool-status", Static)
        if self._status == "running":
            char = _SPINNER_FRAMES[self._frame]
            secs = int(self._elapsed)
            status_w.update(Text(f"{char} Running... ({secs}s)", style="yellow dim"))
        elif self._status == "success":
            # Show first line of result as summary
            summary = self._result_summary()
            status_w.update(Text(f"\u2713 {summary}", style="green dim"))
        elif self._status == "interrupted":
            status_w.update(Text("\u25a0 interrupted", style="yellow dim"))
        elif self._status == "rejected":
            status_w.update(Text("\u25a0 rejected", style="yellow dim"))
        else:
            summary = self._result_summary()
            status_w.update(Text(f"\u2717 {summary}", style="red dim"))

    def _result_summary(self) -> str:
        """One-line summary of the result."""
        if not self._result_content:
            return "done"
        first_line = self._result_content.strip().split("\n")[0]
        if len(first_line) > 60:
            first_line = first_line[:57] + "\u2026"
        return first_line

    def _should_collapse(self) -> bool:
        lines = self._result_content.strip().split("\n")
        return (
            len(lines) > _COLLAPSE_LINES or len(self._result_content) > _COLLAPSE_CHARS
        )

    def set_success(self, content: str) -> None:
        """Mark tool call as successfully completed."""
        self._status = "success"
        self._result_content = content
        self._stop_timer()
        with self.app.batch_update():
            self._render_header()
            self._render_status()
            self._render_output()

    def set_interrupted(self) -> None:
        """Mark tool call as interrupted/cancelled."""
        self._status = "interrupted"
        self._stop_timer()
        with self.app.batch_update():
            self._render_header()
            self._render_status()

    def set_rejected(self) -> None:
        """Mark tool call as rejected by HITL approval."""
        self._status = "rejected"
        self._stop_timer()
        with self.app.batch_update():
            self._render_header()
            self._render_status()

    def set_error(self, content: str) -> None:
        """Mark tool call as failed."""
        self._status = "error"
        self._result_content = content
        self._stop_timer()
        with self.app.batch_update():
            self._render_header()
            self._render_status()
            self._render_output()

    def _render_output(self) -> None:
        output_w = self.query_one(".tool-output", Static)
        if not self._result_content.strip():
            return
        # Diff rendering for edit_file (never truncates — collapses instead)
        if (
            self._tool_name == "edit_file"
            and self._status == "success"
            and self._tool_args
        ):
            old_str = self._tool_args.get("old_string", "")
            new_str = self._tool_args.get("new_string", "")
            path = self._tool_args.get("path", self._tool_args.get("file_path", ""))
            if old_str and new_str and old_str != new_str:
                diff_markup = build_edit_diff(path, old_str, new_str)
                if diff_markup:
                    self._diff_markup = diff_markup
                    if diff_markup.count("\n") > _COLLAPSE_LINES:
                        # Long diff: collapse, click to expand
                        self._collapsed = True
                        output_w.update(
                            Text("  [click to expand diff]", style="dim italic"),
                        )
                    else:
                        # Short diff: show inline
                        self._collapsed = False
                        output_w.update(diff_markup)
                    output_w.add_class("--visible")
                    return
        if self._status == "error" or not self._should_collapse():
            # Show full output for errors or short output
            self._collapsed = False
            style = "red dim" if self._status == "error" else "dim"
            content = self._result_content.strip()
            if len(content) > 800:
                content = content[:800] + "\n... (truncated)"
            output_w.update(Text(content, style=style))
            output_w.add_class("--visible")
        else:
            # Collapsed — show hint, click to expand
            self._collapsed = True
            output_w.update(
                Text("  [click to expand output]", style="dim italic"),
            )
            output_w.add_class("--visible")

    def on_click(self, event: Click) -> None:
        """Toggle collapsed output on click."""
        if self._status == "running" or not self._result_content.strip():
            return
        output_w = self.query_one(".tool-output", Static)

        # Diff toggle (separate path — _should_collapse checks plain text,
        # not the diff markup, so we handle it independently)
        if self._diff_markup is not None:
            if self._collapsed:
                self._collapsed = False
                output_w.update(self._diff_markup)
            else:
                self._collapsed = True
                output_w.update(
                    Text("  [click to expand diff]", style="dim italic"),
                )
            return

        if not self._should_collapse():
            return  # Short output is always visible, nothing to toggle
        if self._collapsed:
            # Expand
            self._collapsed = False
            content = self._result_content.strip()
            if len(content) > 2000:
                content = content[:2000] + "\n... (truncated)"
            style = "red dim" if self._status == "error" else "dim"
            output_w.update(Text(content, style=style))
        else:
            # Collapse back
            self._collapsed = True
            output_w.update(
                Text("  [click to expand output]", style="dim italic"),
            )

    def _stop_timer(self) -> None:
        if self._timer_handle is not None:
            self._timer_handle.stop()
            self._timer_handle = None
