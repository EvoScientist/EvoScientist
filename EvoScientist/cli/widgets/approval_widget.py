"""Textual widget for HITL (Human-in-the-Loop) approval prompts.

Shows each action with forced-confirmation warnings for dangerous patterns.
Keyboard-driven: y/n/a quick keys, arrow keys, Enter, e to expand, Esc to reject.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Container  # type: ignore[import-untyped]
from textual.message import Message  # type: ignore[import-untyped]
from textual.widget import Widget  # type: ignore[import-untyped]
from textual.widgets import Static  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult

_COMMAND_TRUNCATE_LENGTH: int = 120


class ApprovalWidget(Widget):
    """Widget that displays pending tool approvals and collects user decisions."""

    can_focus = True
    can_focus_children = False

    DEFAULT_CSS = """
    ApprovalWidget {
        height: auto;
        max-height: 20;
        margin: 1 0;
        padding: 0 1;
        background: $surface;
        border: solid $warning;
    }
    ApprovalWidget .approval-title { height: 1; text-style: bold; color: $warning; }
    ApprovalWidget .approval-command { height: auto; margin: 0 0 0 2; }
    ApprovalWidget .approval-warning { height: 1; margin: 0 0 0 2; color: $error; text-style: bold; }
    ApprovalWidget .approval-options { height: auto; }
    ApprovalWidget .approval-option { height: 1; padding: 0 1; }
    ApprovalWidget .approval-option-selected { background: $primary; text-style: bold; }
    ApprovalWidget .approval-help { height: 1; color: $text-muted; text-style: italic; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("k", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("j", "move_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("1", "select_approve", "Approve", show=False),
        Binding("y", "select_approve", "Approve", show=False),
        Binding("2", "select_reject", "Reject", show=False),
        Binding("n", "select_reject", "Reject", show=False),
        Binding("3", "select_auto", "Auto-approve", show=False),
        Binding("a", "select_auto", "Auto-approve", show=False),
        Binding("e", "toggle_expand", "Expand", show=False),
        Binding("escape", "select_reject", "Reject", show=False),
    ]

    class Decided(Message):
        def __init__(
            self,
            decisions: list[dict[str, Any]] | None,
            auto_approve_session: bool = False,
        ) -> None:
            super().__init__()
            self.decisions = decisions
            self.auto_approve_session = auto_approve_session

    def __init__(self, action_requests: list, **kwargs) -> None:
        super().__init__(**kwargs)
        self._action_requests = action_requests
        self._selected = 0
        self._expanded = False
        self._options: list[str] = []
        self._option_widgets: list[Static] = []
        self._title_widget: Static | None = None
        self._command_widget: Static | None = None
        self._warning_widget: Static | None = None

    def compose(self) -> ComposeResult:
        self._option_widgets = []
        self._title_widget = Static("", classes="approval-title")
        yield self._title_widget
        self._command_widget = Static("", classes="approval-command")
        yield self._command_widget
        self._warning_widget = Static("", classes="approval-warning")
        yield self._warning_widget

        with Container(classes="approval-options"):
            for _ in range(3):
                widget = Static("", classes="approval-option")
                self._option_widgets.append(widget)
                yield widget

        self._help_widget = Static("", classes="approval-help")
        yield self._help_widget

    def on_mount(self) -> None:
        self._refresh_display()
        self.focus()

    def _refresh_display(self) -> None:
        from EvoScientist.backends import check_forced_confirmation

        count = len(self._action_requests)
        title = f">>> {count} Tool Call{'s' if count > 1 else ''} Require{'s' if count == 1 else ''} Approval <<<"
        if count == 1:
            name = (
                self._action_requests[0].get("name", "")
                if isinstance(self._action_requests[0], dict)
                else ""
            )
            title = f">>> {name} Requires Approval <<<"
        if self._title_widget:
            self._title_widget.update(title)

        # Build command display (all actions)
        lines: list[str] = []
        forced_reason = None
        for req in self._action_requests:
            name = (
                req.get("name", "")
                if isinstance(req, dict)
                else getattr(req, "name", "")
            )
            args = (
                req.get("args", {})
                if isinstance(req, dict)
                else getattr(req, "args", {})
            )
            command = (
                args.get("command", args.get("path", ""))
                if isinstance(args, dict)
                else ""
            )
            if command:
                cmd_str = str(command)
                reason = check_forced_confirmation(cmd_str)
                if reason and not forced_reason:
                    forced_reason = reason
                if not self._expanded and len(cmd_str) > _COMMAND_TRUNCATE_LENGTH:
                    cmd_str = (
                        cmd_str[:_COMMAND_TRUNCATE_LENGTH] + "... [dim](e=expand)[/dim]"
                    )
                lines.append(f"[bold #f59e0b]{cmd_str}[/bold #f59e0b]")
            else:
                lines.append(f"[bold]{name}[/bold]")

        if self._command_widget:
            self._command_widget.update("\n".join(lines))

        if self._warning_widget:
            if forced_reason:
                self._warning_widget.update(f"\u26a0 {forced_reason}")
                self._warning_widget.display = True
            else:
                self._warning_widget.update("")
                self._warning_widget.display = False

        if self._help_widget:
            if forced_reason:
                self._help_widget.update("↑/↓ navigate · Enter select · y/n quick keys · e expand · Esc reject")
            else:
                self._help_widget.update("↑/↓ navigate · Enter select · y/n/a quick keys · e expand · Esc reject")

        n = len(self._action_requests)
        if forced_reason:
            self._options = ["1. Approve (y)", "2. Reject (n)"]
        elif n == 1:
            self._options = [
                "1. Approve (y)",
                "2. Reject (n)",
                "3. Auto-approve for this session (a)",
            ]
        else:
            self._options = [
                f"1. Approve all {n} (y)",
                f"2. Reject all {n} (n)",
                "3. Auto-approve for this session (a)",
            ]

        self._update_options()

    def _update_options(self) -> None:
        for i, widget in enumerate(self._option_widgets):
            if i < len(self._options):
                cursor = "▸ " if i == self._selected else "  "
                widget.update(f"{cursor}{self._options[i]}")
                widget.remove_class("approval-option-selected")
                if i == self._selected:
                    widget.add_class("approval-option-selected")
                widget.display = True
            else:
                widget.update("")
                widget.display = False

    def action_move_up(self) -> None:
        self._selected = (self._selected - 1) % len(self._options)
        self._update_options()

    def action_move_down(self) -> None:
        self._selected = (self._selected + 1) % len(self._options)
        self._update_options()

    def action_select(self) -> None:
        self._handle_selection(self._selected)

    def action_select_approve(self) -> None:
        self._handle_selection(0)

    def action_select_reject(self) -> None:
        self._handle_selection(1)

    def action_select_auto(self) -> None:
        if len(self._options) > 2:
            self._handle_selection(2)

    def action_toggle_expand(self) -> None:
        self._expanded = not self._expanded
        self._refresh_display()

    def _handle_selection(self, option: int) -> None:
        n = len(self._action_requests) or 1
        if option == 0:
            self.post_message(self.Decided([{"type": "approve"} for _ in range(n)]))
        elif option == 1:
            self.post_message(
                self.Decided(
                    [
                        {"type": "reject", "message": "Rejected by user"}
                        for _ in range(n)
                    ]
                )
            )
        elif option == 2:
            self.post_message(
                self.Decided(
                    [{"type": "approve"} for _ in range(n)],
                    auto_approve_session=True,
                )
            )

    def on_blur(self, event: events.Blur) -> None:
        self.call_after_refresh(self.focus)
