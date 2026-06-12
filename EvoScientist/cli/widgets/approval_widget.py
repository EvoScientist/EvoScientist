"""Textual widget for HITL (Human-in-the-Loop) approval prompts.

Follows DeepAgents CLI pattern: keyboard-driven menu with Static text options,
not Button widgets. Compact height: auto layout.
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

# Max length for truncated shell command display
_COMMAND_TRUNCATE_LENGTH: int = 120


class ApprovalWidget(Widget):
    """Widget that displays pending tool approvals and collects user decisions.

    Keyboard-driven: y/n/a or 1/2/3 quick keys, arrow keys to navigate,
    Enter to confirm. Posts ``ApprovalWidget.Decided`` when user chooses.
    """

    can_focus = True
    can_focus_children = False

    DEFAULT_CSS = """
    ApprovalWidget {
        height: auto;
        max-height: 12;
        margin: 1 0;
        padding: 0 1;
        background: $surface;
        border: solid $warning;
    }
    ApprovalWidget .approval-title {
        height: 1;
        text-style: bold;
        color: $warning;
    }
    ApprovalWidget .approval-command {
        height: 1;
        margin: 0 0 0 2;
    }
    ApprovalWidget .approval-options {
        height: auto;
    }
    ApprovalWidget .approval-option {
        height: 1;
        padding: 0 1;
    }
    ApprovalWidget .approval-option-selected {
        background: $primary;
        text-style: bold;
    }
    ApprovalWidget .approval-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
    }
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
        Binding("escape", "select_reject", "Reject", show=False),
    ]

    class Decided(Message):
        """Posted when the user approves, rejects, or auto-approves."""

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
        self._option_widgets: list[Static] = []

    def compose(self) -> ComposeResult:
        self._option_widgets = []
        count = len(self._action_requests)
        if count == 1:
            name = self._action_requests[0].get("name", "")
            title = f">>> {name} Requires Approval <<<"
        else:
            title = f">>> {count} Tool Calls Require Approval <<<"
        yield Static(title, classes="approval-title")

        # Show each action request as a compact line
        for req in self._action_requests:
            name = req.get("name", "")
            args = req.get("args", {})
            if isinstance(args, dict):
                command = args.get("command", args.get("path", ""))
            else:
                command = ""
            if command:
                cmd_str = str(command)
                if len(cmd_str) > _COMMAND_TRUNCATE_LENGTH:
                    cmd_str = cmd_str[:_COMMAND_TRUNCATE_LENGTH] + "..."
                label = f"[bold #f59e0b]{cmd_str}[/bold #f59e0b]"
            else:
                label = f"[bold]{name}[/bold]"
            yield Static(label, classes="approval-command")

        # Options as Static text (not buttons)
        with Container(classes="approval-options"):
            for _ in range(3):
                widget = Static("", classes="approval-option")
                self._option_widgets.append(widget)
                yield widget

        yield Static(
            "↑/↓ navigate · Enter select · y/n/a quick keys · Esc reject",
            classes="approval-help",
        )

    def on_mount(self) -> None:
        self._update_options()
        self.focus()

    def _update_options(self) -> None:
        n = len(self._action_requests)
        if n == 1:
            options = [
                "1. Approve (y)",
                "2. Reject (n)",
                "3. Auto-approve for this session (a)",
            ]
        else:
            options = [
                f"1. Approve all {n} (y)",
                f"2. Reject all {n} (n)",
                "3. Auto-approve for this session (a)",
            ]

        for i, (text, widget) in enumerate(
            zip(options, self._option_widgets, strict=True)
        ):
            cursor = "▸ " if i == self._selected else "  "
            widget.update(f"{cursor}{text}")
            widget.remove_class("approval-option-selected")
            if i == self._selected:
                widget.add_class("approval-option-selected")

    def action_move_up(self) -> None:
        self._selected = (self._selected - 1) % 3
        self._update_options()

    def action_move_down(self) -> None:
        self._selected = (self._selected + 1) % 3
        self._update_options()

    def action_select(self) -> None:
        self._handle_selection(self._selected)

    def action_select_approve(self) -> None:
        self._selected = 0
        self._update_options()
        self._handle_selection(0)

    def action_select_reject(self) -> None:
        self._selected = 1
        self._update_options()
        self._handle_selection(1)

    def action_select_auto(self) -> None:
        self._selected = 2
        self._update_options()
        self._handle_selection(2)

    def _handle_selection(self, option: int) -> None:
        n = len(self._action_requests) or 1
        if option == 0:
            self.post_message(self.Decided([{"type": "approve"} for _ in range(n)]))
        elif option == 2:
            self.post_message(
                self.Decided(
                    [{"type": "approve"} for _ in range(n)],
                    auto_approve_session=True,
                )
            )
        else:
            self.post_message(self.Decided(None))

    def on_blur(self, event: events.Blur) -> None:
        """Re-focus to keep focus trapped until decision is made."""
        self.call_after_refresh(self.focus)
