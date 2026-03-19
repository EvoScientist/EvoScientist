"""Interactive ask_user widget for Textual TUI.

Shows one question at a time with a progress indicator.  Keyboard-driven
like ApprovalWidget: all bindings on the top-level widget, compact layout.

The widget is self-contained: the main ``#prompt`` Input should be
**disabled** while this widget is mounted so it cannot steal focus.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from rich.markup import escape as escape_markup
from textual.binding import Binding, BindingType
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult

logger = logging.getLogger(__name__)

OTHER_CHOICE_LABEL = "Other (type your answer)"
_CURSOR = "▸"


class AskUserWidget(Widget):
    """Interactive widget for asking the user questions one at a time.

    Extends Widget (not Container) to avoid built-in scroll behavior
    that captures arrow keys.  Same pattern as ApprovalWidget.
    """

    can_focus = True
    can_focus_children = True  # needed for text Input

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("k", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("j", "move_down", "Down", show=False),
        Binding("enter", "confirm", "Confirm", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    DEFAULT_CSS = """
    AskUserWidget {
        height: auto;
        margin: 1 0;
        padding: 0 1;
        background: $surface;
        border: solid $success;
    }
    AskUserWidget .ask-title {
        height: 1;
        text-style: bold;
        color: $success;
    }
    AskUserWidget .ask-question-text {
        height: auto;
        margin: 0 0 0 1;
        text-style: bold;
    }
    AskUserWidget .ask-choice {
        height: 1;
        padding: 0 2;
        margin: 0;
    }
    AskUserWidget .ask-choice-selected {
        background: $primary;
        text-style: bold;
    }
    AskUserWidget .ask-text-input {
        height: auto;
        margin: 0 2;
    }
    AskUserWidget .ask-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin: 0;
    }
    """

    class Answered(Message):
        """Posted when the user submits all answers."""

        def __init__(self, answers: list[str]) -> None:
            super().__init__()
            self.answers = answers

    class Cancelled(Message):
        """Posted when the user cancels the ask_user prompt."""

        def __init__(self) -> None:
            super().__init__()

    def __init__(
        self,
        questions: list[dict],
        id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(id=id or "ask-user-widget", **kwargs)
        self._questions = questions
        self._answers: list[str] = []
        self._current_index = 0
        self._future: asyncio.Future | None = None
        self._submitted = False

        # Current question state
        self._q_type: Literal["text", "multiple_choice"] = "text"
        self._choices: list[dict] = []
        self._required: bool = True
        self._selected_choice: int = 0
        self._is_other: bool = False

        # Widgets (composed once, updated per question)
        self._title_w: Static | None = None
        self._question_w: Static | None = None
        self._choice_widgets: list[Static] = []
        self._text_input: Input | None = None
        self._other_input: Input | None = None
        self._help_w: Static | None = None

    def set_future(self, future: asyncio.Future) -> None:
        """Set the future to resolve when user answers."""
        self._future = future

    def compose(self) -> ComposeResult:
        total = len(self._questions)
        if total == 1:
            title = ">>> Quick check-in from EvoScientist <<<"
        else:
            title = f">>> Question 1/{total} — Quick check-in from EvoScientist <<<"

        self._title_w = Static(title, classes="ask-title")
        yield self._title_w

        self._question_w = Static("", classes="ask-question-text")
        yield self._question_w

        # Pre-create max choice slots (choices + Other = up to ~10)
        # Unused slots stay hidden.
        self._choice_widgets = []
        for _ in range(12):
            cw = Static("", classes="ask-choice")
            cw.display = False
            self._choice_widgets.append(cw)
            yield cw

        self._text_input = Input(
            placeholder="Type your answer...",
            classes="ask-text-input",
        )
        self._text_input.display = False
        yield self._text_input

        self._other_input = Input(
            placeholder="Type your answer...",
            classes="ask-text-input",
        )
        self._other_input.display = False
        yield self._other_input

        self._help_w = Static("", classes="ask-help")
        yield self._help_w

    async def on_mount(self) -> None:
        self._show_question(0)

    def focus_active(self) -> None:
        """Focus the appropriate element for the current question."""
        if self._q_type == "text":
            if self._text_input:
                self._text_input.focus()
        elif self._is_other:
            if self._other_input:
                self._other_input.focus()
        else:
            self.focus()

    # ------------------------------------------------------------------
    # Show a question
    # ------------------------------------------------------------------

    def _show_question(self, index: int) -> None:
        """Populate widgets for question at *index*."""
        q = self._questions[index]
        q_text = q.get("question", "")
        q_type = q.get("type", "text")
        self._choices = q.get("choices", [])
        self._required = q.get("required", True)
        self._q_type = "multiple_choice" if q_type == "multiple_choice" else "text"
        self._selected_choice = 0
        self._is_other = False

        # Title
        total = len(self._questions)
        if self._title_w:
            if total == 1:
                self._title_w.update(">>> Quick check-in from EvoScientist <<<")
            else:
                self._title_w.update(
                    f">>> Question {index + 1}/{total}"
                    " — Quick check-in from EvoScientist <<<"
                )

        # Question text
        suffix = (
            " [dim](required)[/dim]" if self._required else " [dim](optional)[/dim]"
        )
        if self._question_w:
            self._question_w.update(
                f"[bold]{index + 1}. {escape_markup(q_text)}[/bold]{suffix}"
            )

        # Reset all choice slots
        for cw in self._choice_widgets:
            cw.display = False
            cw.remove_class("ask-choice-selected")

        if self._text_input:
            self._text_input.display = False
            self._text_input.value = ""
        if self._other_input:
            self._other_input.display = False
            self._other_input.value = ""

        if self._q_type == "multiple_choice" and self._choices:
            # Show choice options + Other
            for i, choice in enumerate(self._choices):
                if i < len(self._choice_widgets):
                    label = escape_markup(choice.get("value", str(choice)))
                    cursor = f"{_CURSOR} " if i == 0 else "  "
                    self._choice_widgets[i].update(f"{cursor}{label}")
                    self._choice_widgets[i].display = True
                    if i == 0:
                        self._choice_widgets[i].add_class("ask-choice-selected")

            other_idx = len(self._choices)
            if other_idx < len(self._choice_widgets):
                self._choice_widgets[other_idx].update(f"  {OTHER_CHOICE_LABEL}")
                self._choice_widgets[other_idx].display = True

            # Help text
            if self._help_w:
                self._help_w.update("↑/↓ select · Enter confirm · Esc cancel")
            self.focus()
        else:
            # Text input
            if self._text_input:
                self._text_input.display = True
                self._text_input.focus()
            if self._help_w:
                self._help_w.update("Enter confirm · Esc cancel")

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def action_move_up(self) -> None:
        if self._q_type != "multiple_choice":
            return
        if self._is_other and self._other_input and self._other_input.has_focus:
            # Jump back from Other input to choice list
            self._is_other = False
            if self._other_input:
                self._other_input.display = False
            self._selected_choice = len(self._choices)  # stay on Other option
            self._update_choices()
            self.focus()
            return
        total_opts = len(self._choices) + 1  # choices + Other
        self._selected_choice = (self._selected_choice - 1) % total_opts
        self._update_choices()

    def action_move_down(self) -> None:
        if self._q_type != "multiple_choice":
            return
        total_opts = len(self._choices) + 1
        self._selected_choice = (self._selected_choice + 1) % total_opts
        self._update_choices()

    def action_confirm(self) -> None:
        if self._q_type == "multiple_choice":
            is_other = self._selected_choice == len(self._choices)
            if is_other and not self._is_other:
                # Show Other text input
                self._is_other = True
                if self._other_input:
                    self._other_input.display = True
                    self._other_input.focus()
                return
            if is_other and self._is_other:
                # Submit Other answer
                answer = self._other_input.value if self._other_input else ""
                if answer.strip() or not self._required:
                    self._advance(answer)
                return
            # Regular choice
            if self._selected_choice < len(self._choices):
                answer = self._choices[self._selected_choice].get("value", "")
                self._advance(answer)
        else:
            # Text question — Enter on widget (not Input) acts as confirm
            answer = self._text_input.value if self._text_input else ""
            if answer.strip() or not self._required:
                self._advance(answer)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter in text Input widgets."""
        event.stop()
        if event.input is self._text_input:
            answer = self._text_input.value if self._text_input else ""
            if answer.strip() or not self._required:
                self._advance(answer)
        elif event.input is self._other_input:
            answer = self._other_input.value if self._other_input else ""
            if answer.strip() or not self._required:
                self._advance(answer)

    def action_cancel(self) -> None:
        if self._submitted:
            return
        self._submitted = True
        if self._future and not self._future.done():
            self._future.set_result({"type": "cancelled"})
        self.post_message(self.Cancelled())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_choices(self) -> None:
        """Update choice display to reflect current selection."""
        total_opts = len(self._choices) + 1
        for i in range(total_opts):
            if i >= len(self._choice_widgets):
                break
            if i < len(self._choices):
                label = escape_markup(self._choices[i].get("value", ""))
            else:
                label = OTHER_CHOICE_LABEL
            cursor = f"{_CURSOR} " if i == self._selected_choice else "  "
            self._choice_widgets[i].update(f"{cursor}{label}")
            self._choice_widgets[i].remove_class("ask-choice-selected")
            if i == self._selected_choice:
                self._choice_widgets[i].add_class("ask-choice-selected")

    def _advance(self, answer: str) -> None:
        """Record answer, show next question or submit."""
        self._answers.append(answer)
        self._current_index += 1

        if self._current_index >= len(self._questions):
            self._submit()
            return

        self._show_question(self._current_index)

    def _submit(self) -> None:
        if self._submitted:
            return
        self._submitted = True
        if self._future and not self._future.done():
            self._future.set_result({"type": "answered", "answers": self._answers})
        self.post_message(self.Answered(self._answers))

    def on_blur(self, event: events.Blur) -> None:
        """Prevent blur from propagating and dismissing the widget."""
        event.stop()
