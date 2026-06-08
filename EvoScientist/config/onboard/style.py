"""Shared visual elements: Console, prompt styles, checkbox helper, headers.

These are used across all wizard steps and helpers. Kept in one place so a
single import in each submodule covers UI styling.
"""

from __future__ import annotations

import questionary
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


# =============================================================================
# Wizard Style
# =============================================================================

WIZARD_STYLE = Style.from_dict(
    {
        "qmark": "fg:#00bcd4 bold",  # Cyan question mark
        "question": "bold",  # Bold question text
        "answer": "fg:#4caf50 bold",  # Green selected answer
        "pointer": "fg:#4caf50",  # Green pointer (»)
        "highlighted": "noreverse bold",  # No background, bold text
        "selected": "fg:#4caf50 bold",  # Green ● indicator
        "separator": "fg:#6c6c6c",  # Dim separator
        "disabled": "fg:#858585",  # Dim disabled indicator (-)
        "instruction": "fg:#858585",  # Dim instructions
        "text": "fg:#858585",  # Dim gray ○ and unselected text
    }
)

CONFIRM_STYLE = Style.from_dict(
    {
        "qmark": "fg:#e69500 bold",  # Orange warning mark (!)
        "question": "bold",
        "answer": "fg:#4caf50 bold",
        "instruction": "fg:#858585",
        "text": "",
    }
)

QMARK = "❯"

# Installed-item indicator style for disabled checkbox choices.
_INSTALLED_INDICATOR = ("fg:#4caf50", "✓ ")


def _checkbox_ask(choices, message: str, **kwargs):
    """``questionary.checkbox`` that renders disabled items with ✓ instead of ``-``.

    Temporarily patches the rendering so the hard-coded ``"- "`` prefix for
    disabled choices is replaced by a green ``"✓ "`` — keeping alignment with
    the ``○`` indicator of normal choices.
    """
    from questionary.prompts.common import InquirerControl

    original = InquirerControl._get_choice_tokens

    def _patched(self):
        tokens = original(self)
        return [
            _INSTALLED_INDICATOR
            if cls == "class:disabled" and text == "- "
            else (cls, text)
            for cls, text in tokens
        ]

    InquirerControl._get_choice_tokens = _patched
    try:
        return questionary.checkbox(
            message,
            choices=choices,
            style=WIZARD_STYLE,
            qmark=QMARK,
            **kwargs,
        ).ask()
    finally:
        InquirerControl._get_choice_tokens = original


# =============================================================================
# Headers / per-step result printers
# =============================================================================


def _print_header() -> None:
    """Print the wizard header."""
    console.print()
    console.print(
        Panel.fit(
            Text.from_markup(
                "[bold cyan]EvoScientist Setup Wizard[/bold cyan]\n\n"
                "This wizard will help you configure EvoScientist.\n"
                "Press Ctrl+C at any time to cancel."
            ),
            border_style="cyan",
        )
    )
    console.print()


_SECTION_WIDTH = 53  # ~ the Setup Wizard panel width (not the full terminal)


def _print_section(title: str) -> None:
    """Print a compact bold-cyan section divider, e.g. ──── Pilot ────.

    Sized to roughly the Setup Wizard panel width and styled like the section
    markers used elsewhere in the CLI, rather than a full-width rule.
    """
    label = f" {title} "
    pad = max(2, _SECTION_WIDTH - len(label))
    left = pad // 2
    console.print(f"[bold cyan]{'─' * left}{label}{'─' * (pad - left)}[/bold cyan]")


def _print_step_result(step_name: str, value: str, success: bool = True) -> None:
    """Print a completed step result inline.

    Args:
        step_name: Name of the step.
        value: The selected/entered value.
        success: Whether the step was successful (affects icon).
    """
    icon = "[green]✓[/green]" if success else "[red]✗[/red]"
    console.print(f"  {icon} [bold]{step_name}:[/bold] [cyan]{value}[/cyan]")


def _print_step_skipped(step_name: str, reason: str = "kept current") -> None:
    """Print a skipped step result inline.

    Args:
        step_name: Name of the step.
        reason: Reason for skipping.
    """
    console.print(f"  [dim]○ {step_name}: {reason}[/dim]")


# =============================================================================
# Step Functions
# =============================================================================
