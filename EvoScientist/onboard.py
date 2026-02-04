"""Interactive onboarding wizard for EvoScientist.

Guides users through initial setup including API keys, model selection,
workspace settings, and agent parameters. Uses flow-style arrow-key selection UI.
"""

from __future__ import annotations

import os
from typing import Any

import questionary
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import Validator, ValidationError
from questionary import Choice
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .config import (
    EvoScientistConfig,
    load_config,
    save_config,
    get_config_path,
)
from .llm import MODELS, DEFAULT_MODEL

console = Console()


# =============================================================================
# Wizard Style
# =============================================================================

WIZARD_STYLE = Style.from_dict({
    "qmark": "fg:#00bcd4 bold",          # Cyan question mark
    "question": "bold",                   # Bold question text
    "answer": "fg:#4caf50 bold",          # Green selected answer
    "pointer": "fg:#4caf50",             # Green pointer (»)
    "highlighted": "noreverse bold",      # No background, bold text
    "selected": "fg:#4caf50 bold",        # Green ● indicator
    "separator": "fg:#6c6c6c",            # Dim separator
    "instruction": "fg:#858585",          # Dim instructions
    "text": "fg:#858585",                 # Dim gray ○ and unselected text
})

CONFIRM_STYLE = Style.from_dict({
    "qmark": "fg:#e69500 bold",           # Orange warning mark (!)
    "question": "bold",
    "answer": "fg:#4caf50 bold",
    "instruction": "fg:#858585",
    "text": "",
})

STEPS = ["Provider", "API Key", "Model", "Tavily Key", "Workspace", "Parameters"]


# =============================================================================
# Validators
# =============================================================================

class IntegerValidator(Validator):
    """Validates that input is a positive integer."""

    def __init__(self, min_value: int = 1, max_value: int = 100):
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, document) -> None:
        text = document.text.strip()
        if not text:
            return  # Allow empty for default
        try:
            value = int(text)
            if value < self.min_value or value > self.max_value:
                raise ValidationError(
                    message=f"Must be between {self.min_value} and {self.max_value}"
                )
        except ValueError:
            raise ValidationError(message="Must be a valid integer")


class ChoiceValidator(Validator):
    """Validates that input is one of the allowed choices."""

    def __init__(self, choices: list[str], allow_empty: bool = True):
        self.choices = choices
        self.allow_empty = allow_empty

    def validate(self, document) -> None:
        text = document.text.strip().lower()
        if not text and self.allow_empty:
            return
        if text not in [c.lower() for c in self.choices]:
            raise ValidationError(
                message=f"Must be one of: {', '.join(self.choices)}"
            )


# =============================================================================
# API Key Validation
# =============================================================================

def validate_anthropic_key(api_key: str) -> tuple[bool, str]:
    """Validate an Anthropic API key by making a test request.

    Args:
        api_key: The API key to validate.

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        # Make a minimal request to validate the key
        client.models.list()
        return True, "Valid"
    except anthropic.AuthenticationError:
        return False, "Invalid API key"
    except Exception as e:
        return False, f"Error: {e}"


def validate_openai_key(api_key: str) -> tuple[bool, str]:
    """Validate an OpenAI API key by making a test request.

    Args:
        api_key: The API key to validate.

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        # Make a minimal request to validate the key
        client.models.list()
        return True, "Valid"
    except openai.AuthenticationError:
        return False, "Invalid API key"
    except Exception as e:
        return False, f"Error: {e}"


def validate_nvidia_key(api_key: str) -> tuple[bool, str]:
    """Validate an NVIDIA API key by making a test request.

    Args:
        api_key: The API key to validate.

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        llm = ChatNVIDIA(api_key=api_key, model="meta/llama-3.1-8b-instruct")
        llm.available_models
        return True, "Valid"
    except Exception as e:
        error_str = str(e).lower()
        if "401" in error_str or "unauthorized" in error_str or "invalid" in error_str or "authentication" in error_str:
            return False, "Invalid API key"
        return False, f"Error: {e}"


def validate_tavily_key(api_key: str) -> tuple[bool, str]:
    """Validate a Tavily API key by making a test request.

    Args:
        api_key: The API key to validate.

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        # Make a minimal search to validate
        client.search("test", max_results=1)
        return True, "Valid"
    except Exception as e:
        error_str = str(e).lower()
        if "invalid" in error_str or "unauthorized" in error_str or "401" in error_str:
            return False, "Invalid API key"
        return False, f"Error: {e}"


# =============================================================================
# Display Helpers
# =============================================================================

def _print_header() -> None:
    """Print the wizard header."""
    console.print()
    console.print(Panel.fit(
        Text.from_markup(
            "[bold cyan]EvoScientist Setup Wizard[/bold cyan]\n\n"
            "This wizard will help you configure EvoScientist.\n"
            "Press Ctrl+C at any time to cancel."
        ),
        border_style="cyan",
    ))
    console.print()


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

def _step_provider(config: EvoScientistConfig) -> str:
    """Step 1: Select LLM provider.

    Args:
        config: Current configuration.

    Returns:
        Selected provider name.
    """
    choices = [
        Choice(title="Anthropic (Claude models)", value="anthropic"),
        Choice(title="OpenAI (GPT models)", value="openai"),
        Choice(title="NVIDIA (GLM, MiniMax, Kimi, etc.)", value="nvidia"),
    ]

    # Set default based on current config
    default = config.provider if config.provider in ["anthropic", "openai", "nvidia"] else "anthropic"

    provider = questionary.select(
        "Select your LLM provider:",
        choices=choices,
        default=default,
        style=WIZARD_STYLE,
        use_indicator=True,
    ).ask()

    if provider is None:
        raise KeyboardInterrupt()

    return provider


def _step_provider_api_key(
    config: EvoScientistConfig,
    provider: str,
    skip_validation: bool = False,
) -> str | None:
    """Step 2: Enter API key for the selected provider.

    Args:
        config: Current configuration.
        provider: Selected provider name.
        skip_validation: Skip API key validation.

    Returns:
        New API key or None if unchanged.
    """
    if provider == "anthropic":
        key_name = "Anthropic"
        current = config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        validate_fn = validate_anthropic_key
    elif provider == "nvidia":
        key_name = "NVIDIA"
        current = config.nvidia_api_key or os.environ.get("NVIDIA_API_KEY", "")
        validate_fn = validate_nvidia_key
    else:
        key_name = "OpenAI"
        current = config.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        validate_fn = validate_openai_key

    # Show current status inline
    if current:
        display_current = f"***{current[-4:]}"
        hint = f"Current: {display_current}"
    else:
        hint = "Not set"

    # Prompt for new key
    new_key = questionary.password(
        f"Enter {key_name} API key ({hint}, Enter to keep):",
        style=WIZARD_STYLE,
    ).ask()

    if new_key is None:
        raise KeyboardInterrupt()

    new_key = new_key.strip()

    # Determine the key to validate: new input or existing current key
    key_to_validate = new_key if new_key else current

    if not key_to_validate:
        return None  # Nothing to validate

    # Validate the key (new or current)
    if not skip_validation:
        console.print("  [dim]Validating...[/dim]", end="")
        valid, msg = validate_fn(key_to_validate)
        if valid:
            console.print(f"\r  [green]✓ {msg}[/green]      ")
            return new_key if new_key else None  # Return new key or None (keep current)
        else:
            console.print(f"\r  [red]✗ {msg}[/red]      ")
            if not new_key:
                return None  # Current key invalid, but keep it
            # Ask if they want to save anyway
            save_anyway = questionary.confirm(
                "Save anyway?",
                default=False,
                style=WIZARD_STYLE,
            ).ask()
            if save_anyway is None:
                raise KeyboardInterrupt()
            if save_anyway:
                return new_key
            return None
    else:
        return new_key if new_key else None


def _step_model(config: EvoScientistConfig, provider: str) -> str:
    """Step 3: Select model for the provider.

    Args:
        config: Current configuration.
        provider: Selected provider name.

    Returns:
        Selected model name.
    """
    # Get models for the selected provider
    provider_models = [
        name for name, (model_id, p) in MODELS.items()
        if p == provider
    ]

    if not provider_models:
        # Fallback if no models for provider
        console.print(f"  [yellow]No registered models for {provider}[/yellow]")
        model = questionary.text(
            "Enter model name:",
            default=config.model,
            style=WIZARD_STYLE,
        ).ask()
        if model is None:
            raise KeyboardInterrupt()
        return model

    # Create choices with model IDs as hints
    choices = []
    for name in provider_models:
        model_id, _ = MODELS[name]
        choices.append(Choice(title=f"{name} ({model_id})", value=name))

    # Determine default
    if config.model in provider_models:
        default = config.model
    else:
        default = provider_models[0]

    model = questionary.select(
        "Select model:",
        choices=choices,
        default=default,
        style=WIZARD_STYLE,
        use_indicator=True,
    ).ask()

    if model is None:
        raise KeyboardInterrupt()

    return model


def _step_tavily_key(
    config: EvoScientistConfig,
    skip_validation: bool = False,
) -> str | None:
    """Step 4: Enter Tavily API key for web search.

    Args:
        config: Current configuration.
        skip_validation: Skip API key validation.

    Returns:
        New API key or None if unchanged.
    """
    current = config.tavily_api_key or os.environ.get("TAVILY_API_KEY", "")

    # Show current status inline
    if current:
        display_current = f"***{current[-4:]}"
        hint = f"Current: {display_current}"
    else:
        hint = "Not set"

    # Prompt for new key
    new_key = questionary.password(
        f"Tavily API key for web search ({hint}, Enter to keep):",
        style=WIZARD_STYLE,
    ).ask()

    if new_key is None:
        raise KeyboardInterrupt()

    new_key = new_key.strip()

    # Determine the key to validate: new input or existing current key
    key_to_validate = new_key if new_key else current

    if not key_to_validate:
        return None  # Nothing to validate

    # Validate the key (new or current)
    if not skip_validation:
        console.print("  [dim]Validating...[/dim]", end="")
        valid, msg = validate_tavily_key(key_to_validate)
        if valid:
            console.print(f"\r  [green]✓ {msg}[/green]      ")
            return new_key if new_key else None  # Return new key or None (keep current)
        else:
            console.print(f"\r  [red]✗ {msg}[/red]      ")
            if not new_key:
                return None  # Current key invalid, but keep it
            # Ask if they want to save anyway
            save_anyway = questionary.confirm(
                "Save anyway?",
                default=False,
                style=WIZARD_STYLE,
            ).ask()
            if save_anyway is None:
                raise KeyboardInterrupt()
            if save_anyway:
                return new_key
            return None
    else:
        return new_key if new_key else None


def _step_workspace(config: EvoScientistConfig) -> tuple[str, str]:
    """Step 5: Configure workspace settings.

    Args:
        config: Current configuration.

    Returns:
        Tuple of (mode, workdir).
    """
    # Mode selection
    mode_choices = [
        Choice(
            title="Daemon (persistent workspace ./workspace/)",
            value="daemon",
        ),
        Choice(
            title="Run (isolated per-session ./workspace/runs/<timestamp>/)",
            value="run",
        ),
    ]

    mode = questionary.select(
        "Default workspace mode:",
        choices=mode_choices,
        default=config.default_mode,
        style=WIZARD_STYLE,
        use_indicator=True,
    ).ask()

    if mode is None:
        raise KeyboardInterrupt()

    # Custom workdir (optional)
    use_custom = questionary.confirm(
        "Use custom workspace directory? (default: ./workspace/)",
        default=bool(config.default_workdir),
        style=WIZARD_STYLE,
    ).ask()

    if use_custom is None:
        raise KeyboardInterrupt()

    workdir = ""
    if use_custom:
        workdir = questionary.text(
            "Workspace directory path:",
            default=config.default_workdir or "",
            style=WIZARD_STYLE,
        ).ask()
        if workdir is None:
            raise KeyboardInterrupt()
        workdir = workdir.strip()

    return mode, workdir


def _step_parameters(config: EvoScientistConfig) -> tuple[int, int, bool]:
    """Step 6: Configure agent parameters.

    Args:
        config: Current configuration.

    Returns:
        Tuple of (max_concurrent, max_iterations, show_thinking).
    """
    # Max concurrent
    max_concurrent_str = questionary.text(
        f"Max concurrent sub-agents (1-10):",
        default=str(config.max_concurrent),
        style=WIZARD_STYLE,
        validate=lambda x: x.strip() == "" or (x.strip().isdigit() and 1 <= int(x.strip()) <= 10),
    ).ask()

    if max_concurrent_str is None:
        raise KeyboardInterrupt()

    max_concurrent = int(max_concurrent_str.strip()) if max_concurrent_str.strip() else config.max_concurrent

    # Max iterations
    max_iterations_str = questionary.text(
        f"Max delegation iterations (1-10):",
        default=str(config.max_iterations),
        style=WIZARD_STYLE,
        validate=lambda x: x.strip() == "" or (x.strip().isdigit() and 1 <= int(x.strip()) <= 10),
    ).ask()

    if max_iterations_str is None:
        raise KeyboardInterrupt()

    max_iterations = int(max_iterations_str.strip()) if max_iterations_str.strip() else config.max_iterations

    # Show thinking
    thinking_choices = [
        Choice(title="On (show model reasoning)", value=True),
        Choice(title="Off (hide model reasoning)", value=False),
    ]

    show_thinking = questionary.select(
        "Show thinking panel in CLI?",
        choices=thinking_choices,
        default=config.show_thinking,
        style=WIZARD_STYLE,
        use_indicator=True,
    ).ask()

    if show_thinking is None:
        raise KeyboardInterrupt()

    return max_concurrent, max_iterations, show_thinking


# =============================================================================
# Progress Rendering (for tests and potential future use)
# =============================================================================

def render_progress(current_step: int, completed: set[int]) -> Panel:
    """Render the progress indicator panel.

    Args:
        current_step: Index of the current step (0-based).
        completed: Set of completed step indices.

    Returns:
        A Rich Panel displaying the progress.
    """
    lines = []
    for i, step_name in enumerate(STEPS):
        if i in completed:
            icon = Text("●", style="green bold")
            label = Text(f" {step_name}", style="green")
        elif i == current_step:
            icon = Text("◉", style="cyan bold")
            label = Text(f" {step_name}", style="cyan bold")
        else:
            icon = Text("○", style="dim")
            label = Text(f" {step_name}", style="dim")

        line = Text()
        line.append_text(icon)
        line.append_text(label)
        lines.append(line)

        # Add connector line between steps
        if i < len(STEPS) - 1:
            if i in completed:
                connector_style = "green"
            elif i == current_step:
                connector_style = "cyan"
            else:
                connector_style = "dim"
            lines.append(Text("│", style=connector_style))

    # Join all lines with newlines
    content = Text("\n").join(lines)
    return Panel(content, title="[bold]EvoScientist Setup[/bold]", border_style="blue")


# =============================================================================
# Main onboard function
# =============================================================================

def run_onboard(skip_validation: bool = False) -> bool:
    """Run the interactive onboarding wizard.

    Args:
        skip_validation: Skip API key validation.

    Returns:
        True if configuration was saved, False if cancelled.
    """
    try:
        # Print header once
        _print_header()

        # Load existing config as starting point
        config = load_config()

        # Step 1: Provider
        provider = _step_provider(config)
        config.provider = provider

        # Step 2: Provider API Key
        new_key = _step_provider_api_key(config, provider, skip_validation)
        if new_key is not None:
            if provider == "anthropic":
                config.anthropic_api_key = new_key
            elif provider == "nvidia":
                config.nvidia_api_key = new_key
            else:
                config.openai_api_key = new_key
        else:
            if provider == "anthropic":
                current = config.anthropic_api_key
            elif provider == "nvidia":
                current = config.nvidia_api_key
            else:
                current = config.openai_api_key
            if not current:
                _print_step_skipped("API Key", "not set")

        # Step 3: Model
        model = _step_model(config, provider)
        config.model = model

        # Step 4: Tavily Key
        new_tavily_key = _step_tavily_key(config, skip_validation)
        if new_tavily_key is not None:
            config.tavily_api_key = new_tavily_key
        else:
            if not config.tavily_api_key:
                _print_step_skipped("Tavily Key", "not set")

        # Step 5: Workspace
        mode, workdir = _step_workspace(config)
        config.default_mode = mode
        config.default_workdir = workdir

        # Step 6: Parameters
        max_concurrent, max_iterations, show_thinking = _step_parameters(config)
        config.max_concurrent = max_concurrent
        config.max_iterations = max_iterations
        config.show_thinking = show_thinking

        # Confirm save
        console.print()
        save = questionary.confirm(
            "Save this configuration?",
            default=True,
            style=CONFIRM_STYLE,
            qmark="!",
        ).ask()

        if save is None:
            raise KeyboardInterrupt()

        if save:
            save_config(config)
            console.print()
            console.print("[green]✓ Configuration saved![/green]")
            console.print(f"[dim]  → {get_config_path()}[/dim]")
            console.print()
            return True
        else:
            console.print()
            console.print("[yellow]Configuration not saved.[/yellow]")
            console.print()
            return False

    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Setup cancelled.[/yellow]")
        console.print()
        return False
