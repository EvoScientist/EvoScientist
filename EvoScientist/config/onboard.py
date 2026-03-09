"""Interactive onboarding wizard for EvoScientist.

Guides users through initial setup including API keys, model selection,
workspace settings, and agent parameters. Uses flow-style arrow-key selection UI.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import questionary
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import Validator, ValidationError
from questionary import Choice
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .settings import (
    EvoScientistConfig,
    load_config,
    save_config,
    get_config_path,
)
from ..llm import get_models_for_provider

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


STEPS = [
    "UI",
    "Provider",
    "API Key",
    "Model",
    "Tavily Key",
    "Workspace",
    "Thinking",
    "Skills",
    "MCP Servers",
    "Channels",
]


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
            raise ValidationError(message=f"Must be one of: {', '.join(self.choices)}")


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
        if (
            "401" in error_str
            or "unauthorized" in error_str
            or "invalid" in error_str
            or "authentication" in error_str
        ):
            return False, "Invalid API key"
        return False, f"Error: {e}"


def validate_google_key(api_key: str) -> tuple[bool, str]:
    """Validate a Google GenAI API key by making a test request.

    Args:
        api_key: The API key to validate.

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        # Make a minimal request to validate the key
        pager = client.models.list(config={"page_size": 1})
        next(iter(pager))  # fetch first model only
        return True, "Valid"
    except StopIteration:
        # Empty result but request succeeded — key is valid
        return True, "Valid"
    except Exception as e:
        error_str = str(e).lower()
        if (
            "400" in error_str
            or "401" in error_str
            or "403" in error_str
            or "unauthorized" in error_str
            or "invalid" in error_str
            or "api key" in error_str
        ):
            return False, "Invalid API key"
        return False, f"Error: {e}"


def validate_siliconflow_key(api_key: str) -> tuple[bool, str]:
    """Validate a SiliconFlow API key by making a test request.

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        import openai

        client = openai.OpenAI(
            api_key=api_key, base_url="https://api.siliconflow.cn/v1"
        )
        client.models.list()
        return True, "Valid"
    except Exception as e:
        error_str = str(e).lower()
        if (
            "401" in error_str
            or "unauthorized" in error_str
            or "invalid" in error_str
            or "authentication" in error_str
        ):
            return False, "Invalid API key"
        return False, f"Error: {e}"


def validate_openrouter_key(api_key: str) -> tuple[bool, str]:
    """Validate an OpenRouter API key by making a test request.

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        import openai

        client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        client.models.list()
        return True, "Valid"
    except Exception as e:
        error_str = str(e).lower()
        if (
            "401" in error_str
            or "unauthorized" in error_str
            or "invalid" in error_str
            or "authentication" in error_str
        ):
            return False, "Invalid API key"
        return False, f"Error: {e}"


def validate_zhipu_key(api_key: str) -> tuple[bool, str]:
    """Validate a ZhipuAI API key by making a test request.

    Uses the general endpoint for validation — both zhipu and zhipu-code
    share the same API key, only the base_url differs at runtime.

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        import openai

        client = openai.OpenAI(
            api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4"
        )
        client.models.list()
        return True, "Valid"
    except Exception as e:
        error_str = str(e).lower()
        if (
            "401" in error_str
            or "unauthorized" in error_str
            or "invalid" in error_str
            or "authentication" in error_str
        ):
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


def validate_ollama_connection(base_url: str) -> tuple[bool, str, list[str]]:
    """Validate that Ollama is reachable at the given base URL.

    Args:
        base_url: The Ollama server base URL.

    Returns:
        Tuple of (is_valid, message, model_names).
        model_names is a list of pulled model names (empty if unreachable).
    """
    if not base_url:
        return True, "Skipped (no URL provided)", []

    try:
        import httpx

        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("models", [])
            names = [m.get("name", "?") for m in models]
            if names:
                preview = ", ".join(names[:5])
                return True, f"Connected — {len(names)} model(s): {preview}", names
            return True, "Connected (no models pulled yet)", []
        return False, f"HTTP {resp.status_code}", []
    except Exception as e:
        return False, f"Cannot reach Ollama: {e}", []


# =============================================================================
# Display Helpers
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


def _step_ui_backend(config: EvoScientistConfig) -> str:
    """Step 0: Select UI backend (Rich CLI or Textual TUI).

    Args:
        config: Current configuration.

    Returns:
        Selected backend name ("rich" or "textual").
    """
    choices = [
        Choice(title="Rich CLI (classic terminal)", value="rich"),
        Choice(title="Textual TUI (full-screen interface)", value="textual"),
    ]

    backend = questionary.select(
        "Select UI mode:",
        choices=choices,
        default=config.ui_backend,
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()

    if backend is None:
        raise KeyboardInterrupt()

    return backend


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
        Choice(title="Google GenAI (Gemini models)", value="google-genai"),
        Choice(title="NVIDIA (third party — limited free requests)", value="nvidia"),
        Choice(
            title="SiliconFlow (third party — GLM, Kimi, MiniMax, etc.)",
            value="siliconflow",
        ),
        Choice(
            title="OpenRouter (third party — Grok, Gemini, Qwen, etc.)",
            value="openrouter",
        ),
        Choice(title="ZhipuAI (智谱 — GLM models)", value="zhipu"),
        Choice(
            title="ZhipuAI CodePlan (智谱代码计划 — GLM models for coding)",
            value="zhipu-code",
        ),
        Choice(title="Ollama (local models)", value="ollama"),
        Choice(title="Other (OpenAI-compatible)", value="custom"),
    ]

    # Set default based on current config
    valid_providers = {c.value for c in choices}
    default = config.provider if config.provider in valid_providers else "anthropic"

    provider = questionary.select(
        "Select your LLM provider:",
        choices=choices,
        default=default,
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()

    if provider is None:
        raise KeyboardInterrupt()

    return provider


def _provider_key_info(config: EvoScientistConfig, provider: str):
    """Return (display_name, current_value, validate_fn) for a provider."""
    mapping = {
        "anthropic": (
            "Anthropic",
            config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            validate_anthropic_key,
        ),
        "nvidia": (
            "NVIDIA",
            config.nvidia_api_key or os.environ.get("NVIDIA_API_KEY", ""),
            validate_nvidia_key,
        ),
        "google-genai": (
            "Google",
            config.google_api_key or os.environ.get("GOOGLE_API_KEY", ""),
            validate_google_key,
        ),
        "siliconflow": (
            "SiliconFlow",
            config.siliconflow_api_key or os.environ.get("SILICONFLOW_API_KEY", ""),
            validate_siliconflow_key,
        ),
        "openrouter": (
            "OpenRouter",
            config.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", ""),
            validate_openrouter_key,
        ),
        "zhipu": (
            "ZhipuAI",
            config.zhipu_api_key or os.environ.get("ZHIPU_API_KEY", ""),
            validate_zhipu_key,
        ),
        "zhipu-code": (
            "ZhipuAI CodePlan",
            config.zhipu_api_key or os.environ.get("ZHIPU_API_KEY", ""),
            validate_zhipu_key,
        ),
        "custom": (
            "Custom",
            config.custom_api_key or os.environ.get("CUSTOM_API_KEY", ""),
            None,
        ),
        "ollama": ("Ollama", "__no_key__", None),
    }
    return mapping.get(
        provider,
        (
            "OpenAI",
            config.openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            validate_openai_key,
        ),
    )


def _prompt_and_validate_api_key(
    prompt_text: str,
    current: str,
    validate_fn,
    skip_validation: bool = False,
    placeholder=None,
) -> str | None:
    """Prompt user for an API key, validate, offer save-anyway on failure.

    Args:
        prompt_text: The question shown to the user.
        current: Currently stored key value (may be empty).
        validate_fn: Callable(key) -> (bool, str).
        skip_validation: If True, skip the validation step entirely.
        placeholder: Optional placeholder for the password input.

    Returns:
        New key string if the user entered one, or None to keep existing.
    """
    kwargs: dict = {"style": WIZARD_STYLE, "qmark": QMARK}
    if placeholder is not None:
        kwargs["placeholder"] = placeholder

    new_key = questionary.password(prompt_text, **kwargs).ask()
    if new_key is None:
        raise KeyboardInterrupt()

    new_key = new_key.strip()

    # Determine which key to validate: new input or existing
    key_to_validate = new_key or current

    if not key_to_validate:
        return None

    if not skip_validation and validate_fn is not None:
        console.print("  [dim]Validating...[/dim]", end="")
        valid, msg = validate_fn(key_to_validate)
        if valid:
            console.print(f"\r  [green]\u2713 {msg}[/green]      ")
            return new_key if new_key else None
        else:
            console.print(f"\r  [red]\u2717 {msg}[/red]      ")
            if not new_key:
                # Existing key is invalid — warn but keep (user didn't change it)
                return None
            save_anyway = questionary.confirm(
                "Save anyway?",
                default=False,
                style=WIZARD_STYLE,
                qmark=QMARK,
            ).ask()
            if save_anyway is None:
                raise KeyboardInterrupt()
            return new_key if save_anyway else None

    return new_key if new_key else None


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
    key_name, current, validate_fn = _provider_key_info(config, provider)

    hint = f"Current: ***{current[-4:]}" if current else "Not set"
    prompt_text = f"Enter {key_name} API key ({hint}, Enter to keep):"

    return _prompt_and_validate_api_key(
        prompt_text,
        current,
        validate_fn,
        skip_validation,
    )


def _step_base_url(config: EvoScientistConfig) -> str:
    """Prompt for custom provider base URL.

    Args:
        config: Current configuration.

    Returns:
        Base URL string.
    """
    current = config.custom_base_url
    hint = f"Current: {current}" if current else ""
    default = current if current else ""

    url = questionary.text(
        f"Base URL{' (' + hint + ', Enter to keep)' if hint else ''}:",
        default=default,
        style=WIZARD_STYLE,
        qmark=QMARK,
        placeholder=FormattedText([("fg:#858585", " e.g. https://api.example.com/v1")])
        if not default
        else None,
    ).ask()
    if url is None:
        raise KeyboardInterrupt()
    return url.strip()


def _step_ollama_base_url(config: EvoScientistConfig) -> tuple[str, list[str]]:
    """Prompt for Ollama server base URL and validate connection.

    Args:
        config: Current configuration.

    Returns:
        Tuple of (base_url, detected_model_names).
    """
    current = config.ollama_base_url or os.environ.get("OLLAMA_BASE_URL", "")
    default = current if current else "http://localhost:11434"

    url = questionary.text(
        f"Ollama base URL (Enter for {default}):",
        default=default,
        style=WIZARD_STYLE,
        qmark=QMARK,
    ).ask()
    if url is None:
        raise KeyboardInterrupt()
    url = url.strip()

    detected_models: list[str] = []
    if url:
        console.print("  [dim]Checking Ollama connection...[/dim]", end="")
        valid, msg, detected_models = validate_ollama_connection(url)
        if valid:
            console.print(f"\r  [green]\u2713 {msg}[/green]      ")
        else:
            console.print(f"\r  [yellow]\u2717 {msg}[/yellow]      ")
            console.print("  [dim]You can start Ollama later and it will work.[/dim]")

    return url, detected_models


def _step_model(
    config: EvoScientistConfig,
    provider: str,
    *,
    ollama_detected_models: list[str] | None = None,
) -> str:
    """Step 3: Select model for the provider.

    Args:
        config: Current configuration.
        provider: Selected provider name.
        ollama_detected_models: Model names detected from a live Ollama server.

    Returns:
        Selected model name.
    """
    # Ollama: show only what's actually pulled on the server
    if provider == "ollama":
        if ollama_detected_models:
            _CUSTOM_SENTINEL = "__custom__"
            choices = [
                Choice(title=name, value=name) for name in ollama_detected_models
            ]
            choices.append(Choice(title="Type a model name...", value=_CUSTOM_SENTINEL))

            default = ollama_detected_models[0]
            if config.model in ollama_detected_models:
                default = config.model

            selected = questionary.select(
                "Select model:",
                choices=choices,
                default=default,
                style=WIZARD_STYLE,
                qmark=QMARK,
                use_indicator=True,
            ).ask()
            if selected is None:
                raise KeyboardInterrupt()

            if selected != _CUSTOM_SENTINEL:
                return selected

        # No detected models (server down or empty) — direct text input
        if not ollama_detected_models:
            console.print(
                "  [dim]No models detected — type the model name you plan to pull.[/dim]"
            )
        model = questionary.text(
            "Model name:",
            style=WIZARD_STYLE,
            qmark=QMARK,
            placeholder=FormattedText([("fg:#858585", " e.g. qwen3-coder-next")]),
        ).ask()
        if model is None:
            raise KeyboardInterrupt()
        model = model.strip()
        if not model:
            model = "qwen3-coder-next"
            console.print(f"  [dim]Using default: {model}[/dim]")
        return model

    # Get models for the selected provider
    entries = get_models_for_provider(provider)

    if not entries:
        # Custom / unknown provider: direct text input
        model = questionary.text(
            "Model name:",
            style=WIZARD_STYLE,
            qmark=QMARK,
            placeholder=FormattedText([("fg:#858585", " e.g. owner/model-name")]),
        ).ask()
        if model is None:
            raise KeyboardInterrupt()
        return model

    provider_models = [name for name, _ in entries]

    # Create choices with model IDs as hints
    _CUSTOM_SENTINEL = "__custom__"
    choices = []
    for name, model_id in entries:
        choices.append(Choice(title=f"{name} ({model_id})", value=name))
    choices.append(Choice(title="Type a model name...", value=_CUSTOM_SENTINEL))

    # Determine default
    if config.model in provider_models:
        default = config.model
    else:
        default = provider_models[0]

    selected = questionary.select(
        "Select model:",
        choices=choices,
        default=default,
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()

    if selected is None:
        raise KeyboardInterrupt()

    if selected != _CUSTOM_SENTINEL:
        return selected

    model = questionary.text(
        "Model name:",
        style=WIZARD_STYLE,
        qmark=QMARK,
        placeholder=FormattedText([("fg:#858585", " e.g. owner/model-name")]),
    ).ask()
    if model is None:
        raise KeyboardInterrupt()
    model = model.strip()
    if not model:
        model = provider_models[0]
        console.print(f"  [dim]Using default: {model}[/dim]")
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

    hint = f"Current: ***{current[-4:]}" if current else "Not set"
    prompt_text = f"Tavily API key for web search ({hint}, Enter to keep):"

    return _prompt_and_validate_api_key(
        prompt_text,
        current,
        validate_tavily_key,
        skip_validation,
        placeholder=FormattedText([("fg:#858585", " (recommended for web search)")]),
    )


def _step_workspace(config: EvoScientistConfig) -> tuple[str, str]:
    """Step 5: Configure workspace settings.

    Args:
        config: Current configuration.

    Returns:
        Tuple of (mode, workdir).
    """
    # Mode selection
    cwd = os.getcwd()
    cwd_short = os.path.basename(cwd) or cwd
    mode_choices = [
        Choice(
            title="Daemon (persistent workspace)",
            value="daemon",
        ),
        Choice(
            title="Run (isolated per-session)",
            value="run",
        ),
    ]

    mode = questionary.select(
        "Default workspace mode:",
        choices=mode_choices,
        default=config.default_mode,
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()

    if mode is None:
        raise KeyboardInterrupt()

    # Custom workdir (optional)
    current_default = config.default_workdir or ""
    if current_default:
        prompt_text = f"Workspace directory (Enter to keep '{current_default}'):"
    else:
        prompt_text = f"Workspace directory (Enter to use ./{cwd_short}/):"
    workdir = questionary.text(
        prompt_text,
        default=current_default,
        style=WIZARD_STYLE,
        qmark=QMARK,
    ).ask()
    if workdir is None:
        raise KeyboardInterrupt()
    workdir = workdir.strip()

    return mode, workdir


def _step_thinking(config: EvoScientistConfig) -> bool:
    """Step 6: Configure thinking panel visibility.

    Args:
        config: Current configuration.

    Returns:
        Whether to show thinking panels in CLI.
    """
    thinking_choices = [
        Choice(title="On (show model reasoning)", value=True),
        Choice(title="Off (hide model reasoning)", value=False),
    ]

    show_thinking = questionary.select(
        "Show thinking panel in CLI?",
        choices=thinking_choices,
        default=config.show_thinking,
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()

    if show_thinking is None:
        raise KeyboardInterrupt()

    return show_thinking


_RECOMMENDED_SKILLS = [
    # ── Official (EvoScientist) ──
    {
        "label": "EvoSci Skills  (optimized for EvoScientist — paper planning, writing, review, etc.)",
        "source": "EvoScientist/EvoSkills@skills",
    },
    # ── Third-party (K-Dense) ──
    {
        "label": "Scientific Skills  (147+ research & experiment skills, third party by K-Dense)",
        "source": "K-Dense-AI/claude-scientific-skills@scientific-skills",
    },
    {
        "label": "Scientific Writer  (23 writing, review & presentation skills, third party by K-Dense)",
        "source": "K-Dense-AI/claude-scientific-writer@skills",
    },
    # ── Third-party (Anthropic) ──
    {
        "label": "Anthropic Skills  (co-authoring, design, etc., third party by Anthropic)",
        "source": "anthropics/skills@skills",
    },
    # ── Third-party (HuggingFace) ──
    {
        "label": "HuggingFace Skills  (dataset creation, model training & evaluation, third party by HuggingFace)",
        "source": "huggingface/skills@skills",
    },
    # ── Third-party (Orchestra Research) ──
    {
        "label": "AI Research Skills  (85 skills for training, evaluation, deployment, etc., third party by Orchestra Research)",
        "source": "Orchestra-Research/AI-Research-SKILLs",
    },
]


def _check_npx() -> bool:
    """Check if npx is available on the system.

    Returns:
        True if npx is found and working.
    """
    try:
        result = subprocess.run(
            ["npx", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _detect_node_install_method() -> tuple[str, str]:
    """Detect the best way to install Node.js for this environment.

    Returns:
        Tuple of (method_name, install_command).
    """
    # Conda environment (any platform)
    if os.environ.get("CONDA_PREFIX"):
        return "conda", "conda install -y nodejs"

    # macOS with Homebrew
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["brew", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return "brew", "brew install node"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return "manual", "https://nodejs.org"


def _install_node(method: str, command: str) -> bool:
    """Install Node.js using the detected method.

    Returns:
        True if installation succeeded.
    """
    if method == "manual":
        return False

    try:
        proc = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.returncode == 0
    except FileNotFoundError:
        console.print(f"  [red]✗ {method} not found[/red]")
        return False
    except subprocess.TimeoutExpired:
        console.print("  [red]✗ Installation timed out[/red]")
        return False
    except Exception as e:
        console.print(f"  [red]✗ Installation failed: {e}[/red]")
        return False


def _ensure_npx(reason: str) -> bool:
    """Check for npx and offer to install Node.js if missing.

    Args:
        reason: Why npx is needed (shown in the warning message).

    Returns:
        True if npx is available (was already present or just installed).
    """
    if _check_npx():
        return True

    console.print(f"  [yellow]✗ npx not found — {reason}[/yellow]")
    method, command = _detect_node_install_method()

    if method != "manual":
        install_node = questionary.confirm(
            f"Install Node.js via {method}? ({command})",
            default=True,
            style=WIZARD_STYLE,
            qmark=f"  {QMARK}",
        ).ask()
        if install_node is None:
            raise KeyboardInterrupt()
        if install_node:
            console.print("  [dim]Installing Node.js...[/dim]")
            if _install_node(method, command):
                if _check_npx():
                    console.print("  [green]✓ npx now available[/green]")
                    return True
                else:
                    console.print(
                        "  [yellow]✗ npx still not found after install[/yellow]"
                    )
            else:
                console.print("  [red]✗ Installation failed[/red]")
    else:
        console.print(f"  [dim]Install Node.js: {command}[/dim]")

    return False


def _step_skills() -> list[str]:
    """Step 7: Optionally install recommended skills.

    Shows checkbox first. Already-installed skills are shown as disabled
    so users don't accidentally reinstall them. If user selects nothing,
    checks npx as an easter egg — confirms skill discovery is available,
    or offers to install Node.js if missing.

    Returns:
        List of skill sources that were selected (empty if skipped).
    """
    from ..paths import USER_SKILLS_DIR

    # Collect names of already-installed user skills
    skills_dir = Path(USER_SKILLS_DIR)
    installed_names: set[str] = set()
    if skills_dir.exists():
        installed_names = {e.name for e in skills_dir.iterdir() if e.is_dir()}

    def _hint_name(source: str) -> str:
        """Derive expected skill directory name from source URL."""
        if "@" in source and "://" not in source:
            return source.split("@", 1)[1].strip()
        return source.rstrip("/").rsplit("/", 1)[-1]

    choices = []
    for skill in _RECOMMENDED_SKILLS:
        if _hint_name(skill["source"]) in installed_names:
            choices.append(
                Choice(
                    title=[
                        ("", skill["label"]),
                        ("class:instruction", "  (already installed)"),
                    ],
                    value=skill["source"],
                    disabled=True,
                )
            )
        else:
            choices.append(Choice(title=skill["label"], value=skill["source"]))

    all_installed = all(
        _hint_name(skill["source"]) in installed_names for skill in _RECOMMENDED_SKILLS
    )
    if all_installed:
        console.print(
            "  [green]✓ All recommended skills are already installed.[/green]"
        )
        return []

    selected = _checkbox_ask(choices, "Install or Sync predefined skills:")

    if selected is None:
        raise KeyboardInterrupt()

    if not selected:
        # Verify skill discovery environment
        console.print("  [dim]Checking skill discovery environment...[/dim]")
        has_npx = _ensure_npx("skill discovery requires Node.js")
        if has_npx:
            _print_step_skipped("Skills", "none selected — good choice!")
            console.print("  [green]✓ npx found — skill discovery available[/green]")
            console.print(
                "  [yellow bold]* Less is more[/yellow bold] [dim](EvoScientist can discover and install skills on its own)[/dim]"
            )
        else:
            _print_step_skipped("Skills", "none selected")

        return []

    from ..tools.skills_manager import install_skill

    installed = []
    for source in selected:
        label = next(s["label"] for s in _RECOMMENDED_SKILLS if s["source"] == source)
        try:
            result = install_skill(source)
            if result.get("success"):
                _print_step_result("Skill", label)
                installed.append(source)
            else:
                _print_step_result(
                    "Skill", f"{label} — {result.get('error', 'failed')}", success=False
                )
        except Exception as e:
            _print_step_result("Skill", f"{label} — {e}", success=False)

    return installed


_RECOMMENDED_MCP_SERVERS = [
    # ── Built-in ──
    {
        "label": "Sequential Thinking  (structured reasoning for non-reasoning models)",
        "name": "sequential-thinking",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
    },
    {
        "label": "Docs by LangChain  (documentation for building agents)",
        "name": "docs-langchain",
        "url": "https://docs.langchain.com/mcp",
    },
    # ── Search & Knowledge ──
    {
        "label": "Perplexity  (AI-powered web search — requires PERPLEXITY_API_KEY)",
        "name": "perplexity",
        "command": "npx",
        "args": ["-y", "@perplexity-ai/mcp-server"],
        "env": {"PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}"},
        "env_key": "PERPLEXITY_API_KEY",
        "env_hint": "export PERPLEXITY_API_KEY=pplx-... (get one at perplexity.ai/settings/api)",
    },
    {
        "label": "Context7  (fast documentation lookup — API key unlocks higher rate limits)",
        "name": "context7",
        "command": "npx",
        "args": ["-y", "@upstash/context7-mcp"],
        "env": {"CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}"},
        "env_key": "CONTEXT7_API_KEY",
        "env_hint": "export CONTEXT7_API_KEY=... (optional — unlocks higher rate limits)",
        "env_optional": True,
    },
    # ── Research ──
    {
        "label": "DeepWiki  (search & read GitHub repo documentation)",
        "name": "deepwiki",
        "url": "https://mcp.deepwiki.com/mcp",
    },
    {
        "label": "ArXiv  (search & fetch academic papers from arXiv)",
        "name": "arxiv",
        "pip_package": "arxiv-mcp-server",
        "command": "arxiv-mcp-server",
        "args": [],
    },
]


def _install_pip_package(package: str) -> bool:
    """Silently install a pip package.

    Returns:
        True if installation succeeded.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", package],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            # Invalidate import caches so newly-installed packages are
            # discoverable in the current process without a restart.
            import importlib

            importlib.invalidate_caches()
            return True
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _step_mcp_servers() -> list[str]:
    """Step 8: Optionally install recommended MCP servers.

    Shows a checkbox list of recommended servers. Already-configured servers
    are shown as disabled so users don't accidentally override them.
    Selected ones are added to the user MCP config via ``add_mcp_server()``.

    Handles env-key prompts, pip package installs, and URL-based servers.

    Returns:
        List of server names that were installed.
    """
    from ..mcp.client import _load_user_config, add_mcp_server

    existing_config = _load_user_config()

    choices = []
    for srv in _RECOMMENDED_MCP_SERVERS:
        if srv["name"] in existing_config:
            choices.append(
                Choice(
                    title=[
                        ("", srv["label"]),
                        ("class:instruction", "  (already configured)"),
                    ],
                    value=srv["name"],
                    disabled=True,
                )
            )
        else:
            choices.append(Choice(title=srv["label"], value=srv["name"]))

    all_installed = all(
        srv["name"] in existing_config for srv in _RECOMMENDED_MCP_SERVERS
    )
    if all_installed:
        console.print(
            "  [green]✓ All recommended MCP servers are already configured.[/green]"
        )
        return []

    selected = _checkbox_ask(choices, "Install recommended MCP servers:")

    if selected is None:
        raise KeyboardInterrupt()

    if not selected:
        _print_step_skipped("MCP Servers", "none selected")
        console.print(
            "  [dim]Add later with: EvoSci mcp add <name> <command> [--env-ref KEY] -- [args][/dim]"
        )
        return []

    # Check if any selected servers require npx
    needs_npx = any(
        srv.get("command") == "npx"
        for srv in _RECOMMENDED_MCP_SERVERS
        if srv["name"] in selected
    )
    if needs_npx:
        if not _ensure_npx("some MCP servers require Node.js"):
            npx_servers = {
                srv["name"]
                for srv in _RECOMMENDED_MCP_SERVERS
                if srv["name"] in selected and srv.get("command") == "npx"
            }
            selected = [s for s in selected if s not in npx_servers]
            if npx_servers:
                console.print(
                    f"  [yellow]\u26a0 Skipping {', '.join(sorted(npx_servers))} (npx not available)[/yellow]"
                )
            if not selected:
                return []

    installed = []
    for name in selected:
        srv = next(s for s in _RECOMMENDED_MCP_SERVERS if s["name"] == name)
        try:
            # Prompt for required API keys
            env_key = srv.get("env_key")
            if env_key:
                is_optional = srv.get("env_optional", False)
                hint = srv.get("env_hint", "")
                if is_optional:
                    console.print(f"  [dim]{hint}[/dim]")
                else:
                    console.print(f"  [yellow]⚠ Requires {env_key}[/yellow]")
                    console.print(f"  [dim]{hint}[/dim]")
                    if not os.environ.get(env_key):
                        console.print(
                            f"  [dim]Set it before running EvoScientist: export {env_key}=...[/dim]"
                        )

            # Install pip package if needed
            pip_pkg = srv.get("pip_package")
            if pip_pkg:
                console.print(f"  [dim]Installing {pip_pkg}...[/dim]")
                if not _install_pip_package(pip_pkg):
                    _print_step_result(
                        "MCP", f"{name} — pip install {pip_pkg} failed", success=False
                    )
                    continue

            # Add to MCP config
            if "url" in srv:
                add_mcp_server(name, "streamable_http", url=srv["url"])
            else:
                add_mcp_server(
                    name,
                    "stdio",
                    command=srv["command"],
                    args=srv.get("args", []),
                    env=srv.get("env"),
                )
            _print_step_result("MCP", f"{name}")
            installed.append(name)
        except Exception as e:
            _print_step_result("MCP", f"{name} — {e}", success=False)

    return installed


def validate_imessage() -> tuple[bool, str]:
    """Validate iMessage environment by checking for the imsg CLI.

    Returns:
        Tuple of (is_valid, message).
    """
    # macOS only
    if sys.platform != "darwin":
        return False, "iMessage requires macOS"

    from ..channels.imessage.probe import find_cli

    cli_path = find_cli()
    if not cli_path:
        return False, "not_installed"

    # Check version
    try:
        result = subprocess.run(
            [cli_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        version = None

    # Check RPC support
    try:
        result = subprocess.run(
            [cli_path, "rpc", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        rpc_ok = result.returncode == 0
    except Exception:
        rpc_ok = False

    if not rpc_ok:
        return (
            False,
            f"imsg found at {cli_path} but RPC not supported (update with: brew upgrade imsg)",
        )

    version_str = f" ({version})" if version else ""
    return True, f"imsg{version_str} at {cli_path}"


def _install_imsg() -> bool:
    """Run brew install for imsg CLI.

    Returns:
        True if installation succeeded.
    """
    try:
        proc = subprocess.run(
            ["brew", "install", "steipete/tap/imsg"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.returncode == 0
    except FileNotFoundError:
        console.print("  [red]✗ Homebrew not found[/red]")
        console.print("  [dim]Install Homebrew first: https://brew.sh[/dim]")
        return False
    except subprocess.TimeoutExpired:
        console.print("  [red]✗ Installation timed out[/red]")
        return False
    except Exception as e:
        console.print(f"  [red]✗ Installation failed: {e}[/red]")
        return False


def _setup_imessage() -> bool:
    """Guide the user through iMessage setup: install, validate, test.

    Returns:
        True if iMessage is ready to use.
    """
    # Step 1: Validate
    console.print("  [dim]Checking iMessage environment...[/dim]")
    valid, msg = validate_imessage()

    if valid:
        console.print(f"  [green]✓ {msg}[/green]")
        return True

    if msg == "iMessage requires macOS":
        console.print(f"  [red]✗ {msg}[/red]")
        return False

    if msg == "not_installed":
        console.print("  [yellow]✗ imsg CLI not installed[/yellow]")
        console.print()

        # Step 2: Offer to install
        install = questionary.confirm(
            "Install imsg via Homebrew? (brew install steipete/tap/imsg)",
            default=True,
            style=WIZARD_STYLE,
            qmark=f"  {QMARK}",
        ).ask()

        if install is None:
            raise KeyboardInterrupt()

        if install:
            console.print()
            if _install_imsg():
                console.print()
                # Re-validate after install
                valid, msg = validate_imessage()
                if valid:
                    console.print(f"  [green]✓ {msg}[/green]")
                    return True
                else:
                    console.print(f"  [red]✗ {msg}[/red]")
                    return False
            else:
                return False
        else:
            console.print(
                "  [dim]Skipped. Install manually: brew install steipete/tap/imsg[/dim]"
            )
            return False
    else:
        # RPC not supported or other issue
        console.print(f"  [red]✗ {msg}[/red]")
        return False


def _step_channels(config: EvoScientistConfig) -> dict[str, object]:
    """Step: Select channels to enable on startup.

    Presents a multi-select list of supported channels.
    For each selected channel, prompts for required credentials
    and validates them via the channel's probe function.

    Args:
        config: Current configuration.

    Returns:
        Dict mapping config field names to their new values.
        Empty dict when the user skips or selects nothing.
    """
    # Currently enabled channels
    _currently_enabled = {
        t.strip()
        for t in (getattr(config, "channel_enabled", "") or "").split(",")
        if t.strip()
    }
    # Legacy iMessage compat
    if (
        getattr(config, "imessage_enabled", False)
        and "imessage" not in _currently_enabled
    ):
        _currently_enabled.add("imessage")

    # Direct pip packages for each channel extra.  Used to install the
    # exact dependency without requiring the evoscientist package itself
    # to be resolvable on PyPI (e.g. editable / dev installs).
    _CHANNEL_PIP_DEPS: dict[str, list[str]] = {
        "telegram": ["python-telegram-bot>=21.0"],
        "discord": ["discord.py>=2.3"],
        "slack": ["slack-sdk>=3.27", "aiohttp>=3.9"],
        "feishu": ["aiohttp>=3.9"],
        "dingtalk": ["aiohttp>=3.9"],
        "wechat": ["pycryptodome>=3.20"],
        "qq": ["qq-botpy>=1.0"],
    }

    # Channel definitions: (value, display_name, required_fields, import_check, pip_extra)
    # import_check: module name to try importing; None = no check needed
    _CHANNELS = [
        (
            "telegram",
            "Telegram",
            [("telegram_bot_token", "Bot token (from @BotFather)")],
            "telegram",
            "telegram",
        ),
        (
            "discord",
            "Discord",
            [("discord_bot_token", "Bot token")],
            "discord",
            "discord",
        ),
        (
            "slack",
            "Slack",
            [
                ("slack_bot_token", "Bot token (xoxb-...)"),
                ("slack_app_token", "App token for Socket Mode (xapp-...)"),
            ],
            "slack_sdk",
            "slack",
        ),
        (
            "feishu",
            "Feishu",
            [("feishu_app_id", "App ID"), ("feishu_app_secret", "App Secret")],
            "aiohttp",
            "feishu",
        ),
        (
            "dingtalk",
            "DingTalk",
            [
                ("dingtalk_client_id", "Client ID (AppKey)"),
                ("dingtalk_client_secret", "Client Secret (AppSecret)"),
            ],
            "aiohttp",
            "dingtalk",
        ),
        (
            "wechat",
            "WeChat",
            [
                ("wechat_wecom_corp_id", "WeCom Corp ID"),
                ("wechat_wecom_agent_id", "WeCom Agent ID"),
                ("wechat_wecom_secret", "WeCom Secret"),
            ],
            "aiohttp",
            "wechat",
        ),
        (
            "email",
            "Email",
            [
                ("email_imap_host", "IMAP host"),
                ("email_imap_username", "IMAP username"),
                ("email_imap_password", "IMAP password"),
                ("email_smtp_host", "SMTP host"),
                ("email_smtp_username", "SMTP username"),
                ("email_smtp_password", "SMTP password"),
                ("email_from_address", "From address"),
            ],
            None,
            None,
        ),
        (
            "qq",
            "QQ",
            [("qq_app_id", "App ID"), ("qq_app_secret", "App Secret")],
            "botpy",
            "qq",
        ),
        (
            "signal",
            "Signal",
            [("signal_phone_number", "Phone number (E.164)")],
            None,
            None,
        ),
        ("imessage", "iMessage", [], None, None),  # handled via _setup_imessage()
    ]

    choices = [
        Choice(
            title=display,
            value=value,
            checked=value in _currently_enabled,
        )
        for value, display, *_ in _CHANNELS
    ]

    selected = questionary.checkbox(
        "Select channels to enable (Space to toggle, Enter to confirm):",
        choices=choices,
        style=WIZARD_STYLE,
        qmark=QMARK,
    ).ask()

    if selected is None:
        raise KeyboardInterrupt()

    updates: dict[str, object] = {}

    if not selected:
        updates["channel_enabled"] = ""
        updates["imessage_enabled"] = False
        return updates

    # Build a lookup for channel definitions
    _ch_lookup = {
        v: (v, d, fields, imp, extra) for v, d, fields, imp, extra in _CHANNELS
    }

    enabled_channels: list[str] = []

    for ch_name in selected:
        _, display, required_fields, import_check, pip_extra = _ch_lookup[ch_name]
        console.print(f"\n  [bold cyan]── {display} ──[/bold cyan]")

        # Check pip dependency before proceeding
        if import_check:
            _pkg_ready = False
            try:
                __import__(import_check)
                _pkg_ready = True
            except ImportError:
                console.print("  [yellow]✗ Required package not installed.[/yellow]")
                # Determine packages to install
                _pip_pkgs = _CHANNEL_PIP_DEPS.get(pip_extra, []) if pip_extra else []
                _pkg_display = ", ".join(_pip_pkgs) if _pip_pkgs else f"evoscientist[{pip_extra}]"
                install_now = questionary.confirm(
                    f"Install {_pkg_display} now?",
                    default=True,
                    style=WIZARD_STYLE,
                    qmark=f"  {QMARK}",
                ).ask()
                if install_now is None:
                    raise KeyboardInterrupt()
                if install_now:
                    console.print(
                        f"  [dim]Installing {_pkg_display}...[/dim]"
                    )
                    if _pip_pkgs:
                        _ok = all(_install_pip_package(p) for p in _pip_pkgs)
                    else:
                        _ok = _install_pip_package(f"evoscientist[{pip_extra}]")
                    if _ok:
                        # Verify the import actually works now
                        try:
                            __import__(import_check)
                            console.print("  [green]✓ Installed successfully.[/green]")
                            _pkg_ready = True
                        except ImportError:
                            console.print("  [red]✗ Package installed but import failed.[/red]")
                            console.print(
                                "  [dim]Try restarting and running:[/dim] evosci channel setup"
                            )
                    else:
                        console.print("  [red]✗ Installation failed.[/red]")
                        console.print(
                            f'  [dim]Run manually:[/dim] pip install {_pkg_display}'
                        )
            if not _pkg_ready:
                continue

        # Special handling for iMessage
        if ch_name == "imessage":
            ready = _setup_imessage()
            if not ready:
                console.print()
                enable_anyway = questionary.confirm(
                    "Enable iMessage anyway? (will try to connect on startup)",
                    default=False,
                    style=WIZARD_STYLE,
                    qmark=f"  {QMARK}",
                ).ask()
                if enable_anyway is None:
                    raise KeyboardInterrupt()
                if not enable_anyway:
                    continue
            # Allowed senders
            senders = questionary.text(
                "Allowed senders (comma-separated, empty = all):",
                default=getattr(config, "imessage_allowed_senders", ""),
                style=WIZARD_STYLE,
                qmark=f"  {QMARK}",
            ).ask()
            if senders is None:
                raise KeyboardInterrupt()
            updates["imessage_enabled"] = True
            updates["imessage_allowed_senders"] = senders.strip()
            enabled_channels.append("imessage")
            continue

        # Prompt for required fields
        for field_name, prompt_label in required_fields:
            current = getattr(config, field_name, "")
            value = questionary.text(
                f"{prompt_label}:",
                default=current,
                style=WIZARD_STYLE,
                qmark=f"  {QMARK}",
            ).ask()
            if value is None:
                raise KeyboardInterrupt()
            updates[field_name] = value.strip()

        # Feishu optional fields (verification_token & encrypt_key)
        if ch_name == "feishu":
            console.print(
                "  [dim]The following fields are optional (press Enter to skip):[/dim]"
            )
            for field_name, prompt_label in [
                ("feishu_verification_token", "Verification Token (optional)"),
                ("feishu_encrypt_key", "Encrypt Key (optional)"),
            ]:
                current = getattr(config, field_name, "")
                value = questionary.text(
                    f"{prompt_label}:",
                    default=current,
                    style=WIZARD_STYLE,
                    qmark=f"  {QMARK}",
                ).ask()
                if value is None:
                    raise KeyboardInterrupt()
                updates[field_name] = value.strip()

        # Allowed senders (common for all channels)
        senders_field = f"{ch_name}_allowed_senders"
        if hasattr(config, senders_field):
            senders = questionary.text(
                "Allowed senders (comma-separated, empty = all):",
                default=getattr(config, senders_field, ""),
                style=WIZARD_STYLE,
                qmark=f"  {QMARK}",
            ).ask()
            if senders is None:
                raise KeyboardInterrupt()
            updates[senders_field] = senders.strip()

        # Probe validation
        _probe_channel(ch_name, config, updates)

        enabled_channels.append(ch_name)

    updates["channel_enabled"] = ",".join(enabled_channels)
    # Keep legacy field in sync
    updates["imessage_enabled"] = "imessage" in enabled_channels

    # --- Common prompt: send thinking (shown when any channel is enabled) ---
    if enabled_channels:
        console.print("\n  [bold cyan]── Channel Settings ──[/bold cyan]")
        thinking_choices = [
            Choice(title="On (forward model reasoning)", value=True),
            Choice(title="Off (only send final responses)", value=False),
        ]

        send_thinking = questionary.select(
            "Send thinking panel in channel?",
            choices=thinking_choices,
            default=config.channel_send_thinking,
            style=WIZARD_STYLE,
            qmark=f"  {QMARK}",
            use_indicator=True,
        ).ask()

        if send_thinking is None:
            raise KeyboardInterrupt()

        updates["channel_send_thinking"] = send_thinking

    return updates


def _probe_channel(
    ch_name: str,
    config: EvoScientistConfig,
    updates: dict[str, object],
) -> None:
    """Run the probe for a channel type and print the result.

    Non-fatal: prints a warning on failure but does not prevent enabling.
    """
    import asyncio

    def _val(key: str, fallback: str = "") -> str:
        """Get a value from updates first, then config, then fallback."""
        if key in updates:
            return str(updates[key])
        return str(getattr(config, key, fallback))

    console.print("  [dim]Validating credentials...[/dim]")

    async def _run() -> tuple[bool, str]:
        if ch_name == "telegram":
            from ..channels.telegram.probe import validate_telegram_token

            return await validate_telegram_token(
                _val("telegram_bot_token"),
                _val("telegram_proxy") or None,
            )
        elif ch_name == "discord":
            from ..channels.discord.probe import validate_discord_token

            return await validate_discord_token(
                _val("discord_bot_token"),
                _val("discord_proxy") or None,
            )
        elif ch_name == "slack":
            from ..channels.slack.probe import validate_slack_tokens

            return await validate_slack_tokens(
                _val("slack_bot_token"),
                _val("slack_app_token") or None,
                _val("slack_proxy") or None,
            )
        elif ch_name == "wechat":
            backend = _val("wechat_backend", "wecom")
            if backend == "wechatmp":
                from ..channels.wechat.probe import validate_wechat_mp

                return await validate_wechat_mp(
                    _val("wechat_mp_app_id"),
                    _val("wechat_mp_app_secret"),
                    _val("wechat_proxy") or None,
                )
            else:
                from ..channels.wechat.probe import validate_wecom

                return await validate_wecom(
                    _val("wechat_wecom_corp_id"),
                    _val("wechat_wecom_secret"),
                    _val("wechat_proxy") or None,
                )
        elif ch_name == "feishu":
            from ..channels.feishu.probe import validate_feishu_credentials

            return await validate_feishu_credentials(
                _val("feishu_app_id"),
                _val("feishu_app_secret"),
                _val("feishu_domain", "https://open.feishu.cn"),
            )
        elif ch_name == "dingtalk":
            from ..channels.dingtalk.probe import validate_dingtalk

            return await validate_dingtalk(
                _val("dingtalk_client_id"),
                _val("dingtalk_client_secret"),
                _val("dingtalk_proxy") or None,
            )
        elif ch_name == "email":
            from ..channels.email.probe import validate_email_imap

            return await validate_email_imap(
                _val("email_imap_host"),
                int(_val("email_imap_port", "993")),
                _val("email_imap_username"),
                _val("email_imap_password"),
                _val("email_imap_use_ssl", "True").lower() not in ("false", "0", "no"),
            )
        elif ch_name == "qq":
            from ..channels.qq.probe import validate_qq

            return await validate_qq(
                _val("qq_app_id"),
                _val("qq_app_secret"),
            )
        elif ch_name == "signal":
            from ..channels.signal.probe import validate_signal

            return await validate_signal(
                _val("signal_phone_number"),
                _val("signal_cli_path", "signal-cli"),
                int(_val("signal_rpc_port", "7583")),
            )
        else:
            return True, "No probe available"

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio  # type: ignore[import-untyped]

                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        ok, detail = loop.run_until_complete(_run())
        if ok:
            console.print(f"  [green]✓ {detail}[/green]")
        else:
            console.print(f"  [yellow]⚠ {detail}[/yellow]")
            console.print(
                "  [dim]Channel will still be enabled — check credentials later.[/dim]"
            )
    except Exception as e:
        console.print(f"  [yellow]⚠ Could not validate: {e}[/yellow]")
        console.print(
            "  [dim]Channel will still be enabled — check credentials later.[/dim]"
        )


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

        # Step 0: UI Backend
        ui_backend = _step_ui_backend(config)
        config.ui_backend = ui_backend

        # Step 1: Provider
        provider = _step_provider(config)
        config.provider = provider

        # Step 2a: Base URL (custom or ollama provider)
        ollama_detected_models: list[str] = []
        if provider == "custom":
            base_url = _step_base_url(config)
            config.custom_base_url = base_url
        elif provider == "ollama":
            ollama_url, ollama_detected_models = _step_ollama_base_url(config)
            config.ollama_base_url = ollama_url

        # Step 2b: Provider API Key (skip for Ollama — no key needed)
        # Maps provider name → config attribute for the API key.
        _PROVIDER_KEY_ATTR = {
            "anthropic": "anthropic_api_key",
            "nvidia": "nvidia_api_key",
            "google-genai": "google_api_key",
            "siliconflow": "siliconflow_api_key",
            "openrouter": "openrouter_api_key",
            "zhipu": "zhipu_api_key",
            "zhipu-code": "zhipu_api_key",
            "custom": "custom_api_key",
        }
        if provider != "ollama":
            new_key = _step_provider_api_key(config, provider, skip_validation)
            key_attr = _PROVIDER_KEY_ATTR.get(provider, "openai_api_key")
            if new_key is not None:
                setattr(config, key_attr, new_key)
            elif not getattr(config, key_attr):
                _print_step_skipped("API Key", "not set")

        # Step 3: Model
        model = _step_model(
            config, provider, ollama_detected_models=ollama_detected_models
        )
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

        # Step 6: Thinking
        show_thinking = _step_thinking(config)
        config.show_thinking = show_thinking

        # Step 7: Skills
        _step_skills()

        # Step 8: MCP Servers
        _step_mcp_servers()

        # Step 9: Channels
        channel_updates = _step_channels(config)
        for key, value in channel_updates.items():
            setattr(config, key, value)

        # Confirm save
        console.print()
        save = questionary.confirm(
            "Save this configuration?",
            default=True,
            style=CONFIRM_STYLE,
            qmark=QMARK,
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
