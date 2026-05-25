"""Onboarding wizard entry point and progress display."""

from __future__ import annotations

import copy
import os

import questionary
from rich.panel import Panel
from rich.text import Text

from ..settings import (
    EvoScientistConfig,
    get_config_path,
    load_config,
    save_config,
)
from .channels import _step_channels
from .steps import (
    _step_anthropic_auth_mode,
    _step_base_url,
    _step_langgraph_dev_port,
    _step_mcp_servers,
    _step_minimax_region,
    _step_model,
    _step_ollama_base_url,
    _step_openai_auth_mode,
    _step_provider,
    _step_provider_api_key,
    _step_reasoning_effort,
    _step_skills,
    _step_tavily_key,
    _step_thinking,
    _step_tinytex,
    _step_ui_backend,
    _step_workspace,
)
from .style import (
    CONFIRM_STYLE,
    QMARK,
    _print_header,
    _print_step_skipped,
    console,
)

STEPS = [
    "UI",
    "LangGraph Port",
    "Provider",
    "API Key",
    "Model",
    "Tavily Key",
    "Workspace",
    "Thinking",
    "Skills",
    "MCP Servers",
    "LaTeX",
    "Channels",
]


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


_PROVIDER_KEY_ATTR = {
    "anthropic": "anthropic_api_key",
    "minimax": "minimax_api_key",
    "nvidia": "nvidia_api_key",
    "google-genai": "google_api_key",
    "siliconflow": "siliconflow_api_key",
    "openrouter": "openrouter_api_key",
    "deepseek": "deepseek_api_key",
    "zhipu": "zhipu_api_key",
    "zhipu-code": "zhipu_api_key",
    "volcengine": "volcengine_api_key",
    "dashscope": "dashscope_api_key",
    "dashscope-code": "dashscope_api_key",
    "moonshot": "moonshot_api_key",
    "kimi-coding": "kimi_api_key",
    "custom-openai": "custom_openai_api_key",
    "custom-anthropic": "custom_anthropic_api_key",
}


def _autosave(config: EvoScientistConfig) -> None:
    """Persist current config to disk between phases.

    Silently swallows IO errors so a transient disk issue doesn't abort the
    wizard — the final save at the end will surface anything broken.
    """
    try:
        save_config(config)
    except Exception:
        pass


# Sections offered in Keep/Modify/Reset → which step labels they enable.
_SECTION_LABELS: list[tuple[str, str]] = [
    ("ui", "UI backend"),
    ("port", "LangGraph server port"),
    ("provider", "LLM provider + auth + API key"),
    ("model", "Model + reasoning effort"),
    ("tavily", "Tavily search key"),
    ("workspace", "Workspace mode"),
    ("thinking", "Thinking panel"),
    ("skills", "Skills"),
    ("mcp", "MCP servers"),
    ("latex", "LaTeX (TinyTeX)"),
    ("channels", "Channels"),
]
_ALL_SECTIONS: frozenset[str] = frozenset(s for s, _ in _SECTION_LABELS)


def _config_has_meaningful_settings(config: EvoScientistConfig) -> bool:
    """True if the user has been through onboarding before.

    Compares ``config`` against fresh ``EvoScientistConfig()`` defaults — any
    non-default field means the user has customised something previously.
    """
    import dataclasses

    default = EvoScientistConfig()
    return any(
        getattr(config, f.name) != getattr(default, f.name)
        for f in dataclasses.fields(config)
    )


def _open_existing_config_prompt(
    config: EvoScientistConfig,
) -> tuple[frozenset[str], EvoScientistConfig] | None:
    """Offer Keep / Modify / Reset on an existing config.

    Returns:
        - ``None`` if user chose Keep (wizard should exit early).
        - ``(sections, config)`` otherwise: the sections to run and the
          (possibly reset) config to operate on.
    """
    from questionary import Choice

    from .style import QMARK, WIZARD_STYLE

    choice = questionary.select(
        "Found existing configuration. What would you like to do?",
        choices=[
            Choice(title="Keep current configuration — exit wizard", value="keep"),
            Choice(title="Modify — pick specific sections to update", value="modify"),
            Choice(title="Reset — start over from defaults", value="reset"),
        ],
        default="modify",
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()

    if choice is None:
        raise KeyboardInterrupt()

    if choice == "keep":
        console.print()
        console.print("[green]✓ Keeping current configuration.[/green]")
        console.print(f"[dim]  → {get_config_path()}[/dim]")
        console.print()
        return None

    if choice == "reset":
        console.print()
        console.print("[yellow]Resetting to defaults …[/yellow]")
        return _ALL_SECTIONS, EvoScientistConfig()

    # Modify: ask which sections.
    from .style import _checkbox_ask

    section_choices = [
        Choice(title=label, value=sid, checked=False) for sid, label in _SECTION_LABELS
    ]
    selected = _checkbox_ask(
        section_choices,
        "Which sections to update? (Space to toggle, Enter to confirm)",
    )
    if selected is None:
        raise KeyboardInterrupt()
    if not selected:
        # No section picked → effectively the same as Keep.
        console.print()
        console.print(
            "[green]✓ Nothing selected. Keeping current configuration.[/green]"
        )
        console.print()
        return None
    return frozenset(selected), config


def run_onboard(
    skip_validation: bool = False,
    prompter=None,
    only_sections: set[str] | frozenset[str] | None = None,
) -> bool:
    """Run the interactive onboarding wizard.

    Args:
        skip_validation: Skip API key validation.
        prompter: Optional :class:`Prompter` instance. When a
            :class:`NonInteractivePrompter` is passed, prompts whose answers
            are pre-supplied (via ``--provider``, ``--model``, …) skip their
            interactive form; sections in the prompter's ``skip_set`` are
            skipped entirely.
        only_sections: If given, restrict the wizard to exactly these section
            ids — the Keep/Modify/Reset prompt is skipped. Used by ``EvoSci
            configure <section>`` to re-run a single phase.

    Returns:
        True if configuration was saved, False if cancelled.

    Behaviour notes
    ---------------
    Config is **autosaved between phases**: each completed section is written
    to ``~/.config/evoscientist/config.yaml`` immediately, so a Ctrl+C does
    not lose what's been answered so far. The final ``Save this configuration?``
    prompt is the user's chance to *revert* — declining writes the original
    snapshot back to disk.

    .. warning::
        Revert covers **the YAML config file only**. Sections with filesystem
        side effects — ``_step_skills`` (downloads + ``npm`` installs),
        ``_step_mcp_servers`` (writes to ``mcp.yaml``), ``_step_tinytex``
        (installs TinyTeX), and ``_step_channels`` (may ``pip install``
        channel deps) — execute their side effects *before* the final
        confirmation and are **not** rolled back when the user declines to
        save. "No" thus restores the YAML but does not uninstall packages,
        delete skill files, or remove MCP server entries.
    """
    from .prompter import NonInteractivePrompter, select_navigation_active

    p = prompter if isinstance(prompter, NonInteractivePrompter) else None
    strict = bool(p and getattr(p, "_strict_non_interactive", False))

    _PROVIDER_API_KEY_FLAG_MAP = {
        "anthropic": "anthropic_api_key",
        "openai": "openai_api_key",
        "minimax": "minimax_api_key",
        "nvidia": "nvidia_api_key",
        "google-genai": "google_api_key",
        "siliconflow": "siliconflow_api_key",
        "openrouter": "openrouter_api_key",
        "deepseek": "deepseek_api_key",
        "zhipu": "zhipu_api_key",
        "volcengine": "volcengine_api_key",
        "dashscope": "dashscope_api_key",
        "moonshot": "moonshot_api_key",
        "kimi-coding": "kimi_api_key",
        "custom-openai": "custom_openai_api_key",
        "custom-anthropic": "custom_anthropic_api_key",
    }

    def _preset(pid: str):
        """Return preset answer for ``pid`` if available, else None."""
        return p.answers.get(pid) if p else None

    def _require(pid: str, label: str) -> None:
        if strict and not (p and p.has(pid)):
            flag = "--" + pid.replace("_", "-")
            raise RuntimeError(
                f"--non-interactive: missing required answer for {label!r}. "
                f"Pass {flag} on the command line."
            )

    try:
        with select_navigation_active():
            # Print header once
            _print_header()

            # Load existing config as starting point + snapshot for revert.
            config = load_config()
            snapshot = copy.deepcopy(config)

            # When `only_sections` is given (e.g. `configure provider`), bypass
            # Keep/Modify/Reset and run only those sections.
            sections_to_run: frozenset[str]
            if only_sections is not None:
                sections_to_run = frozenset(only_sections)
            else:
                sections_to_run = _ALL_SECTIONS
                if not strict and _config_has_meaningful_settings(config):
                    result = _open_existing_config_prompt(config)
                    if result is None:
                        return True  # Keep
                    sections_to_run, config = result
                    # NOTE: ``snapshot`` is intentionally NOT refreshed after Reset.
                    # "Save? = No" must restore the user's pre-wizard config — if
                    # we re-snapped here, declining the save after Reset would
                    # silently overwrite the user's previous settings with
                    # ``EvoScientistConfig()`` defaults.

            # CLI --skip-* flags remove sections entirely.
            if p and p.skip_set:
                sections_to_run = sections_to_run - p.skip_set

            console.print(
                "[dim]  Progress is autosaved after every step. Ctrl+C is safe.[/dim]"
            )
            console.print()

            if "ui" in sections_to_run:
                _require("ui", "UI backend")
                preset_ui = _preset("ui")
                if preset_ui is not None:
                    config.ui_backend = preset_ui
                    console.print(
                        f"  [green]✓ UI: {preset_ui}[/green]   [dim](--ui)[/dim]"
                    )
                else:
                    config.ui_backend = _step_ui_backend(config)
                _autosave(config)

            if "port" in sections_to_run:
                preset_port = _preset("port")
                if preset_port is not None:
                    config.langgraph_dev_port = int(preset_port)
                    console.print(
                        f"  [green]✓ Port: {preset_port}[/green]   [dim](--port)[/dim]"
                    )
                else:
                    config.langgraph_dev_port = _step_langgraph_dev_port(config)
                _autosave(config)

            ollama_detected_models: list[str] = []
            if "provider" in sections_to_run:
                from .prompter import GoBack

                _require("provider", "LLM provider")
                # Provider sub-loop: auth_mode can raise GoBack to re-pick provider.
                # We snapshot config at the top of each iteration so a GoBack can
                # roll back partial writes (base_url, minimax region, ollama URL,
                # provider id itself) — otherwise picking `custom-openai`, entering
                # a base URL, going Back, then picking `anthropic` would leave a
                # stale ``custom_openai_base_url`` in the final saved config.
                while True:
                    loop_snapshot = copy.deepcopy(config)
                    preset_provider = _preset("provider")
                    if preset_provider is not None:
                        provider = preset_provider
                        config.provider = provider
                        console.print(
                            f"  [green]✓ Provider: {provider}[/green]   "
                            "[dim](--provider)[/dim]"
                        )
                    else:
                        provider = _step_provider(config)
                        config.provider = provider

                    # Step 2a: Base URL (custom-openai, custom-anthropic, minimax, ollama)
                    if provider == "custom-openai":
                        current_base_url = (
                            config.custom_openai_base_url
                            or os.environ.get("CUSTOM_OPENAI_BASE_URL", "")
                        )
                        config.custom_openai_base_url = _step_base_url(
                            config, current_value=current_base_url
                        )
                    elif provider == "custom-anthropic":
                        current_base_url = (
                            config.custom_anthropic_base_url
                            or os.environ.get("CUSTOM_ANTHROPIC_BASE_URL", "")
                        )
                        config.custom_anthropic_base_url = _step_base_url(
                            config, current_value=current_base_url
                        )
                    elif provider == "minimax":
                        config.minimax_base_url = _step_minimax_region(config)
                    elif provider == "ollama":
                        ollama_url, ollama_detected_models = _step_ollama_base_url(
                            config
                        )
                        config.ollama_base_url = ollama_url

                    # Step 2b: Auth mode (Anthropic or OpenAI — API key vs OAuth).
                    # In strict non-interactive mode we assume "api_key".
                    # The prompt offers a `← Back` choice that raises GoBack so
                    # the user can re-pick the provider without exiting the wizard.
                    try:
                        if provider == "anthropic":
                            if strict:
                                config.anthropic_auth_mode = "api_key"
                            else:
                                config.anthropic_auth_mode = _step_anthropic_auth_mode(
                                    config
                                )
                        elif provider == "openai":
                            if strict:
                                config.openai_auth_mode = "api_key"
                            else:
                                config.openai_auth_mode = _step_openai_auth_mode(config)
                        else:
                            # Non-Anthropic/OpenAI provider: reset OAuth modes to
                            # avoid stale oauth config triggering ccproxy at startup.
                            config.anthropic_auth_mode = "api_key"
                            config.openai_auth_mode = "api_key"
                    except GoBack:
                        # User picked "← Back" — restore config to its state at the
                        # top of this iteration (drops any base_url / region /
                        # provider writes), then discard the preset and re-prompt.
                        for field_name in vars(loop_snapshot):
                            setattr(
                                config, field_name, getattr(loop_snapshot, field_name)
                            )
                        if p:
                            p.answers.pop("provider", None)
                        ollama_detected_models = []
                        console.print("  [dim]↩ Returning to provider selection.[/dim]")
                        continue
                    break  # auth_mode succeeded — exit sub-loop

                # Step 2c: Provider API Key (skip for Ollama and pure OAuth)
                _skip_api_key = (
                    provider == "ollama"
                    or (
                        provider == "anthropic"
                        and config.anthropic_auth_mode == "oauth"
                    )
                    or (provider == "openai" and config.openai_auth_mode == "oauth")
                )
                if not _skip_api_key:
                    key_attr = _PROVIDER_KEY_ATTR.get(provider, "openai_api_key")
                    preset_api_key = _preset("api_key")
                    if preset_api_key is not None:
                        setattr(config, key_attr, preset_api_key)
                        console.print(
                            f"  [green]✓ API key: ***{preset_api_key[-4:]}[/green]"
                            "   [dim](--api-key)[/dim]"
                        )
                    else:
                        _require("api_key", f"{provider} API key")
                        new_key = _step_provider_api_key(
                            config, provider, skip_validation
                        )
                        if new_key is not None:
                            setattr(config, key_attr, new_key)
                        elif not getattr(config, key_attr):
                            _print_step_skipped("API Key", "not set")
                _autosave(config)
            else:
                # Provider section skipped — keep prior provider value to drive
                # downstream sections that depend on it (e.g., model picker).
                provider = config.provider

            if "model" in sections_to_run:
                _require("model", "Model")
                preset_model = _preset("model")
                if preset_model is not None:
                    config.model = preset_model
                    console.print(
                        f"  [green]✓ Model: {preset_model}[/green]   [dim](--model)[/dim]"
                    )
                else:
                    config.model = _step_model(
                        config, provider, ollama_detected_models=ollama_detected_models
                    )
                if provider == "openrouter" and _preset("model") is None:
                    config.reasoning_effort = _step_reasoning_effort(config)
                _autosave(config)

            if "tavily" in sections_to_run:
                preset_tavily = _preset("tavily_key")
                if preset_tavily is not None:
                    config.tavily_api_key = preset_tavily
                    console.print(
                        f"  [green]✓ Tavily key: ***{preset_tavily[-4:]}[/green]"
                        "   [dim](--tavily-key)[/dim]"
                    )
                else:
                    new_tavily_key = _step_tavily_key(config, skip_validation)
                    if new_tavily_key is not None:
                        config.tavily_api_key = new_tavily_key
                    elif not config.tavily_api_key:
                        _print_step_skipped("Tavily Key", "not set")
                _autosave(config)

            if "workspace" in sections_to_run:
                _require("workspace_mode", "Workspace mode")
                preset_ws = _preset("workspace_mode")
                if preset_ws is not None:
                    config.default_mode = preset_ws
                    console.print(
                        f"  [green]✓ Workspace: {preset_ws}[/green]"
                        "   [dim](--workspace-mode)[/dim]"
                    )
                else:
                    config.default_mode = _step_workspace(config)
                _autosave(config)

            if "thinking" in sections_to_run:
                _require("show_thinking", "Thinking panel")
                preset_thinking = _preset("show_thinking")
                if preset_thinking is not None:
                    config.show_thinking = bool(preset_thinking)
                    console.print(
                        f"  [green]✓ Thinking: {'on' if preset_thinking else 'off'}[/green]"
                        "   [dim](--show-thinking)[/dim]"
                    )
                else:
                    config.show_thinking = _step_thinking(config)
                _autosave(config)

            if "skills" in sections_to_run:
                _step_skills()

            if "mcp" in sections_to_run:
                _step_mcp_servers()

            if "latex" in sections_to_run:
                _step_tinytex()

            if "channels" in sections_to_run:
                for key, value in _step_channels(config).items():
                    setattr(config, key, value)
                _autosave(config)

            # Final confirmation — opportunity to revert. In strict
            # non-interactive mode, skip the prompt and commit silently.
            if strict:
                save = True
            else:
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
                # User declined — restore the pre-wizard snapshot.
                save_config(snapshot)
                console.print()
                console.print(
                    "[yellow]Reverted to previous configuration "
                    "(autosaved progress discarded).[/yellow]"
                )
                console.print()
                return False

    except KeyboardInterrupt:
        console.print()
        console.print(
            "[yellow]Setup interrupted. "
            "Progress through the last completed step has been autosaved.[/yellow]"
        )
        console.print(
            f"[dim]  Run [bold]EvoSci onboard[/bold] again to resume — "
            f"answers persist in {get_config_path()}.[/dim]"
        )
        console.print()
        return False
