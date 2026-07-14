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
    _step_auxiliary_enable,
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
    _step_webui_port,
    _step_workspace,
)
from .style import (
    CONFIRM_STYLE,
    QMARK,
    _print_header,
    _print_section,
    _print_step_skipped,
    console,
)

STEPS = [
    "UI",
    "LangGraph Port",
    "Provider",
    "API Key",
    "Model",
    "Auxiliary Model",
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

_MINIMAX_GLOBAL_BASE_URL = "https://api.minimax.io/anthropic"
_CUSTOM_PROVIDER_BASE_URL = {
    "custom-openai": ("custom_openai_base_url", "CUSTOM_OPENAI_BASE_URL"),
    "custom-anthropic": ("custom_anthropic_base_url", "CUSTOM_ANTHROPIC_BASE_URL"),
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


def _configure_provider_base_url(
    config: EvoScientistConfig,
    provider: str,
    *,
    strict: bool,
) -> list[str]:
    """Configure provider-specific base URL/region and return Ollama models."""
    if provider in _CUSTOM_PROVIDER_BASE_URL:
        attr_name, env_name = _CUSTOM_PROVIDER_BASE_URL[provider]
        current_base_url = getattr(config, attr_name) or os.environ.get(env_name, "")
        if strict:
            if not current_base_url:
                raise RuntimeError(
                    f"--non-interactive: {provider} provider needs a base URL. "
                    f"Set the {env_name} env var or run without --non-interactive."
                )
            setattr(config, attr_name, current_base_url)
        else:
            setattr(
                config,
                attr_name,
                _step_base_url(config, current_value=current_base_url),
            )
    elif provider == "minimax":
        if strict:
            config.minimax_base_url = (
                config.minimax_base_url or _MINIMAX_GLOBAL_BASE_URL
            )
        else:
            config.minimax_base_url = _step_minimax_region(config)
    elif provider == "ollama":
        if strict:
            config.ollama_base_url = (
                config.ollama_base_url
                or os.environ.get("OLLAMA_BASE_URL", "")
                or "http://localhost:11434"
            )
        else:
            ollama_url, ollama_detected_models = _step_ollama_base_url(config)
            config.ollama_base_url = ollama_url
            return ollama_detected_models
    return []


def _configure_provider_auth_mode(
    config: EvoScientistConfig,
    provider: str,
    *,
    strict: bool,
) -> None:
    """Configure Anthropic/OpenAI auth mode for the selected provider."""
    if provider == "anthropic":
        if strict:
            config.anthropic_auth_mode = "api_key"
        else:
            config.anthropic_auth_mode = _step_anthropic_auth_mode(config)
    elif provider == "openai":
        if strict:
            config.openai_auth_mode = "api_key"
        else:
            config.openai_auth_mode = _step_openai_auth_mode(config)


def _active_llm_providers(config: EvoScientistConfig) -> set[str]:
    """Return providers currently selected by the main and auxiliary models."""
    providers = {config.provider}
    if config.auxiliary_provider:
        providers.add(config.auxiliary_provider)
    return providers


def _reconcile_oauth_modes(config: EvoScientistConfig) -> None:
    """Clear OAuth flags for providers no selected model uses."""
    active_providers = _active_llm_providers(config)
    if "anthropic" not in active_providers:
        config.anthropic_auth_mode = "api_key"
    if "openai" not in active_providers:
        config.openai_auth_mode = "api_key"


def _provider_uses_oauth(config: EvoScientistConfig, provider: str) -> bool:
    return (provider == "anthropic" and config.anthropic_auth_mode == "oauth") or (
        provider == "openai" and config.openai_auth_mode == "oauth"
    )


def _apply_preset_provider_api_key(
    config: EvoScientistConfig,
    provider: str,
    preset_api_key: str,
    *,
    skip_validation: bool,
) -> None:
    """Validate and store a CLI-supplied provider API key."""
    if not skip_validation:
        from .helpers import _provider_key_info

        _info = _provider_key_info(config, provider)
        validate_fn = _info[2] if _info else None
        if validate_fn is not None:
            console.print("  [dim]Validating preset API key...[/dim]", end="")
            valid, msg = validate_fn(preset_api_key)
            if valid:
                console.print(f"\r  [green]✓ {msg}[/green]      ")
            else:
                console.print(f"\r  [red]✗ {msg}[/red]      ")
                raise RuntimeError(
                    f"--api-key rejected by {provider} validator: {msg}. "
                    "Pass --skip-validation to override."
                )

    key_attr = _PROVIDER_KEY_ATTR.get(provider, "openai_api_key")
    setattr(config, key_attr, preset_api_key)
    console.print(
        f"  [green]✓ API key: ***{preset_api_key[-4:]}[/green]   [dim](--api-key)[/dim]"
    )


def _configure_provider_api_key(
    config: EvoScientistConfig,
    provider: str,
    *,
    skip_validation: bool,
    preset_api_key: str | None = None,
    require_api_key=None,
) -> None:
    """Configure provider API key unless the provider does not need one."""
    if provider == "ollama" or _provider_uses_oauth(config, provider):
        return

    key_attr = _PROVIDER_KEY_ATTR.get(provider, "openai_api_key")
    if preset_api_key is not None:
        _apply_preset_provider_api_key(
            config,
            provider,
            preset_api_key,
            skip_validation=skip_validation,
        )
        return

    if require_api_key is not None:
        require_api_key()
    new_key = _step_provider_api_key(config, provider, skip_validation)
    if new_key is not None:
        setattr(config, key_attr, new_key)
    elif not getattr(config, key_attr):
        _print_step_skipped("API Key", "not set")


def _provider_connection_configured(config: EvoScientistConfig, provider: str) -> bool:
    """Return True when provider-level setup can be safely reused."""
    if provider == "ollama":
        return bool(config.ollama_base_url)
    if provider == "custom-openai" and not config.custom_openai_base_url:
        return False
    if provider == "custom-anthropic" and not config.custom_anthropic_base_url:
        return False
    if provider == "minimax" and not config.minimax_base_url:
        return False
    if _provider_uses_oauth(config, provider):
        return True
    key_attr = _PROVIDER_KEY_ATTR.get(provider, "openai_api_key")
    return bool(getattr(config, key_attr))


def _configure_provider_connection(
    config: EvoScientistConfig,
    provider: str,
    *,
    strict: bool,
    skip_validation: bool,
    preset_api_key: str | None = None,
    require_api_key=None,
) -> list[str]:
    """Configure provider base URL/region, auth mode, and API key."""
    ollama_detected_models = _configure_provider_base_url(
        config,
        provider,
        strict=strict,
    )
    _configure_provider_auth_mode(
        config,
        provider,
        strict=strict,
    )
    _configure_provider_api_key(
        config,
        provider,
        skip_validation=skip_validation,
        preset_api_key=preset_api_key,
        require_api_key=require_api_key,
    )
    return ollama_detected_models


# Sections offered in Keep/Modify/Reset → which step labels they enable.
_SECTION_LABELS: list[tuple[str, str]] = [
    ("ui", "UI backend"),
    ("port", "LangGraph server port"),
    ("provider", "LLM provider + auth + API key"),
    ("model", "Model + reasoning effort"),
    ("auxiliary_model", "Auxiliary model (optional)"),
    ("tavily", "Tavily search key"),
    ("workspace", "Workspace mode"),
    ("thinking", "Thinking panel"),
    ("skills", "Skills"),
    ("mcp", "MCP servers"),
    ("latex", "LaTeX (TinyTeX)"),
    ("channels", "Channels"),
]
_ALL_SECTIONS: frozenset[str] = frozenset(s for s, _ in _SECTION_LABELS)

# Each preset flag implies the section(s) it would change. ``--provider`` also
# cascades into ``model`` because the model list depends on the provider —
# silently keeping a stale model id would leave the first request broken.
_FLAG_TO_SECTIONS: dict[str, frozenset[str]] = {
    "ui": frozenset({"ui"}),
    "port": frozenset({"port"}),
    "provider": frozenset({"provider", "model"}),
    # ``--api-key`` re-runs the provider section, which can change provider —
    # cascade to model for the same reason ``--provider`` does.
    "api_key": frozenset({"provider", "model"}),
    "model": frozenset({"model"}),
    "tavily_key": frozenset({"tavily"}),
    "workspace_mode": frozenset({"workspace"}),
    "show_thinking": frozenset({"thinking"}),
}


def _sections_implied_by_flags(prompter) -> frozenset[str]:
    """Sections the user's flag-supplied answers imply should run.

    Empty frozenset means no preset flags were passed (only ``--skip-*`` or
    ``--non-interactive`` or no flags at all).
    """
    if prompter is None:
        return frozenset()
    out: set[str] = set()
    for pid in prompter.answers:
        out |= _FLAG_TO_SECTIONS.get(pid, set())
    return frozenset(out)


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
        prompter: Optional :class:`NonInteractivePrompter` carrying
            CLI-supplied answers (``--provider``, ``--model``, …) and
            ``skip_set`` (sections to bypass). When None, all prompts
            fall through to the interactive questionary form.
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
    strict = bool(p and p.strict)

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
            # Also capture raw file state so a "No" at the final save can
            # restore the exact pre-wizard byte content (or remove the file
            # entirely if it did not exist before). ``existed`` and
            # ``bytes`` are tracked independently so a read failure on a
            # file that DID exist doesn't get downgraded to "no file" —
            # which would cause revert to delete the user's config.
            config = load_config()
            snapshot = copy.deepcopy(config)
            config_path = get_config_path()
            original_file_existed = config_path.exists()
            original_file_bytes: bytes | None = None
            if original_file_existed:
                try:
                    original_file_bytes = config_path.read_bytes()
                except OSError as exc:
                    console.print(
                        "[yellow]Warning: could not snapshot existing config "
                        f"bytes ({exc}); revert will fall back to a re-serialized "
                        "snapshot, which may not preserve comments / unknown "
                        "fields.[/yellow]"
                    )

            # Decide which sections this run should cover.
            #
            #   - ``only_sections`` (programmatic, e.g. ``configure provider``):
            #     run exactly those sections, no Keep/Modify/Reset prompt.
            #   - Any preset flag (``--provider``/``--model``/…): treat as
            #     explicit user intent — skip Keep/Modify/Reset and run ONLY
            #     the sections each flag implies (see ``_FLAG_TO_SECTIONS``).
            #   - Strict ``--non-interactive`` with no preset flags: run all
            #     sections; the inner ``_require()`` calls will raise on
            #     missing answers.
            #   - Otherwise: full wizard, with Keep/Modify/Reset offered when
            #     an existing config is detected.
            sections_to_run: frozenset[str]
            implied_sections = _sections_implied_by_flags(p)
            if only_sections is not None:
                sections_to_run = frozenset(only_sections)
            elif implied_sections:
                sections_to_run = implied_sections
                console.print(
                    "[dim]  CLI flags detected — running only the implied "
                    f"sections: {', '.join(sorted(implied_sections))}.[/dim]"
                )
                console.print(
                    "[dim]  (Use 'EvoSci configure <section>' or 'EvoSci "
                    "onboard' with no flags to revisit other sections.)[/dim]"
                )
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

            # In strict --non-interactive mode, optional sections that have
            # no flag-driven equivalent (skills / mcp / latex / channels)
            # would otherwise still open their interactive pickers — that
            # breaks the "no prompts" contract advertised by the flag.
            # Auto-skip them unless the caller explicitly opted in by NOT
            # passing the corresponding --skip-* flag AND providing answers.
            # Today none of these have preset support, so always auto-skip.
            if strict:
                sections_to_run = sections_to_run - {
                    "skills",
                    "mcp",
                    "latex",
                    "channels",
                }

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
                # WebUI mode needs a front-end port; ask right after the mode
                # choice (only when chosen interactively — non-interactive /
                # preset runs keep the config default).
                if config.ui_backend == "webui" and not strict and preset_ui is None:
                    config.webui_port = _step_webui_port(config)
                _autosave(config)

            if "port" in sections_to_run:
                preset_port = _preset("port")
                if preset_port is not None:
                    config.langgraph_dev_port = int(preset_port)
                    console.print(
                        f"  [green]✓ Port: {preset_port}[/green]   [dim](--port)[/dim]"
                    )
                elif strict:
                    # ``--non-interactive`` without ``--port`` — keep the
                    # existing config value (has a sensible default in
                    # ``EvoScientistConfig``) instead of opening the
                    # questionary prompt and hanging the wizard.
                    console.print(
                        f"  [green]✓ Port: {config.langgraph_dev_port} "
                        "(kept)[/green]   [dim](no --port; non-interactive)[/dim]"
                    )
                else:
                    config.langgraph_dev_port = _step_langgraph_dev_port(config)
                _autosave(config)

            ollama_detected_models: list[str] = []
            if "provider" in sections_to_run:
                from .prompter import GoBack

                _print_section("EvoScientist · Pilot (Main model)")
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

                    try:
                        ollama_detected_models = _configure_provider_connection(
                            config,
                            provider,
                            strict=strict,
                            skip_validation=skip_validation,
                            preset_api_key=_preset("api_key"),
                            require_api_key=lambda provider=provider: _require(
                                "api_key", f"{provider} API key"
                            ),
                        )
                    except GoBack:
                        # User picked "← Back" — restore config to its state at the
                        # top of this iteration (drops any base_url / region /
                        # provider writes), then discard ALL provider-coupled
                        # presets and re-prompt. Clearing only ``provider``
                        # leaves a stale ``--model`` / ``--api-key`` that would
                        # be re-applied under a different provider, producing
                        # an invalid pair (e.g. ``provider=openai`` +
                        # ``model=claude-sonnet-4-6``).
                        for field_name in vars(loop_snapshot):
                            setattr(
                                config, field_name, getattr(loop_snapshot, field_name)
                            )
                        if p:
                            for stale_key in ("provider", "model", "api_key"):
                                p.answers.pop(stale_key, None)
                        ollama_detected_models = []
                        console.print("  [dim]↩ Returning to provider selection.[/dim]")
                        continue
                    break  # Provider setup succeeded — exit sub-loop

                _reconcile_oauth_modes(config)
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

            if "auxiliary_model" in sections_to_run:
                _print_section("Co-pilot (Auxiliary model)")
                if strict:
                    # Optional; never prompt under --non-interactive. Keep
                    # current (default empty = use main model).
                    _print_step_skipped(
                        "Auxiliary Model",
                        "kept current" if config.auxiliary_model else "not set",
                    )
                elif _step_auxiliary_enable(config):
                    from .prompter import GoBack

                    aux_ollama_detected_models: list[str] = []
                    while True:
                        loop_snapshot = copy.deepcopy(config)
                        aux_provider = _step_provider(
                            config,
                            label="co-pilot",
                            default_value=config.auxiliary_provider,
                        )
                        config.auxiliary_provider = aux_provider
                        if (
                            aux_provider == config.provider
                            and _provider_connection_configured(config, aux_provider)
                        ):
                            if aux_provider == "ollama":
                                aux_ollama_detected_models = ollama_detected_models
                            _print_step_skipped(
                                "Co-pilot credentials",
                                "reusing main provider settings",
                            )
                        else:
                            try:
                                aux_ollama_detected_models = (
                                    _configure_provider_connection(
                                        config,
                                        aux_provider,
                                        strict=False,
                                        skip_validation=skip_validation,
                                    )
                                )
                            except GoBack:
                                for field_name in vars(loop_snapshot):
                                    setattr(
                                        config,
                                        field_name,
                                        getattr(loop_snapshot, field_name),
                                    )
                                aux_ollama_detected_models = []
                                console.print(
                                    "  [dim]↩ Returning to co-pilot provider "
                                    "selection.[/dim]"
                                )
                                continue
                        break
                    config.auxiliary_model = _step_model(
                        config,
                        aux_provider,
                        ollama_detected_models=aux_ollama_detected_models,
                        label="co-pilot",
                        default_value=config.auxiliary_model,
                    )
                else:
                    # Skip: single driver — clear any prior auxiliary config.
                    config.auxiliary_provider = ""
                    config.auxiliary_model = ""
                _reconcile_oauth_modes(config)
                _autosave(config)

            if "tavily" in sections_to_run:
                preset_tavily = _preset("tavily_key")
                if preset_tavily is not None:
                    # Validate the preset key like the --api-key path does;
                    # the non-interactive flow can't show a "Save anyway?"
                    # prompt, so a failed validation is fatal.
                    if not skip_validation:
                        from .validators import validate_tavily_key

                        console.print(
                            "  [dim]Validating preset Tavily key...[/dim]", end=""
                        )
                        valid, msg = validate_tavily_key(preset_tavily)
                        if valid:
                            console.print(f"\r  [green]✓ {msg}[/green]      ")
                        else:
                            console.print(f"\r  [red]✗ {msg}[/red]      ")
                            raise RuntimeError(
                                f"--tavily-key rejected by validator: {msg}. "
                                "Pass --skip-validation to override."
                            )
                    config.tavily_api_key = preset_tavily
                    console.print(
                        f"  [green]✓ Tavily key: ***{preset_tavily[-4:]}[/green]"
                        "   [dim](--tavily-key)[/dim]"
                    )
                elif strict:
                    # ``--non-interactive`` without ``--tavily-key`` — Tavily
                    # is optional (web search). Keep whatever's in config
                    # (likely empty for first-time setup); never open the
                    # interactive password prompt under strict.
                    if config.tavily_api_key:
                        _print_step_skipped("Tavily Key", "kept current")
                    else:
                        _print_step_skipped("Tavily Key", "not set")
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
                # User declined — restore exact pre-wizard file state.
                # Three cases driven by the capture-time flags:
                #   1. ``existed=False`` → file is new, delete it (autosaves
                #      during the run created it).
                #   2. ``existed=True`` + bytes captured → restore bytes
                #      verbatim, preserves comments / unknown fields.
                #   3. ``existed=True`` + bytes None (read failed at capture)
                #      → fall back to ``save_config(snapshot)`` since we
                #      can't restore the exact bytes; still better than
                #      leaving the mid-wizard state in place.
                try:
                    if not original_file_existed:
                        if config_path.exists():
                            config_path.unlink()
                    elif original_file_bytes is not None:
                        config_path.write_bytes(original_file_bytes)
                    else:
                        save_config(snapshot)
                except OSError as exc:
                    save_config(snapshot)
                    console.print(
                        f"[yellow]Revert via raw bytes failed ({exc}); "
                        "wrote parsed snapshot instead.[/yellow]"
                    )
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
