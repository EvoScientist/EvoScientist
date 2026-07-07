"""Individual wizard step functions.

Each ``_step_*`` prompts the user for one logical decision and returns the
chosen value. Conditional steps (auth mode, base URL) are only called by
``run_onboard`` when the provider needs them.
"""

from __future__ import annotations

import os
from pathlib import Path

import questionary
from prompt_toolkit.formatted_text import FormattedText
from questionary import Choice

from ...llm import get_models_for_provider
from ...llm.ollama_discovery import validate_ollama_connection
from ..settings import EvoScientistConfig
from .helpers import (
    _auto_install_latexmk,
    _check_latex_components,
    _detect_tinytex_install_method,
    _ensure_npx,
    _install_ccproxy,
    _install_tinytex,
    _print_latex_status,
    _prompt_and_validate_api_key,
    _prompt_ccproxy_port,
    _provider_key_info,
    _run_ccproxy_login,
)
from .style import (
    CONFIRM_STYLE,
    QMARK,
    WIZARD_STYLE,
    _checkbox_ask,
    _print_step_result,
    _print_step_skipped,
    console,
)
from .validators import validate_tavily_key


def _step_ui_backend(config: EvoScientistConfig) -> str:
    """Step 0: Select UI backend (desktop WebUI, Textual TUI, or Rich CLI).

    Args:
        config: Current configuration.

    Returns:
        Selected backend name ("tui", "cli", or "webui").
    """
    choices = [
        Choice(title="WebUI (desktop interface, modern)", value="webui"),
        Choice(title="TUI (full-screen interface, recommended)", value="tui"),
        Choice(title="CLI (classic terminal, lightweight)", value="cli"),
    ]

    # Map legacy values to current ones
    _legacy_map = {"textual": "tui", "rich": "cli"}
    default_backend = _legacy_map.get(config.ui_backend, config.ui_backend)
    if default_backend not in ("tui", "cli", "webui"):
        default_backend = "tui"

    backend = questionary.select(
        "Select UI mode:",
        choices=choices,
        default=default_backend,
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()

    if backend is None:
        raise KeyboardInterrupt()

    return backend


def _step_langgraph_dev_port(config: EvoScientistConfig) -> int:
    """Step 0.5: Choose the local TCP port for the langgraph dev subprocess.

    EvoSci auto-starts a ``langgraph dev`` server in the background to host
    deployed sub-agents (writing-agent, data-analysis-agent) when
    ``enable_async_subagents`` is True. This step lets the user pick a free
    port, with a live conflict check on the configured default.

    Returns the chosen port; caller assigns it to ``config.langgraph_dev_port``.
    """
    if not getattr(config, "enable_async_subagents", True):
        # User has async disabled — port is irrelevant, no prompt.
        return getattr(config, "langgraph_dev_port", 6174)

    from ...langgraph_dev.manager import _is_port_occupied, is_langgraph_dev_running

    current_port = getattr(config, "langgraph_dev_port", 6174)
    current_occupied = _is_port_occupied(current_port)
    if current_occupied and is_langgraph_dev_running(port=current_port):
        # Another EvoSci shell is already serving on this port — reuse, don't
        # force the user to renumber.
        current_occupied = False

    # Bake the live status into the prompt label so the user sees it WITH
    # the question, not as a side-effect line that prints before input.
    # Single set of parens, no nesting (mirrors ccproxy's prompt style).
    if current_occupied:
        prompt_label = (
            f"Enter port for EvoScientist server "
            f"(Current: {current_port}, occupied, pick another):"
        )
    else:
        prompt_label = (
            f"Enter port for EvoScientist server "
            f"(Current: {current_port}, available, Enter to keep):"
        )

    def valid_port(value: str) -> bool:
        if not value:
            # Allow keeping the default only if it's actually free; otherwise
            # require the user to pick something else.
            return not current_occupied
        try:
            port = int(value)
        except (ValueError, TypeError):
            return False
        if not (1024 < port < 65536):
            return False
        # Reject user-typed ports that are already occupied UNLESS the
        # occupier is our own langgraph dev (e.g., another EvoSci shell) —
        # in that case the runtime will reuse it.
        if not _is_port_occupied(port):
            return True
        return is_langgraph_dev_running(port=port)

    raw = questionary.text(
        prompt_label,
        validate=valid_port,
        style=WIZARD_STYLE,
        qmark=QMARK,
    ).ask()

    if raw is None:
        raise KeyboardInterrupt()

    port = int(raw) if raw else current_port

    # Final probe — warn (don't fail) if the chosen port is still occupied
    # by something OTHER than our own langgraph dev. Reuse of an existing
    # EvoSci server on that port is fine. They can always change later via:
    # EvoSci config set langgraph_dev_port <port>
    if _is_port_occupied(port) and not is_langgraph_dev_running(port=port):
        console.print(
            f"  [yellow]⚠ Port {port} is occupied. EvoSci may fail to start its "
            f"server. Free the port or change later with: "
            f"EvoSci config set langgraph_dev_port <other-port>[/yellow]"
        )
    else:
        console.print(
            f"  [green]✓ EvoScientist will run on http://127.0.0.1:{port}[/green]"
        )
    return port


def _step_webui_port(config: EvoScientistConfig) -> int:
    """Step 0.6: Choose the local TCP port for the WebUI front-end.

    Only asked when ``ui_backend == "webui"``. This is the Next.js server port
    the browser opens (``@evoscientist/webui``); the backend keeps its own
    ``langgraph_dev_port``. Mirrors the langgraph-dev port prompt's UX.

    Returns the chosen port; caller assigns it to ``config.webui_port``.
    """
    from ...langgraph_dev.manager import _is_port_occupied

    current_port = getattr(config, "webui_port", 4716)
    backend_port = getattr(config, "langgraph_dev_port", 6174)
    occupied = _is_port_occupied(current_port)
    conflicts_backend = current_port == backend_port

    # Bake live availability into the label (same single-paren style as the
    # langgraph-dev / ccproxy prompts) so the status shows WITH the question.
    # The WebUI port must also differ from the backend (langgraph dev) port —
    # run_webui refuses equal ports at startup, so reject them here too instead
    # of saving a config that fails to launch later.
    if occupied or conflicts_backend:
        reason = "occupied" if occupied else f"= backend port {backend_port}"
        prompt_label = (
            f"Enter port for WebUI server "
            f"(Current: {current_port}, {reason}, pick another):"
        )
    else:
        prompt_label = (
            f"Enter port for WebUI server "
            f"(Current: {current_port}, available, Enter to keep):"
        )

    def valid_port(value: str) -> bool:
        if not value:
            # Keep the default only if it's free AND not the backend port.
            return not occupied and not conflicts_backend
        try:
            port = int(value)
        except (ValueError, TypeError):
            return False
        if not (1024 < port < 65536):
            return False
        if port == backend_port:
            return False
        return not _is_port_occupied(port)

    raw = questionary.text(
        prompt_label,
        validate=valid_port,
        style=WIZARD_STYLE,
        qmark=QMARK,
    ).ask()

    if raw is None:
        raise KeyboardInterrupt()

    port = int(raw) if raw else current_port
    console.print(f"  [green]✓ WebUI will open at http://localhost:{port}[/green]")
    console.print(
        "  [yellow]⚠️  Note: the WebUI won't show your CLI/TUI chat history "
        "yet.[/yellow]"
    )
    return port


def _step_provider(
    config: EvoScientistConfig,
    *,
    label: str | None = None,
    default_value: str | None = None,
) -> str:
    """Step 1: Select LLM provider.

    Args:
        config: Current configuration.
        label: Optional role label (e.g. "co-pilot") to clarify which model this
            provider is for. When omitted, the generic main-model prompt is used.
        default_value: Preselect this provider instead of ``config.provider``
            (e.g. the auxiliary provider when configuring the co-pilot).

    Returns:
        Selected provider name.
    """
    choices = [
        # Direct providers
        Choice(title="Anthropic (Claude models — API / OAuth)", value="anthropic"),
        Choice(title="OpenAI (GPT models — API / OAuth)", value="openai"),
        Choice(title="Google GenAI (Gemini models)", value="google-genai"),
        Choice(
            title="MiniMax (M2 — M3 models, up to 1M context, thinking)",
            value="minimax",
        ),
        Choice(title="ZhipuAI (智谱 — GLM models)", value="zhipu"),
        Choice(
            title="ZhipuAI CodePlan (智谱代码计划 — GLM models for coding)",
            value="zhipu-code",
        ),
        Choice(
            title="Volcengine (火山引擎 — Doubao models)",
            value="volcengine",
        ),
        Choice(
            title="DashScope (阿里云 — Qwen models)",
            value="dashscope",
        ),
        Choice(
            title="DashScope Coding Plan (阿里云代码计划 — Qwen models)",
            value="dashscope-code",
        ),
        Choice(
            title="DeepSeek (DeepSeek-R1, DeepSeek-V3)",
            value="deepseek",
        ),
        Choice(
            title="Moonshot (月之暗面 — Moonshot models)",
            value="moonshot",
        ),
        Choice(
            title="Kimi Coding Plan (Kimi 代码计划 — coding-focused)",
            value="kimi-coding",
        ),
        # Local
        Choice(title="Ollama (local models)", value="ollama"),
        # Third-party / aggregator
        Choice(title="NVIDIA (third party — limited free requests)", value="nvidia"),
        Choice(
            title="SiliconFlow (aggregator — GLM, Kimi, MiniMax, etc.)",
            value="siliconflow",
        ),
        Choice(
            title="OpenRouter (aggregator — Grok, Gemini, Qwen, etc.)",
            value="openrouter",
        ),
        Choice(
            title="OpenAI-compatible (third-party OpenAI endpoint)",
            value="custom-openai",
        ),
        Choice(
            title="Claude-compatible (third-party Anthropic endpoint)",
            value="custom-anthropic",
        ),
    ]

    # Set default based on current config (or an explicit override).
    valid_providers = {c.value for c in choices}
    preferred = default_value or config.provider
    default = preferred if preferred in valid_providers else "anthropic"

    provider = questionary.select(
        f"Select {label} provider:" if label else "Select your LLM provider:",
        choices=choices,
        default=default,
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()

    if provider is None:
        raise KeyboardInterrupt()

    return provider


_MINIMAX_REGIONS: dict[str, str] = {
    "global": "https://api.minimax.io/anthropic",
    "cn": "https://api.minimaxi.com/anthropic",
}


def _step_minimax_region(config: EvoScientistConfig) -> str:
    """Step 2a (MiniMax): Select API region.

    MiniMax has two regional endpoints — Global (api.minimax.io) and
    Mainland China (api.minimaxi.com).  API keys are region-bound.

    Returns:
        The selected base URL.
    """
    current = config.minimax_base_url or os.environ.get("MINIMAX_BASE_URL", "")
    if current == _MINIMAX_REGIONS["global"]:
        default = "global"
    else:
        default = "cn"

    region = questionary.select(
        "Select MiniMax API region (must match where your key was created):",
        choices=[
            Choice(
                title="Global (api.minimax.io — platform.minimax.io keys)",
                value="global",
            ),
            Choice(
                title="Mainland China (api.minimaxi.com — platform.minimaxi.com keys)",
                value="cn",
            ),
        ],
        default=default,
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()

    if region is None:
        raise KeyboardInterrupt()

    return _MINIMAX_REGIONS[region]


def _step_oauth_auth_mode(
    config: EvoScientistConfig,
    *,
    provider_label: str,
    ccproxy_provider: str,
    config_attr: str,
    prompt_login_label: str,
    oauth_choice_label: str | None = None,
    status_label: str | None = None,
    question_label: str | None = None,
) -> str:
    """Select API-key vs ccproxy OAuth authentication for a provider.

    Args:
        config: Current configuration.
        provider_label: Provider display name for direct API-key access.
        ccproxy_provider: ccproxy auth provider name.
        config_attr: Config attribute storing this provider's auth mode.
        prompt_login_label: Label used in "Log in to ..." prompts.
        oauth_choice_label: Optional display label for the OAuth choice.
        status_label: Optional display label for status messages.
        question_label: Optional prompt label override.

    Returns:
        Selected auth mode: "api_key" or "oauth".
    """
    from ...ccproxy_manager import check_ccproxy_auth, is_ccproxy_available

    ccproxy_available = is_ccproxy_available()

    from .prompter import BACK_SENTINEL, GoBack, install_navigation_keys

    oauth_label = oauth_choice_label or f"{prompt_login_label} OAuth"
    auth_status_label = status_label or oauth_label
    auth_question_label = question_label or f"{provider_label} authentication mode"

    choices = [
        Choice(title=f"API Key (direct {provider_label} access)", value="api_key"),
        Choice(
            title=f"{oauth_label} (via ccproxy — no API key needed)"
            + (
                ""
                if ccproxy_available
                else " [requires: pip install evoscientist[oauth]]"
            ),
            value="oauth",
        ),
        questionary.Separator(),
        Choice(title="← Back (re-pick provider)", value=BACK_SENTINEL),
    ]

    current = getattr(config, config_attr)
    if current not in ("api_key", "oauth"):
        current = "api_key"

    question = questionary.select(
        f"{auth_question_label}  [Esc/← to go back]:",
        choices=choices,
        default=current,
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    )
    install_navigation_keys(question, with_back=True)
    auth_mode = question.ask()

    if auth_mode is None:
        raise KeyboardInterrupt()
    if auth_mode == BACK_SENTINEL:
        raise GoBack()

    if auth_mode == "oauth" and not ccproxy_available:
        console.print("  [yellow]✗ ccproxy not installed[/yellow]")
        console.print()
        install = questionary.confirm(
            'Install ccproxy now? (pip install "evoscientist[oauth]")',
            default=True,
            style=WIZARD_STYLE,
            qmark=f"  {QMARK}",
        ).ask()
        if install is None:
            raise KeyboardInterrupt()
        if install:
            console.print()
            if _install_ccproxy():
                console.print("  [green]✓ ccproxy installed successfully.[/green]")
            else:
                console.print("  [yellow]Falling back to API key mode.[/yellow]")
                return "api_key"
        else:
            console.print(
                '  [dim]Skipped. Install manually: pip install "evoscientist[oauth]"[/dim]'
            )
            return "api_key"

    if auth_mode == "oauth":
        _prompt_ccproxy_port(config)

        authed, msg = check_ccproxy_auth(ccproxy_provider)
        if authed:
            console.print(f"  [green]✓ {auth_status_label}: {msg}[/green]")
            relogin = questionary.confirm(
                "Re-authenticate to refresh credentials?",
                default=False,
                style=CONFIRM_STYLE,
                qmark=QMARK,
            ).ask()
            if relogin is None:
                raise KeyboardInterrupt()
            if relogin:
                _run_ccproxy_login(ccproxy_provider, auth_status_label)
        else:
            console.print(
                f"  [yellow]{auth_status_label} not authenticated: {msg}[/yellow]"
            )
            login = questionary.confirm(
                f"Log in to {prompt_login_label} now?",
                default=True,
                style=CONFIRM_STYLE,
                qmark=QMARK,
            ).ask()
            if login is None:
                raise KeyboardInterrupt()
            if login:
                _run_ccproxy_login(ccproxy_provider, auth_status_label)

    return auth_mode


def _step_anthropic_auth_mode(config: EvoScientistConfig) -> str:
    """Step 2a: Select Anthropic authentication mode (API key vs OAuth).

    Args:
        config: Current configuration.

    Returns:
        Selected auth mode: "api_key" or "oauth".
    """
    return _step_oauth_auth_mode(
        config,
        provider_label="Anthropic",
        ccproxy_provider="claude_api",
        config_attr="anthropic_auth_mode",
        prompt_login_label="Claude",
        oauth_choice_label="Claude Code OAuth",
        status_label="OAuth",
        question_label="Authentication mode",
    )


def _step_openai_auth_mode(config: EvoScientistConfig) -> str:
    """Step 2b: Select OpenAI authentication mode (API key vs Codex OAuth).

    Args:
        config: Current configuration.

    Returns:
        Selected auth mode: "api_key" or "oauth".
    """
    return _step_oauth_auth_mode(
        config,
        provider_label="OpenAI",
        ccproxy_provider="codex",
        config_attr="openai_auth_mode",
        prompt_login_label="Codex",
        oauth_choice_label="Codex OAuth",
        status_label="Codex OAuth",
        question_label="OpenAI authentication mode",
    )


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


def _step_base_url(config: EvoScientistConfig, current_value: str | None = None) -> str:
    """Prompt for custom provider base URL.

    Args:
        config: Current configuration.
        current_value: Current base URL value (if None, defaults to empty).

    Returns:
        Base URL string.
    """
    current = current_value if current_value is not None else ""
    hint = f"Current: {current}" if current else ""
    default = current or ""

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
    default = current or "http://localhost:11434"

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
    label: str | None = None,
    default_value: str | None = None,
) -> str:
    """Step 3: Select model for the provider.

    Args:
        config: Current configuration.
        provider: Selected provider name.
        ollama_detected_models: Model names detected from a live Ollama server.
        label: Optional role label (e.g. "co-pilot") for the prompt. When omitted,
            the generic main-model prompt is used.
        default_value: Preselect this model instead of ``config.model`` (e.g. the
            auxiliary model when configuring the co-pilot).

    Returns:
        Selected model name.
    """
    model_prompt = f"Select {label} model:" if label else "Select model:"
    model_default = default_value or config.model
    # Ollama: show only what's actually pulled on the server
    if provider == "ollama":
        if ollama_detected_models:
            _CUSTOM_SENTINEL = "__custom__"
            choices = [
                Choice(title=name, value=name) for name in ollama_detected_models
            ]
            choices.append(Choice(title="Type a model name...", value=_CUSTOM_SENTINEL))

            default = ollama_detected_models[0]
            if model_default in ollama_detected_models:
                default = model_default

            selected = questionary.select(
                model_prompt,
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
        # Custom / unknown provider: direct text input.
        # Keep prompting until a non-empty model name is provided — saving an
        # empty string here leaves the first request broken with an opaque
        # "model required" error from the provider SDK.
        while True:
            model = questionary.text(
                "Model name:",
                style=WIZARD_STYLE,
                qmark=QMARK,
                placeholder=FormattedText([("fg:#858585", " e.g. owner/model-name")]),
                default=model_default or "",
            ).ask()
            if model is None:
                raise KeyboardInterrupt()
            model = model.strip()
            if model:
                return model
            console.print(
                "  [yellow]Model name cannot be empty for a custom provider. "
                "Press Ctrl+C to cancel.[/yellow]"
            )

    provider_models = [name for name, _ in entries]

    # Create choices with model IDs as hints
    _CUSTOM_SENTINEL = "__custom__"
    choices = []
    for name, model_id in entries:
        choices.append(Choice(title=f"{name} ({model_id})", value=name))
    choices.append(Choice(title="Type a model name...", value=_CUSTOM_SENTINEL))

    # Determine default. An explicit ``default_value`` override (e.g. a saved
    # co-pilot model on a re-run) that isn't a registry model is a custom name:
    # preselect "Type a model name..." and prefill it. A plain ``config.model``
    # that just isn't in the current provider's list (e.g. the provider was
    # changed) falls back to the first model, NOT the custom entry.
    custom_default = (
        default_value if default_value and default_value not in provider_models else ""
    )
    if model_default in provider_models:
        default = model_default
    elif custom_default:
        default = _CUSTOM_SENTINEL
    else:
        default = provider_models[0]

    selected = questionary.select(
        model_prompt,
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
        default=custom_default,
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


def _step_auxiliary_enable(config: EvoScientistConfig) -> bool:
    """Step 3.25: Choose whether to assemble a co-pilot (auxiliary) model.

    The co-pilot runs background/helper LLM calls — EvoMemory (memory workers)
    and the main agent's tool selector — so it can be a cheaper/faster model.
    Returns True when the user picks "Assemble"; the caller then runs the
    provider/key/model pickers. Returns False to keep the pilot (main model)
    everywhere.
    """
    console.print(
        "  [dim]A cheaper/faster co-pilot for EvoMemory (memory workers).[/dim]"
    )
    choice = questionary.select(
        "Co-pilot (auxiliary model):",
        choices=[
            Choice(
                title="Skip — single pilot (main model handles everything)",
                value="skip",
            ),
            Choice(
                title="Assemble a co-pilot — separate cheaper/faster model",
                value="assemble",
            ),
        ],
        default="assemble" if config.auxiliary_model else "skip",
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()
    if choice is None:
        raise KeyboardInterrupt()
    return choice == "assemble"


def _step_reasoning_effort(config: EvoScientistConfig) -> str:
    """Step 3.5: Configure OpenRouter reasoning effort level.

    Only shown when the selected provider is OpenRouter. See:
    https://openrouter.ai/docs/guides/best-practices/reasoning-tokens

    Args:
        config: Current configuration.

    Returns:
        Selected reasoning effort level, or empty string to use default.
    """
    effort_choices = [
        Choice(title="xhigh  — ~95% of max_tokens for reasoning", value="xhigh"),
        Choice(title="high   — ~80% of max_tokens (recommended)", value="high"),
        Choice(title="medium — ~50% of max_tokens", value="medium"),
        Choice(title="low    — ~20% of max_tokens", value="low"),
        Choice(title="minimal — ~10% of max_tokens", value="minimal"),
        Choice(title="none   — disable reasoning entirely", value="none"),
    ]

    current = config.reasoning_effort or "high"
    effort = questionary.select(
        "Select reasoning effort level:",
        choices=effort_choices,
        default=current,
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()

    if effort is None:
        raise KeyboardInterrupt()

    return effort


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


def _step_workspace(config: EvoScientistConfig) -> str:
    """Step 5: Configure workspace mode.

    Args:
        config: Current configuration.

    Returns:
        Selected mode ("daemon" or "run").
    """
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

    return mode


def _step_thinking(config: EvoScientistConfig) -> bool:
    """Step 6: Configure thinking panel visibility.

    Args:
        config: Current configuration.

    Returns:
        Whether to show thinking panels in the interface.
    """
    thinking_choices = [
        Choice(title="On (show model reasoning)", value=True),
        Choice(title="Off (hide model reasoning)", value=False),
    ]

    show_thinking = questionary.select(
        "Show thinking panel?",
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
        "label": "EvoSci Skills  (optimized for EvoScientist — paper planning, writing, review, etc.) 👈 Recommended",
        "source": "EvoScientist/EvoSkills@skills",
    },
    # ── Third-party (K-Dense) ──
    {
        "label": "Scientific Skills  (143 research & experiment skills, third party by K-Dense)",
        "source": "K-Dense-AI/scientific-agent-skills@skills",
    },
    {
        "label": "Scientific Writer  (27 writing, review & presentation skills, third party by K-Dense)",
        "source": "K-Dense-AI/claude-scientific-writer@skills",
    },
    # ── Third-party (Orchestra Research) ──
    {
        "label": "AI Research Skills  (98 skills for training, evaluation, deployment, etc., third party by Orchestra Research)",
        "source": "Orchestra-Research/AI-Research-SKILLs",
    },
    # ── Third-party (Google DeepMind) ──
    {
        "label": "Science Skills  (37 genomics, structural-biology & literature skills, third party by Google DeepMind)",
        "source": "google-deepmind/science-skills@skills",
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
]


def _step_tinytex() -> None:
    """Step 9: Prepare LaTeX environment (TinyTeX).

    Asks the user whether they want to set up LaTeX for paper compilation.
    If yes, checks for an existing installation and offers to install TinyTeX
    when none is found.  The agent can auto-install missing LaTeX packages at
    runtime via ``tlmgr``, so only the base TinyTeX is needed here.
    """
    latex_choices = [
        Choice(title="No need (skip LaTeX setup)", value=False),
        Choice(title="Install now (TinyTeX compiler)", value=True),
    ]
    prepare = questionary.select(
        "LaTeX environment (needed to compile .tex → .pdf):",
        choices=latex_choices,
        default=False,
        style=WIZARD_STYLE,
        qmark=QMARK,
    ).ask()

    if prepare is None:
        raise KeyboardInterrupt()

    if not prepare:
        _print_step_skipped("LaTeX", "skipped")
        console.print(
            "  [dim]Install later:"
            ' curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh[/dim]'
        )
        return

    # User wants LaTeX — check existing installation
    console.print("  [dim]Checking LaTeX environment...[/dim]")

    components = _check_latex_components()

    if components["pdflatex"]:
        # Already installed — show detailed status
        _print_latex_status(components)
        # Auto-fix missing latexmk if tlmgr is available
        if not components["latexmk"] and components["tlmgr"]:
            _auto_install_latexmk()
        return

    # Not installed — detect install method and offer
    console.print("  [yellow]✗ pdflatex not found[/yellow]")
    method, command = _detect_tinytex_install_method()

    if method == "manual":
        _print_step_skipped("LaTeX", "manual install needed")
        console.print(f"  [dim]Install TinyTeX: {command}[/dim]")
        return

    install = questionary.confirm(
        f"Install TinyTeX via {method}?",
        default=True,
        style=WIZARD_STYLE,
        qmark=f"  {QMARK}",
    ).ask()

    if install is None:
        raise KeyboardInterrupt()

    if not install:
        _print_step_skipped("LaTeX", "skipped")
        console.print(f"  [dim]Install later: {command}[/dim]")
        return

    console.print("  [dim]Installing TinyTeX (this may take a minute)...[/dim]")
    if _install_tinytex(method, command):
        post = _check_latex_components()
        if post["pdflatex"]:
            _print_latex_status(post)
            _print_step_result("LaTeX", "TinyTeX installed")
        else:
            console.print("  [green]✓ TinyTeX installed[/green]")
            console.print(
                "  [yellow]⚠ Restart your terminal"
                " for pdflatex to appear in PATH[/yellow]"
            )
            _print_step_result("LaTeX", "installed (restart terminal for PATH)")
    else:
        console.print(f"  [dim]Install manually: {command}[/dim]")
        _print_step_result("LaTeX", "installation failed", success=False)


def _step_skills() -> list[str]:
    """Step 7: Optionally install recommended skills.

    Shows checkbox first. Already-installed skills are shown as disabled
    so users don't accidentally reinstall them. If user selects nothing,
    checks npx as an easter egg — confirms skill discovery is available,
    or offers to install Node.js if missing.

    Returns:
        List of skill sources that were selected (empty if skipped).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from ...paths import GLOBAL_SKILLS_DIR, USER_SKILLS_DIR
    from ...tools.skills_manager import installed_provenance, resolve_remote_head

    # Collect installed-skill dir names across both tiers. The dir-name match
    # is a best-effort fallback for legacy installs without manifests; the
    # provenance map below is the authoritative signal (handles packs that
    # explode into many child directories with unrelated names).
    installed_names: set[str] = set()
    for skills_dir in (Path(USER_SKILLS_DIR), Path(GLOBAL_SKILLS_DIR)):
        if skills_dir.exists():
            installed_names.update(e.name for e in skills_dir.iterdir() if e.is_dir())
    provenance = installed_provenance()
    installed_src = set(provenance)

    def _hint_name(source: str) -> str:
        """Derive expected skill directory name from source URL."""
        if "@" in source and "://" not in source:
            return source.split("@", 1)[1].strip()
        return source.rstrip("/").rsplit("/", 1)[-1]

    def _is_installed(source: str) -> bool:
        return source in installed_src or _hint_name(source) in installed_names

    # For installed packs with a stored commit, ask GitHub if upstream has
    # moved. Bounded parallel calls so onboard stays snappy; failures are
    # silently treated as "unknown" (label falls back to plain "installed").
    sources_to_check = [
        skill["source"]
        for skill in _RECOMMENDED_SKILLS
        if _is_installed(skill["source"])
        and provenance.get(skill["source"], {}).get("commit")
    ]
    upstream_heads: dict[str, str | None] = {}
    if sources_to_check:
        with ThreadPoolExecutor(max_workers=min(6, len(sources_to_check))) as ex:
            futures = {
                ex.submit(resolve_remote_head, src): src for src in sources_to_check
            }
            for fut in as_completed(futures):
                src = futures[fut]
                try:
                    upstream_heads[src] = fut.result()
                except Exception:
                    upstream_heads[src] = None

    def _has_update(source: str) -> bool:
        head = upstream_heads.get(source)
        recorded = provenance.get(source, {}).get("commit")
        return bool(head and recorded and head != recorded)

    choices = []
    for skill in _RECOMMENDED_SKILLS:
        src = skill["source"]
        if _is_installed(src):
            hint = (
                "  (installed — update available, re-select to sync)"
                if _has_update(src)
                else "  (installed — re-select to sync)"
            )
            choices.append(
                Choice(
                    title=[
                        ("", skill["label"]),
                        ("class:instruction", hint),
                    ],
                    value=src,
                )
            )
        else:
            choices.append(Choice(title=skill["label"], value=src))

    all_installed = all(_is_installed(skill["source"]) for skill in _RECOMMENDED_SKILLS)
    has_updates = any(_has_update(skill["source"]) for skill in _RECOMMENDED_SKILLS)
    if all_installed:
        if has_updates:
            console.print(
                "  [green]✓ All recommended skills installed; "
                "[yellow]updates available[/yellow] — re-select any to sync.[/green]"
            )
        else:
            console.print(
                "  [green]✓ All recommended skills are already installed "
                "and up to date.[/green]"
            )
            # No updates AND nothing new to install — nothing useful the
            # picker can do.
            return []

    prompt_label = (
        "Re-select installed skills to sync, or pick new ones:"
        if all_installed
        else "Install or Sync predefined skills:"
    )
    selected = _checkbox_ask(choices, prompt_label)

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

    from ...tools.skills_manager import install_skill

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


def _step_mcp_servers() -> list[str]:
    """Step 8: Optionally install recommended MCP servers.

    Shows a checkbox list of recommended servers. Already-configured servers
    are shown as disabled so users don't accidentally override them.
    Selected ones are added to the user MCP config via ``install_mcp_server()``.

    Handles env-key prompts, pip package installs, and URL-based servers.

    Returns:
        List of server names that were installed.
    """
    from ...mcp.client import _load_user_config
    from ...mcp.registry import fetch_marketplace_index, install_mcp_server

    try:
        all_servers = fetch_marketplace_index()
    except Exception as exc:
        console.print(
            "  [yellow]\u26a0 Could not fetch MCP marketplace index "
            f"({type(exc).__name__}). Skipping MCP setup \u2014 "
            "you can re-run with [bold]EvoSci configure mcp[/bold] later.[/yellow]"
        )
        return []
    servers = [s for s in all_servers if "onboarding" in s.tags]
    existing_config = _load_user_config()

    choices = []
    for srv in servers:
        if srv.name in existing_config:
            choices.append(
                Choice(
                    title=[
                        ("", srv.label),
                        ("class:instruction", "  (already configured)"),
                    ],
                    value=srv.name,
                    disabled=True,
                )
            )
        else:
            choices.append(Choice(title=srv.label, value=srv.name))

    # Only declare "all configured" when there ARE recommended servers AND
    # every one is already in the user's config \u2014 distinct from "marketplace
    # returned nothing" (a transient failure) which is handled above.
    if servers and all(srv.name in existing_config for srv in servers):
        console.print(
            "[green]\u2713 All recommended MCP servers are already configured.[/green]"
        )
        return []
    if not servers:
        console.print(
            "  [dim]No recommended MCP servers are tagged for onboarding right now.[/dim]"
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
    needs_npx = any(srv.command == "npx" for srv in servers if srv.name in selected)
    if needs_npx:
        if not _ensure_npx("some MCP servers require Node.js"):
            npx_servers = {
                srv.name
                for srv in servers
                if srv.name in selected and srv.command == "npx"
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
        srv = next(s for s in servers if s.name == name)
        try:
            if install_mcp_server(srv):
                _print_step_result("MCP", f"{name}")
                installed.append(name)
            else:
                _print_step_result(
                    "MCP", f"{name} — installation failed", success=False
                )
        except Exception as e:
            _print_step_result("MCP", f"{name} — {e}", success=False)

    return installed
