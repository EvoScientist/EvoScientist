"""Onboarding package — re-exports for backward compatibility.

Originally a single ``onboard.py`` module. Split into submodules in the
configure/refactor pass. Existing callers and tests reference symbols at
``EvoScientist.config.onboard.X`` — keep those imports working by
re-exporting here.

Tests that ``patch("EvoScientist.config.onboard.<symbol>")`` still patch
the attribute on this package, but the *function* that uses that symbol now
lives in a submodule (e.g., ``onboard.steps``) and reads its own globals.
Tests in this repo have been updated to patch the submodule directly. New
tests should follow the same convention.
"""

from __future__ import annotations

# Submodules — exposed so tests/users can address them by full path.
from . import channels as channels
from . import helpers as helpers
from . import steps as steps
from . import style as style
from . import validators as validators
from . import wizard as wizard

# Re-exports for existing imports.
from .channels import _step_channels
from .helpers import (
    _auto_install_latexmk,
    _check_latex_components,
    _check_tinytex,
    _detect_node_install_method,
    _detect_tinytex_install_method,
    _ensure_npx,
    _install_ccproxy,
    _install_imsg,
    _install_node,
    _install_tinytex,
    _print_latex_status,
    _prompt_and_validate_api_key,
    _prompt_ccproxy_port,
    _provider_key_info,
    _run_ccproxy_login,
    _setup_imessage,
    validate_imessage,
)
from .steps import (
    _RECOMMENDED_SKILLS,
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
    WIZARD_STYLE,
    _checkbox_ask,
    _print_header,
    _print_step_result,
    _print_step_skipped,
    console,
)
from .validators import (
    ChoiceValidator,
    IntegerValidator,
    validate_anthropic_key,
    validate_dashscope_code_key,
    validate_dashscope_key,
    validate_deepseek_key,
    validate_google_key,
    validate_kimi_key,
    validate_minimax_key,
    validate_moonshot_key,
    validate_nvidia_key,
    validate_openai_key,
    validate_openrouter_key,
    validate_siliconflow_key,
    validate_tavily_key,
    validate_volcengine_key,
    validate_zhipu_key,
)

# Public API
from .wizard import (
    STEPS,
    render_progress,
    run_onboard,
)

__all__ = [
    "CONFIRM_STYLE",
    "QMARK",
    "STEPS",
    # styles
    "WIZARD_STYLE",
    "_RECOMMENDED_SKILLS",
    "ChoiceValidator",
    # validators
    "IntegerValidator",
    # helpers (tested)
    "_auto_install_latexmk",
    "_check_latex_components",
    "_check_tinytex",
    "_checkbox_ask",
    "_detect_node_install_method",
    "_detect_tinytex_install_method",
    "_ensure_npx",
    "_install_ccproxy",
    "_install_imsg",
    "_install_node",
    "_install_tinytex",
    "_print_header",
    "_print_latex_status",
    "_print_step_result",
    "_print_step_skipped",
    "_prompt_and_validate_api_key",
    "_prompt_ccproxy_port",
    "_provider_key_info",
    "_run_ccproxy_login",
    "_setup_imessage",
    "_step_anthropic_auth_mode",
    "_step_base_url",
    "_step_channels",
    "_step_langgraph_dev_port",
    "_step_mcp_servers",
    "_step_minimax_region",
    "_step_model",
    "_step_ollama_base_url",
    "_step_openai_auth_mode",
    "_step_provider",
    "_step_provider_api_key",
    "_step_reasoning_effort",
    "_step_skills",
    "_step_tavily_key",
    "_step_thinking",
    "_step_tinytex",
    # steps
    "_step_ui_backend",
    "_step_workspace",
    "console",
    "render_progress",
    # public
    "run_onboard",
    "validate_anthropic_key",
    "validate_dashscope_code_key",
    "validate_dashscope_key",
    "validate_deepseek_key",
    "validate_google_key",
    "validate_imessage",
    "validate_kimi_key",
    "validate_minimax_key",
    "validate_moonshot_key",
    "validate_nvidia_key",
    "validate_openai_key",
    "validate_openrouter_key",
    "validate_siliconflow_key",
    "validate_tavily_key",
    "validate_volcengine_key",
    "validate_zhipu_key",
]
