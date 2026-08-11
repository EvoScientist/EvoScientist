"""LLM model configuration based on LangChain init_chat_model.

This module provides a unified interface for creating chat model instances
with support for multiple providers (Anthropic, OpenAI, Google GenAI, Atlas
Cloud, MiniMax (Anthropic-compatible), NVIDIA, SiliconFlow, OpenRouter, Requesty,
ZhipuAI, Volcengine, DashScope, DashScope-Code, DeepSeek, Ollama, and custom
OpenAI/Anthropic-compatible endpoints) and convenient short names for common
models.
"""

from __future__ import annotations

import os
import re
import subprocess
import warnings
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from langchain.chat_models import init_chat_model

from ..config.settings import (
    OPENROUTER_DEFAULT_APP_CATEGORIES,
    OPENROUTER_DEFAULT_APP_TITLE,
    OPENROUTER_DEFAULT_HTTP_REFERER,
)
from .context_window import apply_known_context_window
from .deepseek import EvoChatDeepSeek
from .patches import (
    _is_ccproxy_codex,
    _patch_anthropic_strip_foreign_reasoning,
    _patch_anthropic_structured_output,
    _patch_ccproxy_system_to_developer,
    _patch_openai_compat_content,
    _patch_openrouter_strip_responses_reasoning,
    _patch_openrouter_structured_output,
)
from .registry import (
    _ANTHROPIC_ROUTED_PROVIDERS,
    _MODEL_ENTRIES,
    _OPENAI_ROUTED_PROVIDERS,
    _OPENROUTER_JSON_SCHEMA_STRUCTURED_OUTPUT_MODELS,  # noqa: F401 — re-exported
    _THINKING_CAPABLE_PROVIDERS,
    DEFAULT_MODEL,
    MODELS,
    _is_mandatory_thinking_kimi,
    get_model_info,  # noqa: F401 — re-exported for existing import sites
    get_models_for_provider,  # noqa: F401 — re-exported for existing import sites
    list_model_picker_entries,  # noqa: F401 — re-exported for existing import sites
    list_models,  # noqa: F401 — re-exported for existing import sites
    list_models_by_provider,  # noqa: F401 — re-exported for existing import sites
)

# Minimum Codex CLI version advertised when no explicit override is set. Newer
# installed versions are advertised automatically.
_CODEX_CLIENT_VERSION_FALLBACK = "0.144.1"


@lru_cache(maxsize=1)
def _installed_codex_client_version() -> str:
    """Return the installed Codex CLI version, or an empty string."""
    try:
        result = subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""

    if result.returncode != 0:
        return ""
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", result.stdout + result.stderr)
    return match.group(1) if match else ""


def _resolve_codex_client_version() -> str:
    """Resolve an explicit override or the newer of installed and minimum versions."""
    override = os.environ.get("EVOSCIENTIST_CODEX_CLIENT_VERSION", "").strip()
    if override:
        return override

    installed = _installed_codex_client_version()
    if installed and tuple(map(int, installed.split("."))) >= tuple(
        map(int, _CODEX_CLIENT_VERSION_FALLBACK.split("."))
    ):
        return installed
    return _CODEX_CLIENT_VERSION_FALLBACK


def _resolve_reasoning_effort(default: str) -> str:
    """Return the configured reasoning effort or a provider-specific default."""
    return os.environ.get("EVOSCIENTIST_REASONING_EFFORT", "").strip() or default


def _is_deepseek_endpoint(base_url: str | None) -> bool:
    """Return whether an OpenAI-compatible endpoint is DeepSeek's API."""
    if not base_url:
        return False
    try:
        return urlparse(base_url).hostname == "api.deepseek.com"
    except ValueError:
        return False


_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSEY_ENV_VALUES = {"0", "false", "no", "off"}

# OpenRouter app attribution (issue #339). Default values are the single source
# of truth in config/settings.py (imported above); langchain-openrouter maps
# app_url → HTTP-Referer, app_title → X-Title, app_categories →
# X-OpenRouter-Categories. OpenRouter honors at most this many categories per
# request (server-side limit) and silently ignores the rest, so the sent list is
# capped to this many below. https://openrouter.ai/docs/app-attribution
_OPENROUTER_MAX_CATEGORIES_PER_REQUEST = 2


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY_ENV_VALUES


def _env_flag_disabled(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in _FALSEY_ENV_VALUES


def _supports_openrouter_anthropic_prompt_cache(
    provider: str | None, model_id: str
) -> bool:
    """Return whether EvoScientist should declare Claude caching for a router.

    Both OpenRouter and Requesty are OpenAI-compatible routers that forward an
    Anthropic-style ``cache_control`` declaration through to Claude models
    addressed as ``anthropic/...``. Implicit caching is handled upstream for
    most providers, but Claude prompt caching needs the explicit declaration.
    """
    return provider in ("openrouter", "requesty") and model_id.startswith(
        ("anthropic/", "~anthropic/")
    )


def _has_cache_control_override(kwargs: dict[str, Any]) -> bool:
    """Return whether the caller already supplied cache-control settings."""
    if "cache_control" in kwargs:
        return True
    model_kwargs = kwargs.get("model_kwargs")
    if model_kwargs is None:
        return False
    if not isinstance(model_kwargs, dict):
        warnings.warn(
            "OpenRouter Anthropic prompt caching was not applied because "
            "`model_kwargs` is not a dict; pass cache_control explicitly or use "
            "a dict-shaped model_kwargs.",
            UserWarning,
            stacklevel=3,
        )
        return True
    return "cache_control" in model_kwargs


def _apply_openrouter_anthropic_prompt_cache(
    provider: str | None,
    model_id: str,
    kwargs: dict[str, Any],
) -> None:
    """Declare router Claude prompt caching unless explicitly disabled.

    OpenRouter and Requesty both handle implicit caching for most providers,
    but Claude prompt caching needs an Anthropic-style cache-control
    declaration. Each router honours its own opt-out env flag
    (``EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE`` /
    ``EVOSCIENTIST_REQUESTY_ANTHROPIC_PROMPT_CACHE``).
    """
    if provider is None:
        return
    disable_flag = {
        "openrouter": "EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE",
        "requesty": "EVOSCIENTIST_REQUESTY_ANTHROPIC_PROMPT_CACHE",
    }.get(provider)
    if disable_flag is not None and _env_flag_disabled(disable_flag):
        return
    if not _supports_openrouter_anthropic_prompt_cache(provider, model_id):
        return
    if _has_cache_control_override(kwargs):
        return
    kwargs.setdefault("model_kwargs", {})["cache_control"] = {"type": "ephemeral"}


def _enable_openrouter_429_retry(chat_model: Any) -> None:
    """Add 429 to the OpenRouter SDK's retryable status codes (default ["5XX"]).

    Upstream rate limits ("temporarily rate-limited upstream", whose
    Retry-After the SDK backoff already honors) otherwise fail the run outright.
    """
    sdk_config = getattr(getattr(chat_model, "client", None), "sdk_configuration", None)
    retry_config: Any = getattr(sdk_config, "retry_config", None)
    # Skip the UNSET sentinel (max_retries=0) and explicit caller overrides.
    if not hasattr(retry_config, "status_codes_override"):
        return
    if retry_config.status_codes_override:
        return
    retry_config.status_codes_override = ["429", "5XX"]


def _apply_auto_config(
    provider: str,
    model_id: str,
    is_third_party: bool,
    kwargs: dict[str, Any],
    original_provider: str | None = None,
) -> None:
    """Auto-enable provider-specific features (thinking, reasoning, etc.).

    Mutates *kwargs* in place.  Only sets keys that the caller hasn't already
    provided, so explicit user settings are never overridden.
    """
    # Anthropic: extended thinking
    if provider == "anthropic" and "thinking" not in kwargs:
        _supports_thinking = original_provider in _THINKING_CAPABLE_PROVIDERS
        # Detect local proxy (e.g. ccproxy): thinking blocks in conversation
        # history cause 422 errors because the proxy doesn't accept 'thinking'
        # as a valid content block type on round-trip.
        if not is_third_party:
            base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
            _is_proxy = "127.0.0.1" in base_url or "localhost" in base_url
        else:
            _is_proxy = False
        if _is_proxy or (is_third_party and not _supports_thinking):
            # Mandatory-thinking Kimi models (K3 / Kimi For Coding) must declare
            # thinking so with_structured_output avoids forced tool_choice (400).
            # max_tokens must exceed budget_tokens (default resolves to 4096).
            if is_third_party and _is_mandatory_thinking_kimi(model_id):
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}
                kwargs.setdefault("max_tokens", 16000)
        elif "fable" in model_id or model_id.endswith(
            ("opus-5", "sonnet-5", "4-6", "4-7", "4-8")
        ):
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
            kwargs.setdefault("effort", "max")
        else:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

    # OpenAI (native, not third-party routed): reasoning
    if provider == "openai" and not is_third_party and "reasoning" not in kwargs:
        _default_effort = (
            "xhigh"
            if (
                "5.4" in model_id
                or "5.5" in model_id
                or "5.6" in model_id
                or "codex" in model_id
            )
            else "high"
        )
        _eff = _resolve_reasoning_effort(_default_effort)
        kwargs["reasoning"] = {"effort": _eff, "summary": "auto"}

    # Google GenAI: surface thinking traces
    if provider == "google-genai":
        kwargs.setdefault("include_thoughts", True)

    # Ollama: separate reasoning content from response for thinking models
    if provider == "ollama" and "reasoning" not in kwargs:
        kwargs["reasoning"] = True


def get_chat_model(
    model: str | None = None,
    provider: str | None = None,
    **kwargs: Any,
) -> Any:
    """Get a chat model instance.

    Args:
        model: Model name (short name like 'claude-sonnet-4-6' or full ID
               like 'claude-sonnet-4-6-20250929'). Defaults to DEFAULT_MODEL.
        provider: Override the provider (e.g., 'anthropic', 'openai').
                  If not specified, inferred from model name or defaults to 'anthropic'.
        **kwargs: Additional arguments passed to init_chat_model (e.g., temperature).

    Returns:
        A LangChain chat model instance.

    Examples:
        >>> model = get_chat_model()  # Uses default (claude-sonnet-4-6)
        >>> model = get_chat_model("claude-opus-4-8")  # Use short name
        >>> model = get_chat_model("gpt-4o")  # OpenAI model
        >>> model = get_chat_model("claude-3-opus-20240229", provider="anthropic")  # Full ID
    """
    model = model or DEFAULT_MODEL

    # Look up short name in registry (provider-aware)
    model_id = None
    if provider:
        # Try exact match with provider first
        for name, mid, p in _MODEL_ENTRIES:
            if name == model and p == provider:
                model_id = mid
                break
    if model_id is None and model in MODELS:
        model_id, default_provider = MODELS[model]
        provider = provider or default_provider

    if model_id is None:
        # Assume it's a full model ID
        model_id = model
        # Try to infer provider from model ID prefix
        if provider is None:
            if model_id.startswith(("claude-", "anthropic")):
                provider = "anthropic"
            elif model_id.startswith(("gpt-", "o1", "davinci", "text-")):
                provider = "openai"
            elif model_id.startswith("gemini"):
                provider = "google-genai"
            elif model_id.startswith("ollama:"):
                provider = "ollama"
                model_id = model_id.removeprefix("ollama:")
            else:
                provider = "anthropic"  # Default fallback

    # Anthropic base_url override (e.g. ccproxy at localhost:8000/api/v1)
    _is_third_party = (
        provider in _OPENAI_ROUTED_PROVIDERS or provider in _ANTHROPIC_ROUTED_PROVIDERS
    )
    _is_openai_proxy = False
    _original_provider: str | None = None
    if provider == "anthropic":
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
        if base_url:
            kwargs["base_url"] = base_url
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            kwargs["api_key"] = api_key

    # Native OpenAI base_url override (e.g. ccproxy Codex at localhost:8000/codex/v1)
    elif provider == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        if base_url:
            kwargs["base_url"] = base_url
            _is_openai_proxy = _is_ccproxy_codex()
            if _is_openai_proxy:
                # Use Responses API for ccproxy: bypasses the format chain
                # converter (Chat→Responses→Chat) which returns 502 on
                # complex responses.  System messages are converted to
                # developer role by _patch_ccproxy_system_to_developer().
                kwargs.setdefault("use_responses_api", True)
                # Streaming must stay ON for Responses API: ccproxy's
                # StreamingBufferService loses output when assembling
                # non-streaming responses.  (The old streaming=False was
                # for Chat Completions tool_call duplication — not an issue
                # with the Responses API SSE format.)
                kwargs.pop("streaming", None)  # remove if set elsewhere
                # ccproxy forwards client headers upstream and only
                # gap-fills its own, so the Codex backend sees this
                # client's identity. Without Codex-CLI-shaped headers it
                # rejects current models ("The '<model>' model requires
                # a newer version of Codex").
                _codex_ver = _resolve_codex_client_version()
                _headers = kwargs.get("default_headers") or {}
                kwargs["default_headers"] = _headers
                _headers.setdefault("originator", "codex_cli_rs")
                _headers.setdefault("version", _codex_ver)
                _headers.setdefault(
                    "User-Agent",
                    f"codex_cli_rs/{_headers['version']} (EvoScientist)",
                )
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            kwargs["api_key"] = api_key

    elif provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if api_key:
            kwargs["api_key"] = api_key

    # OpenAI-routed providers → route through OpenAI provider with base_url
    elif provider in _OPENAI_ROUTED_PROVIDERS:
        _original_provider = provider
        base_url_default, api_key_env = _OPENAI_ROUTED_PROVIDERS[provider]
        if provider == "custom-openai":
            base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL", "")
            if not base_url:
                raise ValueError(
                    "CUSTOM_OPENAI_BASE_URL environment variable is required when using "
                    "the 'custom-openai' provider. Please set it to your "
                    "OpenAI-compatible API endpoint URL (e.g. https://api.openai.com/v1)."
                )
            base_url = base_url.rstrip("/")
        else:
            base_url = base_url_default
        if base_url:
            kwargs["base_url"] = base_url
        api_key = os.environ.get(api_key_env, "")
        if api_key:
            kwargs["api_key"] = api_key
        # SiliconFlow: disable thinking — LangChain drops reasoning_content
        # from history, causing error 20015 on multi-turn requests.
        if provider == "siliconflow":
            kwargs.setdefault("extra_body", {})["enable_thinking"] = False
        # Moonshot: disable thinking for pre-K3 models to prevent LangChain from
        # dropping reasoning_content, which causes multi-turn conversation errors
        # (error 20015). Even native thinking models like kimi-k2-thinking operate
        # in non-thinking mode. kimi-k3+ is exempt: always-thinking, and
        # Moonshot's K3 guide forbids the K2.x `thinking` parameter for it.
        if provider == "moonshot" and not model_id.startswith("kimi-k3"):
            kwargs.setdefault("extra_body", {})["thinking"] = {"type": "disabled"}
        provider = "openai"

    # OpenRouter → native ChatOpenRouter via init_chat_model.
    elif provider == "openrouter":
        _is_third_party = True
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if api_key:
            kwargs["api_key"] = api_key
        # Reasoning via `effort` + `summary: "auto"` so a readable reasoning
        # summary is returned for display. OpenAI-Responses also emits encrypted
        # reasoning items (`rs_*` id) that can't be replayed on multi-turn
        # passback (OpenRouter's `/responses` beta is stateless, store=false —
        # "Item with id 'rs_...' not found"); the patch strips them on passback,
        # so enabling `summary` is safe. See langchain-ai/langchain#37777.
        # Note: mandatory-reasoning endpoints (kimi-k3, grok-4.5, …) reject
        # effort "none" with HTTP 400 — that error is surfaced to the user
        # as-is; pick a real effort (low+) for those models.
        effort = _resolve_reasoning_effort("high")
        kwargs.setdefault("reasoning", {"effort": effort, "summary": "auto"})
        # App attribution (issue #339): identify EvoScientist to OpenRouter so
        # usage is credited to the project (app rankings, model app tabs,
        # analytics) rather than langchain-openrouter's LangChain-branded
        # defaults. setdefault so an explicit caller kwarg wins; values are
        # configurable via EVOSCIENTIST_OPENROUTER_* env (fed from the config
        # file by apply_config_to_env). Applied only here, so no other provider
        # ever receives these kwargs.
        kwargs.setdefault(
            "app_url",
            os.environ.get("EVOSCIENTIST_OPENROUTER_HTTP_REFERER", "").strip()
            or OPENROUTER_DEFAULT_HTTP_REFERER,
        )
        kwargs.setdefault(
            "app_title",
            os.environ.get("EVOSCIENTIST_OPENROUTER_APP_TITLE", "").strip()
            or OPENROUTER_DEFAULT_APP_TITLE,
        )
        # app_categories must be a list[str] (langchain-openrouter joins it into
        # the X-OpenRouter-Categories header); split the comma-separated config
        # value and drop blanks so a stray comma/space can't emit an empty one.
        _app_categories_raw = (
            os.environ.get("EVOSCIENTIST_OPENROUTER_APP_CATEGORIES", "").strip()
            or OPENROUTER_DEFAULT_APP_CATEGORIES
        )
        _app_categories = [
            c.strip() for c in _app_categories_raw.split(",") if c.strip()
        ]
        # Cap to the per-request limit and warn, so a misconfigured extra is
        # dropped predictably here (and surfaced to the user) rather than being
        # silently truncated server-side.
        _limit = _OPENROUTER_MAX_CATEGORIES_PER_REQUEST
        if len(_app_categories) > _limit:
            warnings.warn(
                f"OpenRouter accepts at most {_limit} app categories per "
                f"request, so only the first {_limit} are sent: "
                f"{_app_categories[:_limit]}. Ignoring the rest: "
                f"{_app_categories[_limit:]}. Set "
                f"EVOSCIENTIST_OPENROUTER_APP_CATEGORIES (or the "
                f"openrouter_app_categories config) to at most {_limit} "
                f"categories to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
            _app_categories = _app_categories[:_limit]
        if _app_categories:
            kwargs.setdefault("app_categories", _app_categories)
        _patch_openrouter_strip_responses_reasoning()
        _patch_openrouter_structured_output()

    # Anthropic-routed providers → route through Anthropic provider with base_url
    elif provider in _ANTHROPIC_ROUTED_PROVIDERS:
        _original_provider = provider
        base_url_default, api_key_env = _ANTHROPIC_ROUTED_PROVIDERS[provider]
        if provider == "custom-anthropic":
            base_url = os.environ.get("CUSTOM_ANTHROPIC_BASE_URL", "")
            if not base_url:
                raise ValueError(
                    "CUSTOM_ANTHROPIC_BASE_URL environment variable is required when using "
                    "the 'custom-anthropic' provider. Please set it to your "
                    "Anthropic-compatible API endpoint URL (e.g. https://api.anthropic.com)."
                )
            base_url = base_url.rstrip("/")
        elif provider == "minimax":
            base_url = os.environ.get("MINIMAX_BASE_URL", base_url_default).rstrip("/")
        else:
            base_url = base_url_default
        if base_url:
            kwargs["base_url"] = base_url
        api_key = os.environ.get(api_key_env, "")
        if api_key:
            kwargs["api_key"] = api_key
        # Kimi Coding Plan requires claude-code User-Agent header
        if provider == "kimi-coding":
            kwargs.setdefault("default_headers", {})["User-Agent"] = "claude-code/0.1.0"
        provider = "anthropic"

    elif provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "")
        if base_url:
            kwargs["base_url"] = base_url

    _apply_auto_config(provider, model_id, _is_third_party, kwargs, _original_provider)
    # OpenAI-routed routers (e.g. Requesty) reassign ``provider`` to "openai"
    # above, so use the original provider name to detect router-level caching.
    _cache_provider = _original_provider or provider
    _apply_openrouter_anthropic_prompt_cache(_cache_provider, model_id, kwargs)

    _uses_native_deepseek = provider == "deepseek" or (
        provider == "openai"
        and _original_provider == "custom-openai"
        and _is_deepseek_endpoint(kwargs.get("base_url"))
    )

    # User-level override for the OpenAI Responses API vs Chat Completions.
    # When "false", force Chat Completions and drop reasoning (which triggers
    # the Responses API path in langchain-openai). Only applies to OpenAI.
    if _uses_native_deepseek:
        if kwargs.get("use_responses_api") is True:
            raise ValueError(
                "DeepSeek does not support the OpenAI Responses API. "
                "Remove use_responses_api=True."
            )
        kwargs.pop("use_responses_api", None)
    elif provider == "openai":
        _responses_api_setting = (
            os.environ.get("EVOSCIENTIST_USE_RESPONSES_API", "").strip().lower()
        )
        if _responses_api_setting == "false":
            kwargs["use_responses_api"] = False
            kwargs.pop("reasoning", None)
        elif _responses_api_setting == "true":
            kwargs["use_responses_api"] = True

    if _is_openai_proxy and kwargs.get("use_responses_api") is True:
        reasoning = kwargs.setdefault("reasoning", {})
        if isinstance(reasoning, dict):
            reasoning = dict(reasoning)
            reasoning.setdefault("context", "all_turns")
            kwargs["reasoning"] = reasoning

    if _uses_native_deepseek:
        chat_model = EvoChatDeepSeek(model=model_id, **kwargs)
    else:
        chat_model = init_chat_model(model=model_id, model_provider=provider, **kwargs)

    # Flatten list content to strings for strict OpenAI-compatible providers
    # (SiliconFlow, OpenRouter, custom-openai, etc.) and
    # native OpenAI through a proxy, to avoid "sequence expected string" errors.
    # Moonshot and Kimi Coding support standard format, no patch needed.
    # Mandatory-thinking Kimi models on Anthropic-routed endpoints are exempt:
    # flatten drops thinking blocks, which Kimi requires on tool-call turns.
    _no_patch_providers = {"moonshot", "kimi-coding"}
    if (
        (_is_third_party or _is_openai_proxy)
        and _original_provider not in _no_patch_providers
        and not _uses_native_deepseek
        and not (provider == "anthropic" and _is_mandatory_thinking_kimi(model_id))
    ):
        # Anthropic-routed providers accept media in tool results natively;
        # only OpenAI-compatible providers need tool-media hoisting.
        _hoist = _original_provider not in _ANTHROPIC_ROUTED_PROVIDERS
        _patch_openai_compat_content(chat_model, hoist_tool_media=_hoist)

    if _is_openai_proxy:
        _patch_ccproxy_system_to_developer(chat_model)

    if provider == "openrouter":
        _enable_openrouter_429_retry(chat_model)

    if provider == "anthropic":
        _patch_anthropic_strip_foreign_reasoning()
        _patch_anthropic_structured_output()

    apply_known_context_window(chat_model)

    return chat_model
