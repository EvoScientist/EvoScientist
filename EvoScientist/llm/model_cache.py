"""Dynamic model list fetching and caching for OpenAI-compatible providers.

Fetches available models from providers' OpenAI-compatible ``GET /v1/models``
endpoint and caches results locally so the API is not hit on every request.
Cached entries are valid for :data:`CACHE_TTL` seconds (24 h by default);
users can bypass the cache at any time via ``force=True``.

Supported providers are those that expose a standard OpenAI-compatible
``/v1/models`` endpoint (i.e. the OpenAI-routed providers defined in
``models.py``).  Anthropic-compatible and non-standard providers (e.g.
``ollama``, ``google-genai``, ``nvidia``) are out of scope.

Cache file: ``~/.config/evoscientist/model_cache.json``
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

CACHE_TTL: int = 86400  # seconds — 24 hours

# Provider → (default_base_url, api_key_env_var)
# ``None`` for base_url means the value is resolved at runtime from env vars.
_SUPPORTED: dict[str, tuple[str | None, str]] = {
    "openai": (None, "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "moonshot": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "siliconflow": ("https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY"),
    "zhipu": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY"),
    "zhipu-code": ("https://open.bigmodel.cn/api/coding/paas/v4", "ZHIPU_API_KEY"),
    "volcengine": ("https://ark.cn-beijing.volces.com/api/v3", "VOLCENGINE_API_KEY"),
    "dashscope": (
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY",
    ),
    "custom-openai": (None, "CUSTOM_OPENAI_API_KEY"),
}


def is_supported(provider: str) -> bool:
    """Return ``True`` if *provider* supports dynamic model fetching."""
    return provider in _SUPPORTED


# =============================================================================
# Cache helpers
# =============================================================================


def _get_cache_path() -> Path:
    from ..config.settings import get_config_dir

    return get_config_dir() / "model_cache.json"


def _load_cache() -> dict:
    path = _get_cache_path()
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    path = _get_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def get_cached_models(provider: str) -> list[str] | None:
    """Return cached model IDs for *provider* if still within the TTL.

    Args:
        provider: Provider name (e.g. ``"openai"``, ``"deepseek"``).

    Returns:
        A list of model ID strings when the cache is valid, otherwise ``None``.
    """
    entry = _load_cache().get(provider)
    if not entry:
        return None
    if time.time() - entry.get("fetched_at", 0) > CACHE_TTL:
        return None
    return entry.get("models") or None


# =============================================================================
# Base-URL / key resolution
# =============================================================================


def _resolve(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str | None, str | None]:
    """Return ``(resolved_base_url, resolved_api_key)`` for *provider*."""
    if provider not in _SUPPORTED:
        return None, None

    default_base_url, api_key_env = _SUPPORTED[provider]

    resolved_key = api_key or os.environ.get(api_key_env, "") or None

    if base_url:
        resolved_base_url = base_url.rstrip("/")
    elif provider == "openai":
        resolved_base_url = (
            os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
            or "https://api.openai.com/v1"
        )
    elif provider == "custom-openai":
        resolved_base_url = (
            os.environ.get("CUSTOM_OPENAI_BASE_URL", "").rstrip("/") or None
        )
    else:
        resolved_base_url = default_base_url

    return resolved_base_url, resolved_key


# =============================================================================
# Public API — sync + async fetch
# =============================================================================


def fetch_models(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    force: bool = False,
) -> list[str] | None:
    """Return model IDs for *provider*, using cache when fresh.

    Fetches from the provider's OpenAI-compatible ``GET /v1/models`` endpoint
    when the cache is absent, stale, or *force* is ``True``.  The result is
    written back to the cache on a successful fetch.

    Never raises — returns ``None`` on any failure or unsupported provider.

    Args:
        provider: Provider name (must be in the supported set).
        api_key: API key override; falls back to the provider's env var.
        base_url: Base URL override; falls back to the provider's default.
        force: Bypass the cache and always contact the API.

    Returns:
        List of model ID strings, or ``None`` on failure / unsupported provider.
    """
    if provider not in _SUPPORTED:
        return None

    if not force:
        cached = get_cached_models(provider)
        if cached is not None:
            return cached

    resolved_base_url, resolved_key = _resolve(
        provider, api_key=api_key, base_url=base_url
    )
    if not resolved_base_url or not resolved_key:
        return None

    try:
        import httpx

        resp = httpx.get(
            f"{resolved_base_url}/models",
            headers={"Authorization": f"Bearer {resolved_key}"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        models = [m["id"] for m in resp.json().get("data", []) if "id" in m]
        if not models:
            return None

        cache = _load_cache()
        cache[provider] = {"models": models, "fetched_at": time.time()}
        _save_cache(cache)
        return models
    except Exception:
        return None


async def fetch_models_async(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    force: bool = False,
) -> list[str] | None:
    """Async variant of :func:`fetch_models`.

    Never raises — returns ``None`` on any failure or unsupported provider.

    Args:
        provider: Provider name (must be in the supported set).
        api_key: API key override; falls back to the provider's env var.
        base_url: Base URL override; falls back to the provider's default.
        force: Bypass the cache and always contact the API.

    Returns:
        List of model ID strings, or ``None`` on failure / unsupported provider.
    """
    if provider not in _SUPPORTED:
        return None

    if not force:
        cached = get_cached_models(provider)
        if cached is not None:
            return cached

    resolved_base_url, resolved_key = _resolve(
        provider, api_key=api_key, base_url=base_url
    )
    if not resolved_base_url or not resolved_key:
        return None

    try:
        import httpx

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{resolved_base_url}/models",
                headers={"Authorization": f"Bearer {resolved_key}"},
            )
        if resp.status_code != 200:
            return None
        models = [m["id"] for m in resp.json().get("data", []) if "id" in m]
        if not models:
            return None

        cache = _load_cache()
        cache[provider] = {"models": models, "fetched_at": time.time()}
        _save_cache(cache)
        return models
    except Exception:
        return None
