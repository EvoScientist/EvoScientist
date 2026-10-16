"""API key checks for the model providers and Tavily.

Kept free of UI imports so the backend's HTTP routes can use them without
loading the onboarding wizard. :data:`PROVIDERS` maps each provider to its
key field, its base URL field and its check; :func:`check_provider_key` runs
a check and returns a :class:`KeyCheck`.

Every check makes one live request with a 10 s timeout and no retries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

CheckStatus = Literal["valid", "invalid", "inconclusive", "unavailable", "empty"]

_TIMEOUT = 10


class KeyCheck(NamedTuple):
    """Result of a key check.

    ``valid`` and ``invalid`` are conclusive. ``inconclusive`` covers a
    transient or unexpected failure (offline, timeout, 5xx), ``unavailable``
    a provider or endpoint that cannot be checked, ``empty`` a missing key.
    """

    status: CheckStatus
    message: str
    models: tuple[str, ...] = ()

    @property
    def ok(self) -> bool | None:
        """``True`` or ``False`` for a conclusive result, else ``None``."""
        if self.status == "valid":
            return True
        if self.status == "invalid":
            return False
        return None


_SKIPPED = KeyCheck("empty", "Skipped (no key provided)")
_INVALID = KeyCheck("invalid", "Invalid API key")

# =============================================================================
# API Key Validation
# =============================================================================


# Avoids bare "invalid" — collides with "Invalid request: model X not found".
_AUTH_FAILURE_HINTS = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "authentication",
    "invalid api key",
    "invalid_api_key",
    "incorrect api key",
    "incorrect_api_key",
    "api key not valid",
    "invalid token",
)

_TRANSIENT_HINTS = (
    "429",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "connection",
    "service unavailable",
    "temporarily unavailable",
    "upstream",
)


def _classify_validation_error(error: BaseException) -> KeyCheck:
    """Classify a check's exception as an auth failure or an inconclusive result.

    An exception that matches neither hint list is inconclusive too, with the
    error text as its message.
    """
    s = str(error).lower()
    if any(h in s for h in _AUTH_FAILURE_HINTS):
        return _INVALID
    if any(h in s for h in _TRANSIENT_HINTS):
        return KeyCheck(
            "inconclusive",
            "Validation inconclusive — transient error, try again later",
        )
    return KeyCheck("inconclusive", f"Error: {error}")


def _status_result(status_code: int) -> KeyCheck:
    """Result for an HTTP status other than the ones a check accepts."""
    # Only 401/403 mean the key is actually rejected. 429 (rate-limit) and 5xx
    # (provider incident) leave the key validity unknown — surface the real
    # status so the user doesn't go re-roll a good key during an outage.
    if status_code in (401, 403):
        return _INVALID
    return KeyCheck("inconclusive", f"Validation inconclusive (HTTP {status_code})")


def check_anthropic_key(api_key: str) -> KeyCheck:
    """Check an Anthropic API key by listing models."""
    if not api_key:
        return _SKIPPED

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, timeout=_TIMEOUT, max_retries=0)
        client.models.list()
        return KeyCheck("valid", "Valid")
    except anthropic.AuthenticationError:
        return _INVALID
    except Exception as e:
        return _classify_validation_error(e)


def check_openai_key(api_key: str) -> KeyCheck:
    """Check an OpenAI API key by listing models."""
    if not api_key:
        return _SKIPPED

    try:
        import openai

        client = openai.OpenAI(api_key=api_key, timeout=_TIMEOUT, max_retries=0)
        client.models.list()
        return KeyCheck("valid", "Valid")
    except openai.AuthenticationError:
        return _INVALID
    except Exception as e:
        return _classify_validation_error(e)


def check_nvidia_key(api_key: str) -> KeyCheck:
    """Check an NVIDIA API key by listing models with it.

    ``ChatNVIDIA(api_key=...)`` sends no request, and the classmethod
    ``get_available_models()`` builds a fresh client that reads
    ``NVIDIA_API_KEY`` from the environment instead of the key under test;
    the instance property ``available_models`` uses the client's own key.
    """
    if not api_key:
        return _SKIPPED

    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        client = ChatNVIDIA(
            api_key=api_key, model="meta/llama-3.1-8b-instruct", timeout=_TIMEOUT
        )
        client.available_models  # noqa: B018  (property makes the request)
        return KeyCheck("valid", "Valid")
    except Exception as e:
        return _classify_validation_error(e)


def check_google_key(api_key: str) -> KeyCheck:
    """Check a Google GenAI API key by fetching the first model."""
    if not api_key:
        return _SKIPPED

    try:
        from google import genai

        # google-genai counts the timeout in milliseconds.
        client = genai.Client(
            api_key=api_key, http_options={"timeout": _TIMEOUT * 1000}
        )
        pager = client.models.list(config={"page_size": 1})
        next(iter(pager))  # fetch first model only
        return KeyCheck("valid", "Valid")
    except StopIteration:
        # Empty result but request succeeded — key is valid
        return KeyCheck("valid", "Valid")
    except Exception as e:
        # Google-specific 400 phrasing not in the shared hint list.
        error_str = str(e).lower()
        if "api_key_invalid" in error_str or "api key invalid" in error_str:
            return _INVALID
        return _classify_validation_error(e)


def check_minimax_key(
    api_key: str,
    base_url: str = "https://api.minimaxi.com/anthropic",
) -> KeyCheck:
    """Check a MiniMax API key without consuming tokens.

    Sends a messages.create() with an empty model string.  MiniMax checks
    auth *before* validating request params, so a valid key returns 400
    (bad model) while an invalid key returns 401.

    Args:
        api_key: The MiniMax API key to check.
        base_url: Anthropic-compatible endpoint (global or mainland China).
    """
    if not api_key:
        return _SKIPPED

    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=_TIMEOUT,
            max_retries=0,
        )
        client.messages.create(
            model="",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        # Unexpected success — treat as valid
        return KeyCheck("valid", "Valid")
    except anthropic.AuthenticationError:
        return _INVALID
    except anthropic.APIStatusError:
        # Any non-auth HTTP error (400 bad model, 500 insufficient balance,
        # etc.) means the key itself was accepted → treat as valid.
        return KeyCheck("valid", "Valid")
    except Exception as e:
        return _classify_validation_error(e)


def _check_openai_compatible_key(api_key: str, base_url: str) -> KeyCheck:
    """Check a key by listing models on a fixed OpenAI-compatible endpoint."""
    if not api_key:
        return _SKIPPED

    try:
        import openai

        client = openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=_TIMEOUT, max_retries=0
        )
        client.models.list()
        return KeyCheck("valid", "Valid")
    except Exception as e:
        return _classify_validation_error(e)


def check_siliconflow_key(api_key: str) -> KeyCheck:
    """Check a SiliconFlow API key by listing models."""
    return _check_openai_compatible_key(api_key, "https://api.siliconflow.cn/v1")


def check_openrouter_key(api_key: str) -> KeyCheck:
    """Check an OpenRouter API key via the authenticated /auth/key endpoint."""
    if not api_key:
        return _SKIPPED

    try:
        import httpx

        resp = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return KeyCheck("valid", "Valid")
        return _status_result(resp.status_code)
    except Exception as e:
        return _classify_validation_error(e)


def check_opper_key(api_key: str) -> KeyCheck:
    """Check an Opper API key via the authenticated ``/models`` endpoint.

    Unlike the Requesty/Atlas Cloud/Novita routers, Opper's OpenAI-compatible
    ``/v3/compat/models`` endpoint is authenticated: a missing or invalid key
    returns 401 (verified against the live endpoint), so no sentinel-model
    request is needed.

    - valid key → 200 (the account's model catalog);
    - invalid/missing key → 401/403;
    - 429 (rate-limit) / 5xx (gateway incident) leave validity unknown, so a
      transient outage doesn't reject a good key.
    """
    if not api_key:
        return _SKIPPED

    try:
        import httpx

        resp = httpx.get(
            "https://api.opper.ai/v3/compat/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return KeyCheck("valid", "Valid")
        return _status_result(resp.status_code)
    except Exception as e:
        return _classify_validation_error(e)


def check_atlascloud_key(api_key: str) -> KeyCheck:
    """Check an Atlas Cloud key with a nonexistent sentinel model.

    The probe deliberately targets a nonexistent sentinel model. A 404 means
    authentication passed and model resolution failed; 200 also confirms
    authentication if the sentinel unexpectedly resolves. A 401/403 means the
    key was rejected. Other statuses remain inconclusive until verified.
    """
    if not api_key:
        return _SKIPPED

    try:
        import httpx

        resp = httpx.post(
            "https://api.atlascloud.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "atlascloud/auth-preflight",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout=_TIMEOUT,
        )
        if resp.status_code in (200, 404):
            return KeyCheck("valid", "Valid")
        # Atlas checks account balance before model resolution: a valid key
        # on an uncredited account gets 402 from the sentinel probe.
        if resp.status_code == 402:
            return KeyCheck("valid", "Valid (insufficient balance — top up to use)")
        return _status_result(resp.status_code)
    except Exception as e:
        return _classify_validation_error(e)


def check_requesty_key(api_key: str) -> KeyCheck:
    """Check a Requesty API key against the router's auth layer.

    Unlike OpenRouter, Requesty's ``/v1/models`` endpoint returns HTTP 200
    (the public model catalog) even for a missing or invalid key, so it
    cannot be used to check a key. We instead issue a minimal
    ``/v1/chat/completions`` request, but deliberately target a nonexistent
    sentinel model: the router checks auth *before* resolving the model, so
    the response distinguishes the two failures without depending on any
    real model staying available upstream.

    - valid key → 404 ("Model and/or policy not supported"), i.e. auth passed
      (or 200 in the unlikely event the sentinel ever resolves);
    - invalid/missing key → 401/403 ("Invalid authorization token");
    - 429 (rate-limit) / 5xx (router incident) leave validity unknown, so a
      transient outage doesn't reject a good key.
    """
    if not api_key:
        return _SKIPPED

    try:
        import httpx

        resp = httpx.post(
            "https://router.requesty.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                # Deliberately nonexistent sentinel: auth is resolved before
                # the model, so a valid key gets a 404 (model-not-found)
                # rather than depending on a specific model being available.
                "model": "requesty/auth-preflight",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout=_TIMEOUT,
        )
        # 200 (accepted) or 404 (auth passed, model not found) → key is good.
        if resp.status_code in (200, 404):
            return KeyCheck("valid", "Valid")
        return _status_result(resp.status_code)
    except Exception as e:
        return _classify_validation_error(e)


def check_novita_key(api_key: str) -> KeyCheck:
    """Check a Novita API key against the router's auth layer.

    Like Requesty and Atlas Cloud, Novita's ``/v1/models`` endpoint returns
    HTTP 200 (the public model catalog) even for a missing or invalid key, so
    it cannot be used to check a key (verified against the live endpoint). We
    instead issue a minimal ``/v1/chat/completions`` request with a
    deliberately nonexistent sentinel model: auth is resolved before the
    model, so a valid key doesn't depend on any real model staying available
    upstream.

    - invalid/missing key → 401/403 (confirmed against the live endpoint);
    - valid key → 200 or 404 (model-not-found, auth passed), mirroring the
      Requesty/Atlas Cloud sentinel pattern;
    - 429 (rate-limit) / 5xx (service incident) leave validity unknown, so a
      transient outage doesn't reject a good key.
    """
    if not api_key:
        return _SKIPPED

    try:
        import httpx

        resp = httpx.post(
            "https://api.novita.ai/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "novita/auth-preflight",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout=_TIMEOUT,
        )
        if resp.status_code in (200, 404):
            return KeyCheck("valid", "Valid")
        return _status_result(resp.status_code)
    except Exception as e:
        return _classify_validation_error(e)


def check_deepseek_key(api_key: str) -> KeyCheck:
    """Check a DeepSeek API key by listing models."""
    return _check_openai_compatible_key(api_key, "https://api.deepseek.com")


def check_zhipu_key(api_key: str) -> KeyCheck:
    """Check a ZhipuAI API key by listing models.

    Uses the general endpoint for validation — both zhipu and zhipu-code
    share the same API key, only the base_url differs at runtime.
    """
    return _check_openai_compatible_key(api_key, "https://open.bigmodel.cn/api/paas/v4")


def check_volcengine_key(api_key: str) -> KeyCheck:
    """Check a Volcengine API key by listing models.

    Uses the general endpoint for validation; volcengine and volcengine-code
    share the same API key and only differ in their runtime base URL.
    """
    return _check_openai_compatible_key(
        api_key, "https://ark.cn-beijing.volces.com/api/v3"
    )


def check_dashscope_key(api_key: str) -> KeyCheck:
    """Check a DashScope API key by listing models."""
    return _check_openai_compatible_key(
        api_key, "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def check_dashscope_code_key(api_key: str) -> KeyCheck:
    """Check a DashScope Coding Plan API key (sk-sp-* subscription keys).

    The coding endpoint at coding.dashscope.aliyuncs.com does not expose
    /models (returns 404), so validation issues a minimal chat completion
    instead of the usual models.list() probe.
    """
    if not api_key:
        return _SKIPPED

    try:
        import openai

        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://coding.dashscope.aliyuncs.com/v1",
            timeout=_TIMEOUT,
            max_retries=0,
        )
        client.chat.completions.create(
            model="qwen3-coder-plus",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return KeyCheck("valid", "Valid")
    except Exception as e:
        return _classify_validation_error(e)


def check_moonshot_key(api_key: str) -> KeyCheck:
    """Check a Moonshot API key by listing models."""
    return _check_openai_compatible_key(api_key, "https://api.moonshot.cn/v1")


def check_kimi_key(api_key: str) -> KeyCheck:
    """Check a Kimi Coding Plan API key by listing models.

    Uses the Anthropic-compatible endpoint at api.kimi.com/coding/.
    """
    if not api_key:
        return _SKIPPED

    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=api_key,
            base_url="https://api.kimi.com/coding/",
            default_headers={"User-Agent": "claude-code/0.1.0"},
            timeout=_TIMEOUT,
            max_retries=0,
        )
        client.models.list()
        return KeyCheck("valid", "Valid")
    except Exception as e:
        return _classify_validation_error(e)


def check_xiaomi_key(
    api_key: str,
    base_url: str = "https://api.xiaomimimo.com/anthropic",
) -> KeyCheck:
    """Check a Xiaomi MiMo API key by listing models.

    ``/v1/models`` requires auth (401 on a bad key), so no tokens are spent.
    ``base_url`` is the Anthropic-compatible endpoint (pay-as-you-go or a
    Token Plan region); its sibling ``/v1`` root serves the model list.
    """
    return _check_openai_compatible_key(
        api_key, base_url.rstrip("/").removesuffix("/anthropic") + "/v1"
    )


def check_tavily_key(api_key: str) -> KeyCheck:
    """Check a Tavily API key by running a one-result search."""
    if not api_key:
        return _SKIPPED

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        client.search("test", max_results=1, timeout=_TIMEOUT)
        return KeyCheck("valid", "Valid")
    except Exception as e:
        return _classify_validation_error(e)


def _listed_models(body: Any) -> tuple[str, ...] | None:
    """Model ids from a ``/models`` body (``{"data": [{"id": ...}]}``), or
    ``None`` when the body is not such a list."""
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        return None
    return tuple(
        str(item["id"])
        for item in body["data"]
        if isinstance(item, dict) and item.get("id")
    )


def _models_listed(models: tuple[str, ...]) -> KeyCheck:
    # A listing proves the URL, not the key: some servers list models for any
    # key, so the message says what was actually checked.
    return KeyCheck(
        "valid",
        f"The endpoint listed {len(models)} model(s) with this key",
        models,
    )


def check_custom_openai(api_key: str, base_url: str) -> KeyCheck:
    """List models on an OpenAI-compatible endpoint (``{base_url}/models``).

    ``base_url`` includes the ``/v1`` part, as the runtime expects. The key is
    passed explicitly even when empty, so the SDK never falls back to
    ``OPENAI_API_KEY`` and sends the user's OpenAI key to another server.
    """
    if not base_url:
        return KeyCheck("unavailable", "No base URL to check")

    try:
        import openai

        client = openai.OpenAI(
            api_key=api_key or "none",
            base_url=base_url.rstrip("/"),
            timeout=_TIMEOUT,
            max_retries=0,
        )
        # The SDK does not validate the body, so decide on the raw JSON.
        raw = client.models.with_raw_response.list()
        models = _listed_models(raw.http_response.json())
    except (openai.AuthenticationError, openai.PermissionDeniedError):
        return _INVALID
    except openai.NotFoundError:
        return KeyCheck("unavailable", "The endpoint has no model list")
    except openai.APIStatusError as e:
        return _status_result(e.status_code)
    except ValueError:
        return KeyCheck("unavailable", "The endpoint's model list is not JSON")
    except Exception as e:
        return _classify_validation_error(e)
    if models is None:
        return KeyCheck("unavailable", "The endpoint's model list has no data")
    return _models_listed(models)


def check_custom_anthropic(api_key: str, base_url: str) -> KeyCheck:
    """List models on an Anthropic-compatible endpoint (``{base_url}/v1/models``).

    Reads only the first page of the list.
    """
    if not base_url:
        return KeyCheck("unavailable", "No base URL to check")

    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=api_key or "none",
            base_url=base_url.rstrip("/"),
            timeout=_TIMEOUT,
            max_retries=0,
        )
        raw = client.models.with_raw_response.list()
        models = _listed_models(raw.http_response.json())
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
        return _INVALID
    except anthropic.NotFoundError:
        return KeyCheck("unavailable", "The endpoint has no model list")
    except anthropic.APIStatusError as e:
        return _status_result(e.status_code)
    except ValueError:
        return KeyCheck("unavailable", "The endpoint's model list is not JSON")
    except Exception as e:
        return _classify_validation_error(e)
    if models is None:
        return KeyCheck("unavailable", "The endpoint's model list has no data")
    return _models_listed(models)


def check_ollama(api_key: str, base_url: str) -> KeyCheck:
    """List the models of an Ollama server; Ollama takes no key."""
    from ..llm.ollama_discovery import validate_ollama_connection

    reachable, message, models = validate_ollama_connection(
        base_url or OLLAMA_DEFAULT_BASE_URL
    )
    if not reachable:
        return KeyCheck("inconclusive", message)
    return KeyCheck("valid", message, tuple(models))


# =============================================================================
# Provider table
# =============================================================================

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True)
class KeySpec:
    """How one provider (or Tavily) is configured and checked.

    ``check`` names a function in this module, looked up when the check runs,
    so tests can patch it. A check with a ``base_url_field`` takes
    ``(key, base_url)``, the others ``(key)``.
    """

    label: str
    key_field: str | None
    key_env: str | None
    check: str | None
    base_url_field: str | None = None
    base_url_env: str | None = None
    default_base_url: str | None = None
    needs_base_url: bool = False
    lists_models: bool = False

    @property
    def needs_key(self) -> bool:
        return self.key_field is not None

    @property
    def has_check(self) -> bool:
        return self.check is not None


def _key(label: str, field: str, env: str, check: str) -> KeySpec:
    return KeySpec(label=label, key_field=field, key_env=env, check=check)


PROVIDERS: dict[str, KeySpec] = {
    "anthropic": _key(
        "Anthropic", "anthropic_api_key", "ANTHROPIC_API_KEY", "check_anthropic_key"
    ),
    "openai": _key("OpenAI", "openai_api_key", "OPENAI_API_KEY", "check_openai_key"),
    "google-genai": _key(
        "Google", "google_api_key", "GOOGLE_API_KEY", "check_google_key"
    ),
    "minimax": KeySpec(
        label="MiniMax",
        key_field="minimax_api_key",
        key_env="MINIMAX_API_KEY",
        check="check_minimax_key",
        base_url_field="minimax_base_url",
        base_url_env="MINIMAX_BASE_URL",
        default_base_url="https://api.minimaxi.com/anthropic",
    ),
    "nvidia": _key("NVIDIA", "nvidia_api_key", "NVIDIA_API_KEY", "check_nvidia_key"),
    "siliconflow": _key(
        "SiliconFlow",
        "siliconflow_api_key",
        "SILICONFLOW_API_KEY",
        "check_siliconflow_key",
    ),
    "openrouter": _key(
        "OpenRouter", "openrouter_api_key", "OPENROUTER_API_KEY", "check_openrouter_key"
    ),
    "atlascloud": _key(
        "Atlas Cloud",
        "atlascloud_api_key",
        "ATLASCLOUD_API_KEY",
        "check_atlascloud_key",
    ),
    "requesty": _key(
        "Requesty", "requesty_api_key", "REQUESTY_API_KEY", "check_requesty_key"
    ),
    "opper": _key("Opper", "opper_api_key", "OPPER_API_KEY", "check_opper_key"),
    "novita": _key("Novita", "novita_api_key", "NOVITA_API_KEY", "check_novita_key"),
    "deepseek": _key(
        "DeepSeek", "deepseek_api_key", "DEEPSEEK_API_KEY", "check_deepseek_key"
    ),
    "zhipu": _key("ZhipuAI", "zhipu_api_key", "ZHIPU_API_KEY", "check_zhipu_key"),
    "zhipu-code": _key(
        "ZhipuAI CodePlan", "zhipu_api_key", "ZHIPU_API_KEY", "check_zhipu_key"
    ),
    "volcengine": _key(
        "Volcengine", "volcengine_api_key", "VOLCENGINE_API_KEY", "check_volcengine_key"
    ),
    "volcengine-code": _key(
        "Volcengine Coding Plan",
        "volcengine_api_key",
        "VOLCENGINE_API_KEY",
        "check_volcengine_key",
    ),
    "dashscope": _key(
        "DashScope", "dashscope_api_key", "DASHSCOPE_API_KEY", "check_dashscope_key"
    ),
    "dashscope-code": _key(
        "DashScope Coding Plan",
        "dashscope_api_key",
        "DASHSCOPE_API_KEY",
        "check_dashscope_code_key",
    ),
    "moonshot": _key(
        "Moonshot", "moonshot_api_key", "MOONSHOT_API_KEY", "check_moonshot_key"
    ),
    "kimi-coding": _key(
        "Kimi Coding Plan", "kimi_api_key", "KIMI_API_KEY", "check_kimi_key"
    ),
    "xiaomi": _key("Xiaomi MiMo", "mimo_api_key", "MIMO_API_KEY", "check_xiaomi_key"),
    "xiaomi-token-plan": KeySpec(
        label="Xiaomi MiMo Token Plan",
        key_field="mimo_token_plan_api_key",
        key_env="MIMO_TOKEN_PLAN_API_KEY",
        check="check_xiaomi_key",
        base_url_field="mimo_token_plan_base_url",
        base_url_env="MIMO_TOKEN_PLAN_BASE_URL",
        default_base_url="https://token-plan-cn.xiaomimimo.com/anthropic",
    ),
    "custom-openai": KeySpec(
        label="OpenAI-compatible",
        key_field="custom_openai_api_key",
        key_env="CUSTOM_OPENAI_API_KEY",
        check="check_custom_openai",
        base_url_field="custom_openai_base_url",
        base_url_env="CUSTOM_OPENAI_BASE_URL",
        needs_base_url=True,
        lists_models=True,
    ),
    "custom-anthropic": KeySpec(
        label="Custom Anthropic",
        key_field="custom_anthropic_api_key",
        key_env="CUSTOM_ANTHROPIC_API_KEY",
        check="check_custom_anthropic",
        base_url_field="custom_anthropic_base_url",
        base_url_env="CUSTOM_ANTHROPIC_BASE_URL",
        needs_base_url=True,
        lists_models=True,
    ),
    "ollama": KeySpec(
        label="Ollama",
        key_field=None,
        key_env=None,
        check="check_ollama",
        base_url_field="ollama_base_url",
        base_url_env="OLLAMA_BASE_URL",
        default_base_url=OLLAMA_DEFAULT_BASE_URL,
        lists_models=True,
    ),
}

TAVILY = _key("Tavily", "tavily_api_key", "TAVILY_API_KEY", "check_tavily_key")

# Everything ``test-key`` accepts: the providers plus Tavily.
KEY_SPECS: dict[str, KeySpec] = {**PROVIDERS, "tavily": TAVILY}


def provider_spec(provider: str) -> KeySpec:
    """The table entry for ``provider``; ``KeyError`` for an unknown one."""
    return PROVIDERS[provider]


def configured_base_url(spec: KeySpec, config: Any) -> str:
    """The base URL a check uses: the config's, else the environment's, else
    the provider's default (blank values count as unset)."""
    if spec.base_url_field is None:
        return ""
    return (
        str(getattr(config, spec.base_url_field, "") or "").strip()
        or os.environ.get(spec.base_url_env or "", "").strip()
        or (spec.default_base_url or "")
    )


def run_check(spec: KeySpec, key: str, base_url: str = "") -> KeyCheck:
    """Run ``spec``'s check on ``key`` (and ``base_url`` where it takes one)."""
    if spec.check is None:
        return KeyCheck("unavailable", f"No check is available for {spec.label}")
    check = globals()[spec.check]
    if spec.base_url_field is not None:
        return check(key, base_url=base_url or spec.default_base_url or "")
    return check(key)


def check_provider_key(name: str, key: str, base_url: str = "") -> KeyCheck:
    """Run the check for a provider or ``"tavily"``; ``KeyError`` if unknown."""
    return run_check(KEY_SPECS[name], key, base_url)
