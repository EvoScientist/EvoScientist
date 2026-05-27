"""Input validators for the onboarding wizard.

- IntegerValidator / ChoiceValidator: prompt_toolkit Validators
- validate_*_key: per-provider API key validators (live HTTP probes)
"""

from __future__ import annotations

from prompt_toolkit.validation import ValidationError, Validator


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
        except ValueError as e:
            raise ValidationError(message="Must be a valid integer") from e


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


def _classify_validation_error(error: BaseException) -> tuple[bool, str] | None:
    """Classify a validator exception as auth failure, transient, or unknown.

    Returns ``(False, msg)`` for the first two, ``None`` for unknown so the
    caller can fall back to ``f"Error: {e}"``.
    """
    s = str(error).lower()
    if any(h in s for h in _AUTH_FAILURE_HINTS):
        return False, "Invalid API key"
    if any(h in s for h in _TRANSIENT_HINTS):
        return False, "Validation inconclusive — transient error, try again later"
    return None


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
        classified = _classify_validation_error(e)
        if classified is not None:
            return classified
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
        classified = _classify_validation_error(e)
        if classified is not None:
            return classified
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

    # NOTE: ``ChatNVIDIA(api_key=...)`` does NOT send a network request — it
    # only stores the key in a client object. We must actually invoke the
    # API (e.g. ``get_available_models()``) to verify the key is good.
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        client = ChatNVIDIA(api_key=api_key, model="meta/llama-3.1-8b-instruct")
        # Force a real authenticated request via model discovery.
        client.get_available_models()
        return True, "Valid"

    except Exception as e:
        classified = _classify_validation_error(e)
        if classified is not None:
            return classified
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
        classified = _classify_validation_error(e)
        if classified is not None:
            return classified
        # Google-specific 400 phrasing not in the shared hint list.
        error_str = str(e).lower()
        if "api_key_invalid" in error_str or "api key invalid" in error_str:
            return False, "Invalid API key"
        return False, f"Error: {e}"


def validate_minimax_key(
    api_key: str,
    base_url: str = "https://api.minimaxi.com/anthropic",
) -> tuple[bool, str]:
    """Validate a MiniMax API key without consuming tokens.

    Sends a messages.create() with an empty model string.  MiniMax checks
    auth *before* validating request params, so a valid key returns 400
    (bad model) while an invalid key returns 401.

    Args:
        api_key: The MiniMax API key to validate.
        base_url: Anthropic-compatible endpoint (global or mainland China).

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url,
        )
        client.messages.create(
            model="",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        # Unexpected success — treat as valid
        return True, "Valid"
    except anthropic.AuthenticationError:
        return False, "Invalid API key"
    except anthropic.APIStatusError:
        # Any non-auth HTTP error (400 bad model, 500 insufficient balance,
        # etc.) means the key itself was accepted → treat as valid.
        return True, "Valid"
    except Exception as e:
        classified = _classify_validation_error(e)
        if classified is not None:
            return classified
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
        classified = _classify_validation_error(e)
        if classified is not None:
            return classified
        return False, f"Error: {e}"


def validate_openrouter_key(api_key: str) -> tuple[bool, str]:
    """Validate an OpenRouter API key via the authenticated /auth/key endpoint.

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        import httpx

        resp = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return True, "Valid"
        # Only 401/403 mean the key is actually rejected. 429 (rate-limit)
        # and 5xx (OpenRouter incident) leave the key validity unknown —
        # surface the real status so the user doesn't go re-roll a good key
        # during an outage.
        if resp.status_code in (401, 403):
            return False, "Invalid API key"
        return False, f"Validation inconclusive (HTTP {resp.status_code})"
    except Exception as e:
        classified = _classify_validation_error(e)
        if classified is not None:
            return classified
        return False, f"Error: {e}"


def validate_deepseek_key(api_key: str) -> tuple[bool, str]:
    """Validate a DeepSeek API key by making a test request.

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        import openai

        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        client.models.list()
        return True, "Valid"
    except Exception as e:
        classified = _classify_validation_error(e)
        if classified is not None:
            return classified
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
        classified = _classify_validation_error(e)
        if classified is not None:
            return classified
        return False, f"Error: {e}"


def validate_volcengine_key(api_key: str) -> tuple[bool, str]:
    """Validate a Volcengine API key by making a test request.

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        import openai

        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        client.models.list()
        return True, "Valid"
    except Exception as e:
        classified = _classify_validation_error(e)
        if classified is not None:
            return classified
        return False, f"Error: {e}"


def validate_dashscope_key(api_key: str) -> tuple[bool, str]:
    """Validate a DashScope API key by making a test request.

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        import openai

        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        client.models.list()
        return True, "Valid"
    except Exception as e:
        classified = _classify_validation_error(e)
        if classified is not None:
            return classified
        return False, f"Error: {e}"


def validate_dashscope_code_key(api_key: str) -> tuple[bool, str]:
    """Validate a DashScope Coding Plan API key (sk-sp-* subscription keys).

    The coding endpoint at coding.dashscope.aliyuncs.com does not expose
    /models (returns 404), so validation issues a minimal chat completion
    instead of the usual models.list() probe.

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        import openai

        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://coding.dashscope.aliyuncs.com/v1",
        )
        client.chat.completions.create(
            model="qwen3-coder-plus",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return True, "Valid"
    except Exception as e:
        classified = _classify_validation_error(e)
        if classified is not None:
            return classified
        return False, f"Error: {e}"


def validate_moonshot_key(api_key: str) -> tuple[bool, str]:
    """Validate a Moonshot API key by making a test request.

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        import openai

        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
        )
        client.models.list()
        return True, "Valid"
    except Exception as e:
        classified = _classify_validation_error(e)
        if classified is not None:
            return classified
        return False, f"Error: {e}"


def validate_kimi_key(api_key: str) -> tuple[bool, str]:
    """Validate a Kimi Coding Plan API key by making a test request.

    Uses the Anthropic-compatible endpoint at api.kimi.com/coding/.

    Returns:
        Tuple of (is_valid, message).
    """
    if not api_key:
        return True, "Skipped (no key provided)"

    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=api_key,
            base_url="https://api.kimi.com/coding/",
            default_headers={"User-Agent": "claude-code/0.1.0"},
        )
        client.models.list()
        return True, "Valid"
    except Exception as e:
        classified = _classify_validation_error(e)
        if classified is not None:
            return classified
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
        classified = _classify_validation_error(e)
        if classified is not None:
            return classified
        return False, f"Error: {e}"


# =============================================================================
# Display Helpers
# =============================================================================
