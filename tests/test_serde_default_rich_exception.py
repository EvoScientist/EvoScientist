"""Regression tests for the helpers ``ErrorNormalizationMiddleware``
uses to build the SSE error envelope.

- ``_redact_api_keys`` + ``_build_env_key_redaction_re`` — scrubs
  deployed credentials that the SDK might echo back.
- ``_extract_status_code`` / ``_extract_provider_code`` /
  ``_extract_error_type`` — read SDK-specific fields off the raised
  exception.

Middleware wire behavior + ``_provider_from_model`` live in
``test_error_normalization_middleware.py``. One end-to-end orjson test
at the bottom guards that a ``ProviderStreamError`` survives
langgraph_api's UNPATCHED ``serde.default`` under
``OPT_SERIALIZE_DATACLASS`` — the whole reason the wrapper exists.
"""

from __future__ import annotations

import os

import langgraph_api.serde as _serde_mod

from EvoScientist.llm.errors import (
    _API_KEY_ENV_SUFFIXES,
    _build_env_key_redaction_re,
    _extract_error_type,
    _extract_provider_code,
    _extract_status_code,
    _redact_api_keys,
)

# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_env_deployed_key_redacted_in_message(monkeypatch):
    """A credential exported via env var is scrubbed by
    ``_redact_api_keys``. The redaction table is rebuilt per call —
    ``monkeypatch.setenv`` alone is enough, no attribute reassignment.
    """
    key = "sk-proj-aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890"
    monkeypatch.setenv("OPENAI_API_KEY", key)

    msg = (
        f"Invalid API key: {key}. Get a new one at https://platform.openai.com/api-keys"
    )
    redacted = _redact_api_keys(msg)
    assert key not in redacted
    assert "<redacted>" in redacted
    assert "Invalid API key" in redacted
    assert "platform.openai.com" in redacted


def test_multiple_env_keys_redacted_independently(monkeypatch):
    """Each ``*_API_KEY`` / ``*_TOKEN`` / ``*_SECRET`` env var
    contributes its own prefix to the alternation.
    """
    k1 = "sk-or-aBcDeFg012345678901234"
    k2 = "AIzaABCDEFGHIJ0123456789"
    k3 = "ghp_p4t70k3n0123456789abcdef"
    monkeypatch.setenv("OPENROUTER_API_KEY", k1)
    monkeypatch.setenv("GOOGLE_API_KEY", k2)
    monkeypatch.setenv("GITHUB_TOKEN", k3)

    msg = _redact_api_keys(f"Failures: {k1}, {k2}, {k3}")
    assert k1 not in msg
    assert k2 not in msg
    assert k3 not in msg
    assert msg.count("<redacted>") == 3


def test_base64_suffix_secret_fully_redacted(monkeypatch):
    """A base64-style secret (``/`` ``+`` ``=``) must redact end-to-end,
    not leak its tail past the first padding char.
    """
    key = "AbCdEfGh/secret+tail=="
    monkeypatch.setenv("SOME_SECRET", key)

    msg = _redact_api_keys(f"auth failed with token={key} on retry")
    assert "secret" not in msg
    assert "tail" not in msg
    assert "<redacted>" in msg
    assert "auth failed" in msg
    assert "on retry" in msg


def test_unknown_shape_not_redacted_without_env(monkeypatch):
    """Env-only redaction: a key-shaped string not deployed via env is
    left alone. Tradeoff — we only scrub what we know is a secret.
    """
    for k in list(os.environ):
        if k.endswith(_API_KEY_ENV_SUFFIXES):
            monkeypatch.delenv(k, raising=False)

    msg = _redact_api_keys("Unknown key seen: sk-or-aBcDeFg012345678901234")
    assert "sk-or-aBcDeFg012345678901234" in msg
    assert "<redacted>" not in msg


def test_env_key_loaded_after_first_call_is_redacted(monkeypatch):
    """The pattern rebuilds every call so keys loaded after
    ``patches.py`` imports (typical ``load_dotenv`` sequence) are
    still scrubbed on the next call.
    """
    for k in list(os.environ):
        if k.endswith(_API_KEY_ENV_SUFFIXES):
            monkeypatch.delenv(k, raising=False)

    key = "sk-proj-loaded_after_import_1234567890abcdef"
    # Pass 1: env empty — key leaks.
    assert key in _redact_api_keys(f"leak: {key}")

    # Pass 2: after simulated load_dotenv.
    monkeypatch.setenv("OPENAI_API_KEY", key)
    redacted = _redact_api_keys(f"leak: {key}")
    assert key not in redacted
    assert "<redacted>" in redacted


def test_redaction_regex_holds_only_prefix(monkeypatch):
    """Defense-in-depth: the compiled regex must not embed the full key.
    A process-memory leak (traceback locals, debugger) exposes at most
    the first 8 chars — not the secret.
    """
    key = "sk-proj-aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890_secret_suffix"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    pattern = _build_env_key_redaction_re()
    assert pattern is not None
    assert key not in pattern.pattern
    assert "aBcDeFgHiJkLmNoPqRs" not in pattern.pattern
    # Sanity: still matches the full key at runtime via prefix + suffix
    # greedy.
    m = pattern.search(f"err: {key}")
    assert m is not None
    assert m.group(0) == key


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------


def _fake_exc(**attrs):
    return type("APIError", (Exception,), attrs)("boom")


def test_status_code_read_from_direct_attribute():
    """openai / anthropic ``APIStatusError`` carries integer
    ``.status_code`` — the primary path.
    """
    assert _extract_status_code(_fake_exc(status_code=429)) == 429


def test_status_code_read_via_response_attribute():
    """Wrappers that don't promote status to top level expose it via
    ``.response.status_code`` (httpx pattern).
    """

    class FakeResponse:
        status_code = 504

    assert _extract_status_code(_fake_exc(response=FakeResponse())) == 504


def test_status_code_read_via_integer_code_attribute():
    """``google.genai.errors.APIError`` stores HTTP status as integer
    ``.code`` — type-disambiguated from openai/anthropic's string
    ``.code`` (provider error code).
    """
    assert _extract_status_code(_fake_exc(code=400)) == 400


def test_provider_code_read_from_string_code_attribute():
    """Provider error code (``insufficient_quota`` etc.) is a string
    ``.code`` — higher signal than the integer HTTP status alone.
    """
    assert (
        _extract_provider_code(_fake_exc(code="insufficient_quota"))
        == "insufficient_quota"
    )


def test_provider_code_ignores_integer_code():
    """An integer ``.code`` is HTTP status (see above); must not bleed
    into the provider-code path.
    """
    assert _extract_provider_code(_fake_exc(code=429)) is None


def test_error_type_read_from_type_attribute():
    """openai exposes a ``.type`` label (``rate_limit_error``)."""
    assert _extract_error_type(_fake_exc(type="rate_limit_error")) == "rate_limit_error"


def test_extractors_return_none_when_attributes_absent():
    """A bare exception with no SDK-shape attributes — every extractor
    returns None so the envelope drops the optional fields.
    """
    exc = _fake_exc()
    assert _extract_status_code(exc) is None
    assert _extract_provider_code(exc) is None
    assert _extract_error_type(exc) is None


# ---------------------------------------------------------------------------
# End-to-end: ProviderStreamError survives orjson under
# OPT_SERIALIZE_DATACLASS via upstream's UNPATCHED serde.default.
# ---------------------------------------------------------------------------


def test_provider_stream_error_survives_orjson_dataclass_option():
    """Guard: ``ProviderStreamError`` — a plain Exception subclass with
    a ``model_dump()`` hook — must emerge as the envelope on the wire
    even under ``OPT_SERIALIZE_DATACLASS``, using ONLY upstream's
    stock ``serde.default``. Proof that we no longer need to patch
    the serde module.
    """
    import orjson

    from EvoScientist.llm.errors import ProviderStreamError

    err = ProviderStreamError(
        provider="openrouter",
        class_qualname="openrouter.errors.foo.UnauthorizedResponseError",
        message="User not found.",
        status_code=401,
    )

    wire = orjson.dumps(
        err,
        default=_serde_mod.default,  # upstream, unpatched
        option=orjson.OPT_SERIALIZE_DATACLASS,
    )
    decoded = orjson.loads(wire)
    assert decoded == {
        "error": "UnauthorizedResponseError",
        "class": "openrouter.errors.foo.UnauthorizedResponseError",
        "message": "User not found.",
        "provider": "openrouter",
        "status_code": 401,
    }
