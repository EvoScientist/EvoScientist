"""Tests for EvoScientist.config.provider_keys (key checks and provider table)."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

from EvoScientist.config import provider_keys
from EvoScientist.config.provider_keys import (
    KeyCheck,
    check_custom_anthropic,
    check_custom_openai,
    check_ollama,
    check_provider_key,
)


def _mock_client(real, handler, http=httpx):
    """Wrap an SDK client class so its requests go to ``handler``.

    The Anthropic SDK is built on ``httpx2``; pass it as ``http`` there.
    """
    return lambda **kw: real(
        **kw, http_client=http.Client(transport=http.MockTransport(handler))
    )


class TestResult:
    @pytest.mark.parametrize(
        ("status", "ok"),
        [
            ("valid", True),
            ("invalid", False),
            ("inconclusive", None),
            ("unavailable", None),
            ("empty", None),
        ],
    )
    def test_ok_is_set_only_for_conclusive_results(self, status, ok):
        assert KeyCheck(status, "m").ok is ok

    @pytest.mark.parametrize(
        ("error", "status", "message"),
        [
            ("Error code: 401", "invalid", "Invalid API key"),
            (
                "Connection refused",
                "inconclusive",
                "Validation inconclusive — transient error, try again later",
            ),
            ("something odd", "inconclusive", "Error: something odd"),
        ],
    )
    def test_exceptions_are_classified(self, error, status, message):
        result = provider_keys._classify_validation_error(RuntimeError(error))
        assert (result.status, result.message) == (status, message)

    def test_empty_key_is_empty_not_valid(self):
        result = check_provider_key("openai", "")
        assert result.status == "empty"
        assert result.ok is None

    def test_unknown_provider_raises(self):
        with pytest.raises(KeyError):
            check_provider_key("no-such-provider", "k")


class TestWizardAdapter:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("valid", True),
            ("empty", True),
            ("invalid", False),
            ("inconclusive", False),
            ("unavailable", False),
        ],
    )
    def test_as_pair(self, status, expected):
        from EvoScientist.config.onboard.validators import as_pair

        assert as_pair(KeyCheck(status, "msg")) == (expected, "msg")

    def test_validator_follows_a_patched_check(self):
        from EvoScientist.config.onboard.validators import validate_openai_key

        with patch.object(
            provider_keys, "check_openai_key", return_value=KeyCheck("invalid", "no")
        ):
            assert validate_openai_key("k") == (False, "no")


class TestTimeouts:
    def test_sdk_clients_get_timeout_and_no_retries(self):
        import anthropic

        with patch("anthropic.Anthropic") as client_cls:
            client_cls.return_value.models.list.side_effect = (
                anthropic.AuthenticationError(
                    "bad", response=MagicMock(status_code=401), body=None
                )
            )
            result = provider_keys.check_anthropic_key("k")
        assert result.status == "invalid"
        assert client_cls.call_args.kwargs["timeout"] == 10
        assert client_cls.call_args.kwargs["max_retries"] == 0

    def test_google_timeout_is_in_milliseconds(self):
        with patch("google.genai.Client") as client_cls:
            client_cls.return_value.models.list.return_value = iter(["m"])
            assert provider_keys.check_google_key("k").status == "valid"
        assert client_cls.call_args.kwargs["http_options"] == {"timeout": 10000}

    def test_tavily_search_gets_timeout(self):
        with patch("tavily.TavilyClient") as client_cls:
            assert provider_keys.check_tavily_key("k").status == "valid"
        assert client_cls.return_value.search.call_args.kwargs["timeout"] == 10


class TestNvidia:
    def test_checks_the_key_under_test(self, monkeypatch):
        """The classmethod would build a client from ``NVIDIA_API_KEY``;
        the instance property uses the key that is being checked."""
        monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
        with patch("langchain_nvidia_ai_endpoints.ChatNVIDIA") as client_cls:
            assert provider_keys.check_nvidia_key("k").status == "valid"
        assert client_cls.call_args.kwargs["api_key"] == "k"
        assert client_cls.call_args.kwargs["timeout"] == 10
        client_cls.get_available_models.assert_not_called()
        client_cls.return_value.get_available_models.assert_not_called()


class TestCustomOpenAI:
    def _run(self, status, body, key="sk-x", base_url="https://llm.test/v1"):
        import openai

        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(status, json=body)

        with patch("openai.OpenAI", side_effect=_mock_client(openai.OpenAI, handler)):
            return check_custom_openai(key, base_url), seen

    def test_lists_models(self):
        result, seen = self._run(200, {"object": "list", "data": [{"id": "m1"}]})
        assert result.status == "valid"
        assert result.models == ("m1",)
        assert "listed 1 model(s)" in result.message
        assert str(seen[0].url) == "https://llm.test/v1/models"
        assert seen[0].headers["authorization"] == "Bearer sk-x"

    @pytest.mark.parametrize(
        ("status", "body", "expected"),
        [
            (401, {"error": "no"}, "invalid"),
            (403, {"error": "no"}, "invalid"),
            (404, {"error": "no"}, "unavailable"),
            (200, {"models": ["m1"]}, "unavailable"),
            (500, {"error": "down"}, "inconclusive"),
        ],
    )
    def test_statuses(self, status, body, expected):
        result, _ = self._run(status, body)
        assert result.status == expected
        assert result.models == ()

    def test_never_sends_the_openai_key_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
        _, seen = self._run(200, {"data": []}, key="")
        assert "real-openai-key" not in seen[0].headers["authorization"]

    def test_no_base_url_is_unavailable(self):
        assert check_custom_openai("k", "").status == "unavailable"


class TestCustomAnthropic:
    def test_lists_models_on_v1(self):
        import anthropic
        import httpx2

        seen = []

        def handler(request):
            seen.append(request)
            return httpx2.Response(
                200,
                json={"data": [{"id": "claude-x", "type": "model"}], "has_more": True},
            )

        with patch(
            "anthropic.Anthropic",
            side_effect=_mock_client(anthropic.Anthropic, handler, httpx2),
        ):
            result = check_custom_anthropic("k", "https://proxy.test/")
        assert result.models == ("claude-x",)
        assert str(seen[0].url).startswith("https://proxy.test/v1/models")
        assert len(seen) == 1  # first page only

    def test_auth_rejection_is_invalid(self):
        import anthropic
        import httpx2

        with patch(
            "anthropic.Anthropic",
            side_effect=_mock_client(
                anthropic.Anthropic, lambda r: httpx2.Response(401, json={}), httpx2
            ),
        ):
            assert check_custom_anthropic("k", "https://proxy.test").status == "invalid"


class TestOllama:
    def test_empty_base_url_probes_the_default(self):
        with patch(
            "EvoScientist.llm.ollama_discovery.validate_ollama_connection",
            return_value=(True, "Connected", ["llama3.2"]),
        ) as probe:
            result = check_ollama("", "")
        probe.assert_called_once_with("http://localhost:11434")
        assert (result.status, result.models) == ("valid", ("llama3.2",))

    def test_unreachable_is_inconclusive(self):
        with patch(
            "EvoScientist.llm.ollama_discovery.validate_ollama_connection",
            return_value=(False, "Cannot reach Ollama: refused", []),
        ):
            assert check_ollama("", "http://x").status == "inconclusive"


class TestTable:
    def test_run_check_passes_the_base_url(self):
        with patch.object(
            provider_keys, "check_minimax_key", return_value=KeyCheck("valid", "ok")
        ) as check:
            check_provider_key("minimax", "k", "https://mm.test")
        check.assert_called_once_with("k", base_url="https://mm.test")

    def test_run_check_falls_back_to_the_default_base_url(self):
        with patch.object(
            provider_keys, "check_xiaomi_key", return_value=KeyCheck("valid", "ok")
        ) as check:
            check_provider_key("xiaomi-token-plan", "k")
        assert check.call_args.kwargs["base_url"] == (
            "https://token-plan-cn.xiaomimimo.com/anthropic"
        )

    def test_only_custom_endpoints_need_a_base_url(self):
        assert {
            name
            for name, spec in provider_keys.PROVIDERS.items()
            if spec.needs_base_url
        } == {"custom-openai", "custom-anthropic"}

    def test_tavily_is_checkable_but_not_a_provider(self):
        assert "tavily" not in provider_keys.PROVIDERS
        assert provider_keys.KEY_SPECS["tavily"].key_field == "tavily_api_key"


def test_import_loads_no_ui_library():
    """The backend imports this module; it must not pull in the wizard."""
    code = (
        "import sys; import EvoScientist.config.provider_keys; "
        "bad = [m for m in ('questionary', 'prompt_toolkit', "
        "'EvoScientist.config.onboard') if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
