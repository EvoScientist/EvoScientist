"""Tests for EvoScientist.llm.model_cache.

Covers sync/async fetch, caching behaviour, TTL expiry, cache persistence,
and the ``is_supported`` helper.  HTTP calls are always mocked so no real
network traffic is generated.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from tests.conftest import run_async as _run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_models_response(*ids: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(
        return_value={"data": [{"id": mid} for mid in ids], "object": "list"}
    )
    return resp


# ---------------------------------------------------------------------------
# is_supported
# ---------------------------------------------------------------------------


class TestIsSupported:
    def test_known_providers_are_supported(self):
        from EvoScientist.llm.model_cache import is_supported

        for provider in ("openai", "deepseek", "moonshot", "siliconflow", "openrouter"):
            assert is_supported(provider), f"{provider!r} should be supported"

    def test_unsupported_providers(self):
        from EvoScientist.llm.model_cache import is_supported

        for provider in ("anthropic", "ollama", "google-genai", "nvidia", "minimax"):
            assert not is_supported(provider), f"{provider!r} should not be supported"


# ---------------------------------------------------------------------------
# get_cached_models
# ---------------------------------------------------------------------------


class TestGetCachedModels:
    def test_no_cache_file_returns_none(self, tmp_path):
        from EvoScientist.llm import model_cache

        with patch.object(model_cache, "_get_cache_path", return_value=tmp_path / "model_cache.json"):
            assert model_cache.get_cached_models("deepseek") is None

    def test_fresh_cache_returns_models(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"
        data = {"deepseek": {"models": ["deepseek-chat", "deepseek-reasoner"], "fetched_at": time.time()}}
        cache_path.write_text(json.dumps(data))

        with patch.object(model_cache, "_get_cache_path", return_value=cache_path):
            result = model_cache.get_cached_models("deepseek")

        assert result == ["deepseek-chat", "deepseek-reasoner"]

    def test_stale_cache_returns_none(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"
        stale_time = time.time() - model_cache.CACHE_TTL - 1
        data = {"deepseek": {"models": ["deepseek-chat"], "fetched_at": stale_time}}
        cache_path.write_text(json.dumps(data))

        with patch.object(model_cache, "_get_cache_path", return_value=cache_path):
            result = model_cache.get_cached_models("deepseek")

        assert result is None

    def test_missing_provider_returns_none(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"
        data = {"openai": {"models": ["gpt-4o"], "fetched_at": time.time()}}
        cache_path.write_text(json.dumps(data))

        with patch.object(model_cache, "_get_cache_path", return_value=cache_path):
            assert model_cache.get_cached_models("deepseek") is None

    def test_corrupt_cache_file_returns_none(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"
        cache_path.write_text("not json {{{")

        with patch.object(model_cache, "_get_cache_path", return_value=cache_path):
            assert model_cache.get_cached_models("openai") is None


# ---------------------------------------------------------------------------
# fetch_models (sync)
# ---------------------------------------------------------------------------


class TestFetchModels:
    def test_unsupported_provider_returns_none(self):
        from EvoScientist.llm.model_cache import fetch_models

        assert fetch_models("anthropic") is None
        assert fetch_models("ollama") is None

    def test_returns_cached_when_fresh(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"
        data = {"deepseek": {"models": ["deepseek-chat"], "fetched_at": time.time()}}
        cache_path.write_text(json.dumps(data))

        with patch.object(model_cache, "_get_cache_path", return_value=cache_path):
            result = model_cache.fetch_models("deepseek", api_key="sk-test")

        assert result == ["deepseek-chat"]

    def test_fetches_from_api_when_no_cache(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"

        with (
            patch.object(model_cache, "_get_cache_path", return_value=cache_path),
            patch(
                "httpx.get",
                return_value=_make_models_response("deepseek-chat", "deepseek-reasoner"),
            ),
        ):
            result = model_cache.fetch_models(
                "deepseek", api_key="sk-test", base_url="https://api.deepseek.com/v1"
            )

        assert result == ["deepseek-chat", "deepseek-reasoner"]

    def test_result_is_written_to_cache(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"

        with (
            patch.object(model_cache, "_get_cache_path", return_value=cache_path),
            patch(
                "httpx.get",
                return_value=_make_models_response("deepseek-chat"),
            ),
        ):
            model_cache.fetch_models(
                "deepseek", api_key="sk-test", base_url="https://api.deepseek.com/v1"
            )

        written = json.loads(cache_path.read_text())
        assert "deepseek" in written
        assert written["deepseek"]["models"] == ["deepseek-chat"]

    def test_force_bypasses_fresh_cache(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"
        data = {"deepseek": {"models": ["old-model"], "fetched_at": time.time()}}
        cache_path.write_text(json.dumps(data))

        with (
            patch.object(model_cache, "_get_cache_path", return_value=cache_path),
            patch(
                "httpx.get",
                return_value=_make_models_response("new-model"),
            ),
        ):
            result = model_cache.fetch_models(
                "deepseek",
                api_key="sk-test",
                base_url="https://api.deepseek.com/v1",
                force=True,
            )

        assert result == ["new-model"]

    def test_non_200_response_returns_none(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"
        resp = MagicMock()
        resp.status_code = 401

        with (
            patch.object(model_cache, "_get_cache_path", return_value=cache_path),
            patch("httpx.get", return_value=resp),
        ):
            result = model_cache.fetch_models(
                "deepseek", api_key="sk-bad", base_url="https://api.deepseek.com/v1"
            )

        assert result is None

    def test_network_error_returns_none(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"

        with (
            patch.object(model_cache, "_get_cache_path", return_value=cache_path),
            patch("httpx.get", side_effect=httpx.ConnectError("refused")),
        ):
            result = model_cache.fetch_models(
                "deepseek", api_key="sk-test", base_url="https://api.deepseek.com/v1"
            )

        assert result is None

    def test_empty_data_list_returns_none(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"
        resp = MagicMock()
        resp.status_code = 200
        resp.json = MagicMock(return_value={"data": [], "object": "list"})

        with (
            patch.object(model_cache, "_get_cache_path", return_value=cache_path),
            patch("httpx.get", return_value=resp),
        ):
            result = model_cache.fetch_models(
                "openai", api_key="sk-test", base_url="https://api.openai.com/v1"
            )

        assert result is None

    def test_missing_api_key_returns_none(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"

        with (
            patch.object(model_cache, "_get_cache_path", return_value=cache_path),
            patch.dict("os.environ", {}, clear=True),
        ):
            # deepseek has a default base URL but no key is available
            result = model_cache.fetch_models("deepseek")

        assert result is None


# ---------------------------------------------------------------------------
# fetch_models_async
# ---------------------------------------------------------------------------


class TestFetchModelsAsync:
    def test_unsupported_provider_returns_none(self):
        from EvoScientist.llm.model_cache import fetch_models_async

        assert _run(fetch_models_async("anthropic")) is None

    def test_returns_cached_when_fresh(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"
        data = {"openai": {"models": ["gpt-4o"], "fetched_at": time.time()}}
        cache_path.write_text(json.dumps(data))

        with patch.object(model_cache, "_get_cache_path", return_value=cache_path):
            result = _run(model_cache.fetch_models_async("openai", api_key="sk-test"))

        assert result == ["gpt-4o"]

    def test_fetches_from_api_when_no_cache(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"

        async def fake_get(self, url, **kwargs):
            return _make_models_response("gpt-4o", "gpt-4-turbo")

        with (
            patch.object(model_cache, "_get_cache_path", return_value=cache_path),
            patch.object(httpx.AsyncClient, "get", fake_get),
        ):
            result = _run(
                model_cache.fetch_models_async(
                    "openai", api_key="sk-test", base_url="https://api.openai.com/v1"
                )
            )

        assert result == ["gpt-4o", "gpt-4-turbo"]

    def test_force_bypasses_cache(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"
        data = {"openai": {"models": ["old-model"], "fetched_at": time.time()}}
        cache_path.write_text(json.dumps(data))

        async def fake_get(self, url, **kwargs):
            return _make_models_response("new-model")

        with (
            patch.object(model_cache, "_get_cache_path", return_value=cache_path),
            patch.object(httpx.AsyncClient, "get", fake_get),
        ):
            result = _run(
                model_cache.fetch_models_async(
                    "openai",
                    api_key="sk-test",
                    base_url="https://api.openai.com/v1",
                    force=True,
                )
            )

        assert result == ["new-model"]

    def test_timeout_returns_none(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"

        async def fake_get(self, url, **kwargs):
            raise httpx.TimeoutException("timed out")

        with (
            patch.object(model_cache, "_get_cache_path", return_value=cache_path),
            patch.object(httpx.AsyncClient, "get", fake_get),
        ):
            result = _run(
                model_cache.fetch_models_async(
                    "deepseek", api_key="sk-test", base_url="https://api.deepseek.com/v1"
                )
            )

        assert result is None

    def test_connect_error_returns_none(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"

        async def fake_get(self, url, **kwargs):
            raise httpx.ConnectError("refused")

        with (
            patch.object(model_cache, "_get_cache_path", return_value=cache_path),
            patch.object(httpx.AsyncClient, "get", fake_get),
        ):
            result = _run(
                model_cache.fetch_models_async(
                    "deepseek", api_key="sk-test", base_url="https://api.deepseek.com/v1"
                )
            )

        assert result is None

    def test_non_200_returns_none(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"

        async def fake_get(self, url, **kwargs):
            resp = MagicMock()
            resp.status_code = 403
            return resp

        with (
            patch.object(model_cache, "_get_cache_path", return_value=cache_path),
            patch.object(httpx.AsyncClient, "get", fake_get),
        ):
            result = _run(
                model_cache.fetch_models_async(
                    "openrouter",
                    api_key="sk-test",
                    base_url="https://openrouter.ai/api/v1",
                )
            )

        assert result is None

    def test_result_written_to_cache(self, tmp_path):
        from EvoScientist.llm import model_cache

        cache_path = tmp_path / "model_cache.json"

        async def fake_get(self, url, **kwargs):
            return _make_models_response("moonshot-v1-8k")

        with (
            patch.object(model_cache, "_get_cache_path", return_value=cache_path),
            patch.object(httpx.AsyncClient, "get", fake_get),
        ):
            _run(
                model_cache.fetch_models_async(
                    "moonshot",
                    api_key="sk-test",
                    base_url="https://api.moonshot.cn/v1",
                )
            )

        written = json.loads(cache_path.read_text())
        assert "moonshot" in written
        assert written["moonshot"]["models"] == ["moonshot-v1-8k"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
