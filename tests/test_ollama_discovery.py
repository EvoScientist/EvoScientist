"""Tests for EvoScientist.llm.ollama_discovery.

Covers both the sync ``validate_ollama_connection`` (used by the onboarding
wizard) and the async ``discover_ollama_models`` (used by the /model
picker). The async variant must never raise — the picker's UX depends on
a silent empty-list fallback when the daemon is down or misbehaving.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from EvoScientist.llm.ollama_discovery import (
    discover_ollama_models,
    validate_ollama_connection,
)
from tests.conftest import run_async as _run


class TestValidateOllamaConnection:
    """Sync probe — guards onboard's existing contract."""

    def test_empty_base_url_returns_skipped(self):
        ok, msg, names = validate_ollama_connection("")
        assert ok is True
        assert "Skipped" in msg
        assert names == []

    def test_200_with_models_returns_names(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "models": [{"name": "llama3.3:latest"}, {"name": "qwen3:8b"}]
        }
        with patch("httpx.get", return_value=resp):
            ok, msg, names = validate_ollama_connection("http://localhost:11434")
        assert ok is True
        assert "Connected" in msg
        assert names == ["llama3.3:latest", "qwen3:8b"]

    def test_200_with_no_models_still_ok(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"models": []}
        with patch("httpx.get", return_value=resp):
            ok, msg, names = validate_ollama_connection("http://localhost:11434")
        assert ok is True
        assert "no models pulled" in msg
        assert names == []

    def test_non_200_returns_error(self):
        resp = MagicMock()
        resp.status_code = 500
        with patch("httpx.get", return_value=resp):
            ok, msg, names = validate_ollama_connection("http://localhost:11434")
        assert ok is False
        assert "500" in msg
        assert names == []

    def test_connect_error_returns_error(self):
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            ok, msg, names = validate_ollama_connection("http://localhost:11434")
        assert ok is False
        assert "Cannot reach Ollama" in msg
        assert names == []


class TestDiscoverOllamaModels:
    """Async probe — contract: never raise, return list[str]."""

    def test_empty_base_url_returns_empty_without_http(self):
        # No HTTP call should be made for an empty base_url — verified by
        # the fact that no mock is set up and the test completes.
        names = _run(discover_ollama_models(""))
        assert names == []

    def test_none_base_url_returns_empty(self):
        names = _run(discover_ollama_models(None))
        assert names == []

    def test_200_returns_names(self):
        async def fake_get(self, url):
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(
                return_value={
                    "models": [{"name": "llama3.3:latest"}, {"name": "qwen3:8b"}]
                }
            )
            return resp

        with patch.object(httpx.AsyncClient, "get", fake_get):
            names = _run(discover_ollama_models("http://localhost:11434"))
        assert names == ["llama3.3:latest", "qwen3:8b"]

    def test_strips_entries_without_name(self):
        async def fake_get(self, url):
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(
                return_value={
                    "models": [
                        {"name": "llama3.3"},
                        {"name": ""},  # dropped
                        {},  # dropped
                    ]
                }
            )
            return resp

        with patch.object(httpx.AsyncClient, "get", fake_get):
            names = _run(discover_ollama_models("http://localhost:11434"))
        assert names == ["llama3.3"]

    def test_timeout_returns_empty(self):
        async def fake_get(self, url):
            raise httpx.TimeoutException("timed out")

        with patch.object(httpx.AsyncClient, "get", fake_get):
            names = _run(discover_ollama_models("http://localhost:11434"))
        assert names == []

    def test_connect_error_returns_empty(self):
        async def fake_get(self, url):
            raise httpx.ConnectError("refused")

        with patch.object(httpx.AsyncClient, "get", fake_get):
            names = _run(discover_ollama_models("http://localhost:11434"))
        assert names == []

    def test_non_200_returns_empty(self):
        async def fake_get(self, url):
            resp = MagicMock()
            resp.status_code = 500
            return resp

        with patch.object(httpx.AsyncClient, "get", fake_get):
            names = _run(discover_ollama_models("http://localhost:11434"))
        assert names == []

    def test_malformed_json_returns_empty(self):
        async def fake_get(self, url):
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(side_effect=ValueError("bad json"))
            return resp

        with patch.object(httpx.AsyncClient, "get", fake_get):
            names = _run(discover_ollama_models("http://localhost:11434"))
        assert names == []

    def test_missing_models_key_returns_empty(self):
        async def fake_get(self, url):
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(return_value={"unexpected": "shape"})
            return resp

        with patch.object(httpx.AsyncClient, "get", fake_get):
            names = _run(discover_ollama_models("http://localhost:11434"))
        assert names == []

    def test_trailing_slash_stripped_from_url(self):
        called = {}

        async def fake_get(self, url):
            called["url"] = url
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(return_value={"models": []})
            return resp

        with patch.object(httpx.AsyncClient, "get", fake_get):
            _run(discover_ollama_models("http://localhost:11434/"))
        assert called["url"] == "http://localhost:11434/api/tags"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
