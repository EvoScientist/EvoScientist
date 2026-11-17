"""Tests for EvoScientist.llm.custom_openai_discovery.

Covers the sync ``list_custom_openai_models`` (onboarding wizard) and the
async ``discover_custom_openai_models`` (model pickers). Both must never
raise — the picker falls back to the static defaults when the endpoint is
down or misbehaving. The key is sent only when one was configured; a keyless
local endpoint is still probed, and ``OPENAI_API_KEY`` is never attached.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from EvoScientist.llm.custom_openai_discovery import (
    discover_custom_openai_models,
    list_custom_openai_models,
)

_LISTING = {"data": [{"id": "fake-agent"}, {"id": "other-model"}]}


def _response(status_code: int = 200, body: object = None, *, json_error: bool = False):
    resp = MagicMock()
    resp.status_code = status_code
    if json_error:
        resp.json = MagicMock(side_effect=ValueError("bad json"))
    else:
        resp.json = MagicMock(return_value=body if body is not None else _LISTING)
    return resp


class TestListCustomOpenaiModels:
    """Sync probe — the wizard calls this while questionary is blocking."""

    def test_empty_base_url_returns_empty_without_http(self):
        with patch("httpx.get") as get:
            assert list_custom_openai_models("") == []
            assert list_custom_openai_models(None) == []
        get.assert_not_called()

    def test_200_returns_ids(self):
        with patch("httpx.get", return_value=_response()) as get:
            names = list_custom_openai_models("https://llm.test/v1", "sk-test")
        assert names == ["fake-agent", "other-model"]
        args, kwargs = get.call_args
        assert args[0] == "https://llm.test/v1/models"
        assert kwargs["headers"] == {"Authorization": "Bearer sk-test"}

    def test_empty_key_sends_no_authorization(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
        with patch("httpx.get", return_value=_response()) as get:
            names = list_custom_openai_models("https://llm.test/v1", "")
        assert names == ["fake-agent", "other-model"]
        assert get.call_args.kwargs["headers"] == {}

    def test_whitespace_key_sends_no_authorization(self):
        with patch("httpx.get", return_value=_response()) as get:
            list_custom_openai_models("https://llm.test/v1", "   ")
        assert get.call_args.kwargs["headers"] == {}

    def test_non_200_returns_empty(self):
        for status in (401, 404, 500):
            with patch("httpx.get", return_value=_response(status)):
                assert list_custom_openai_models("https://llm.test/v1", "sk") == []

    def test_timeout_returns_empty(self):
        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            assert list_custom_openai_models("https://llm.test/v1") == []

    def test_connect_error_returns_empty(self):
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            assert list_custom_openai_models("https://llm.test/v1") == []

    def test_malformed_json_returns_empty(self):
        with patch("httpx.get", return_value=_response(json_error=True)):
            assert list_custom_openai_models("https://llm.test/v1") == []

    def test_missing_data_returns_empty(self):
        with patch("httpx.get", return_value=_response(body={"unexpected": []})):
            assert list_custom_openai_models("https://llm.test/v1") == []

    def test_data_not_a_list_returns_empty(self):
        with patch("httpx.get", return_value=_response(body={"data": "nope"})):
            assert list_custom_openai_models("https://llm.test/v1") == []

    def test_drops_entries_without_id(self):
        body = {"data": [{"id": "fake-agent"}, {"id": ""}, {}, "bare"]}
        with patch("httpx.get", return_value=_response(body=body)):
            assert list_custom_openai_models("https://llm.test/v1") == ["fake-agent"]

    def test_trailing_slash_stripped_from_url(self):
        with patch("httpx.get", return_value=_response(body={"data": []})) as get:
            assert list_custom_openai_models("https://llm.test/v1/") == []
        assert get.call_args.args[0] == "https://llm.test/v1/models"


class TestDiscoverCustomOpenaiModels:
    """Async probe — contract: never raise, return list[str]."""

    async def test_empty_base_url_returns_empty_without_http(self):
        names = await discover_custom_openai_models("")
        assert names == []

    async def test_none_base_url_returns_empty(self):
        names = await discover_custom_openai_models(None)
        assert names == []

    async def test_200_returns_ids_and_sends_key(self):
        seen: dict[str, object] = {}

        async def fake_get(self, url, headers=None):
            seen["url"] = url
            seen["headers"] = headers
            return _response()

        with patch.object(httpx.AsyncClient, "get", fake_get):
            names = await discover_custom_openai_models(
                "https://llm.test/v1", "sk-test"
            )
        assert names == ["fake-agent", "other-model"]
        assert seen["url"] == "https://llm.test/v1/models"
        assert seen["headers"] == {"Authorization": "Bearer sk-test"}

    async def test_empty_key_sends_no_authorization(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
        seen: dict[str, object] = {}

        async def fake_get(self, url, headers=None):
            seen["headers"] = headers
            return _response()

        with patch.object(httpx.AsyncClient, "get", fake_get):
            names = await discover_custom_openai_models("https://llm.test/v1", "")
        assert names == ["fake-agent", "other-model"]
        assert seen["headers"] == {}

    async def test_401_returns_empty(self):
        async def fake_get(self, url, headers=None):
            return _response(401)

        with patch.object(httpx.AsyncClient, "get", fake_get):
            assert await discover_custom_openai_models("https://llm.test/v1") == []

    async def test_404_returns_empty(self):
        async def fake_get(self, url, headers=None):
            return _response(404)

        with patch.object(httpx.AsyncClient, "get", fake_get):
            assert await discover_custom_openai_models("https://llm.test/v1") == []

    async def test_500_returns_empty(self):
        async def fake_get(self, url, headers=None):
            return _response(500)

        with patch.object(httpx.AsyncClient, "get", fake_get):
            assert await discover_custom_openai_models("https://llm.test/v1") == []

    async def test_timeout_returns_empty(self):
        async def fake_get(self, url, headers=None):
            raise httpx.TimeoutException("timed out")

        with patch.object(httpx.AsyncClient, "get", fake_get):
            assert await discover_custom_openai_models("https://llm.test/v1") == []

    async def test_connect_error_returns_empty(self):
        async def fake_get(self, url, headers=None):
            raise httpx.ConnectError("refused")

        with patch.object(httpx.AsyncClient, "get", fake_get):
            assert await discover_custom_openai_models("https://llm.test/v1") == []

    async def test_malformed_json_returns_empty(self):
        async def fake_get(self, url, headers=None):
            return _response(json_error=True)

        with patch.object(httpx.AsyncClient, "get", fake_get):
            assert await discover_custom_openai_models("https://llm.test/v1") == []

    async def test_missing_data_returns_empty(self):
        async def fake_get(self, url, headers=None):
            return _response(body={"object": "list"})

        with patch.object(httpx.AsyncClient, "get", fake_get):
            assert await discover_custom_openai_models("https://llm.test/v1") == []

    async def test_data_not_a_list_returns_empty(self):
        async def fake_get(self, url, headers=None):
            return _response(body=[{"id": "fake-agent"}])

        with patch.object(httpx.AsyncClient, "get", fake_get):
            assert await discover_custom_openai_models("https://llm.test/v1") == []

    async def test_trailing_slash_stripped_from_url(self):
        called: dict[str, str] = {}

        async def fake_get(self, url, headers=None):
            called["url"] = url
            return _response(body={"data": []})

        with patch.object(httpx.AsyncClient, "get", fake_get):
            await discover_custom_openai_models("https://llm.test/v1/")
        assert called["url"] == "https://llm.test/v1/models"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
