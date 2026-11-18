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


def _patch_get(response=None, *, side_effect=None):
    """Patch the client's request; return the patcher and what it recorded."""
    seen: dict[str, object] = {}

    async def fake_get(self, url, headers=None, **kwargs):
        seen["url"] = url
        seen["headers"] = headers
        if side_effect is not None:
            raise side_effect
        return response

    return patch.object(httpx.AsyncClient, "get", fake_get), seen


class TestListCustomOpenaiModels:
    """Sync probe - the wizard calls this while questionary is blocking."""

    def test_empty_base_url_returns_empty_without_http(self):
        patcher, seen = _patch_get(_response())
        with patcher:
            assert list_custom_openai_models("") == []
            assert list_custom_openai_models(None) == []
        assert seen == {}

    def test_200_returns_ids(self):
        patcher, seen = _patch_get(_response())
        with patcher:
            names = list_custom_openai_models("https://llm.test/v1", "sk-test")
        assert names == ["fake-agent", "other-model"]
        assert seen["url"] == "https://llm.test/v1/models"
        assert seen["headers"] == {"Authorization": "Bearer sk-test"}

    def test_empty_key_sends_no_authorization(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-sho...leak")
        patcher, seen = _patch_get(_response())
        with patcher:
            names = list_custom_openai_models("https://llm.test/v1", "")
        assert names == ["fake-agent", "other-model"]
        assert seen["headers"] == {}

    def test_whitespace_key_sends_no_authorization(self):
        patcher, seen = _patch_get(_response())
        with patcher:
            list_custom_openai_models("https://llm.test/v1", "   ")
        assert seen["headers"] == {}

    def test_non_200_returns_empty(self):
        for status in (401, 404, 500):
            patcher, _ = _patch_get(_response(status))
            with patcher:
                assert list_custom_openai_models("https://llm.test/v1", "sk") == []

    def test_timeout_returns_empty(self):
        patcher, _ = _patch_get(side_effect=httpx.TimeoutException("timed out"))
        with patcher:
            assert list_custom_openai_models("https://llm.test/v1") == []

    def test_connect_error_returns_empty(self):
        patcher, _ = _patch_get(side_effect=httpx.ConnectError("refused"))
        with patcher:
            assert list_custom_openai_models("https://llm.test/v1") == []

    def test_malformed_json_returns_empty(self):
        patcher, _ = _patch_get(_response(json_error=True))
        with patcher:
            assert list_custom_openai_models("https://llm.test/v1") == []

    def test_missing_data_returns_empty(self):
        patcher, _ = _patch_get(_response(body={"unexpected": []}))
        with patcher:
            assert list_custom_openai_models("https://llm.test/v1") == []

    def test_data_not_a_list_returns_empty(self):
        patcher, _ = _patch_get(_response(body={"data": "nope"}))
        with patcher:
            assert list_custom_openai_models("https://llm.test/v1") == []

    def test_drops_entries_without_id(self):
        body = {"data": [{"id": "fake-agent"}, {"id": ""}, {}, "bare"]}
        patcher, _ = _patch_get(_response(body=body))
        with patcher:
            assert list_custom_openai_models("https://llm.test/v1") == ["fake-agent"]

    def test_trailing_slash_stripped_from_url(self):
        patcher, seen = _patch_get(_response(body={"data": []}))
        with patcher:
            assert list_custom_openai_models("https://llm.test/v1/") == []
        assert seen["url"] == "https://llm.test/v1/models"

    def test_a_timed_out_probe_leaves_no_worker_running(self):
        """An endpoint that never answers does not leave a thread behind.

        The deadline cancels the request, so the worker ends with it. A probe
        that only stopped waiting would leave the thread and its connection
        behind, and every further probe would leave another one.
        """
        import asyncio
        import threading
        import time

        async def never_answers(self, url, headers=None, **kwargs):
            await asyncio.sleep(3600)

        def never_answers_sync(self, url, headers=None, **kwargs):
            time.sleep(3600)

        before = threading.active_count()
        with (
            patch.object(httpx.AsyncClient, "get", never_answers),
            patch.object(httpx.Client, "get", never_answers_sync),
        ):
            assert (
                list_custom_openai_models("https://llm.test/v1", "sk", timeout=0.2)
                == []
            )

        deadline = time.monotonic() + 5.0
        while threading.active_count() > before and time.monotonic() < deadline:
            time.sleep(0.05)
        assert threading.active_count() <= before


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
