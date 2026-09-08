"""Tests for the langgraph dev SDK client helpers."""

from __future__ import annotations

import pytest

from EvoScientist.langgraph_dev import sdk
from EvoScientist.langgraph_dev.sdk import (
    LANGGRAPH_DEV_AUTH_HEADERS,
    cached_langgraph_async_client,
)


@pytest.fixture(autouse=True)
def _clean_client_cache():
    sdk._ASYNC_CLIENT_CACHE.clear()
    yield
    sdk._ASYNC_CLIENT_CACHE.clear()


def _patched_builder(monkeypatch):
    built = []

    def _fake_builder(*, url, headers):
        client = object()
        built.append((client, url, dict(headers)))
        return client

    monkeypatch.setattr(sdk, "get_langgraph_async_client", _fake_builder)
    return built


def test_same_url_and_headers_reuse_one_client(monkeypatch):
    built = _patched_builder(monkeypatch)

    first = cached_langgraph_async_client("http://127.0.0.1:8123")
    second = cached_langgraph_async_client("http://127.0.0.1:8123")

    assert first is second
    assert len(built) == 1
    assert built[0][2] == dict(LANGGRAPH_DEV_AUTH_HEADERS)


def test_none_and_explicit_default_headers_share_one_client(monkeypatch):
    built = _patched_builder(monkeypatch)

    via_none = cached_langgraph_async_client("http://127.0.0.1:8123")
    via_explicit = cached_langgraph_async_client(
        "http://127.0.0.1:8123", headers=dict(LANGGRAPH_DEV_AUTH_HEADERS)
    )

    assert via_none is via_explicit
    assert len(built) == 1


def test_different_headers_get_separate_clients(monkeypatch):
    built = _patched_builder(monkeypatch)

    default = cached_langgraph_async_client("http://127.0.0.1:8123")
    custom = cached_langgraph_async_client(
        "http://127.0.0.1:8123", headers={"x-auth-scheme": "other"}
    )
    default_again = cached_langgraph_async_client("http://127.0.0.1:8123")

    assert custom is not default
    assert default_again is default
    assert len(built) == 2
    assert built[1][2] == {"x-auth-scheme": "other"}
