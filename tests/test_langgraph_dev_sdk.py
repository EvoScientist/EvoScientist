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

    def _fake_builder(*, url, headers, timeout=None):
        client = object()
        built.append((client, url, dict(headers), timeout))
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


def test_cached_read_client_capped_with_read_timeout(monkeypatch):
    """The cached read-only client is built with the 10s cap so a jammed dev
    server cannot block channel/serve dispatch for the SDK-default read=300s."""
    built = _patched_builder(monkeypatch)

    cached_langgraph_async_client("http://127.0.0.1:8123")

    assert built[0][3] == sdk._READ_CLIENT_TIMEOUT_SECONDS


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


# --- #880: one client, several event loops -----------------------------------


def _serve_keep_alive():
    """A local HTTP/1.1 server that keeps connections open, in a thread.

    Without keep-alive no connection is reused and the bug cannot show.
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            body = json.dumps({"thread_id": "t-1", "values": {}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _reply

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _use_on_two_loops(client) -> dict:
    """Create a thread on one loop and, while it stays alive, read state on
    another loop in a worker thread: the Rich CLI's shape."""
    import asyncio

    result: dict = {}

    def on_second_loop() -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(client.threads.get_state("t-1"))
            result["ok"] = True
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            loop.close()

    async def main() -> None:
        await client.threads.create()
        await asyncio.to_thread(on_second_loop)

    asyncio.run(main())
    return result


def test_one_client_works_from_two_event_loops():
    server, url = _serve_keep_alive()
    try:
        assert _use_on_two_loops(sdk.get_langgraph_async_client(url=url)) == {
            "ok": True
        }
    finally:
        server.shutdown()


def test_the_sdk_default_client_fails_from_two_event_loops():
    """The premise of the test above, on this server: without a pool per loop
    the second loop reuses the first loop's connection."""
    from langgraph_sdk import get_client

    server, url = _serve_keep_alive()
    try:
        result = _use_on_two_loops(get_client(url=url))
    finally:
        server.shutdown()
    assert "bound to a different event loop" in result.get("error", "")


def test_each_event_loop_gets_its_own_pool_and_keeps_it(monkeypatch):
    import asyncio

    import httpx

    made: list[object] = []
    real = httpx.AsyncHTTPTransport

    class _Counting(real):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            made.append(kwargs)

    server, url = _serve_keep_alive()
    # Built first: ``get_client`` inside it makes a transport of its own.
    client = sdk.get_langgraph_async_client(url=url)
    monkeypatch.setattr(sdk.httpx, "AsyncHTTPTransport", _Counting)
    try:

        async def two_reads() -> None:
            await client.threads.get_state("t-1")
            await client.threads.get_state("t-1")

        asyncio.run(two_reads())
        assert made == [{"retries": 5}]
        asyncio.run(two_reads())
        assert made == [{"retries": 5}, {"retries": 5}]
    finally:
        server.shutdown()


def test_the_client_keeps_the_sdk_defaults():
    """Base URL, headers (including the API key the SDK reads from the
    environment) and timeouts are those ``get_client`` sets."""
    from langgraph_sdk import get_client

    url = "http://127.0.0.1:8123"
    ours = sdk.get_langgraph_async_client(url=url).http.client
    theirs = get_client(url=url, headers=dict(LANGGRAPH_DEV_AUTH_HEADERS)).http.client

    assert ours.base_url == theirs.base_url
    assert dict(ours.headers) == dict(theirs.headers)
    assert ours.timeout == theirs.timeout
    assert sdk.get_langgraph_async_client(url=url, timeout=10).http.client.timeout == (
        get_client(url=url, timeout=10).http.client.timeout
    )


def test_closing_the_client_closes_the_calling_loops_pool():
    import asyncio

    server, url = _serve_keep_alive()
    client = sdk.get_langgraph_async_client(url=url)
    transport = client.http.client._transport
    try:

        async def read_then_close() -> int:
            await client.threads.get_state("t-1")
            before = len(transport._pools)
            await client.http.client.aclose()
            return before - len(transport._pools)

        assert asyncio.run(read_then_close()) == 1
    finally:
        server.shutdown()
