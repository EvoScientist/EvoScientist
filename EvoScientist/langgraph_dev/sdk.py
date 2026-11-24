"""Shared LangGraph SDK plumbing for the local langgraph-dev server."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx

DEFAULT_LANGGRAPH_DEV_PORT = 6174
# Mirrors ``config.langgraph_dev_host`` / ``manager._DEFAULT_HOST``. The value
# only matters as a stand-in for the *bind* host — ``_format_hostport`` runs it
# through ``_probe_host``, so both this and "0.0.0.0" yield the same client URL.
DEFAULT_LANGGRAPH_DEV_HOST = "127.0.0.1"
LANGGRAPH_DEV_AUTH_HEADERS = {"x-auth-scheme": "langsmith"}


def langgraph_dev_url(
    config: object | None = None,
    *,
    port: int | None = None,
    host: str | None = None,
) -> str:
    """Return the local langgraph-dev base URL for a config or explicit port/host.

    This is a *client* URL, so the configured bind interface is mapped through
    ``manager._probe_host``: a wildcard bind (``0.0.0.0``) still resolves to
    loopback here, while a specific interface is honored so self-dispatch keeps
    working when the server is pinned to one address.
    """
    from .manager import _format_hostport

    selected_port = (
        int(port)
        if port is not None
        else int(getattr(config, "langgraph_dev_port", DEFAULT_LANGGRAPH_DEV_PORT))
    )
    selected_host = (
        host
        if host is not None
        else str(
            getattr(config, "langgraph_dev_host", DEFAULT_LANGGRAPH_DEV_HOST)
            or DEFAULT_LANGGRAPH_DEV_HOST
        )
    )
    return f"http://{_format_hostport(selected_host, selected_port)}"


# The address of the server ``ensure_server`` found or started for this
# process's session: the one its record names. None in the server itself, and
# before (or without) a server.
_SERVED_URL: str | None = None


def set_served_url(url: str | None) -> None:
    """Record the address of the server this process's session uses."""
    global _SERVED_URL
    _SERVED_URL = url


def configured_langgraph_dev_url(config: object | None = None) -> str:
    """Return the local langgraph-dev URL: the server this process's session
    uses, else the one *config* (by default the effective application
    config) names."""
    if _SERVED_URL is not None:
        return _SERVED_URL
    if config is None:
        from ..EvoScientist import _ensure_config

        config = _ensure_config()
    return langgraph_dev_url(config)


def langgraph_dev_headers(headers: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return SDK headers, defaulting to the local langgraph-dev auth scheme."""
    return dict(LANGGRAPH_DEV_AUTH_HEADERS if headers is None else headers)


def get_langgraph_sync_client(
    *,
    url: str,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
):
    """Build a sync LangGraph SDK client with EvoScientist's default headers.

    ``timeout`` (seconds) caps the underlying ``httpx.Client``; ``None``
    leaves the SDK default (``read=300s``) in place.
    """
    from langgraph_sdk import get_sync_client

    return get_sync_client(
        url=url, headers=langgraph_dev_headers(headers), timeout=timeout
    )


class _PerLoopTransport(httpx.AsyncBaseTransport):
    """One connection pool per event loop.

    A pooled connection belongs to the loop that opened it: anyio waits on an
    ``asyncio.Event`` created on that loop, so a request from another loop that
    reuses the connection fails with "... is bound to a different event loop".
    The Rich CLI uses one client on its main loop and on the async runtime's
    loop (#880). A pool lives until it is closed on its own loop or the
    client is closed: its idle connections hold their loop, so it is not
    dropped when the loop ends.
    """

    def __init__(self, **transport_kwargs: Any) -> None:
        self._transport_kwargs = transport_kwargs
        self._lock = threading.Lock()
        self._pools: dict[asyncio.AbstractEventLoop, httpx.AsyncHTTPTransport] = {}

    def _pool(self) -> httpx.AsyncHTTPTransport:
        loop = asyncio.get_running_loop()
        with self._lock:
            pool = self._pools.get(loop)
            if pool is None:
                pool = httpx.AsyncHTTPTransport(**self._transport_kwargs)
                self._pools[loop] = pool
        return pool

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._pool().handle_async_request(request)

    async def aclose(self) -> None:
        """Close the calling loop's pool and forget the others.

        Another loop's pool cannot be closed from here: once that loop is
        gone its close fails with "Event loop is closed", and while it runs
        its sockets would be closed from the wrong thread.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            pool = self._pools.pop(loop, None)
            self._pools.clear()
        if pool is not None:
            await pool.aclose()


def get_langgraph_async_client(
    *,
    url: str,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
):
    """Build an async LangGraph SDK client with EvoScientist's default headers.

    ``timeout`` (seconds) caps the underlying ``httpx.AsyncClient``; ``None``
    leaves the SDK default (``read=300s``) in place. The client keeps one
    connection pool per event loop, so it can be used from any loop.
    """
    from langgraph_sdk import get_client
    from langgraph_sdk.client import LangGraphClient

    # ``get_client`` resolves the base URL, the headers (with the API key it
    # reads from the environment) and the timeout, but takes no transport:
    # build the same client again on the per-loop one. ``retries=5`` is the
    # transport ``get_client`` uses.
    built = get_client(url=url, headers=langgraph_dev_headers(headers), timeout=timeout)
    http = built.http.client
    return LangGraphClient(
        httpx.AsyncClient(
            base_url=http.base_url,
            headers=http.headers,
            timeout=http.timeout,
            transport=_PerLoopTransport(retries=5),
        )
    )


_ASYNC_CLIENT_CACHE: dict[tuple[str, tuple[tuple[str, str], ...]], object] = {}

# The cached client backs the read-only completion poll, which is awaited
# inline on the channel/serve dispatch path. Without a cap it inherits the SDK
# default ``read=300s``, so a jammed dev server (the known stuck-queue
# condition) would block dispatch for up to five minutes per poll. A timed-out
# poll takes the caller's ``except Exception`` branch and retries next tick.
_READ_CLIENT_TIMEOUT_SECONDS = 10.0


def cached_langgraph_async_client(
    url: str, *, headers: Mapping[str, str] | None = None
):
    """Return a cached async SDK client, keyed on URL and normalized headers.

    Caching avoids a new connection pool on every call. The key includes the
    *normalized* headers, so ``headers=None`` (defaults) and an explicit copy
    of the default headers share one client, while genuinely different header
    sets get their own — a caller passing different headers must never
    silently receive a client built for someone else's. The client keeps one
    connection pool per event loop, so one cached client per key serves every
    loop of the process (the Rich CLI has two). Callers that only read (e.g.
    the async-task completion poll) reuse this rather than constructing per
    poll.
    """
    normalized = langgraph_dev_headers(headers)
    key = (url, tuple(sorted(normalized.items())))
    client = _ASYNC_CLIENT_CACHE.get(key)
    if client is None:
        client = get_langgraph_async_client(
            url=url, headers=normalized, timeout=_READ_CLIENT_TIMEOUT_SECONDS
        )
        _ASYNC_CLIENT_CACHE[key] = client
    return client


@asynccontextmanager
async def langgraph_async_client_scoped(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = _READ_CLIENT_TIMEOUT_SECONDS,
) -> AsyncIterator[Any]:
    """An async SDK client for one read, closed on exit.

    Built here instead of through ``get_client`` so the transport makes no
    retries: a single read must not spend 7.5 s retrying a connection the
    caller already knows is down.
    """
    import httpx
    from langgraph_sdk.client import LangGraphClient

    http = httpx.AsyncClient(
        base_url=url,
        transport=httpx.AsyncHTTPTransport(retries=0),
        timeout=httpx.Timeout(timeout),
        headers=langgraph_dev_headers(headers),
    )
    try:
        yield LangGraphClient(http)
    finally:
        await http.aclose()


def default_scheduler_timezone(config: object | None = None) -> str | None:
    """Return configured scheduler timezone, falling back to the host timezone."""
    if config is None:
        from ..EvoScientist import _ensure_config

        config = _ensure_config()
    timezone = str(getattr(config, "scheduler_default_timezone", "") or "")
    if timezone:
        return timezone
    try:
        from tzlocal import get_localzone_name

        return get_localzone_name()
    except Exception:
        return None


def messages_input(content: str) -> dict[str, list[dict[str, str]]]:
    """Return the standard LangGraph chat input shape for one user message."""
    return {"messages": [{"role": "user", "content": content}]}
