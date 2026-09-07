"""Shared LangGraph SDK plumbing for the local langgraph-dev server."""

from __future__ import annotations

from collections.abc import Mapping

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


def configured_langgraph_dev_url() -> str:
    """Return the local langgraph-dev URL from the effective application config."""
    from ..EvoScientist import _ensure_config

    return langgraph_dev_url(_ensure_config())


def langgraph_dev_headers(headers: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return SDK headers, defaulting to the local langgraph-dev auth scheme."""
    return dict(LANGGRAPH_DEV_AUTH_HEADERS if headers is None else headers)


def get_langgraph_sync_client(*, url: str, headers: Mapping[str, str] | None = None):
    """Build a sync LangGraph SDK client with EvoScientist's default headers."""
    from langgraph_sdk import get_sync_client

    return get_sync_client(url=url, headers=langgraph_dev_headers(headers))


def get_langgraph_async_client(*, url: str, headers: Mapping[str, str] | None = None):
    """Build an async LangGraph SDK client with EvoScientist's default headers."""
    from langgraph_sdk import get_client

    return get_client(url=url, headers=langgraph_dev_headers(headers))


_ASYNC_CLIENT_CACHE: dict[str, object] = {}


def cached_langgraph_async_client(
    url: str, *, headers: Mapping[str, str] | None = None
):
    """Return a per-URL cached async SDK client.

    The async client wraps an ``httpx.AsyncClient`` bound to the event loop it
    is first used on; caching avoids leaking a new connection pool on every
    call. The CLI drives its whole session from a single ``asyncio.run`` loop,
    so one cached client per URL is safe. Callers that only read (e.g. the
    async-task completion poll) reuse this rather than constructing per poll.
    """
    client = _ASYNC_CLIENT_CACHE.get(url)
    if client is None:
        client = get_langgraph_async_client(url=url, headers=headers)
        _ASYNC_CLIENT_CACHE[url] = client
    return client


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
