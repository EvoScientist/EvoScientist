"""Shared LangGraph SDK plumbing for the local langgraph-dev server."""

from __future__ import annotations

from collections.abc import Mapping

DEFAULT_LANGGRAPH_DEV_PORT = 6174
LANGGRAPH_DEV_AUTH_HEADERS = {"x-auth-scheme": "langsmith"}


def langgraph_dev_url(config: object | None = None, *, port: int | None = None) -> str:
    """Return the local langgraph-dev base URL for a config or explicit port."""
    selected_port = (
        int(port)
        if port is not None
        else int(getattr(config, "langgraph_dev_port", DEFAULT_LANGGRAPH_DEV_PORT))
    )
    return f"http://localhost:{selected_port}"


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
