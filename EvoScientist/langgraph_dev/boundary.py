"""The backend's browser boundary.

A browser sends every request a page makes, to any address, with the page's
``Origin`` and the host name it used in ``Host``. Without a boundary any web
page open in the user's browser can create threads and start runs on a
loopback backend, also through DNS rebinding (a name that resolves to
``127.0.0.1`` looks same-origin to the browser) and over WebSockets, which CORS
does not cover. The backend therefore accepts browser requests only from pages
on trusted host names and only when ``Host`` is one of them:

- the loopback names, on any port;
- the configured bind host and the machine's own addresses, when the backend
  listens on the network;
- the origins in ``backend_allowed_origins`` (exact), whose host names also
  join the ``Host`` list.

Under ``--tunnel`` the quick-tunnel name joins the ``Host`` list only: a page
on any other origin that calls the tunnel URL needs that origin in
``backend_allowed_origins``.

Requests without ``Origin`` (the CLI, SDK scripts, the WebUI's Node server)
are only checked for ``Host``. Non-browser programs can send any header, so
this stops pages, not programs; the local secret covers the routes that need
more.

The manager computes the trusted names when it starts the server
(:func:`server_env`) and passes them twice from one set: as langgraph-api's
``CORS_CONFIG``, which langgraph-api reads before this code runs, and as
:data:`BOUNDARY_ENV` for :class:`BrowserBoundary`, which refuses requests in the
server. A plain ``langgraph dev`` gets the loopback names from
``langgraph.json``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")
BOUNDARY_ENV = "EVOSCIENTIST_BROWSER_BOUNDARY"
# What the WebUI reads from responses; a ``cors`` block replaces these defaults.
_EXPOSE_HEADERS = ["x-pagination-total", "x-pagination-next", "content-location"]
_WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::"})
# A quick-tunnel name; ``api`` is cloudflared's own API, which langgraph-api
# takes for the tunnel URL when the tunnel cannot be created.
_QUICK_TUNNEL = re.compile(r"[a-z0-9-]+\.trycloudflare\.com")


def parse_allowed_origins(value: str) -> list[str]:
    """The valid origins in a comma-separated ``backend_allowed_origins``.

    An origin is ``http`` or ``https``, a host and an optional port, with no
    path; a trailing ``/``, as copied from the address bar, is dropped. Other
    entries are left out with a warning.
    """
    origins = []
    for item in (part.strip().removesuffix("/") for part in value.split(",")):
        if not item:
            continue
        try:
            parts = urlsplit(item)
            parts.port  # noqa: B018 - raises on a malformed port
        except ValueError:
            parts = None
        if (
            parts is None
            or parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.path
            or parts.query
            or parts.fragment
            or parts.username
        ):
            logger.warning(
                "Ignoring %r in backend_allowed_origins: an origin is http or "
                "https, a host and an optional port, for example "
                "https://smith.langchain.com",
                item,
            )
            continue
        if item != f"{parts.scheme}://{parts.netloc.lower()}" or parts.port == (
            443 if parts.scheme == "https" else 80
        ):
            logger.warning(
                "%r in backend_allowed_origins never matches a page: browsers "
                "send the scheme and host in lowercase and leave out the "
                "default port",
                item,
            )
        origins.append(item)
    return origins


def server_env(bind_host: str, allowed_origins: str) -> dict[str, str]:
    """Environment variables that set the boundary of a server bound to *bind_host*."""
    names = _trusted_names(bind_host)
    origins = parse_allowed_origins(allowed_origins)
    return {
        "CORS_CONFIG": json.dumps(cors_config(names, origins)),
        BOUNDARY_ENV: json.dumps({"names": names, "origins": origins}),
    }


def cors_config(names: list[str], origins: list[str]) -> dict:
    """langgraph-api's CORS settings for pages on *names* (any port) and *origins*.

    ``allow_origins`` is always set: langgraph-api fills a missing one with
    ``["*"]``, which would allow every origin.
    """
    hosts = "|".join(re.escape(f"[{n}]" if ":" in n else n) for n in names)
    return {
        "allow_origins": origins,
        "allow_origin_regex": rf"https?://({hosts})(:\d+)?",
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "expose_headers": _EXPOSE_HEADERS,
    }


def _trusted_names(bind_host: str) -> list[str]:
    names = list(LOOPBACK_NAMES)
    bind = bind_host.strip().strip("[]").lower()
    if bind in names:
        return names
    if bind not in _WILDCARD_HOSTS:
        # The name our own clients use for a server bound to one interface.
        names.append(bind)
    import psutil

    for addresses in psutil.net_if_addrs().values():
        for address in addresses:
            ip = address.address.lower()
            if address.family in (socket.AF_INET, socket.AF_INET6) and ip not in names:
                names.append(ip)
    return names


@dataclass(frozen=True)
class Trust:
    """What a server accepts: page host names (any port), exact origins, and
    the names ``Host`` may carry."""

    names: frozenset[str]
    origins: frozenset[str]
    hosts: frozenset[str]

    @classmethod
    def from_env(cls) -> Trust:
        raw = os.environ.get(BOUNDARY_ENV)
        try:
            data = json.loads(raw) if raw else {}
            names = frozenset(data.get("names") or LOOPBACK_NAMES)
            origins = frozenset(data.get("origins") or ())
            if not all(isinstance(item, str) for item in names | origins):
                raise TypeError(f"{BOUNDARY_ENV} lists a non-string")
            hosts = set(names) | {urlsplit(o).hostname for o in origins}
        except (ValueError, TypeError, AttributeError):
            logger.error(
                "%s is not valid; the backend trusts only loopback pages.",
                BOUNDARY_ENV,
            )
            names, origins = frozenset(LOOPBACK_NAMES), frozenset()
            hosts = set(names)
        tunnel = urlsplit(os.environ.get("LANGGRAPH_API_URL", "")).hostname
        if (
            tunnel
            and tunnel != "api.trycloudflare.com"
            and _QUICK_TUNNEL.fullmatch(tunnel)
        ):
            hosts.add(tunnel)
        return cls(names, origins, frozenset(hosts))

    def refusal(
        self, host: str | None, origin: str | None
    ) -> tuple[str, str, str] | None:
        """``(error, field, value)`` when a request must be refused, else None."""
        if _host_name(host) not in self.hosts:
            return "host_not_allowed", "host", host or ""
        if origin is not None and not self._origin_trusted(origin):
            return "origin_not_allowed", "origin", origin
        return None

    def _origin_trusted(self, origin: str) -> bool:
        if origin in self.origins:
            return True
        try:
            parts = urlsplit(origin)
            parts.port  # noqa: B018 - raises on a malformed port
        except ValueError:
            return False
        return parts.scheme in ("http", "https") and parts.hostname in self.names


def _host_name(host: str | None) -> str:
    """The name in a ``Host`` header, lowercased, without port or IPv6 brackets."""
    if not host:
        return ""
    host = host.lower()
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else ""
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def _detail(error: str, value: str) -> str:
    if error == "host_not_allowed":
        return (
            f"This backend does not answer to the host name in {value!r}. To "
            "reach it under that name, add an origin with it to "
            "backend_allowed_origins in config.yaml and restart the backend."
        )
    if value == "null":
        return (
            "This backend does not accept requests from pages without an origin "
            "of their own, such as sandboxed frames and local files."
        )
    return (
        f"This backend does not accept requests from pages on {value}. To allow "
        "them, add it to backend_allowed_origins in config.yaml (a "
        "comma-separated list) and restart the backend."
    )


class BrowserBoundary:
    """ASGI middleware that refuses requests outside the trusted names.

    Installed on the custom app, which langgraph-api places outside its own
    middleware, so it runs before CORS and every route, WebSockets included.
    Requests langgraph-api makes to itself in process (``root_path``
    ``/noauth``, which no request from outside can set) pass, as they pass its
    auth. *trust* defaults to :meth:`Trust.from_env` on the first request.
    """

    def __init__(self, app, trust: Trust | None = None) -> None:
        self.app = app
        self._trust = trust

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket") or (
            scope.get("root_path") == "/noauth"
        ):
            await self.app(scope, receive, send)
            return
        if self._trust is None:
            self._trust = Trust.from_env()
        headers = {
            k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]
        }
        refusal = self._trust.refusal(headers.get("host"), headers.get("origin"))
        if refusal is None:
            await self.app(scope, receive, send)
            return
        error, field, value = refusal
        detail = _detail(error, value)
        logger.warning("Refused a request: %s", detail)
        if scope["type"] == "websocket":
            # Closing before accepting answers the handshake with 403.
            await send({"type": "websocket.close", "code": 1008})
            return
        from starlette.responses import JSONResponse

        response = JSONResponse(
            {"error": error, field: value, "detail": detail}, status_code=403
        )
        await response(scope, receive, send)
