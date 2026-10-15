"""Tests for the backend's browser boundary (Host/Origin guard and CORS settings)."""

from __future__ import annotations

import json
import logging
import os
import re
import socket
from collections import namedtuple
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from EvoScientist.langgraph_dev import boundary
from EvoScientist.langgraph_dev.http import app as server_app

EVIL = "https://evil.example"


@pytest.fixture
def reached():
    return []


@pytest.fixture
def guarded(reached):
    """A small app behind the guard; each test gets a fresh trusted set."""

    async def ok(_request):
        reached.append("http")
        return PlainTextResponse("ok")

    async def ws(websocket):
        reached.append("websocket")
        await websocket.accept()
        await websocket.send_text("ok")
        await websocket.close()

    def make(**client_kw):
        app = Starlette(
            routes=[
                Route("/threads", ok, methods=["GET", "POST", "OPTIONS"]),
                WebSocketRoute("/threads/t/stream/events", ws),
            ],
            middleware=[Middleware(boundary.BrowserBoundary)],
        )
        return TestClient(app, base_url="http://127.0.0.1:6174", **client_kw)

    return make


def _set_trust(monkeypatch, bind_host="127.0.0.1", allowed=""):
    env = boundary.server_env(bind_host, allowed)
    monkeypatch.setenv(boundary.BOUNDARY_ENV, env[boundary.BOUNDARY_ENV])


# --- allowed origins --------------------------------------------------------


def test_parse_allowed_origins_keeps_only_origins(caplog):
    value = (
        "https://smith.langchain.com, http://10.0.0.5:3000,"
        "smith.langchain.com,https://x.example/path,ftp://x.example,"
        "https://x.example:port,https://user@x.example"
    )
    with caplog.at_level(logging.WARNING):
        origins = boundary.parse_allowed_origins(value)
    assert origins == ["https://smith.langchain.com", "http://10.0.0.5:3000"]
    assert caplog.text.count("Ignoring") == 5


def test_parse_allowed_origins_strips_a_trailing_slash(caplog):
    value = "https://smith.langchain.com/,https://x.example/path/"
    with caplog.at_level(logging.WARNING):
        origins = boundary.parse_allowed_origins(value)
    assert origins == ["https://smith.langchain.com"]
    assert caplog.text.count("Ignoring") == 1


@pytest.mark.parametrize(
    "origin", ["HTTPS://Smith.LangChain.com", "https://x.example:443"]
)
def test_parse_allowed_origins_warns_about_spellings_browsers_never_send(
    origin, caplog
):
    """Kept as written (no normalising), with a warning: it would never match."""
    with caplog.at_level(logging.WARNING):
        assert boundary.parse_allowed_origins(origin) == [origin]
    assert "never matches a page" in caplog.text


# --- the trusted names the manager computes -------------------------------

_Addr = namedtuple("_Addr", "family address netmask broadcast ptp")


@pytest.fixture
def interfaces(monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {
            "lo": [_Addr(socket.AF_INET, "127.0.0.1", None, None, None)],
            "eth0": [
                _Addr(socket.AF_INET, "192.168.1.20", None, None, None),
                _Addr(socket.AF_INET6, "fe80::1%eth0", None, None, None),
                _Addr(getattr(socket, "AF_PACKET", 17), "aa:bb:cc", None, None, None),
            ],
        },
    )


def test_server_env_loopback_bind_trusts_only_loopback(interfaces):
    data = json.loads(boundary.server_env("127.0.0.1", "")[boundary.BOUNDARY_ENV])
    assert data == {"names": list(boundary.LOOPBACK_NAMES), "origins": []}


@pytest.mark.parametrize("bind", ["0.0.0.0", "myhost.local"])
def test_server_env_network_bind_trusts_own_addresses(bind, interfaces):
    names = json.loads(boundary.server_env(bind, "")[boundary.BOUNDARY_ENV])["names"]
    assert "192.168.1.20" in names
    assert "aa:bb:cc" not in names
    assert ("myhost.local" in names) == (bind == "myhost.local")


def test_server_env_cors_and_guard_trust_the_same_pages(interfaces):
    """One set, two serialisations: CORS must allow exactly the guard's pages."""
    env = boundary.server_env("0.0.0.0", "https://smith.langchain.com")
    cors = json.loads(env["CORS_CONFIG"])
    guard = json.loads(env[boundary.BOUNDARY_ENV])
    assert cors["allow_origins"] == guard["origins"] == ["https://smith.langchain.com"]
    regex = re.compile(cors["allow_origin_regex"])
    for name in guard["names"]:
        host = f"[{name}]" if ":" in name else name
        assert regex.fullmatch(f"http://{host}:4716"), name
    assert not regex.fullmatch(EVIL)


def test_langgraph_json_cors_is_loopback_only():
    """A plain ``langgraph dev`` gets loopback pages and an explicit
    ``allow_origins``: langgraph-api fills a missing one with ``["*"]``."""
    path = Path(boundary.__file__).with_name("langgraph.json")
    cors = json.loads(path.read_text())["http"]["cors"]
    assert cors == boundary.cors_config(list(boundary.LOOPBACK_NAMES), [])


# --- CORS settings, as langgraph-api applies them --------------------------


def _cors_client():
    async def ok(_request):
        return PlainTextResponse("ok", headers={"x-pagination-total": "3"})

    app = Starlette(
        routes=[Route("/threads", ok, methods=["GET", "POST"])],
        middleware=[
            Middleware(
                CORSMiddleware,
                **boundary.cors_config(list(boundary.LOOPBACK_NAMES), []),
            )
        ],
    )
    return TestClient(app, base_url="http://127.0.0.1:6174")


def test_cors_config_allows_a_json_preflight_from_a_loopback_page():
    response = _cors_client().options(
        "/threads",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-api-key",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_config_keeps_the_headers_the_webui_reads():
    response = _cors_client().get(
        "/threads", headers={"Origin": "http://127.0.0.1:4716"}
    )
    exposed = response.headers["access-control-expose-headers"].lower()
    for header in ("x-pagination-total", "x-pagination-next", "content-location"):
        assert header in exposed


# --- the guard -------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "headers"),
    [
        ("POST", {"Content-Type": "text/plain"}),  # no preflight: CORS refuses nothing
        ("POST", {"Content-Type": "application/json"}),
        ("OPTIONS", {"Access-Control-Request-Method": "POST"}),
    ],
)
@pytest.mark.parametrize("origin", [EVIL, "null"])
def test_boundary_refuses_a_foreign_origin(
    origin, method, headers, guarded, reached, monkeypatch
):
    _set_trust(monkeypatch)
    response = guarded().request(
        method, "/threads", headers={"Origin": origin, **headers}, content="{}"
    )
    assert response.status_code == 403
    assert response.json()["error"] == "origin_not_allowed"
    assert response.json()["origin"] == origin
    if origin == EVIL:
        assert "add it to backend_allowed_origins" in response.json()["detail"]
    assert reached == []


@pytest.mark.parametrize(
    "origin", ["http://localhost:3000", "http://127.0.0.1:4716", "http://[::1]:5173"]
)
def test_boundary_accepts_a_loopback_page_on_any_port(
    origin, guarded, reached, monkeypatch
):
    _set_trust(monkeypatch)
    assert guarded().post("/threads", headers={"Origin": origin}).status_code == 200
    assert reached == ["http"]


def test_boundary_accepts_a_request_without_origin(guarded, monkeypatch):
    """The CLI, SDK scripts and the WebUI's Node server send no Origin."""
    _set_trust(monkeypatch)
    assert guarded().post("/threads").status_code == 200


@pytest.mark.parametrize(
    ("host", "accepted"),
    [
        ("LOCALHOST:6174", True),
        ("[::1]:6174", True),
        ("attacker.example:6174", False),
        ("localhost.:6174", False),
        ("", False),  # HTTP/1.0 without Host
    ],
)
def test_boundary_checks_the_host_name(host, accepted, guarded, monkeypatch):
    _set_trust(monkeypatch)
    response = guarded().get("/threads", headers={"Host": host})
    if accepted:
        assert response.status_code == 200
    else:
        assert response.status_code == 403
        assert response.json()["error"] == "host_not_allowed"


def test_boundary_accepts_own_addresses_on_a_network_bind(
    guarded, interfaces, monkeypatch
):
    """A LAN browser reaches the WebUI and the backend by the machine's address."""
    _set_trust(monkeypatch, bind_host="0.0.0.0")
    response = guarded().post(
        "/threads",
        headers={"Host": "192.168.1.20:6174", "Origin": "http://192.168.1.20:4716"},
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    "value", ["{not json", '{"names": ["localhost"], "origins": [1]}', "[]"]
)
def test_boundary_with_a_broken_setting_trusts_loopback_only(
    value, guarded, monkeypatch, caplog
):
    monkeypatch.setenv(boundary.BOUNDARY_ENV, value)
    client = guarded()
    assert client.get("/threads").status_code == 200
    assert client.get("/threads", headers={"Host": "10.0.0.5:6174"}).status_code == 403
    assert boundary.BOUNDARY_ENV in caplog.text


def test_boundary_accepts_an_allowed_origin_and_its_host(guarded, monkeypatch):
    _set_trust(monkeypatch, allowed="https://smith.langchain.com")
    client = guarded()
    origin = {"Origin": "https://smith.langchain.com"}
    assert client.post("/threads", headers=origin).status_code == 200
    host = {"Host": "smith.langchain.com"}
    assert client.get("/threads", headers=host).status_code == 200
    other = {"Origin": "https://smith.langchain.com:8443"}
    assert client.post("/threads", headers=other).status_code == 403


def test_boundary_trusts_the_configured_bind_host_name(guarded, monkeypatch):
    """Our own clients use the configured host name when the server is bound to it."""
    _set_trust(monkeypatch, bind_host="myhost.local")
    response = guarded().get("/threads", headers={"Host": "myhost.local:6174"})
    assert response.status_code == 200


def test_boundary_refuses_a_foreign_websocket(guarded, reached, monkeypatch, caplog):
    """WebSockets bypass CORS; a page could start runs over the event stream."""
    _set_trust(monkeypatch)
    headers = {"Host": "127.0.0.1:6174", "Origin": EVIL}
    with (
        pytest.raises(WebSocketDisconnect),
        guarded().websocket_connect("/threads/t/stream/events", headers=headers),
    ):
        pass
    assert reached == []
    assert f"pages on {EVIL}" in caplog.text


def test_boundary_accepts_a_loopback_websocket(guarded, reached, monkeypatch):
    _set_trust(monkeypatch)
    # The test client sends Host "testserver" on WebSockets unless told otherwise.
    headers = {"Host": "127.0.0.1:6174", "Origin": "http://localhost:4716"}
    with guarded().websocket_connect("/threads/t/stream/events", headers=headers) as ws:
        assert ws.receive_text() == "ok"
    assert reached == ["websocket"]


@pytest.mark.parametrize(
    ("api_url", "host", "trusted"),
    [
        (
            "https://quiet-river-bend.trycloudflare.com",
            "quiet-river-bend.trycloudflare.com",
            True,
        ),
        # the tunnel could not be created
        ("https://api.trycloudflare.com", "api.trycloudflare.com", False),
        ("https://attacker.example", "attacker.example", False),
        ("http://127.0.0.1:6174", "quiet-river-bend.trycloudflare.com", False),
        (None, "quiet-river-bend.trycloudflare.com", False),
    ],
)
def test_boundary_trusts_only_a_quick_tunnel_name(
    api_url, host, trusted, guarded, monkeypatch
):
    _set_trust(monkeypatch)
    if api_url is None:
        monkeypatch.delenv("LANGGRAPH_API_URL", raising=False)
    else:
        monkeypatch.setenv("LANGGRAPH_API_URL", api_url)
    client = guarded()
    tunnel = client.get("/threads", headers={"Host": host})
    assert (tunnel.status_code == 200) == trusted
    other = client.get("/threads", headers={"Host": "other.trycloudflare.com"})
    assert other.status_code == 403


def test_boundary_logs_each_refusal(guarded, monkeypatch, caplog):
    _set_trust(monkeypatch)
    client = guarded()
    with caplog.at_level(logging.WARNING, logger="EvoScientist.langgraph_dev.boundary"):
        client.get("/threads", headers={"Origin": EVIL})
        client.get("/threads", headers={"Origin": EVIL})
    refusals = [r for r in caplog.records if "Refused a request" in r.getMessage()]
    assert len(refusals) == 2
    assert EVIL in refusals[0].getMessage()


def test_boundary_without_settings_trusts_loopback_only(guarded, monkeypatch):
    """A plain ``langgraph dev``: no boundary variable from the manager."""
    monkeypatch.delenv(boundary.BOUNDARY_ENV, raising=False)
    client = guarded()
    assert client.get("/threads").status_code == 200
    assert client.get("/threads", headers={"Host": "10.0.0.5:6174"}).status_code == 403


def test_server_app_is_behind_the_boundary(monkeypatch):
    _set_trust(monkeypatch)
    middleware = [m.cls for m in server_app.user_middleware]
    assert boundary.BrowserBoundary in middleware


def test_server_app_keeps_its_boundary_from_the_agent_commands(monkeypatch):
    """The agent's shell inherits the server's environment, and a ``langgraph
    dev`` started there would take this CORS_CONFIG over its own settings."""
    from EvoScientist.langgraph_dev import http

    env = boundary.server_env("127.0.0.1", "https://smith.langchain.com")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    app = http._build_app()
    (guard,) = app.user_middleware
    assert "https://smith.langchain.com" in guard.kwargs["trust"].origins
    assert "CORS_CONFIG" not in os.environ
    assert boundary.BOUNDARY_ENV not in os.environ


def test_boundary_passes_langgraph_api_in_process_calls(guarded, monkeypatch):
    """langgraph-api calls itself in process (relative webhooks, ``get_client()``
    without a URL) with ``Host: api`` and ``root_path`` ``/noauth``."""
    import asyncio

    import httpx

    _set_trust(monkeypatch)
    app = guarded().app

    async def call(**transport_kw):
        transport = httpx.ASGITransport(app, **transport_kw)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as c:
            return (await c.get("/threads")).status_code

    assert asyncio.run(call(root_path="/noauth")) == 200
    assert asyncio.run(call()) == 403


def test_boundary_keeps_the_trust_it_was_given(reached, monkeypatch):
    """The server app reads its trust at import; a ``.env`` merged later cannot
    change the tunnel name it accepts."""
    monkeypatch.setenv(
        "LANGGRAPH_API_URL", "https://quiet-river-bend.trycloudflare.com"
    )
    trust = boundary.Trust.from_env()
    monkeypatch.setenv("LANGGRAPH_API_URL", "https://planted.trycloudflare.com")

    async def ok(_request):
        return PlainTextResponse("ok")

    app = Starlette(
        routes=[Route("/threads", ok)],
        middleware=[Middleware(boundary.BrowserBoundary, trust=trust)],
    )
    client = TestClient(app, base_url="http://127.0.0.1:6174")
    tunnel = {"Host": "quiet-river-bend.trycloudflare.com"}
    assert client.get("/threads", headers=tunnel).status_code == 200
    planted = {"Host": "planted.trycloudflare.com"}
    assert client.get("/threads", headers=planted).status_code == 403
