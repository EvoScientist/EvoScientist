"""Tests for the shared channel mixins."""

import logging
from types import SimpleNamespace

import aiohttp

from EvoScientist.channels.mixins import WebSocketMixin


class _WsChannel(WebSocketMixin):
    name = "ws-test"

    def __init__(self):
        self._running = True
        self.config = SimpleNamespace(proxy="http://user:proxy-secret@proxy:3128")

    async def _get_ws_url(self) -> str:
        return "wss://gateway.example.com/connect?ticket=ws-ticket"


async def test_ws_loop_debug_log_hides_ticket_and_proxy_password(monkeypatch, caplog):
    channel = _WsChannel()

    def _fail_once():
        channel._running = False
        raise RuntimeError("connect refused")

    monkeypatch.setattr(aiohttp, "ClientSession", _fail_once)
    caplog.set_level(logging.DEBUG, logger="EvoScientist.channels.mixins")

    await channel._ws_loop()

    assert "gateway.example.com/connect" in caplog.text
    assert "ws-ticket" not in caplog.text
    assert "proxy-secret" not in caplog.text


async def test_ws_loop_error_log_hides_ticket_on_rejected_handshake(
    monkeypatch, caplog
):
    from aiohttp.client_reqrep import RequestInfo
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL

    channel = _WsChannel()
    url = URL(await channel._get_ws_url())

    def _reject_handshake():
        channel._running = False
        request_info = RequestInfo(url, "GET", CIMultiDictProxy(CIMultiDict()), url)
        raise aiohttp.WSServerHandshakeError(
            request_info, (), status=403, message="Invalid response status"
        )

    monkeypatch.setattr(aiohttp, "ClientSession", _reject_handshake)
    caplog.set_level(logging.ERROR, logger="EvoScientist.channels.mixins")

    await channel._ws_loop()

    assert "403" in caplog.text
    assert "ws-ticket" not in caplog.text
