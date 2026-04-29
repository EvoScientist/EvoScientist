"""Reusable channel mixins for common architecture patterns.

Three mixins that eliminate boilerplate across channels:

- ``WebhookMixin``  — aiohttp webhook server + httpx client + token refresh
- ``WebSocketMixin`` — WS connect/reconnect/heartbeat loop
- ``PollingMixin``   — async poll loop with backoff

Each mixin works with the Channel base class. Subclasses override
a small set of abstract/hook methods to define platform-specific behavior.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# Token refresh mixin (shared by Webhook & WebSocket channels)
# ═════════════════════════════════════════════════════════════════════


class TokenMixin:
    """Mixin for channels that need OAuth-style token management.

    Subclass must implement ``_fetch_token()`` returning
    ``(access_token, expires_in_seconds)``.
    """

    _access_token: str | None = None
    _token_expires: float = 0
    _http_client: Any = None  # httpx.AsyncClient

    async def _fetch_token(self) -> tuple[str, int]:
        """Fetch a new access token. Return (token, expires_in_seconds).

        Must be implemented by the channel.
        """
        raise NotImplementedError

    async def _refresh_token(self) -> None:
        token, expire = await self._fetch_token()
        self._access_token = token
        self._token_expires = time.monotonic() + expire - 300
        logger.debug(
            f"{getattr(self, 'name', '?')} token refreshed, expires in {expire}s"
        )

    async def _ensure_token(self) -> str:
        if not self._access_token or time.monotonic() >= self._token_expires:
            await self._refresh_token()
        return self._access_token


# ═════════════════════════════════════════════════════════════════════
# Webhook + REST mixin
# ═════════════════════════════════════════════════════════════════════


class WebhookMixin:
    """Mixin for channels that use an HTTP webhook server for inbound
    and REST API for outbound.

    Provides:
    - aiohttp web server lifecycle (start/stop)
    - httpx async client lifecycle
    - Route registration via ``_webhook_routes()``

    Subclass must implement:
    - ``_webhook_routes()`` → list of (method, path, handler)
    - ``_get_webhook_port()`` → int
    """

    _http_client: Any = None
    _runner: Any = None
    _site: Any = None

    def _get_webhook_port(self) -> int:
        return getattr(self.config, "webhook_port", 9000)

    def _get_webhook_host(self) -> str:
        return (
            getattr(self.config, "webhook_host", None)
            or getattr(self.config, "bind_host", None)
            or "0.0.0.0"
        )

    def _webhook_routes(self) -> list[tuple[str, str, Any]]:
        """Return [(method, path, handler), ...]. Override in subclass."""
        return []

    async def _start_webhook_server(self) -> None:
        """Start aiohttp webhook server + httpx client.

        If ``_shared_webhook_server`` is set (by ChannelManager), the
        aiohttp server is already running on the shared port — only
        create the httpx outbound client.
        """
        import httpx

        proxy = getattr(self.config, "proxy", None) or None
        self._http_client = httpx.AsyncClient(timeout=15, proxy=proxy)

        # Shared webhook mode: routes already registered on shared server
        if getattr(self, "_shared_webhook_server", None):
            logger.info(f"{getattr(self, 'name', '?')} using shared webhook server")
            return

        from aiohttp import web

        app = web.Application()
        for method, path, handler in self._webhook_routes():
            method = method.upper()
            if method == "GET":
                app.router.add_get(path, handler)
            elif method == "POST":
                app.router.add_post(path, handler)
            else:
                app.router.add_route(method, path, handler)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        port = self._get_webhook_port()
        host = self._get_webhook_host()
        self._site = web.TCPSite(self._runner, host, port)
        await self._site.start()
        logger.info(f"{getattr(self, 'name', '?')} webhook on {host}:{port}")

    async def _stop_webhook_server(self) -> None:
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def _api_post(
        self, url: str, body: dict, headers: dict | None = None
    ) -> dict:
        """POST JSON to API, return parsed response. Raises on HTTP error."""
        resp = await self._http_client.post(url, json=body, headers=headers)
        data = resp.json()
        return data

    async def _api_get(self, url: str, headers: dict | None = None) -> dict:
        resp = await self._http_client.get(url, headers=headers)
        return resp.json()


# ═════════════════════════════════════════════════════════════════════
# WebSocket mixin
# ═════════════════════════════════════════════════════════════════════


class WebSocketMixin:
    """Mixin for channels that receive messages via WebSocket.

    Provides:
    - Connect/reconnect loop with exponential backoff
    - Heartbeat task management
    - Message dispatch

    Subclass must implement:
    - ``_get_ws_url()``       → WebSocket URL to connect to
    - ``_on_ws_message(data)`` → handle a parsed message dict
    - ``_on_ws_connected(ws)`` → called after connection (send identify, etc.)

    Optional overrides:
    - ``_ws_heartbeat_interval`` → seconds between heartbeats (0 = disabled)
    - ``_on_ws_heartbeat(ws)``  → send heartbeat
    """

    _ws_session: Any = None
    _ws_heartbeat_task: asyncio.Task | None = None
    _ws_heartbeat_interval: float = 0  # 0 = no heartbeat
    _ws_reconnect_delay: float = 5.0

    async def _get_ws_url(self) -> str:
        raise NotImplementedError

    async def _on_ws_connected(self, ws) -> None:
        """Called after WebSocket connects. Send identify/auth here."""
        pass

    async def _on_ws_message(self, data: dict | str) -> None:
        """Handle a single WebSocket message."""
        raise NotImplementedError

    async def _on_ws_heartbeat(self, ws) -> None:
        """Send a heartbeat. Override if needed."""
        pass

    async def _ws_loop(self) -> None:
        """Main WebSocket loop with auto-reconnect."""
        import os

        import aiohttp

        while getattr(self, "_running", False):
            try:
                ws_url = await self._get_ws_url()
                # Resolve proxy: channel config > environment variable
                proxy = getattr(getattr(self, "config", None), "proxy", None)
                if not proxy:
                    proxy = (
                        os.environ.get("https_proxy")
                        or os.environ.get("HTTPS_PROXY")
                        or os.environ.get("http_proxy")
                        or os.environ.get("HTTP_PROXY")
                        or None
                    )
                logger.debug(
                    f"{getattr(self, 'name', '?')} WS connecting to {ws_url[:60]}... proxy={proxy}"
                )
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        ws_url,
                        proxy=proxy,
                        timeout=aiohttp.ClientWSTimeout(ws_close=30),
                    ) as ws:
                        logger.info(f"{getattr(self, 'name', '?')} WebSocket connected")
                        self._ws_session = ws
                        await self._on_ws_connected(ws)

                        # Start heartbeat if configured
                        if self._ws_heartbeat_interval > 0:
                            self._ws_heartbeat_task = asyncio.create_task(
                                self._ws_heartbeat_loop(ws)
                            )

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                except (json.JSONDecodeError, TypeError):
                                    data = msg.data
                                await self._on_ws_message(data)
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{getattr(self, 'name', '?')} WS error: {e}")

            await self._ws_cleanup_heartbeat()
            self._ws_session = None

            if getattr(self, "_running", False):
                logger.info(
                    f"{getattr(self, 'name', '?')} reconnecting in {self._ws_reconnect_delay}s..."
                )
                await asyncio.sleep(self._ws_reconnect_delay)

    async def _ws_heartbeat_loop(self, ws) -> None:
        while True:
            try:
                await self._on_ws_heartbeat(ws)
            except Exception:
                break
            await asyncio.sleep(self._ws_heartbeat_interval)

    async def _ws_cleanup_heartbeat(self) -> None:
        if self._ws_heartbeat_task:
            task = self._ws_heartbeat_task
            self._ws_heartbeat_task = None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling() > 0:
                    raise
            except Exception:
                pass

    async def _ws_send_json(self, data: dict) -> None:
        """Send JSON to the active WebSocket."""
        if self._ws_session:
            await self._ws_session.send_str(json.dumps(data))

    async def _stop_ws(self) -> None:
        await self._ws_cleanup_heartbeat()
        if self._ws_session:
            await self._ws_session.close()
            self._ws_session = None


# ═════════════════════════════════════════════════════════════════════
# Polling mixin
# ═════════════════════════════════════════════════════════════════════


class PollingMixin:
    """Mixin for channels that poll for new messages.

    Provides:
    - Poll loop with configurable interval
    - Error handling + reconnect

    Subclass must implement:
    - ``_poll_once()`` → fetch and enqueue new messages
    - ``_get_poll_interval()`` → seconds between polls
    """

    _poll_task: asyncio.Task | None = None

    def _get_poll_interval(self) -> float:
        return getattr(self.config, "poll_interval", 30)

    async def _poll_once(self) -> None:
        """Fetch new messages and enqueue them. Override in subclass."""
        raise NotImplementedError

    async def _start_polling(self) -> None:
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        interval = self._get_poll_interval()
        while getattr(self, "_running", False):
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{getattr(self, 'name', '?')} poll error: {e}")
            await asyncio.sleep(interval)

    async def _stop_polling(self) -> None:
        if self._poll_task:
            task = self._poll_task
            self._poll_task = None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling() > 0:
                    raise
            except Exception:
                pass
