"""Unified channel manager for coordinating chat channels.

Manages channel lifecycle (start/stop), wires each channel to the
message bus, and routes outbound messages to the correct channel.

Also provides the global channel registry (formerly in ``registry.py``),
account management (formerly ``account.py``), and pipeline assembly
(formerly ``pipeline.py``).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import pkgutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .base import Channel, OutboundMessage
from .bus import MessageBus
from .middleware import OutboundMiddlewareBase
from .plugin import ChannelPlugin

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# Account management (formerly account.py)
# ═════════════════════════════════════════════════════════════════════

@dataclass
class ChannelAccountSnapshot:
    """Point-in-time snapshot of a single account's connection state."""

    account_id: str
    channel: str
    connected: bool = False
    started_at: float = 0.0
    last_outbound_at: float = 0.0
    error: str | None = None

    def mark_connected(self) -> None:
        self.connected = True
        self.started_at = time.monotonic()
        self.error = None

    def mark_disconnected(self, error: str | None = None) -> None:
        self.connected = False
        self.error = error

    def mark_outbound(self) -> None:
        self.last_outbound_at = time.monotonic()


@dataclass
class AccountConfig:
    """Per-account configuration wrapper."""

    account_id: str
    channel_id: str  # which plugin
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountState:
    """Runtime state for a single account."""

    account_id: str
    channel_id: str
    status: str = "stopped"  # stopped | starting | running | error
    snapshot: ChannelAccountSnapshot | None = None
    error: str | None = None
    started_at: float = 0.0


class AccountManager:
    """Manages multiple accounts across channel plugins.

    Works with the ``ConfigAdapter`` protocol on each plugin to discover
    accounts and manage their lifecycle independently.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, ChannelPlugin] = {}
        self._states: dict[str, AccountState] = {}  # key: "{channel_id}:{account_id}"

    @staticmethod
    def _key(channel_id: str, account_id: str) -> str:
        return f"{channel_id}:{account_id}"

    def register_plugin(self, plugin: ChannelPlugin) -> None:
        """Register a plugin that supports multi-account."""
        self._plugins[plugin.id] = plugin
        logger.info(f"AccountManager: registered plugin '{plugin.id}'")

    async def start_account(
        self,
        channel_id: str,
        account_id: str,
        config: Any = None,
    ) -> None:
        """Start a specific account on a plugin."""
        plugin = self._plugins.get(channel_id)
        if plugin is None:
            raise ValueError(f"No plugin registered for channel '{channel_id}'")

        key = self._key(channel_id, account_id)
        state = self._states.get(key)
        if state is None:
            state = AccountState(account_id=account_id, channel_id=channel_id)
            self._states[key] = state

        if state.status == "running":
            logger.warning(f"Account {key} is already running")
            return

        state.status = "starting"
        state.error = None
        try:
            account_config = config
            if plugin.config_adapter is not None and config is not None:
                account_config = plugin.config_adapter.resolve_account(config, account_id)

            await plugin.start(account_config, account_id=account_id)
            state.status = "running"
            state.started_at = time.monotonic()
            state.snapshot = ChannelAccountSnapshot(
                account_id=account_id, channel=channel_id,
            )
            state.snapshot.mark_connected()
            logger.info(f"Account {key} started")
        except Exception as e:
            state.status = "error"
            state.error = str(e)
            logger.error(f"Failed to start account {key}: {e}")
            raise

    async def stop_account(self, channel_id: str, account_id: str) -> None:
        """Stop a specific account on a plugin."""
        plugin = self._plugins.get(channel_id)
        if plugin is None:
            raise ValueError(f"No plugin registered for channel '{channel_id}'")

        key = self._key(channel_id, account_id)
        state = self._states.get(key)
        if state is None or state.status == "stopped":
            logger.debug(f"Account {key} is already stopped")
            return

        try:
            await plugin.stop(account_id=account_id)
            state.status = "stopped"
            if state.snapshot is not None:
                state.snapshot.mark_disconnected()
            logger.info(f"Account {key} stopped")
        except Exception as e:
            state.status = "error"
            state.error = str(e)
            if state.snapshot is not None:
                state.snapshot.mark_disconnected(error=str(e))
            logger.error(f"Error stopping account {key}: {e}")
            raise

    async def restart_account(
        self,
        channel_id: str,
        account_id: str,
        config: Any = None,
    ) -> None:
        """Restart a specific account."""
        await self.stop_account(channel_id, account_id)
        await self.start_account(channel_id, account_id, config)

    async def start_all(self, channel_id: str, config: Any = None) -> None:
        """Start all accounts for a given channel plugin."""
        plugin = self._plugins.get(channel_id)
        if plugin is None:
            raise ValueError(f"No plugin registered for channel '{channel_id}'")

        adapter = plugin.config_adapter
        if adapter is None:
            await self.start_account(channel_id, "default", config)
            return

        if config is None:
            logger.warning(f"No config provided for start_all on '{channel_id}'")
            return

        for account_id in adapter.list_account_ids(config):
            if adapter.is_enabled(
                adapter.resolve_account(config, account_id), config,
            ):
                try:
                    await self.start_account(channel_id, account_id, config)
                except Exception as e:
                    logger.error(
                        f"Failed to start account {channel_id}:{account_id}: {e}"
                    )

    async def stop_all(self, channel_id: str) -> None:
        """Stop all accounts for a given channel plugin."""
        keys_to_stop = [
            (state.channel_id, state.account_id)
            for state in self._states.values()
            if state.channel_id == channel_id and state.status != "stopped"
        ]
        for cid, aid in keys_to_stop:
            try:
                await self.stop_account(cid, aid)
            except Exception as e:
                logger.error(f"Failed to stop account {cid}:{aid}: {e}")

    def get_state(
        self, channel_id: str, account_id: str,
    ) -> AccountState | None:
        """Get the runtime state for a specific account."""
        return self._states.get(self._key(channel_id, account_id))

    def list_accounts(
        self, channel_id: str | None = None,
    ) -> list[AccountState]:
        """List account states, optionally filtered by channel."""
        if channel_id is None:
            return list(self._states.values())
        return [
            s for s in self._states.values() if s.channel_id == channel_id
        ]

    def get_snapshot(
        self, channel_id: str, account_id: str,
    ) -> ChannelAccountSnapshot | None:
        """Get the connection snapshot for a specific account."""
        state = self._states.get(self._key(channel_id, account_id))
        return state.snapshot if state else None


# ═════════════════════════════════════════════════════════════════════
# Inbound / outbound pipelines (formerly pipeline.py)
# ═════════════════════════════════════════════════════════════════════


class OutboundPipeline:
    """Processes outgoing messages through a middleware chain."""

    def __init__(
        self,
        plugin: ChannelPlugin,
        middlewares: list[OutboundMiddlewareBase],
    ) -> None:
        self.plugin = plugin
        self.middlewares = middlewares

    async def process(
        self,
        message: OutboundMessage,
        context: dict[str, Any] | None = None,
    ) -> OutboundMessage | None:
        """Run *message* through each middleware.  Returns ``None`` if dropped."""
        ctx = context or {}
        current: OutboundMessage | None = message
        for mw in self.middlewares:
            if current is None:
                return None
            current = await mw.process_outbound(current, ctx)
        return current


def build_outbound_pipeline(
    plugin: ChannelPlugin,
    config: Any,
) -> OutboundPipeline:
    """Auto-assemble outbound pipeline based on plugin capabilities.

    FormattingMiddleware has been removed — Channel.send() handles
    formatting + chunking via _format_chunk() / _prepare_chunks().
    """
    middlewares: list[OutboundMiddlewareBase] = []
    return OutboundPipeline(plugin, middlewares)


# ── Per-channel health tracking ──────────────────────────────────────

@dataclass
class ChannelHealth:
    """Tracks send success / failure metrics for a single channel."""

    consecutive_failures: int = 0
    last_failure_time: float | None = None
    last_failure_error: str | None = None
    total_failures: int = 0
    total_successes: int = 0


# ── Minimal HTTP health-check server ────────────────────────────────

class _HealthServer:
    """Zero-dependency HTTP health-check endpoint using ``asyncio.start_server``.

    Responds to ``GET /healthz`` with a JSON status payload; all other
    requests receive a 404.  A per-connection timeout prevents slow
    clients from tying up the server.
    """

    _CONNECTION_TIMEOUT = 5.0  # seconds

    def __init__(self, manager: ChannelManager, port: int) -> None:
        self._manager = manager
        self._port = port
        self._server: asyncio.AbstractServer | None = None
        self._start_time: float = 0.0

    async def start(self) -> None:
        self._start_time = time.monotonic()
        self._server = await asyncio.start_server(
            self._handle_connection, "0.0.0.0", self._port,
        )
        addrs = [s.getsockname() for s in self._server.sockets]
        logger.info(f"Health server listening on {addrs}")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("Health server stopped")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            await asyncio.wait_for(
                self._process_request(reader, writer),
                timeout=self._CONNECTION_TIMEOUT,
            )
        except (asyncio.TimeoutError, ConnectionError, OSError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _process_request(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        request_line = await reader.readline()
        # Consume remaining headers
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break

        parts = request_line.decode("utf-8", errors="replace").split()
        if len(parts) >= 2 and parts[0] == "GET" and parts[1] == "/healthz":
            body = self._build_response()
            payload = json.dumps(body).encode()
            header = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\n"
                "Connection: close\r\n"
                "\r\n"
            )
        else:
            payload = b'{"error":"not found"}'
            header = (
                "HTTP/1.1 404 Not Found\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\n"
                "Connection: close\r\n"
                "\r\n"
            )
        writer.write(header.encode() + payload)
        await writer.drain()

    def _build_response(self) -> dict[str, Any]:
        mgr = self._manager
        health_map: dict[str, Any] = {}
        for name, h in mgr._health.items():
            health_map[name] = {
                "consecutive_failures": h.consecutive_failures,
                "total_successes": h.total_successes,
                "total_failures": h.total_failures,
            }
        accounts_map: dict[str, Any] = {}
        for state in mgr._account_manager.list_accounts():
            key = f"{state.channel_id}:{state.account_id}"
            accounts_map[key] = {
                "account_id": state.account_id,
                "channel": state.channel_id,
                "status": state.status,
                "error": state.error,
            }
        resp: dict[str, Any] = {
            "status": "healthy",
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "channels": {
                "enabled": mgr.enabled_channels,
                "running": mgr.running_channels(),
            },
            "queues": {
                "inbound_size": mgr.bus.inbound_size,
                "outbound_size": mgr.bus.outbound_size,
            },
            "health": health_map,
            "accounts": accounts_map,
        }
        for pname, provider in mgr._health_providers.items():
            try:
                resp[pname] = provider()
            except Exception:
                resp[pname] = {"error": "provider failed"}
        return resp


# ── Channel registry ──────────────────────────────────────────────────

ChannelFactory = Callable[..., Channel]

_CHANNEL_REGISTRY: dict[str, ChannelFactory] = {}


def _parse_csv(value: str) -> set[str] | None:
    """Parse comma-separated string into a set, or ``None`` if empty."""
    if not value or not value.strip():
        return None
    items = {s.strip() for s in value.split(",") if s.strip()}
    return items if items else None


def register_channel(name: str, factory: ChannelFactory) -> None:
    """Register a channel factory under *name*."""
    _CHANNEL_REGISTRY[name] = factory


def create_channel(name: str, config) -> Channel:
    """Create a channel instance using the registered factory for *name*."""
    factory = _CHANNEL_REGISTRY.get(name)
    if not factory:
        raise ValueError(
            f"Unknown channel type: {name}. "
            f"Available: {list(_CHANNEL_REGISTRY.keys())}"
        )
    return factory(config)


def available_channels() -> list[str]:
    """Return the names of all available channel types.

    Triggers auto-discovery if the registry is empty.
    """
    if not _CHANNEL_REGISTRY:
        _ensure_channels_registered()
    return list(_CHANNEL_REGISTRY.keys())


def _discover_channel_subpackages() -> list[str]:
    """Discover all channel sub-packages under the channels directory.

    Returns a list of sub-package names (e.g. ["telegram", "discord", ...]).
    Excludes non-channel directories (bus, __pycache__) and plain modules.
    """
    channels_dir = Path(__file__).parent
    _EXCLUDED = {"bus", "__pycache__"}
    names = []
    for info in pkgutil.iter_modules([str(channels_dir)]):
        if info.ispkg and info.name not in _EXCLUDED:
            names.append(info.name)
    return sorted(names)


def _ensure_channels_registered(types: list[str] | None = None) -> None:
    """Lazily import channel sub-packages to trigger registration.

    If *types* is given, only those channels are imported.
    If *types* is ``None``, all discovered channel sub-packages are imported.
    """
    if types is None:
        targets = _discover_channel_subpackages()
    else:
        # Only import the ones that exist as sub-packages
        available = set(_discover_channel_subpackages())
        targets = [t for t in types if t in available]

    for t in targets:
        module_name = f"EvoScientist.channels.{t}"
        if t not in _CHANNEL_REGISTRY:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                logger.debug(f"Could not import channel {t}: {e}")


# ── Shared webhook server ─────────────────────────────────────────

class SharedWebhookServer:
    """Single aiohttp server that hosts routes from multiple HTTP channels.

    When ``shared_webhook_port`` is configured, ``ChannelManager`` collects
    routes from every channel that exposes ``_webhook_routes()`` and starts
    one server instead of letting each channel bind its own port.
    """

    def __init__(self, port: int) -> None:
        self._port = port
        self._app: Any = None
        self._runner: Any = None
        self._site: Any = None

    async def start(self, routes: list[tuple[str, str, Any]]) -> None:
        from aiohttp import web

        self._app = web.Application()
        for method, path, handler in routes:
            if method.upper() == "GET":
                self._app.router.add_get(path, handler)
            else:
                self._app.router.add_post(path, handler)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "0.0.0.0", self._port)
        await self._site.start()
        logger.info(
            f"Shared webhook server started on 0.0.0.0:{self._port} "
            f"with {len(routes)} route(s)"
        )

    async def stop(self) -> None:
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        logger.info("Shared webhook server stopped")


class ChannelManager:
    """Manages all chat channels and coordinates message routing.

    Responsibilities:
    - Register channels and inject bus reference
    - Start / stop all channels
    - Route outbound messages from the bus to the correct channel
    """

    def __init__(
        self,
        bus: MessageBus,
        *,
        health_port: int = 8080,
        drain_timeout: float = 30.0,
        shared_webhook_port: int = 0,
    ):
        self.bus = bus
        self._channels: dict[str, Channel] = {}
        self._tasks: list[asyncio.Task] = []
        self._dispatch_task: asyncio.Task | None = None
        self._start_times: dict[str, datetime] = {}
        self._message_counts: dict[str, dict[str, int]] = {}
        self._health: dict[str, ChannelHealth] = {}
        self._is_running: bool = False
        self._health_port = health_port
        self._health_server: _HealthServer | None = None
        self._drain_timeout = drain_timeout
        self._health_providers: dict[str, Callable[[], dict]] = {}
        self._account_manager = AccountManager()
        # Pipelines (built during registration)
        self._outbound_pipelines: dict[str, OutboundPipeline] = {}
        # Shared webhook
        self._shared_webhook_port = shared_webhook_port
        self._shared_webhook_server: SharedWebhookServer | None = None

    @classmethod
    def from_config(cls, config, bus: MessageBus | None = None) -> "ChannelManager":
        """Create a ChannelManager from application config.

        Parses ``config.channel_enabled`` (comma-separated channel types),
        creates each Channel instance, and registers them.

        Args:
            config: Application config with channel settings.
            bus: Optional MessageBus instance. A new one is created if not provided.

        Returns:
            A fully configured ChannelManager.
        """
        if bus is None:
            bus = MessageBus()
        shared_webhook_port = getattr(config, "shared_webhook_port", 0) or 0
        manager = cls(bus, shared_webhook_port=shared_webhook_port)
        types = [t.strip() for t in (config.channel_enabled or "").split(",") if t.strip()]
        if not types:
            raise ValueError("No channels enabled")
        _ensure_channels_registered(types)
        for ct in types:
            channel = create_channel(ct, config)
            manager.register(channel, config=config)
        return manager

    # ── registration ──

    def register(
        self,
        channel: Channel,
        *,
        config: Any = None,
        **kwargs: Any,
    ) -> Channel:
        """Register a channel and inject the bus reference.

        Since Channel IS-A ChannelPlugin, the channel is also registered
        in the plugin registry.  If *config* is provided, inbound/outbound
        pipelines are built for the channel.

        Args:
            channel: The channel instance (must have a unique ``name``).
            config: Optional app config for building pipelines.
            **kwargs: Extra kwargs applied to the channel
                (e.g. ``send_thinking=True``, ``initial_debounce=3.0``).

        Returns:
            The channel instance.
        """
        name = channel.name
        if name in self._channels:
            raise ValueError(f"Channel '{name}' already registered")

        channel.set_bus(self.bus)
        for key, value in kwargs.items():
            if hasattr(channel, key):
                setattr(channel, key, value)
        self._channels[name] = channel
        self._health[name] = ChannelHealth()
        if channel.config_adapter is not None:
            self._account_manager.register_plugin(channel)
        if config is not None:
            self._outbound_pipelines[name] = build_outbound_pipeline(channel, config)
        logger.info(f"Registered channel: {name} (slots: {channel.filled_slots()})")
        return channel

    # ── lifecycle ──

    async def start_all(self) -> None:
        """Start the outbound dispatcher and all registered channels."""
        if not self._channels:
            logger.warning("No channels registered")
            return

        self._is_running = True

        await self.start_health()

        # Start shared webhook server before individual channels
        await self._setup_shared_webhook()

        self._dispatch_task = asyncio.create_task(
            self._dispatch_outbound()
        )

        now = datetime.now()
        for name, channel in self._channels.items():
            logger.info(f"Starting channel: {name}")
            self._start_times[name] = now
            if name not in self._message_counts:
                self._message_counts[name] = {"received": 0, "sent": 0}
            task = asyncio.create_task(channel.run())
            self._tasks.append(task)

        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        """Stop all channels and the outbound dispatcher.

        Before shutting down channels, attempts to drain the outbound
        queue so that pending replies are delivered.
        """
        logger.info("Stopping all channels...")
        self._is_running = False

        # Drain outbound queue — try to send pending replies
        drained = 0
        deadline = time.monotonic() + self._drain_timeout
        while time.monotonic() < deadline:
            try:
                msg = self.bus.outbound.get_nowait()
            except asyncio.QueueEmpty:
                break
            channel = self._channels.get(msg.channel)
            if channel and msg.content:
                try:
                    await asyncio.wait_for(
                        channel.send(msg),
                        timeout=max(1.0, deadline - time.monotonic()),
                    )
                    drained += 1
                except Exception:
                    pass
        dropped = self.bus.outbound.qsize()
        if drained or dropped:
            logger.info(f"Outbound drain: {drained} sent, {dropped} dropped")

        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        for name, channel in self._channels.items():
            try:
                channel._running = False
                await channel.stop()
                logger.info(f"Stopped channel: {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")

        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        # Stop shared webhook server
        if self._shared_webhook_server is not None:
            await self._shared_webhook_server.stop()
            self._shared_webhook_server = None

        await self.stop_health()

    # ── health server ──

    async def start_health(self) -> None:
        """Start the HTTP health-check endpoint (if configured)."""
        if self._health_port and self._health_server is None:
            self._health_server = _HealthServer(self, self._health_port)
            await self._health_server.start()

    async def stop_health(self) -> None:
        """Stop the HTTP health-check endpoint."""
        if self._health_server is not None:
            await self._health_server.stop()
            self._health_server = None

    # ── shared webhook ──

    async def _setup_shared_webhook(self) -> None:
        """Collect routes from HTTP channels and start a shared server.

        Only active when ``shared_webhook_port > 0``.  For each channel
        that exposes ``_webhook_routes()``, the routes are gathered and
        a sentinel attribute (``_shared_webhook_server``) is set so the
        channel's own ``start()`` skips creating its own aiohttp server.
        """
        if not self._shared_webhook_port:
            return

        all_routes: list[tuple[str, str, Any]] = []
        for name, channel in self._channels.items():
            routes_fn = getattr(channel, "_webhook_routes", None)
            if routes_fn is None:
                continue
            routes = routes_fn()
            if not routes:
                continue
            # Set sentinel so the channel skips its own server
            channel._shared_webhook_server = True  # type: ignore[attr-defined]
            all_routes.extend(routes)
            logger.debug(
                f"Shared webhook: collected {len(routes)} route(s) "
                f"from '{name}'"
            )

        if not all_routes:
            logger.info("Shared webhook: no HTTP channels found, skipping")
            return

        self._shared_webhook_server = SharedWebhookServer(
            self._shared_webhook_port,
        )
        await self._shared_webhook_server.start(all_routes)

    def register_health_provider(
        self, name: str, provider: Callable[[], dict],
    ) -> None:
        """Register a callable that returns extra data for ``/healthz``."""
        self._health_providers[name] = provider

    # ── outbound routing ──

    async def _dispatch_outbound(self) -> None:
        """Route outbound messages from the bus to the correct channel."""
        logger.info("Outbound dispatcher started")
        while True:
            try:
                msg: OutboundMessage = await asyncio.wait_for(
                    self.bus.consume_outbound(), timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            channel = self._channels.get(msg.channel)

            if not channel:
                logger.warning(f"Unknown channel: {msg.channel}")
                continue

            try:
                # Run outbound pipeline if available (formatting, etc.)
                if msg.channel in self._outbound_pipelines:
                    processed = await self._outbound_pipelines[msg.channel].process(msg)
                    if processed is None:
                        continue  # dropped by pipeline
                    msg = processed

                if msg.content:
                    await channel.send(msg)

                for media_path in msg.media:
                    try:
                        await channel.send_media(
                            recipient=msg.chat_id,
                            file_path=media_path,
                            metadata=msg.metadata,
                        )
                    except Exception as e:
                        logger.error(
                            f"Error sending media to {msg.channel}: {e}"
                        )

                # Success
                health = self._health.get(msg.channel)
                if health is not None:
                    health.consecutive_failures = 0
                    health.total_successes += 1
            except Exception as e:
                logger.error(
                    f"Error sending to {msg.channel}: {e}"
                )
                health = self._health.get(msg.channel)
                if health is not None:
                    health.consecutive_failures += 1
                    health.total_failures += 1
                    health.last_failure_time = time.monotonic()
                    health.last_failure_error = str(e)

    # ── per-account lifecycle ──

    async def start_account(
        self,
        channel_id: str,
        account_id: str,
        config: Any = None,
    ) -> None:
        """Start a specific account on a registered plugin."""
        await self._account_manager.start_account(channel_id, account_id, config)

    async def stop_account(
        self,
        channel_id: str,
        account_id: str,
    ) -> None:
        """Stop a specific account on a registered plugin."""
        await self._account_manager.stop_account(channel_id, account_id)

    def list_accounts(
        self,
        channel_id: str | None = None,
    ) -> list[AccountState]:
        """List account states, optionally filtered by channel."""
        return self._account_manager.list_accounts(channel_id)

    @property
    def account_manager(self) -> AccountManager:
        """Access the underlying AccountManager."""
        return self._account_manager

    # ── queries ──

    def get_channel(self, name: str) -> Channel | None:
        """Get a channel by name."""
        return self._channels.get(name)

    def get_server(self, name: str) -> Channel | None:
        """Backward compat: returns the Channel (was ChannelServer)."""
        return self._channels.get(name)

    def get_status(self) -> dict[str, Any]:
        """Get status of all registered channels."""
        return {
            name: {
                "registered": True,
                "running": channel._running,
                "slots": channel.filled_slots(),
            }
            for name, channel in self._channels.items()
        }

    @property
    def is_running(self) -> bool:
        """Whether the manager is currently running."""
        return self._is_running

    @property
    def enabled_channels(self) -> list[str]:
        """List of registered channel names."""
        return list(self._channels.keys())

    def running_channels(self) -> list[str]:
        """Return names of currently running channels."""
        return [name for name, ch in self._channels.items() if ch._running]

    def get_stats(self) -> dict:
        """Return summary stats for all channels."""
        return {
            "channels": self.enabled_channels,
            "running": self.running_channels(),
            "message_counts": dict(self._message_counts),
        }

    async def add_channel(self, channel_type: str, config) -> Channel:
        """Dynamically add and start a channel at runtime."""
        _ensure_channels_registered([channel_type])
        channel = create_channel(channel_type, config)
        self.register(channel)
        self._start_times[channel_type] = datetime.now()
        if channel_type not in self._message_counts:
            self._message_counts[channel_type] = {"received": 0, "sent": 0}
        task = asyncio.create_task(channel.run())
        self._tasks.append(task)
        return channel

    async def remove_channel(self, channel_type: str) -> None:
        """Stop and remove a channel at runtime."""
        channel = self._channels.pop(channel_type, None)
        if channel:
            channel._running = False
            await channel.stop()
            logger.info(f"Removed channel: {channel_type}")

    def record_message(self, channel_name: str, direction: str) -> None:
        """Record a message for tracking.

        Args:
            channel_name: Channel name (e.g. "telegram").
            direction: "received" or "sent".
        """
        if channel_name not in self._message_counts:
            self._message_counts[channel_name] = {"received": 0, "sent": 0}
        if direction in self._message_counts[channel_name]:
            self._message_counts[channel_name][direction] += 1

    def get_detailed_status(self) -> dict[str, Any]:
        """Get detailed status of all registered channels.

        Returns:
            Dict keyed by channel name with running, start_time, message
            counts, health, and plugin information.
        """
        now = datetime.now()
        result = {}
        for name, channel in self._channels.items():
            start = self._start_times.get(name)
            counts = self._message_counts.get(name, {"received": 0, "sent": 0})
            health = self._health.get(name, ChannelHealth())
            result[name] = {
                "registered": True,
                "running": channel._running,
                "start_time": start,
                "uptime_seconds": (now - start).total_seconds() if start else 0,
                "received": counts["received"],
                "sent": counts["sent"],
                "health": {
                    "consecutive_failures": health.consecutive_failures,
                    "last_failure_time": health.last_failure_time,
                    "last_failure_error": health.last_failure_error,
                    "total_failures": health.total_failures,
                    "total_successes": health.total_successes,
                },
                "plugin_slots": channel.filled_slots(),
                "has_outbound_pipeline": name in self._outbound_pipelines,
            }
        return result
