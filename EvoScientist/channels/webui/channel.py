"""Lightweight WebUI control channel.

Chat state and streaming are owned by LangGraph Dev. This channel only exposes
small browser-facing control endpoints used by the Next.js WebUI.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import signal
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..base import Channel, ChannelError
from ..capabilities import ChannelCapabilities
from ..config import BaseChannelConfig

logger = logging.getLogger(__name__)

_UUID_THREAD_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_MAX_FILE_PREVIEW_BYTES = 256 * 1024
_MAX_WORKSPACE_TREE_ENTRIES = 500
_BINARY_PROBE_BYTES = 8192
_LANGUAGE_BY_SUFFIX = {
    ".css": "css",
    ".go": "go",
    ".html": "html",
    ".java": "java",
    ".js": "javascript",
    ".json": "json",
    ".jsx": "javascript",
    ".md": "markdown",
    ".py": "python",
    ".rs": "rust",
    ".sh": "bash",
    ".sql": "sql",
    ".toml": "ini",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".txt": "plaintext",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
}


@dataclass
class WebUIConfig(BaseChannelConfig):
    bind_host: str = "127.0.0.1"
    webhook_port: int = 8010
    base_path: str = "/webui"
    api_key: str = ""
    workspace_mode: str = "daemon"
    workspace_root: str = ""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _split_host_port(value: str) -> str:
    host = (value or "").strip()
    if host.startswith("["):
        end = host.find("]")
        return host[1:end] if end > 0 else host
    if host.count(":") == 1:
        return host.rsplit(":", 1)[0]
    return host


def _is_safe_web_host(host: str) -> bool:
    normalized = _split_host_port(host).strip().lower().rstrip(".")
    if normalized in {"", "localhost"} or normalized.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return bool(ip.is_loopback or ip.is_private or ip.is_link_local)


def _normalize_relative_path(path: str) -> str:
    normalized = (path or "").strip().replace("\\", "/").lstrip("/")
    if not normalized or normalized == ".":
        return ""
    parts = [part for part in normalized.split("/") if part and part != "."]
    if any(part == ".." for part in parts):
        raise ValueError("path traversal is not allowed")
    return "/".join(parts)


def _looks_binary(data: bytes) -> bool:
    if not data:
        return False
    sample = data[:_BINARY_PROBE_BYTES]
    if b"\x00" in sample:
        return True
    non_printable = sum(
        1 for byte in sample if byte < 32 and byte not in (9, 10, 12, 13)
    )
    return (non_printable / len(sample)) > 0.3


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


class WebUIChannel(Channel):
    """Dedicated local HTTP control channel for the browser WebUI."""

    name = "webui"
    capabilities = ChannelCapabilities(
        format_type="plain",
        max_text_length=999_999,
        streaming=True,
        threading=True,
        chat_types=("direct", "thread"),
    )

    def __init__(self, config: WebUIConfig):
        super().__init__(config)
        self._runner: Any = None
        self._site: Any = None
        self._thread_registry_lock = asyncio.Lock()

    async def start(self) -> None:
        try:
            from aiohttp import web
        except ImportError:
            raise ChannelError(
                "aiohttp not installed. Install with: pip install aiohttp"
            ) from None

        app = web.Application()
        for method, path, handler in self._routes():
            app.router.add_route(method, path, handler)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(
            self._runner,
            self.config.bind_host,
            int(self.config.webhook_port),
        )
        await self._site.start()
        self._running = True
        logger.info(
            "WebUI control channel started on %s:%s%s",
            self.config.bind_host,
            self.config.webhook_port,
            self.config.base_path,
        )

    async def _cleanup(self) -> None:
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def receive(self):
        while self._running:
            await asyncio.sleep(1.0)
            if False:  # pragma: no cover
                yield None

    async def _send_chunk(
        self,
        _chat_id: str,
        _formatted_text: str,
        _raw_text: str,
        _reply_to: str | None,
        _metadata: dict,
    ) -> None:
        return None

    def _route(self, suffix: str) -> str:
        base = str(self.config.base_path or "/webui").strip() or "/webui"
        if not base.startswith("/"):
            base = f"/{base}"
        base = base.rstrip("/") or "/webui"
        return f"{base}{suffix}"

    def _routes(self) -> list[tuple[str, str, Any]]:
        route_specs = [
            ("GET", "/healthz", self._handle_health),
            ("GET", "/runtime/status", self._handle_runtime_status),
            ("POST", "/runtime/prepare", self._handle_runtime_prepare),
            ("POST", "/runtime/release", self._handle_runtime_release),
            (
                "*",
                "/runtime/langgraph/{thread_id}/{tail:.*}",
                self._handle_runtime_langgraph_proxy,
            ),
            ("GET", "/threads", self._handle_threads_list),
            ("POST", "/threads/ensure", self._handle_threads_ensure),
            ("DELETE", "/threads/{thread_id}", self._handle_threads_delete),
            ("GET", "/commands/catalog", self._handle_commands_catalog),
            ("POST", "/commands/execute", self._handle_commands_execute),
            ("GET", "/ui/stats", self._handle_ui_stats),
            ("GET", "/ui/skills", self._handle_ui_skills),
            ("POST", "/ui/skills/install", self._handle_ui_skills_install),
            ("POST", "/ui/skills/uninstall", self._handle_ui_skills_uninstall),
            ("GET", "/ui/models", self._handle_ui_models),
            ("POST", "/ui/models/switch", self._handle_ui_models_switch),
            ("POST", "/ui/providers/key", self._handle_ui_provider_key),
            ("POST", "/ui/providers/base-url", self._handle_ui_provider_base_url),
            ("GET", "/ui/mcp", self._handle_ui_mcp),
            ("POST", "/ui/mcp/install", self._handle_ui_mcp_install),
            ("POST", "/ui/mcp/remove", self._handle_ui_mcp_remove),
            ("GET", "/ui/channels", self._handle_ui_channels),
            ("POST", "/ui/channels/start", self._handle_ui_channels_start),
            ("POST", "/ui/channels/stop", self._handle_ui_channels_stop),
            ("GET", "/ui/files/tree", self._handle_ui_files_tree),
            ("GET", "/ui/files/read", self._handle_ui_files_read),
            ("GET", "/ui/files/download-all", self._handle_ui_files_download_all),
            ("POST", "/ui/session/shutdown", self._handle_ui_session_shutdown),
        ]
        routes: list[tuple[str, str, Any]] = []
        for method, suffix, handler in route_specs:
            if method != "*":
                routes.append(("OPTIONS", self._route(suffix), self._handle_options))
            routes.append((method, self._route(suffix), handler))
        return routes

    def _cors_headers(self, request) -> dict[str, str]:
        origin = request.headers.get("Origin", "")
        allow_origin = (
            origin
            if origin and self._is_origin_allowed(origin)
            else ("*" if not origin else "")
        )
        headers = {
            "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": (
                "Content-Type, Authorization, X-API-Key, X-Thread-Id"
            ),
            "Access-Control-Expose-Headers": "Content-Disposition",
            "Access-Control-Max-Age": "86400",
            "Vary": "Origin",
        }
        if allow_origin:
            headers["Access-Control-Allow-Origin"] = allow_origin
        return headers

    def _request_host_allowed(self, request) -> bool:
        host = request.headers.get("Host", "")
        configured = (self.config.bind_host or "").strip().lower()
        if configured not in {"", "0.0.0.0", "::"}:
            configured_hosts = {configured, "localhost"}
            if configured in {"127.0.0.1", "::1"}:
                configured_hosts.update({"127.0.0.1", "::1"})
            return (
                _split_host_port(host).strip().lower().rstrip(".") in configured_hosts
            )
        return _is_safe_web_host(host)

    def _is_origin_allowed(self, origin: str) -> bool:
        try:
            parsed = urlparse(origin)
        except Exception:
            return False
        return parsed.scheme in {"http", "https"} and _is_safe_web_host(parsed.netloc)

    def _request_origin_allowed(self, request) -> bool:
        origin = request.headers.get("Origin", "")
        return not origin or self._is_origin_allowed(origin)

    def _check_auth(self, request) -> bool:
        if not self._request_host_allowed(request) or not self._request_origin_allowed(
            request
        ):
            return False
        expected = (self.config.api_key or "").strip()
        if not expected:
            return True
        supplied = (
            request.headers.get("X-API-Key")
            or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        )
        return supplied == expected

    async def _handle_options(self, request):
        from aiohttp import web

        if not self._request_host_allowed(request) or not self._request_origin_allowed(
            request
        ):
            return web.Response(status=403, headers={"Vary": "Origin"})
        return web.Response(status=204, headers=self._cors_headers(request))

    def _json(self, request, payload: Any, *, status: int = 200):
        from aiohttp import web

        return web.json_response(
            _json_safe(payload),
            status=status,
            headers=self._cors_headers(request),
        )

    def _unauthorized(self, request):
        return self._json(request, {"error": "unauthorized"}, status=401)

    async def _request_payload(self, request) -> dict[str, Any]:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    def _extract_thread_id(self, request, payload: dict[str, Any] | None = None) -> str:
        payload = payload or {}
        return str(
            payload.get("threadId")
            or payload.get("thread_id")
            or request.query.get("threadId")
            or request.headers.get("X-Thread-Id")
            or ""
        ).strip()

    async def _handle_health(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        return self._json(request, {"ok": True, "channel": "webui"})

    async def _runtime_status_payload(self) -> dict[str, Any]:
        base_url = self._global_langgraph_base_url()
        return {
            "isRunning": False,
            "activeThreadId": None,
            "activeThreadIds": [],
            "activeWorkspaceDir": str(self._workspace_root()),
            "activeWorkspaces": {},
            "createdAt": None,
            "langGraphBaseUrl": base_url,
            "langGraphApiUrl": "/api/langgraph",
            "workspaceDir": str(self._workspace_root()),
        }

    async def _handle_runtime_status(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        return self._json(request, await self._runtime_status_payload())

    async def _handle_runtime_prepare(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        thread_id = self._extract_thread_id(request, payload)
        if thread_id and not _UUID_THREAD_ID_RE.match(thread_id):
            return self._json(
                request,
                {"error": "direct WebUI runtime requires a UUID threadId"},
                status=400,
            )
        if thread_id:
            await self._ensure_thread_record(thread_id)
        workspace_dir = str(self._workspace_root())
        base_url = self._global_langgraph_base_url()

        return self._json(
            request,
            {
                "ok": True,
                "threadId": thread_id,
                "workspaceDir": workspace_dir,
                "leaseId": None,
                "langGraphBaseUrl": base_url,
                "langGraphApiUrl": "/api/langgraph",
                "runtimeStatus": await self._runtime_status_payload(),
            },
        )

    async def _handle_threads_list(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        async with self._thread_registry_lock:
            threads = await self._load_thread_registry_unlocked()
            await self._save_thread_registry_unlocked(threads)
        return self._json(
            request,
            sorted(
                threads,
                key=lambda item: str(
                    item.get("createdAt") or item.get("updatedAt") or ""
                ),
                reverse=True,
            ),
        )

    async def _handle_threads_ensure(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        thread_id = self._extract_thread_id(request, payload)
        if not _UUID_THREAD_ID_RE.match(thread_id):
            return self._json(
                request,
                {"error": "direct WebUI runtime requires a UUID threadId"},
                status=400,
            )
        title = str(payload.get("title") or "").strip() or None
        title_source = (
            str(payload.get("titleSource") or payload.get("title_source") or "").strip()
            or None
        )
        try:
            record = await self._ensure_thread_record(
                thread_id,
                title=title,
                title_source=title_source,
            )
        except Exception as exc:
            logger.exception("Failed to ensure WebUI thread")
            return self._json(request, {"error": str(exc)}, status=409)
        return self._json(request, {"ok": True, **record})

    async def _handle_threads_delete(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        thread_id = str(request.match_info.get("thread_id") or "").strip()
        if not _UUID_THREAD_ID_RE.match(thread_id):
            return self._json(
                request,
                {"error": "direct WebUI runtime requires a UUID threadId"},
                status=400,
            )
        async with self._thread_registry_lock:
            threads = await self._load_thread_registry_unlocked()
            next_threads = [
                item for item in threads if item.get("threadId") != thread_id
            ]
            await self._save_thread_registry_unlocked(next_threads)
        return self._json(
            request, {"ok": True, "deleted": len(next_threads) != len(threads)}
        )

    async def _handle_runtime_release(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        return self._json(
            request,
            {
                "ok": True,
                "released": False,
                "runtimeStatus": await self._runtime_status_payload(),
            },
        )

    async def _handle_runtime_langgraph_proxy(self, request):
        from aiohttp import ClientSession, ClientTimeout, web

        if request.method == "OPTIONS":
            return await self._handle_options(request)
        if not self._check_auth(request):
            return self._unauthorized(request)

        target_base = self._global_langgraph_base_url()

        tail = str(request.match_info.get("tail") or "").lstrip("/")
        target_url = f"{target_base}/{tail}"
        if request.query_string:
            target_url = f"{target_url}?{request.query_string}"

        headers = self._proxied_request_headers(request.headers)

        body = await request.read()
        timeout = ClientTimeout(total=None, sock_connect=30)
        async with ClientSession(timeout=timeout) as session:
            async with session.request(
                request.method,
                target_url,
                headers=headers,
                data=body if request.method not in {"GET", "HEAD"} else None,
            ) as upstream:
                response_headers = self._cors_headers(request)
                for key, value in upstream.headers.items():
                    if key.lower() not in {
                        "connection",
                        "content-encoding",
                        "content-length",
                        "keep-alive",
                        "proxy-authenticate",
                        "proxy-authorization",
                        "te",
                        "trailer",
                        "transfer-encoding",
                        "upgrade",
                    }:
                        response_headers[key] = value
                response_headers["Cache-Control"] = "no-store"

                stream = web.StreamResponse(
                    status=upstream.status,
                    reason=upstream.reason,
                    headers=response_headers,
                )
                await stream.prepare(request)
                async for chunk in upstream.content.iter_chunked(65536):
                    await stream.write(chunk)
                await stream.write_eof()
                return stream

    def _proxied_request_headers(self, request_headers) -> dict[str, str]:
        excluded = {
            "authorization",
            "connection",
            "content-encoding",
            "content-length",
            "host",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
            "x-api-key",
        }
        return {
            key: value
            for key, value in request_headers.items()
            if key.lower() not in excluded
        }

    def _workspace_root(self) -> Path:
        configured = (self.config.workspace_root or "").strip()
        return Path(configured).expanduser().resolve() if configured else Path.cwd()

    def _thread_registry_path(self) -> Path:
        return self._workspace_root() / ".evoscientist_webui_threads.json"

    def _load_thread_registry_sync(self) -> list[dict[str, Any]]:
        path = self._thread_registry_path()
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            raw_threads = []
        else:
            raw_threads = (
                payload.get("threads") if isinstance(payload, dict) else payload
            )
            if not isinstance(raw_threads, list):
                raw_threads = []
        threads_by_id: dict[str, dict[str, Any]] = {}
        for item in raw_threads:
            if not isinstance(item, dict):
                continue
            thread_id = str(item.get("threadId") or item.get("thread_id") or "").strip()
            if not _UUID_THREAD_ID_RE.match(thread_id):
                continue
            title = str(item.get("title") or f"Session {thread_id}").strip()
            created_at = str(
                item.get("createdAt") or item.get("created_at") or ""
            ).strip()
            updated_at = str(
                item.get("updatedAt") or item.get("updated_at") or ""
            ).strip()
            workspace_dir = str(item.get("workspaceDir") or "").strip()
            title_source = str(
                item.get("titleSource") or item.get("title_source") or ""
            ).strip()
            threads_by_id[thread_id] = {
                "threadId": thread_id,
                "title": title or f"Session {thread_id}",
                "titleSource": title_source or "default",
                "createdAt": created_at or updated_at or _utc_now_iso(),
                "updatedAt": updated_at or created_at or _utc_now_iso(),
                "workspaceDir": workspace_dir
                or str(self._workspace_path_for_thread(thread_id)),
                "legacy": False,
                "readOnly": False,
            }

        return list(threads_by_id.values())

    def _save_thread_registry_sync(self, threads: list[dict[str, Any]]) -> None:
        path = self._thread_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(
            threads,
            key=lambda item: str(item.get("createdAt") or item.get("updatedAt") or ""),
            reverse=True,
        )
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(
                    json.dumps({"threads": ordered}, indent=2, sort_keys=True)
                )
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    async def _load_thread_registry_unlocked(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._load_thread_registry_sync)

    async def _save_thread_registry_unlocked(
        self, threads: list[dict[str, Any]]
    ) -> None:
        await asyncio.to_thread(self._save_thread_registry_sync, threads)

    async def _load_thread_registry(self) -> list[dict[str, Any]]:
        async with self._thread_registry_lock:
            return await self._load_thread_registry_unlocked()

    async def _save_thread_registry(self, threads: list[dict[str, Any]]) -> None:
        async with self._thread_registry_lock:
            await self._save_thread_registry_unlocked(threads)

    async def _ensure_thread_record(
        self,
        thread_id: str,
        *,
        title: str | None = None,
        title_source: str | None = None,
        touch: bool = True,
    ) -> dict[str, Any]:
        if not _UUID_THREAD_ID_RE.match(thread_id):
            raise ValueError("direct WebUI runtime requires a UUID threadId")
        workspace_dir = await self._resolve_or_create_workspace(thread_id)
        async with self._thread_registry_lock:
            threads = await self._load_thread_registry_unlocked()
            now = _utc_now_iso()
            existing_index = next(
                (
                    index
                    for index, item in enumerate(threads)
                    if item.get("threadId") == thread_id
                ),
                None,
            )
            if existing_index is None:
                record = {
                    "threadId": thread_id,
                    "title": title or f"Session {thread_id}",
                    "titleSource": title_source or ("manual" if title else "default"),
                    "createdAt": now,
                    "updatedAt": now,
                    "workspaceDir": workspace_dir,
                    "legacy": False,
                    "readOnly": False,
                }
                threads.append(record)
            else:
                record = {
                    **threads[existing_index],
                    "workspaceDir": workspace_dir,
                    "title": title
                    or threads[existing_index].get("title")
                    or f"Session {thread_id}",
                    "titleSource": title_source
                    or threads[existing_index].get("titleSource")
                    or "default",
                }
                if touch:
                    record["updatedAt"] = now
                threads[existing_index] = record
            await self._save_thread_registry_unlocked(threads)
            return record

    def _workspace_path_for_thread(self, thread_id: str) -> Path:
        return self._workspace_root()

    async def _resolve_or_create_workspace(self, thread_id: str) -> str:
        workspace = self._workspace_path_for_thread(thread_id)
        await asyncio.to_thread(workspace.mkdir, parents=True, exist_ok=True)
        return str(workspace)

    def _global_langgraph_base_url(self) -> str:
        from ...config.settings import get_effective_config

        cfg = get_effective_config()
        port = int(getattr(cfg, "langgraph_dev_port", 6174) or 6174)
        return f"http://localhost:{port}"

    def _command_catalog(self) -> list[dict[str, Any]]:
        commands = {
            "/help": ("Show available WebUI commands", "show_help"),
            "/new": ("Create a new LangGraph thread", "new_thread"),
            "/threads": ("Open the thread picker", "show_threads"),
            "/resume": ("Switch to a thread", "switch_thread"),
            "/delete": ("Delete a thread", "delete_thread"),
            "/clear": ("Clear the current WebUI view", "clear_thread"),
            "/model": ("Open or switch model", "switch_model"),
            "/model-fallback": ("Configure model fallbacks", "switch_model"),
            "/skills": ("Open installed skills", "show_skills"),
            "/evoskills": ("Open skill browser", "browse_skills"),
            "/install-skill": ("Install a skill", "install_skill"),
            "/uninstall-skill": ("Uninstall a skill", "uninstall_skill"),
            "/mcp": ("Open MCP manager", "manage_mcp"),
            "/install-mcp": ("Open MCP marketplace", "browse_mcp"),
            "/channel": ("Open channel manager", "manage_channels"),
            "/current": ("Show current WebUI runtime settings", "show_current"),
            "/compact": ("Compact conversation context", None),
            "/exit": ("Stop the backend process", "exit_session"),
        }
        return [
            {
                "name": name,
                "description": description,
                "aliases": ["/quit", "/q"] if name == "/exit" else [],
                "arguments": [],
                "nativeAction": action,
            }
            for name, (description, action) in commands.items()
        ]

    async def _handle_commands_catalog(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        return self._json(request, {"commands": self._command_catalog()})

    async def _handle_commands_execute(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        command = str(payload.get("command") or "").strip()
        thread_id = self._extract_thread_id(request, payload)
        workspace_dir = await self._resolve_or_create_workspace(
            thread_id or uuid.uuid4().hex
        )
        if not command.startswith("/"):
            command = f"/{command}" if command else ""
        name, *args = command.split()
        name = name.lower()
        outputs: list[dict[str, str]] = []
        actions: list[dict[str, Any]] = []

        if name in {"/exit", "/quit", "/q"}:
            actions.append({"type": "request_quit"})
            outputs.append({"kind": "system", "text": "Shutdown requested."})
        elif name == "/clear":
            actions.append({"type": "clear_chat"})
            outputs.append({"kind": "system", "text": "Cleared the current view."})
        elif name == "/new":
            actions.append({"type": "start_new_session"})
            outputs.append({"kind": "system", "text": "Created a new WebUI thread."})
        elif name == "/resume" and args:
            actions.append({"type": "resume_session", "threadId": args[0]})
            outputs.append({"kind": "system", "text": f"Switching to {args[0]}."})
        elif name == "/delete" and args:
            actions.append({"type": "delete_thread", "threadId": args[0]})
            outputs.append(
                {"kind": "system", "text": f"Delete requested for {args[0]}."}
            )
        elif name == "/current":
            cfg = self._load_config()
            outputs.append(
                {
                    "kind": "system",
                    "text": (
                        f"Model: {cfg.model} ({cfg.provider})\n"
                        f"Workspace: {workspace_dir}\n"
                        "Runtime: LangGraph Dev"
                    ),
                }
            )
        elif name == "/help":
            outputs.append(
                {
                    "kind": "system",
                    "text": "\n".join(
                        f"{item['name']} - {item['description']}"
                        for item in self._command_catalog()
                    ),
                }
            )
        elif name == "/model" and args:
            model = args[0]
            provider = args[1] if len(args) > 1 else ""
            if not provider:
                from ...llm.models import MODELS

                provider = MODELS.get(model, ("", ""))[1]
            if not provider:
                return self._json(
                    request,
                    {
                        "ok": False,
                        "executed": False,
                        "command": command,
                        "outputs": [],
                        "actions": [],
                        "threadId": thread_id,
                        "workspaceDir": workspace_dir,
                        "error": f"Unknown model '{model}'. Specify provider explicitly.",
                    },
                    status=400,
                )
            from ...config.settings import set_config_value

            set_config_value("model", model)
            set_config_value("provider", provider)
            outputs.append(
                {
                    "kind": "system",
                    "text": f"Switched model to {model} ({provider}). Restart the LangGraph run to apply it.",
                }
            )
        elif name == "/install-skill" and args:
            from ...tools.skills_manager import install_skill

            result = await asyncio.to_thread(install_skill, " ".join(args))
            if not result.get("success"):
                return self._json(
                    request,
                    {
                        "ok": False,
                        "executed": False,
                        "command": command,
                        "outputs": [],
                        "actions": [],
                        "threadId": thread_id,
                        "workspaceDir": workspace_dir,
                        "error": str(result.get("error") or "Skill install failed."),
                    },
                    status=400,
                )
            outputs.append({"kind": "system", "text": "Skill installed."})
        elif name == "/uninstall-skill" and args:
            from ...tools.skills_manager import uninstall_skill

            result = await asyncio.to_thread(uninstall_skill, args[0])
            if not result.get("success"):
                return self._json(
                    request,
                    {
                        "ok": False,
                        "executed": False,
                        "command": command,
                        "outputs": [],
                        "actions": [],
                        "threadId": thread_id,
                        "workspaceDir": workspace_dir,
                        "error": str(result.get("error") or "Skill uninstall failed."),
                    },
                    status=400,
                )
            outputs.append({"kind": "system", "text": f"Uninstalled {args[0]}."})
        elif name == "/install-mcp" and args:
            from ...mcp.registry import fetch_marketplace_index, find_server_by_name
            from ...mcp.registry import install_mcp_servers as install_entries

            index = await asyncio.to_thread(fetch_marketplace_index)
            selected = [
                entry
                for entry in (find_server_by_name(arg, index) for arg in args)
                if entry is not None
            ]
            count = await asyncio.to_thread(install_entries, selected)
            outputs.append(
                {"kind": "system", "text": f"Configured {count} MCP server(s)."}
            )
        elif name == "/mcp" and len(args) >= 2 and args[0].lower() == "remove":
            from ...mcp import remove_mcp_server

            removed = await asyncio.to_thread(remove_mcp_server, args[1])
            if not removed:
                return self._json(
                    request,
                    {
                        "ok": False,
                        "executed": False,
                        "command": command,
                        "outputs": [],
                        "actions": [],
                        "threadId": thread_id,
                        "workspaceDir": workspace_dir,
                        "error": f"MCP server '{args[1]}' not found.",
                    },
                    status=400,
                )
            outputs.append({"kind": "system", "text": f"Removed MCP server {args[1]}."})
        elif name == "/compact":
            return self._json(
                request,
                {
                    "ok": False,
                    "executed": False,
                    "command": command,
                    "outputs": [],
                    "actions": [],
                    "threadId": thread_id,
                    "workspaceDir": workspace_dir,
                    "error": "/compact is not available in LangGraph WebUI yet.",
                },
                status=400,
            )
        else:
            outputs.append(
                {
                    "kind": "system",
                    "text": f"{name or command} is handled by native WebUI controls.",
                }
            )
        return self._json(
            request,
            {
                "ok": True,
                "executed": True,
                "command": command,
                "outputs": outputs,
                "actions": actions,
                "threadId": thread_id,
                "workspaceDir": workspace_dir,
            },
        )

    def _load_config(self):
        from ...config.settings import load_config

        return load_config()

    async def _handle_ui_stats(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        skills = self._list_installed_skills()
        memories = self._list_memories()
        return self._json(
            request,
            {
                "skills": {"total": len(skills), "items": skills},
                "memories": {"total": len(memories), "items": memories},
            },
        )

    def _list_installed_skills(self) -> list[dict[str, Any]]:
        from ...tools.skills_manager import list_skills

        return [
            {
                "key": skill.name,
                "name": skill.name,
                "description": skill.description,
                "source": skill.source,
                "tags": skill.tags,
                "path": str(skill.path),
                "updatedAt": self._mtime_iso(skill.path / "SKILL.md"),
            }
            for skill in list_skills(include_system=True)
        ]

    def _list_memories(self) -> list[dict[str, Any]]:
        from ... import paths

        root = Path(paths.MEMORIES_DIR).expanduser()
        if not root.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for entry in root.rglob("*"):
            if not entry.is_file() or entry.is_symlink():
                continue
            try:
                relative = entry.relative_to(root).as_posix()
                stats = entry.stat()
            except OSError:
                continue
            if any(part.startswith(".") for part in relative.split("/")):
                continue
            items.append(
                {
                    "key": relative,
                    "path": relative,
                    "size": int(stats.st_size),
                    "updatedAt": self._mtime_iso(entry),
                }
            )
        return sorted(items, key=lambda item: item["path"].lower())

    @staticmethod
    def _mtime_iso(path: Path) -> str | None:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
        except OSError:
            return None

    async def _handle_ui_skills(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        installed = self._list_installed_skills()
        installed_names = {item["name"] for item in installed}
        marketplace: list[dict[str, Any]] = []
        marketplace_error: str | None = None
        try:
            from ...tools.skills_manager import fetch_remote_skill_index

            index = await asyncio.to_thread(fetch_remote_skill_index)
            marketplace = [
                {
                    "name": item.get("name", ""),
                    "description": item.get("description", ""),
                    "tags": item.get("tags", []),
                    "installSource": item.get("install_source", ""),
                    "installed": item.get("name") in installed_names,
                }
                for item in index
            ]
        except Exception as exc:
            marketplace_error = str(exc)
        return self._json(
            request,
            {
                "installed": installed,
                "marketplace": marketplace,
                "marketplaceError": marketplace_error,
            },
        )

    async def _handle_ui_skills_install(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        source = str(payload.get("source") or payload.get("name") or "").strip()
        if not source:
            return self._json(request, {"error": "source is required"}, status=400)
        from ...tools.skills_manager import install_skill

        result = await asyncio.to_thread(install_skill, source)
        return self._json(request, {"ok": bool(result.get("success")), **result})

    async def _handle_ui_skills_uninstall(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        name = str(payload.get("name") or "").strip()
        if not name:
            return self._json(request, {"error": "name is required"}, status=400)
        from ...tools.skills_manager import uninstall_skill

        result = await asyncio.to_thread(uninstall_skill, name)
        return self._json(request, {"ok": bool(result.get("success")), **result})

    async def _handle_ui_models(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        return self._json(request, await asyncio.to_thread(self._models_overview))

    def _models_overview(self) -> dict[str, Any]:
        from ...llm.models import list_models_by_provider

        cfg = self._load_config()
        provider_meta = self._provider_metadata(cfg)
        by_provider: dict[str, list[dict[str, Any]]] = {}
        for name, model_id, provider in list_models_by_provider():
            by_provider.setdefault(provider, []).append(
                {
                    "name": name,
                    "modelId": model_id,
                    "provider": provider,
                    "current": name == cfg.model and provider == cfg.provider,
                }
            )
        providers = []
        for provider, models in sorted(by_provider.items()):
            meta = provider_meta.get(provider, {})
            providers.append({"provider": provider, "models": models, **meta})
        return {
            "current": {"model": cfg.model, "provider": cfg.provider},
            "providers": providers,
            "ollama": {
                "baseUrl": getattr(cfg, "ollama_base_url", ""),
                "detected": [],
                "error": None,
            },
        }

    def _provider_metadata(self, cfg) -> dict[str, dict[str, Any]]:
        api_fields = {
            "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
            "openai": ("openai_api_key", "OPENAI_API_KEY"),
            "google-genai": ("google_api_key", "GOOGLE_API_KEY"),
            "nvidia": ("nvidia_api_key", "NVIDIA_API_KEY"),
            "minimax": ("minimax_api_key", "MINIMAX_API_KEY"),
            "siliconflow": ("siliconflow_api_key", "SILICONFLOW_API_KEY"),
            "openrouter": ("openrouter_api_key", "OPENROUTER_API_KEY"),
            "deepseek": ("deepseek_api_key", "DEEPSEEK_API_KEY"),
            "zhipu": ("zhipu_api_key", "ZHIPU_API_KEY"),
            "zhipu-code": ("zhipu_api_key", "ZHIPU_API_KEY"),
            "volcengine": ("volcengine_api_key", "VOLCENGINE_API_KEY"),
            "dashscope": ("dashscope_api_key", "DASHSCOPE_API_KEY"),
            "dashscope-code": ("dashscope_api_key", "DASHSCOPE_API_KEY"),
            "moonshot": ("moonshot_api_key", "MOONSHOT_API_KEY"),
            "kimi-coding": ("kimi_api_key", "KIMI_API_KEY"),
            "custom-openai": ("custom_openai_api_key", "CUSTOM_OPENAI_API_KEY"),
            "custom-anthropic": (
                "custom_anthropic_api_key",
                "CUSTOM_ANTHROPIC_API_KEY",
            ),
        }
        base_fields = {
            "anthropic": ("anthropic_base_url", "ANTHROPIC_BASE_URL"),
            "minimax": ("minimax_base_url", "MINIMAX_BASE_URL"),
            "custom-openai": ("custom_openai_base_url", "CUSTOM_OPENAI_BASE_URL"),
            "custom-anthropic": (
                "custom_anthropic_base_url",
                "CUSTOM_ANTHROPIC_BASE_URL",
            ),
            "ollama": ("ollama_base_url", "OLLAMA_BASE_URL"),
        }
        providers = set(api_fields) | set(base_fields) | {"ollama"}
        result: dict[str, dict[str, Any]] = {}
        for provider in providers:
            api_field, api_env = api_fields.get(provider, ("", ""))
            base_field, base_env = base_fields.get(provider, ("", ""))
            has_key = bool(
                api_field and (getattr(cfg, api_field, "") or os.getenv(api_env))
            )
            result[provider] = {
                "displayName": provider,
                "needsApiKey": bool(api_field),
                "hasApiKey": has_key,
                "hasCredential": has_key or provider == "ollama",
                "apiKeyEnvVar": api_env or None,
                "baseUrlEnvVar": base_env or None,
                "baseUrl": getattr(cfg, base_field, "") if base_field else "",
                "supportsApiKeyUpdate": bool(api_field),
                "supportsBaseUrlUpdate": bool(base_field),
            }
        return result

    async def _handle_ui_models_switch(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        model = str(payload.get("model") or payload.get("name") or "").strip()
        provider = str(payload.get("provider") or "").strip()
        if not model or not provider:
            return self._json(
                request, {"error": "model and provider are required"}, status=400
            )
        from ...config.settings import set_config_value

        set_config_value("model", model)
        set_config_value("provider", provider)
        return self._json(
            request,
            {
                "ok": True,
                "message": "Model updated. Restart the LangGraph run to apply it.",
                "models": self._models_overview(),
            },
        )

    async def _handle_ui_provider_key(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        provider = str(payload.get("provider") or "").strip()
        api_key = str(payload.get("apiKey") or payload.get("api_key") or "").strip()
        field = (
            self._provider_metadata(self._load_config())
            .get(provider, {})
            .get("apiKeyEnvVar")
        )
        if not field:
            return self._json(request, {"error": "unsupported provider"}, status=400)
        config_field = field.lower()
        from ...config.settings import set_config_value

        set_config_value(config_field, api_key)
        return self._json(request, {"ok": True, "models": self._models_overview()})

    async def _handle_ui_provider_base_url(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        provider = str(payload.get("provider") or "").strip()
        base_url = str(payload.get("baseUrl") or payload.get("base_url") or "").strip()
        meta = self._provider_metadata(self._load_config()).get(provider, {})
        env_var = meta.get("baseUrlEnvVar")
        if not env_var:
            return self._json(request, {"error": "unsupported provider"}, status=400)
        from ...config.settings import set_config_value

        set_config_value(str(env_var).lower(), base_url)
        return self._json(request, {"ok": True, "models": self._models_overview()})

    async def _handle_ui_mcp(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        return self._json(request, await asyncio.to_thread(self._mcp_overview))

    def _mcp_overview(self) -> dict[str, Any]:
        from ...mcp import load_mcp_config

        configured_raw = load_mcp_config()
        configured = []
        for name, server in configured_raw.items():
            configured.append(
                {
                    "name": name,
                    "transport": server.get("transport", ""),
                    "command": server.get("command"),
                    "args": server.get("args"),
                    "url": server.get("url"),
                    "tools": server.get("tools"),
                    "exposeTo": server.get("expose_to"),
                }
            )
        configured_names = {item["name"] for item in configured}
        marketplace = []
        marketplace_error = None
        try:
            from ...mcp.registry import fetch_marketplace_index

            for entry in fetch_marketplace_index():
                marketplace.append(
                    {
                        "name": entry.name,
                        "label": entry.label,
                        "description": entry.description,
                        "tags": entry.tags,
                        "transport": entry.transport,
                        "installed": entry.name in configured_names,
                        "configured": entry.name in configured_names,
                        "envKey": entry.env_key or "",
                        "envHint": entry.env_hint,
                        "envOptional": entry.env_optional,
                    }
                )
        except Exception as exc:
            marketplace_error = str(exc)
        return {
            "configured": configured,
            "marketplace": marketplace,
            "marketplaceError": marketplace_error,
        }

    async def _handle_ui_mcp_install(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        names = payload.get("names") or payload.get("name")
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, list) or not names:
            return self._json(request, {"error": "names are required"}, status=400)

        from ...mcp.registry import fetch_marketplace_index, find_server_by_name
        from ...mcp.registry import install_mcp_servers as install_entries

        index = await asyncio.to_thread(fetch_marketplace_index)
        entries = [find_server_by_name(str(name), index) for name in names]
        selected = [entry for entry in entries if entry is not None]
        count = await asyncio.to_thread(install_entries, selected)
        return self._json(
            request,
            {
                "ok": count == len(selected),
                "installed": count,
                "mcp": self._mcp_overview(),
            },
        )

    async def _handle_ui_mcp_remove(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        name = str(payload.get("name") or "").strip()
        if not name:
            return self._json(request, {"error": "name is required"}, status=400)
        from ...mcp import remove_mcp_server

        removed = await asyncio.to_thread(remove_mcp_server, name)
        return self._json(request, {"ok": removed, "mcp": self._mcp_overview()})

    async def _handle_ui_channels(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        return self._json(request, self._channels_overview())

    def _channels_overview(self) -> dict[str, Any]:
        from ..channel_manager import available_channels

        configured = [
            part.strip()
            for part in str(self._load_config().channel_enabled or "").split(",")
            if part.strip()
        ]
        return {
            "available": available_channels(),
            "configured": configured,
            "running": ["webui"] if "webui" in configured else [],
            "isRunning": True,
            "details": {"webui": {"running": True}},
            "hasAgent": False,
            "threadId": None,
        }

    async def _handle_ui_channels_start(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        name = str(payload.get("name") or payload.get("channel") or "").strip()
        cfg = self._load_config()
        enabled = [
            part.strip() for part in cfg.channel_enabled.split(",") if part.strip()
        ]
        if name and name not in enabled:
            enabled.append(name)
        from ...config.settings import set_config_value

        set_config_value("channel_enabled", ",".join(enabled))
        return self._json(request, {"ok": True, "channels": self._channels_overview()})

    async def _handle_ui_channels_stop(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        name = str(payload.get("name") or payload.get("channel") or "").strip()
        cfg = self._load_config()
        enabled = [
            part.strip()
            for part in cfg.channel_enabled.split(",")
            if part.strip() and part.strip() != name
        ]
        from ...config.settings import set_config_value

        set_config_value("channel_enabled", ",".join(enabled))
        return self._json(request, {"ok": True, "channels": self._channels_overview()})

    def _resolve_workspace_path(
        self, workspace_dir: str, relative_path: str
    ) -> tuple[Path, str]:
        root = Path(workspace_dir).expanduser().resolve()
        normalized = _normalize_relative_path(relative_path)
        candidate = root if not normalized else (root / normalized).resolve()
        candidate.relative_to(root)
        return candidate, normalized

    async def _handle_ui_files_tree(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        thread_id = self._extract_thread_id(request)
        path = request.query.get("path", "")
        workspace_dir = await self._resolve_or_create_workspace(
            thread_id or uuid.uuid4().hex
        )
        root_path, normalized = self._resolve_workspace_path(workspace_dir, path)
        if not root_path.exists() or not root_path.is_dir():
            return self._json(request, {"error": "directory not found"}, status=404)
        entries = []
        for entry in sorted(
            root_path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())
        ):
            if entry.is_symlink():
                continue
            try:
                relative = (
                    entry.resolve()
                    .relative_to(Path(workspace_dir).resolve())
                    .as_posix()
                )
                stats = entry.stat()
            except OSError:
                continue
            entries.append(
                {
                    "name": entry.name,
                    "relativePath": relative,
                    "kind": "directory" if entry.is_dir() else "file",
                    "size": None if entry.is_dir() else int(stats.st_size),
                    "modifiedAt": datetime.fromtimestamp(
                        stats.st_mtime, UTC
                    ).isoformat(),
                }
            )
            if len(entries) >= _MAX_WORKSPACE_TREE_ENTRIES:
                break
        return self._json(
            request,
            {
                "ok": True,
                "threadId": thread_id,
                "workspaceDir": workspace_dir,
                "path": normalized,
                "entries": entries,
            },
        )

    async def _handle_ui_files_read(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        thread_id = self._extract_thread_id(request)
        relative_path = request.query.get("path", "")
        workspace_dir = await self._resolve_or_create_workspace(
            thread_id or uuid.uuid4().hex
        )
        file_path, normalized = self._resolve_workspace_path(
            workspace_dir, relative_path
        )
        if not file_path.exists() or not file_path.is_file():
            return self._json(request, {"error": "file not found"}, status=404)
        size = file_path.stat().st_size
        data = file_path.read_bytes()[: _MAX_FILE_PREVIEW_BYTES + 1]
        is_text = not _looks_binary(data)
        truncated = len(data) > _MAX_FILE_PREVIEW_BYTES
        content = ""
        message = None
        if is_text:
            content = data[:_MAX_FILE_PREVIEW_BYTES].decode("utf-8", errors="replace")
        else:
            message = "Binary file preview is not available."
        return self._json(
            request,
            {
                "ok": True,
                "threadId": thread_id,
                "workspaceDir": workspace_dir,
                "path": normalized,
                "name": file_path.name,
                "extension": file_path.suffix,
                "language": _LANGUAGE_BY_SUFFIX.get(file_path.suffix.lower()),
                "isText": is_text,
                "content": content,
                "truncated": truncated,
                "size": size,
                "message": message,
            },
        )

    def _create_workspace_zip_sync(self, root: Path) -> str:
        fd, archive_path = tempfile.mkstemp(prefix="evosci-webui-", suffix=".zip")
        os.close(fd)
        try:
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for current_root, dirs, files in os.walk(root, followlinks=False):
                    dirs[:] = sorted(dirs)
                    for filename in sorted(files):
                        candidate = Path(current_root) / filename
                        if candidate.is_symlink() or not candidate.is_file():
                            continue
                        relative = candidate.resolve().relative_to(root).as_posix()
                        archive.write(candidate, arcname=relative)
        except Exception:
            try:
                os.unlink(archive_path)
            except OSError:
                pass
            raise
        return archive_path

    async def _handle_ui_files_download_all(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)
        from aiohttp import web

        thread_id = self._extract_thread_id(request)
        workspace_dir = await self._resolve_or_create_workspace(
            thread_id or uuid.uuid4().hex
        )
        root = Path(workspace_dir).resolve()
        archive_path = await asyncio.to_thread(self._create_workspace_zip_sync, root)
        response = web.StreamResponse(
            headers={
                **self._cors_headers(request),
                "Content-Disposition": 'attachment; filename="workspace.zip"',
                "Content-Type": "application/zip",
            },
        )
        try:
            await response.prepare(request)
            with open(archive_path, "rb") as archive:
                while True:
                    chunk = await asyncio.to_thread(archive.read, 65536)
                    if not chunk:
                        break
                    await response.write(chunk)
            await response.write_eof()
            return response
        finally:
            try:
                os.unlink(archive_path)
            except OSError:
                pass

    async def _handle_ui_session_shutdown(self, request):
        if not self._check_auth(request):
            return self._unauthorized(request)

        def _shutdown() -> None:
            os.kill(os.getpid(), signal.SIGTERM)

        asyncio.get_running_loop().call_later(0.2, _shutdown)
        return self._json(request, {"ok": True})
