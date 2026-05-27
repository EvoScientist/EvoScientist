"""Browser-facing EvoScientist WebUI control API for ``EvoSci deploy``.

LangGraph Dev owns chat execution, streaming, and thread history. This module
hosts only EvoScientist-specific controls that a browser UI cannot get from the
LangGraph API: model/provider configuration, skills, MCP, stats, and workspace
file browsing.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import signal
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_FILE_PREVIEW_BYTES = 256 * 1024
_MAX_WORKSPACE_TREE_ENTRIES = 500
_BINARY_PROBE_BYTES = 8192
_INTERNAL_FILE_NAMES = {".evoscientist_webui_threads.json"}
_INTERNAL_DIR_NAMES = {".langgraph_api"}
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
class WebUIConfig:
    enabled: bool = True
    bind_host: str = "127.0.0.1"
    port: int = 8010
    base_path: str = "/webui"
    api_key: str = ""
    allowed_origins: str = ""
    allowed_hosts: str = ""


class WebUIConfigurationError(RuntimeError):
    """Raised when the configured WebUI control API would be unsafe to start."""


def config_from_evosci(config: Any) -> WebUIConfig:
    return WebUIConfig(
        enabled=bool(getattr(config, "webui_enabled", True)),
        bind_host=getattr(config, "webui_bind_host", "127.0.0.1") or "127.0.0.1",
        port=int(getattr(config, "webui_port", 8010) or 8010),
        base_path=normalize_base_path(
            getattr(config, "webui_base_path", "/webui") or "/webui"
        ),
        api_key=getattr(config, "webui_api_key", "") or "",
        allowed_origins=getattr(config, "webui_allowed_origins", "") or "",
        allowed_hosts=getattr(config, "webui_allowed_hosts", "") or "",
    )


def normalize_base_path(base_path: str) -> str:
    normalized = str(base_path or "/webui").strip() or "/webui"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized.rstrip("/") or "/webui"


def format_url_host(bind_host: str) -> str:
    host = (bind_host or "127.0.0.1").strip() or "127.0.0.1"
    if host in {"127.0.0.1", "::1", "localhost", "0.0.0.0", "::"}:
        return "localhost"
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def webui_base_url(config: WebUIConfig) -> str:
    return f"http://{format_url_host(config.bind_host)}:{config.port}{config.base_path}"


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


def _split_host_port(value: str) -> str:
    host = (value or "").strip()
    if host.startswith("["):
        end = host.find("]")
        return host[1:end] if end > 0 else host
    if host.count(":") == 1:
        return host.rsplit(":", 1)[0]
    return host


def _normalize_origin(value: str) -> str:
    return (value or "").strip().rstrip("/")


def _split_config_list(value: str) -> set[str]:
    return {part.strip() for part in (value or "").split(",") if part.strip()}


def _is_loopback_bind(host: str) -> bool:
    normalized = _split_host_port(host).strip().lower().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _is_safe_direct_host(host: str) -> bool:
    normalized = _split_host_port(host).strip().lower().rstrip(".")
    if normalized in {"", "localhost"} or normalized.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return bool(ip.is_loopback or ip.is_private or ip.is_link_local)


def _default_allowed_origins() -> set[str]:
    return {
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    }


def _normalize_relative_path(path: str) -> str:
    normalized = (path or "").strip().replace("\\", "/").lstrip("/")
    if not normalized or normalized == ".":
        return ""
    parts = [part for part in normalized.split("/") if part and part != "."]
    if any(part == ".." for part in parts):
        raise ValueError("path traversal is not allowed")
    return "/".join(parts)


def _is_hidden_or_internal(relative: str) -> bool:
    parts = [part for part in relative.split("/") if part]
    if not parts:
        return False
    return any(part.startswith(".") for part in parts) or any(
        part in _INTERNAL_DIR_NAMES or part in _INTERNAL_FILE_NAMES for part in parts
    )


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


class WebUIControlServer:
    """Small aiohttp server for deploy-mode WebUI control endpoints."""

    def __init__(
        self,
        config: WebUIConfig,
        *,
        workspace_dir: Path,
        langgraph_base_url: str,
        assistant_id: str = "EvoScientist",
    ) -> None:
        self.config = config
        self.workspace_dir = Path(workspace_dir).expanduser().resolve()
        self.langgraph_base_url = langgraph_base_url.rstrip("/")
        self.assistant_id = assistant_id
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: Any = None
        self._site: Any = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    def start(self, timeout: float = 10.0) -> None:
        self._validate_start_security()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._thread_main,
            name="evosci-webui",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise RuntimeError("WebUI control API did not start in time")
        if self._error is not None:
            raise RuntimeError(f"WebUI control API failed to start: {self._error}")

    def stop(self) -> None:
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None:
            return
        future = asyncio.run_coroutine_threadsafe(self._stop_async(), loop)
        try:
            future.result(timeout=5.0)
        except Exception:
            logger.exception("Failed to stop WebUI control API cleanly")
        thread.join(timeout=5.0)
        self._thread = None
        self._loop = None

    def _validate_start_security(self) -> None:
        if not (1 <= int(self.config.port) <= 65535):
            raise WebUIConfigurationError(
                f"Invalid WebUI port {self.config.port}; use [1, 65535]."
            )
        if not _is_loopback_bind(self.config.bind_host) and not self.config.api_key:
            raise WebUIConfigurationError(
                "webui_api_key is required when webui_bind_host is not loopback."
            )

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._start_async())
            self._ready.set()
            loop.run_forever()
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            if self._runner is not None:
                loop.run_until_complete(self._cleanup_async())
            loop.close()

    async def _start_async(self) -> None:
        try:
            from aiohttp import web
        except ImportError as exc:
            raise RuntimeError(
                "aiohttp is required for the WebUI control API. "
                "Reinstall EvoScientist or install aiohttp>=3.9."
            ) from exc

        app = web.Application()
        for method, path, handler in self._routes():
            app.router.add_route(method, path, handler)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(
            self._runner,
            self.config.bind_host,
            int(self.config.port),
        )
        await self._site.start()
        logger.info(
            "WebUI control API started on %s:%s%s",
            self.config.bind_host,
            self.config.port,
            self.config.base_path,
        )

    async def _stop_async(self) -> None:
        await self._cleanup_async()
        if self._loop is not None:
            self._loop.stop()

    async def _cleanup_async(self) -> None:
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def _route(self, suffix: str) -> str:
        return f"{normalize_base_path(self.config.base_path)}{suffix}"

    def _routes(self) -> list[tuple[str, str, Any]]:
        route_specs = [
            ("GET", "/healthz", self._handle_health),
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
            ("GET", "/ui/files/tree", self._handle_ui_files_tree),
            ("GET", "/ui/files/read", self._handle_ui_files_read),
            ("GET", "/ui/files/download-all", self._handle_ui_files_download_all),
            ("POST", "/ui/session/shutdown", self._handle_ui_session_shutdown),
        ]
        routes: list[tuple[str, str, Any]] = []
        for method, suffix, handler in route_specs:
            routes.append(("OPTIONS", self._route(suffix), self._handle_options))
            routes.append((method, self._route(suffix), handler))
        return routes

    def _configured_allowed_hosts(self) -> set[str]:
        return {
            host.lower().rstrip(".")
            for host in _split_config_list(self.config.allowed_hosts)
        }

    def _configured_allowed_origins(self) -> set[str]:
        return {
            _normalize_origin(origin)
            for origin in _split_config_list(self.config.allowed_origins)
        }

    def _request_host_allowed(self, request: Any) -> bool:
        host = request.headers.get("Host", "")
        normalized_host = host.strip().lower().rstrip(".")
        host_without_port = _split_host_port(normalized_host).lower().rstrip(".")
        allowed_hosts = self._configured_allowed_hosts()
        if "*" in allowed_hosts and self.config.api_key:
            return True
        if normalized_host in allowed_hosts or host_without_port in allowed_hosts:
            return True
        configured = (self.config.bind_host or "").strip().lower()
        if _is_loopback_bind(configured):
            return host_without_port in {"localhost", "127.0.0.1", "::1"}
        return _is_safe_direct_host(host)

    def _is_origin_allowed(self, origin: str) -> bool:
        normalized_origin = _normalize_origin(origin)
        allowed_origins = self._configured_allowed_origins()
        if "*" in allowed_origins and self.config.api_key:
            return True
        if normalized_origin in allowed_origins:
            return True
        return normalized_origin in _default_allowed_origins()

    def _request_origin_allowed(self, request: Any) -> bool:
        origin = request.headers.get("Origin", "")
        return not origin or self._is_origin_allowed(origin)

    def _supplied_api_key(self, request: Any) -> str:
        auth = request.headers.get("Authorization", "")
        bearer = auth.removeprefix("Bearer ").strip() if auth else ""
        return (request.headers.get("X-API-Key") or bearer).strip()

    def _is_authenticated(self, request: Any) -> bool:
        expected = (self.config.api_key or "").strip()
        return not expected or self._supplied_api_key(request) == expected

    def _check_auth(self, request: Any) -> bool:
        return (
            self._request_host_allowed(request)
            and self._request_origin_allowed(request)
            and self._is_authenticated(request)
        )

    def _cors_headers(self, request: Any) -> dict[str, str]:
        origin = request.headers.get("Origin", "")
        headers = {
            "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": (
                "Content-Type, Authorization, X-API-Key, X-Thread-Id"
            ),
            "Access-Control-Expose-Headers": "Content-Disposition",
            "Access-Control-Max-Age": "86400",
            "Vary": "Origin",
        }
        if origin and self._is_origin_allowed(origin):
            headers["Access-Control-Allow-Origin"] = _normalize_origin(origin)
        return headers

    async def _handle_options(self, request: Any):
        from aiohttp import web

        if not self._request_host_allowed(request) or not self._request_origin_allowed(
            request
        ):
            return web.Response(status=403, headers={"Vary": "Origin"})
        return web.Response(status=204, headers=self._cors_headers(request))

    def _json(self, request: Any, payload: Any, *, status: int = 200):
        from aiohttp import web

        return web.json_response(
            _json_safe(payload),
            status=status,
            headers=self._cors_headers(request),
        )

    def _unauthorized(self, request: Any):
        return self._json(request, {"error": "unauthorized"}, status=401)

    async def _request_payload(self, request: Any) -> dict[str, Any]:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    async def _handle_health(self, request: Any):
        if not self._request_host_allowed(request) or not self._request_origin_allowed(
            request
        ):
            return self._json(request, {"error": "forbidden"}, status=403)
        auth_required = bool((self.config.api_key or "").strip())
        authenticated = self._is_authenticated(request)
        return self._json(
            request,
            {
                "ok": True,
                "service": "evoscientist-webui",
                "authRequired": auth_required,
                "authenticated": authenticated,
                "assistantId": self.assistant_id,
            },
        )

    def _load_config(self):
        from ..config.settings import load_config

        return load_config()

    async def _handle_ui_stats(self, request: Any):
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
        from ..tools.skills_manager import list_skills

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
        from .. import paths

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
            if _is_hidden_or_internal(relative):
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

    async def _handle_ui_skills(self, request: Any):
        if not self._check_auth(request):
            return self._unauthorized(request)
        installed = self._list_installed_skills()
        installed_names = {item["name"] for item in installed}
        marketplace: list[dict[str, Any]] = []
        marketplace_error: str | None = None
        try:
            from ..tools.skills_manager import fetch_remote_skill_index

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

    async def _handle_ui_skills_install(self, request: Any):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        source = str(payload.get("source") or payload.get("name") or "").strip()
        if not source:
            return self._json(request, {"error": "source is required"}, status=400)
        from ..tools.skills_manager import install_skill

        result = await asyncio.to_thread(install_skill, source)
        status = 200 if result.get("success") else 400
        return self._json(
            request, {"ok": bool(result.get("success")), **result}, status=status
        )

    async def _handle_ui_skills_uninstall(self, request: Any):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        name = str(payload.get("name") or "").strip()
        if not name:
            return self._json(request, {"error": "name is required"}, status=400)
        from ..tools.skills_manager import uninstall_skill

        result = await asyncio.to_thread(uninstall_skill, name)
        status = 200 if result.get("success") else 400
        return self._json(
            request, {"ok": bool(result.get("success")), **result}, status=status
        )

    async def _handle_ui_models(self, request: Any):
        if not self._check_auth(request):
            return self._unauthorized(request)
        return self._json(request, await asyncio.to_thread(self._models_overview))

    def _models_overview(self) -> dict[str, Any]:
        from ..llm.models import list_models_by_provider

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

    def _provider_metadata(self, cfg: Any) -> dict[str, dict[str, Any]]:
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
                "apiKeyConfigField": api_field or None,
                "baseUrlEnvVar": base_env or None,
                "baseUrlConfigField": base_field or None,
                "baseUrl": getattr(cfg, base_field, "") if base_field else "",
                "supportsApiKeyUpdate": bool(api_field),
                "supportsBaseUrlUpdate": bool(base_field),
            }
        return result

    async def _handle_ui_models_switch(self, request: Any):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        model = str(payload.get("model") or payload.get("name") or "").strip()
        provider = str(payload.get("provider") or "").strip()
        if not model or not provider:
            return self._json(
                request, {"error": "model and provider are required"}, status=400
            )
        from ..config.settings import set_config_value

        ok = set_config_value("model", model) and set_config_value("provider", provider)
        if not ok:
            return self._json(request, {"error": "failed to update model"}, status=500)
        return self._json(
            request,
            {
                "ok": True,
                "message": "Model updated. Restart the LangGraph run to apply it.",
                "models": self._models_overview(),
            },
        )

    async def _handle_ui_provider_key(self, request: Any):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        provider = str(payload.get("provider") or "").strip()
        api_key = str(payload.get("apiKey") or payload.get("api_key") or "").strip()
        config_field = (
            self._provider_metadata(self._load_config())
            .get(provider, {})
            .get("apiKeyConfigField")
        )
        if not config_field:
            return self._json(request, {"error": "unsupported provider"}, status=400)
        from ..config.settings import set_config_value

        if not set_config_value(str(config_field), api_key):
            return self._json(
                request, {"error": f"failed to update {config_field}"}, status=500
            )
        return self._json(request, {"ok": True, "models": self._models_overview()})

    async def _handle_ui_provider_base_url(self, request: Any):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        provider = str(payload.get("provider") or "").strip()
        base_url = str(payload.get("baseUrl") or payload.get("base_url") or "").strip()
        config_field = (
            self._provider_metadata(self._load_config())
            .get(provider, {})
            .get("baseUrlConfigField")
        )
        if not config_field:
            return self._json(request, {"error": "unsupported provider"}, status=400)
        from ..config.settings import set_config_value

        if not set_config_value(str(config_field), base_url):
            return self._json(
                request, {"error": f"failed to update {config_field}"}, status=500
            )
        return self._json(request, {"ok": True, "models": self._models_overview()})

    async def _handle_ui_mcp(self, request: Any):
        if not self._check_auth(request):
            return self._unauthorized(request)
        return self._json(request, await asyncio.to_thread(self._mcp_overview))

    def _mcp_overview(self) -> dict[str, Any]:
        from ..mcp import load_mcp_config

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
            from ..mcp.registry import fetch_marketplace_index

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

    async def _handle_ui_mcp_install(self, request: Any):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        names = payload.get("names") or payload.get("name")
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, list) or not names:
            return self._json(request, {"error": "names are required"}, status=400)

        from ..mcp.registry import fetch_marketplace_index, find_server_by_name
        from ..mcp.registry import install_mcp_servers as install_entries

        index = await asyncio.to_thread(fetch_marketplace_index)
        entries = [(str(name), find_server_by_name(str(name), index)) for name in names]
        missing = [name for name, entry in entries if entry is None]
        if missing:
            return self._json(
                request,
                {"error": f"Unknown MCP server(s): {', '.join(missing)}"},
                status=400,
            )
        selected = [entry for _, entry in entries if entry is not None]
        count = await asyncio.to_thread(install_entries, selected)
        return self._json(
            request,
            {
                "ok": count == len(selected),
                "installed": count,
                "mcp": self._mcp_overview(),
            },
            status=200 if count == len(selected) else 500,
        )

    async def _handle_ui_mcp_remove(self, request: Any):
        if not self._check_auth(request):
            return self._unauthorized(request)
        payload = await self._request_payload(request)
        name = str(payload.get("name") or "").strip()
        if not name:
            return self._json(request, {"error": "name is required"}, status=400)
        from ..mcp import remove_mcp_server

        removed = await asyncio.to_thread(remove_mcp_server, name)
        return self._json(
            request,
            {"ok": removed, "mcp": self._mcp_overview()},
            status=200 if removed else 404,
        )

    def _resolve_workspace_path(self, relative_path: str) -> tuple[Path, str]:
        root = self.workspace_dir
        normalized = _normalize_relative_path(relative_path)
        candidate = root if not normalized else (root / normalized).resolve()
        candidate.relative_to(root)
        return candidate, normalized

    async def _handle_ui_files_tree(self, request: Any):
        if not self._check_auth(request):
            return self._unauthorized(request)
        path = request.query.get("path", "")
        try:
            root_path, normalized = self._resolve_workspace_path(path)
        except ValueError:
            return self._json(request, {"error": "invalid path"}, status=400)
        if not root_path.exists() or not root_path.is_dir():
            return self._json(request, {"error": "directory not found"}, status=404)
        entries = []
        for entry in sorted(
            root_path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())
        ):
            if entry.is_symlink():
                continue
            try:
                relative = entry.resolve().relative_to(self.workspace_dir).as_posix()
                stats = entry.stat()
            except OSError:
                continue
            if _is_hidden_or_internal(relative):
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
                "workspaceDir": str(self.workspace_dir),
                "path": normalized,
                "entries": entries,
            },
        )

    @staticmethod
    def _read_preview_bytes(file_path: Path) -> bytes:
        with file_path.open("rb") as handle:
            return handle.read(_MAX_FILE_PREVIEW_BYTES + 1)

    async def _handle_ui_files_read(self, request: Any):
        if not self._check_auth(request):
            return self._unauthorized(request)
        relative_path = request.query.get("path", "")
        try:
            file_path, normalized = self._resolve_workspace_path(relative_path)
        except ValueError:
            return self._json(request, {"error": "invalid path"}, status=400)
        if not file_path.exists() or not file_path.is_file():
            return self._json(request, {"error": "file not found"}, status=404)
        if _is_hidden_or_internal(normalized):
            return self._json(request, {"error": "file not found"}, status=404)
        size = file_path.stat().st_size
        data = await asyncio.to_thread(self._read_preview_bytes, file_path)
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
                "workspaceDir": str(self.workspace_dir),
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
                    kept_dirs = []
                    for dirname in sorted(dirs):
                        candidate_dir = Path(current_root) / dirname
                        if candidate_dir.is_symlink():
                            continue
                        try:
                            relative_dir = candidate_dir.resolve().relative_to(root)
                        except (OSError, ValueError):
                            continue
                        if not _is_hidden_or_internal(relative_dir.as_posix()):
                            kept_dirs.append(dirname)
                    dirs[:] = kept_dirs
                    for filename in sorted(files):
                        candidate = Path(current_root) / filename
                        if candidate.is_symlink() or not candidate.is_file():
                            continue
                        try:
                            relative = candidate.resolve().relative_to(root).as_posix()
                        except (OSError, ValueError):
                            continue
                        if _is_hidden_or_internal(relative):
                            continue
                        archive.write(candidate, arcname=relative)
        except Exception:
            try:
                os.unlink(archive_path)
            except OSError:
                pass
            raise
        return archive_path

    async def _handle_ui_files_download_all(self, request: Any):
        if not self._check_auth(request):
            return self._unauthorized(request)
        from aiohttp import web

        root = self.workspace_dir
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

    async def _handle_ui_session_shutdown(self, request: Any):
        if not self._check_auth(request):
            return self._unauthorized(request)

        def _shutdown() -> None:
            os.kill(os.getpid(), signal.SIGTERM)

        asyncio.get_running_loop().call_later(0.2, _shutdown)
        return self._json(request, {"ok": True})
