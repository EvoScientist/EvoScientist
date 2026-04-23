"""Browser-facing channel for assistant-ui style web frontends.

The transport exposed here is intentionally aligned with assistant-ui's
Assistant Transport runtime rather than a messaging-platform webhook:

- request body: ``{state, commands, threadId, ...}``
- streamed response lines: ``aui-state:<json-ops>`` and ``3:<json-error>``

This lets a future Next.js app use ``useAssistantTransportRuntime`` while the
backend still reuses EvoScientist's channel plumbing for message handling.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import shlex
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ...runtime import (
    build_command_catalog,
    execute_command_line,
    get_thread_runtime_registry,
)
from ...runtime.native_ui import (
    get_channels_overview,
    get_mcp_overview,
    get_skills_overview,
    install_mcp_servers,
    install_skills,
    remove_mcp,
    schedule_shutdown,
    start_channels,
    stop_channels,
    uninstall_skill,
)
from ..base import Channel, ChannelError
from ..bus.events import InboundMessage
from ..capabilities import WEBUI as WEBUI_CAPS
from ..config import BaseChannelConfig
from ..mixins import WebhookMixin

logger = logging.getLogger(__name__)
_MAX_TOOL_RESULT_CHARS = 8000


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _sse_data(value: Any) -> bytes:
    return f"data: {_json_dumps(value)}\n\n".encode()


def _text_from_transport_message(message: dict[str, Any]) -> str:
    parts = message.get("parts", [])
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)
    return "".join(chunks).strip()


def _text_from_langchain_message(message: Any) -> str:
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        content = "".join(parts)
    return str(content).strip()


def _stable_message_id(
    thread_id: str,
    role: str,
    index: int,
    content: str,
) -> str:
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return f"{role}_{thread_id}_{index}_{digest}"


def _role_from_langchain_message(message: Any) -> str | None:
    msg_type = str(getattr(message, "type", "") or "")
    if msg_type == "human":
        return "user"
    if msg_type == "ai":
        return "assistant"
    return None


def _normalize_tool_result(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:_MAX_TOOL_RESULT_CHARS]
    try:
        rendered = json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        rendered = str(value)
    return rendered[:_MAX_TOOL_RESULT_CHARS]


@dataclass
class WebUIConfig(BaseChannelConfig):
    webhook_port: int = 8010
    base_path: str = "/webui"
    api_key: str = ""
    text_chunk_limit: int = 8192


@dataclass
class _PendingRun:
    thread_id: str
    state: dict[str, Any]
    assistant_message_id: str
    queue: asyncio.Queue[tuple[str, Any]] = field(default_factory=asyncio.Queue)

    def find_assistant_index(self) -> int | None:
        messages = self.state.get("messages", [])
        if not isinstance(messages, list):
            return None
        for i, message in enumerate(messages):
            if (
                isinstance(message, dict)
                and message.get("id") == self.assistant_message_id
            ):
                return i
        return None

    def upsert_subagent(self, subagent_id: str, **patch: Any) -> None:
        subagents = self.state.setdefault("subagents", [])
        if not isinstance(subagents, list):
            subagents = []
            self.state["subagents"] = subagents

        for subagent in subagents:
            if isinstance(subagent, dict) and subagent.get("id") == subagent_id:
                subagent.update({k: v for k, v in patch.items() if v is not None})
                return

        subagents.append(
            {
                "id": subagent_id,
                "name": patch.get("name") or subagent_id,
                "description": patch.get("description") or "",
                "status": patch.get("status") or "running",
                "detail": patch.get("detail") or "",
            }
        )

    def append_activity(self, event: dict[str, Any], limit: int = 200) -> None:
        activity = self.state.setdefault("activity", [])
        if not isinstance(activity, list):
            activity = []
            self.state["activity"] = activity
        activity.append(event)
        if len(activity) > limit:
            del activity[: len(activity) - limit]


@dataclass
class _PendingApproval:
    event: threading.Event
    decision: str | None = None
    count: int = 0


@dataclass
class _PendingAskUser:
    event: threading.Event
    reply: str | None = None


class WebUIChannel(Channel, WebhookMixin):
    """HTTP channel that streams assistant-ui transport state."""

    name = "webui"
    capabilities = WEBUI_CAPS
    _disable_shared_webhook = True

    def __init__(self, config: WebUIConfig):
        super().__init__(config)
        self._pending_runs: dict[str, _PendingRun] = {}
        self._pending_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._approval_waiters: dict[str, _PendingApproval] = {}
        self._approval_lock = threading.Lock()
        self._ask_user_waiters: dict[str, _PendingAskUser] = {}
        self._ask_user_lock = threading.Lock()
        self._runtime = get_thread_runtime_registry()

    def _get_webhook_port(self) -> int:
        return self.config.webhook_port

    def _route(self, suffix: str) -> str:
        base = self.config.base_path.rstrip("/") or "/webui"
        return f"{base}{suffix}"

    def _webhook_routes(self) -> list[tuple[str, str, Any]]:
        return [
            ("GET", self._route("/healthz"), self._handle_health),
            ("OPTIONS", self._route("/threads"), self._handle_options),
            ("GET", self._route("/threads"), self._handle_threads),
            ("OPTIONS", self._route("/thread-state"), self._handle_options),
            ("GET", self._route("/thread-state"), self._handle_thread_state),
            ("OPTIONS", self._route("/thread-events"), self._handle_options),
            ("GET", self._route("/thread-events"), self._handle_thread_events),
            ("OPTIONS", self._route("/commands/catalog"), self._handle_options),
            ("GET", self._route("/commands/catalog"), self._handle_commands_catalog),
            ("OPTIONS", self._route("/commands/execute"), self._handle_options),
            ("POST", self._route("/commands/execute"), self._handle_commands_execute),
            ("OPTIONS", self._route("/ui/skills"), self._handle_options),
            ("GET", self._route("/ui/skills"), self._handle_ui_skills),
            ("OPTIONS", self._route("/ui/skills/install"), self._handle_options),
            ("POST", self._route("/ui/skills/install"), self._handle_ui_skills_install),
            ("OPTIONS", self._route("/ui/skills/uninstall"), self._handle_options),
            (
                "POST",
                self._route("/ui/skills/uninstall"),
                self._handle_ui_skills_uninstall,
            ),
            ("OPTIONS", self._route("/ui/mcp"), self._handle_options),
            ("GET", self._route("/ui/mcp"), self._handle_ui_mcp),
            ("OPTIONS", self._route("/ui/mcp/install"), self._handle_options),
            ("POST", self._route("/ui/mcp/install"), self._handle_ui_mcp_install),
            ("OPTIONS", self._route("/ui/mcp/remove"), self._handle_options),
            ("POST", self._route("/ui/mcp/remove"), self._handle_ui_mcp_remove),
            ("OPTIONS", self._route("/ui/channels"), self._handle_options),
            ("GET", self._route("/ui/channels"), self._handle_ui_channels),
            ("OPTIONS", self._route("/ui/channels/start"), self._handle_options),
            (
                "POST",
                self._route("/ui/channels/start"),
                self._handle_ui_channels_start,
            ),
            ("OPTIONS", self._route("/ui/channels/stop"), self._handle_options),
            ("POST", self._route("/ui/channels/stop"), self._handle_ui_channels_stop),
            ("OPTIONS", self._route("/ui/session/shutdown"), self._handle_options),
            (
                "POST",
                self._route("/ui/session/shutdown"),
                self._handle_ui_session_shutdown,
            ),
            ("OPTIONS", self._route("/run/stop"), self._handle_options),
            ("POST", self._route("/run/stop"), self._handle_run_stop),
            ("OPTIONS", self._route("/approval"), self._handle_approval_options),
            ("POST", self._route("/approval"), self._handle_approval),
            ("OPTIONS", self._route("/ask-user"), self._handle_ask_user_options),
            ("POST", self._route("/ask-user"), self._handle_ask_user),
            ("OPTIONS", self._route("/assistant"), self._handle_assistant_options),
            ("POST", self._route("/assistant"), self._handle_assistant),
        ]

    def _cors_headers(self, request) -> dict[str, str]:
        origin = request.headers.get("Origin", "*")
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": (
                "Content-Type, Authorization, X-API-Key, X-Thread-Id"
            ),
            "Access-Control-Max-Age": "86400",
            "Vary": "Origin",
        }

    async def _handle_options(self, request):
        from aiohttp import web

        return web.Response(status=204, headers=self._cors_headers(request))

    async def start(self) -> None:
        try:
            from aiohttp import web  # noqa: F401
        except ImportError:
            raise ChannelError(
                "aiohttp not installed. Install with: pip install aiohttp"
            ) from None

        self._loop = asyncio.get_running_loop()
        await self._start_webhook_server()
        self._running = True

        try:
            from ...config.settings import load_config

            cfg = load_config()
            if cfg.enable_ask_user or not cfg.auto_approve:
                logger.warning(
                    "webui works best with enable_ask_user=false and auto_approve=true; "
                    "interactive approval flows can stall a single HTTP request."
                )
        except Exception:
            pass

        logger.info(
            "Web UI channel started on port %s at %s/assistant",
            self.config.webhook_port,
            self.config.base_path.rstrip("/") or "/webui",
        )

    async def _cleanup(self) -> None:
        await self._stop_webhook_server()

    async def receive(self):
        while self._running:
            await asyncio.sleep(1.0)
            if False:  # pragma: no cover
                yield None

    async def _send_chunk(
        self,
        chat_id: str,
        _formatted_text: str,
        raw_text: str,
        _reply_to: str | None,
        _metadata: dict,
    ) -> None:
        pending = await self._get_pending_run(chat_id)
        if pending is None:
            logger.debug(
                "Dropping outbound webui chunk for inactive thread %s", chat_id
            )
            return

        idx = pending.find_assistant_index()
        if idx is None:
            return

        pending.state["status"] = None
        pending.state["isRunning"] = True
        pending.state["messages"][idx]["content"] += raw_text
        ops = [
            {
                "type": "append-text",
                "path": ["messages", idx, "content"],
                "value": raw_text,
            },
            {"type": "set", "path": ["status"], "value": None},
            {"type": "set", "path": ["isRunning"], "value": True},
        ]
        await pending.queue.put(("ops", ops))

    async def send(self, message) -> bool:
        ok = await super().send(message)
        self._runtime.set_status(message.chat_id, status=None, is_running=False)
        self._runtime.set_prompt(message.chat_id, approval=None, ask_user=None)
        pending = await self._get_pending_run(message.chat_id)
        if pending is not None:
            pending.state["isRunning"] = False
            await pending.queue.put(
                ("ops", [{"type": "set", "path": ["isRunning"], "value": False}])
            )
            await pending.queue.put(("close", None))
        return ok

    async def send_thinking_message(
        self,
        sender: str,
        thinking: str,
        metadata: dict | None = None,
    ) -> None:
        thread_id = str((metadata or {}).get("chat_id", sender) or sender)
        self._runtime.set_status(thread_id, status=thinking, is_running=True)
        pending = await self._get_pending_run(thread_id)
        if pending is None or not self.send_thinking:
            return
        pending.state["status"] = thinking
        await pending.queue.put(
            ("ops", [{"type": "set", "path": ["status"], "value": thinking}])
        )

    async def send_todo_message(
        self,
        sender: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        thread_id = str((metadata or {}).get("chat_id", sender) or sender)
        self._runtime.set_status(thread_id, status=content, is_running=True)
        pending = await self._get_pending_run(thread_id)
        if pending is None:
            return
        pending.state["status"] = content
        await pending.queue.put(
            ("ops", [{"type": "set", "path": ["status"], "value": content}])
        )

    def send_subagent_event_nowait(
        self,
        sender: str,
        event_type: str,
        payload: dict[str, Any],
        metadata: dict | None = None,
    ) -> bool:
        thread_id = str((metadata or {}).get("chat_id", sender) or sender)
        self._runtime.apply_subagent_event(thread_id, event_type, payload)

        def _build_ops(pending: _PendingRun) -> list[dict[str, Any]]:
            return self._apply_subagent_event(pending, event_type, payload)

        return self._enqueue_pending_ops_nowait(thread_id, _build_ops)

    def _apply_subagent_event(
        self,
        pending: _PendingRun,
        event_type: str,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        raw_name = payload.get("subagent") or payload.get("name") or "sub-agent"
        subagent_id = str(payload.get("instance_id") or raw_name)
        subagent_name = str(raw_name)
        activity_entry: dict[str, Any] | None = None

        if event_type == "subagent_start":
            description = str(payload.get("description", ""))
            pending.upsert_subagent(
                subagent_id,
                name=subagent_name,
                description=description,
                status="running",
                detail="Delegated",
            )
            activity_entry = {
                "id": f"activity_{uuid.uuid4().hex}",
                "type": "subagent_start",
                "agent": subagent_name,
                "title": f"Started {subagent_name}",
                "detail": description,
                "status": "running",
                "createdAt": _utc_now_iso(),
            }
        elif event_type == "subagent_tool_call":
            tool_name = str(payload.get("name", "unknown"))
            pending.upsert_subagent(
                subagent_id,
                name=subagent_name,
                status="running",
                detail=f"Tool: {tool_name}",
            )
            activity_entry = {
                "id": f"activity_{uuid.uuid4().hex}",
                "type": "subagent_tool_call",
                "agent": subagent_name,
                "title": f"{subagent_name} called {tool_name}",
                "detail": "",
                "status": "running",
                "createdAt": _utc_now_iso(),
            }
        elif event_type == "subagent_tool_result":
            tool_name = str(payload.get("name", "unknown"))
            success = bool(payload.get("success", True))
            result_content = _normalize_tool_result(payload.get("content", ""))
            pending.upsert_subagent(
                subagent_id,
                name=subagent_name,
                status="complete" if success else "error",
                detail=f"Finished: {tool_name}",
            )
            activity_entry = {
                "id": f"activity_{uuid.uuid4().hex}",
                "type": "subagent_tool_result",
                "agent": subagent_name,
                "title": f"{subagent_name} finished {tool_name}",
                "detail": "",
                "result": result_content,
                "status": "complete" if success else "error",
                "createdAt": _utc_now_iso(),
            }
        elif event_type == "subagent_end":
            pending.upsert_subagent(
                subagent_id,
                name=subagent_name,
                status="complete",
                detail="Completed",
            )
            activity_entry = {
                "id": f"activity_{uuid.uuid4().hex}",
                "type": "subagent_end",
                "agent": subagent_name,
                "title": f"Completed {subagent_name}",
                "detail": "",
                "status": "complete",
                "createdAt": _utc_now_iso(),
            }
        elif event_type == "subagent_text":
            content = str(payload.get("content", ""))[:240]
            pending.upsert_subagent(
                subagent_id,
                name=subagent_name,
                status="running",
                detail=content[:160],
            )
            activity_entry = {
                "id": f"activity_{uuid.uuid4().hex}",
                "type": "subagent_text",
                "agent": subagent_name,
                "title": subagent_name,
                "detail": content,
                "status": "running",
                "createdAt": _utc_now_iso(),
            }
        else:
            return []

        pending.append_activity(activity_entry)
        return [
            {"type": "set", "path": ["subagents"], "value": pending.state["subagents"]},
            {"type": "set", "path": ["activity"], "value": pending.state["activity"]},
        ]

    async def send_subagent_event(
        self,
        sender: str,
        event_type: str,
        payload: dict[str, Any],
        metadata: dict | None = None,
    ) -> None:
        thread_id = str((metadata or {}).get("chat_id", sender) or sender)
        self._runtime.apply_subagent_event(thread_id, event_type, payload)
        pending = await self._get_pending_run(thread_id)
        if pending is None:
            return
        ops = self._apply_subagent_event(pending, event_type, payload)
        if not ops:
            return
        await pending.queue.put(("ops", ops))

    async def send_stream_state(
        self,
        sender: str,
        stream_state: Any,
        metadata: dict | None = None,
    ) -> None:
        thread_id = str((metadata or {}).get("chat_id", sender) or sender)
        self._runtime.update_subagents_from_stream_state(thread_id, stream_state)
        pending = await self._get_pending_run(thread_id)
        if pending is None:
            return

        thinking_text = str(getattr(stream_state, "thinking_text", "") or "").rstrip()
        is_thinking = bool(getattr(stream_state, "is_thinking", False))
        status_value = thinking_text if is_thinking and thinking_text else None

        next_subagents: list[dict[str, Any]] = []
        for subagent in getattr(stream_state, "subagents", []) or []:
            name = str(getattr(subagent, "name", "sub-agent") or "sub-agent")
            description = str(getattr(subagent, "description", "") or "")
            is_active = bool(getattr(subagent, "is_active", False))
            tool_calls = list(getattr(subagent, "tool_calls", []) or [])
            detail = ""
            if tool_calls:
                last_tool = tool_calls[-1]
                detail = f"Tool: {last_tool.get('name', 'unknown')}"

            next_subagents.append(
                {
                    "id": name,
                    "name": name,
                    "description": description,
                    "status": "running" if is_active else "complete",
                    "detail": detail,
                }
            )

        pending.state["status"] = status_value
        pending.state["subagents"] = next_subagents
        await pending.queue.put(
            (
                "ops",
                [
                    {"type": "set", "path": ["status"], "value": status_value},
                    {"type": "set", "path": ["subagents"], "value": next_subagents},
                ],
            )
        )

    async def _handle_health(self, _request):
        from aiohttp import web

        return web.json_response(
            {
                "ok": True,
                "channel": self.name,
                "active_threads": len(self._pending_runs),
                "base_path": self.config.base_path,
            }
        )

    async def _handle_threads(self, request):
        from aiohttp import web

        from ...sessions import list_threads

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        threads = await list_threads(limit=100, include_preview=True)
        persisted_by_id = {
            str(thread.get("thread_id", "") or ""): thread for thread in threads
        }
        for runtime_thread_id in self._runtime.thread_ids():
            if not runtime_thread_id:
                continue
            if runtime_thread_id in persisted_by_id:
                continue
            runtime_state = self._runtime.snapshot(runtime_thread_id)
            persisted_by_id[runtime_thread_id] = {
                "thread_id": runtime_thread_id,
                "updated_at": runtime_state.get("updatedAt"),
                "workspace_dir": "",
                "model": "",
                "preview": "",
            }

        merged_threads = list(persisted_by_id.values())
        merged_threads.sort(
            key=lambda item: str(item.get("updated_at", "") or ""),
            reverse=True,
        )
        payload = []
        for thread in merged_threads[:100]:
            thread_id = str(thread.get("thread_id", "") or "")
            preview = str(thread.get("preview", "") or "").strip()
            payload.append(
                {
                    "threadId": thread_id,
                    "title": preview or f"Session {thread_id}",
                    "updatedAt": thread.get("updated_at"),
                    "workspaceDir": thread.get("workspace_dir"),
                    "model": thread.get("model"),
                    "preview": preview,
                }
            )

        return web.json_response(payload, headers=self._cors_headers(request))

    async def _handle_thread_state(self, request):
        from aiohttp import web

        from ...sessions import get_thread_messages, get_thread_metadata

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        thread_id = str(request.query.get("threadId", "") or "").strip()
        if not thread_id:
            return web.json_response(
                {"error": "threadId is required"},
                status=400,
                headers=self._cors_headers(request),
            )

        metadata = await get_thread_metadata(thread_id)
        messages = await get_thread_messages(thread_id)
        persisted_messages = []
        for index, message in enumerate(messages):
            role = _role_from_langchain_message(message)
            if role is None:
                continue
            content = _text_from_langchain_message(message)
            if not content:
                continue
            persisted_messages.append(
                {
                    "id": _stable_message_id(thread_id, role, index, content),
                    "role": role,
                    "content": content,
                }
            )

        pending = await self._get_pending_run(thread_id)
        state: dict[str, Any] = {
            "messages": persisted_messages,
            "approval": None,
            "askUser": None,
            "activity": [],
            "status": None,
            "subagents": [],
            "isRunning": False,
        }
        if pending is not None:
            state.update(copy.deepcopy(pending.state))
            pending_messages = pending.state.get("messages")
            if isinstance(pending_messages, list) and len(pending_messages) >= len(
                persisted_messages
            ):
                state["messages"] = copy.deepcopy(pending_messages)
            else:
                state["messages"] = persisted_messages

        runtime_state = self._runtime.snapshot(thread_id)
        if runtime_state:
            for field in (
                "status",
                "isRunning",
                "subagents",
                "activity",
                "approval",
                "askUser",
            ):
                if field in runtime_state:
                    state[field] = copy.deepcopy(runtime_state[field])

        return web.json_response(
            {
                "threadId": thread_id,
                "title": (
                    str(metadata.get("workspace_dir", "") or "").split("/")[-1]
                    if metadata and str(metadata.get("workspace_dir", "") or "").strip()
                    else f"Session {thread_id}"
                ),
                "updatedAt": metadata.get("updated_at") if metadata else None,
                "state": state,
            },
            headers=self._cors_headers(request),
        )

    async def _handle_thread_events(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        thread_id = str(request.query.get("threadId", "") or "").strip()
        if not thread_id:
            return web.json_response(
                {"error": "threadId is required"},
                status=400,
                headers=self._cors_headers(request),
            )

        cursor_raw = str(request.query.get("cursor", "0") or "0").strip()
        limit_raw = str(request.query.get("limit", "200") or "200").strip()
        try:
            cursor = max(0, int(cursor_raw))
        except ValueError:
            cursor = 0
        try:
            limit = max(1, min(500, int(limit_raw)))
        except ValueError:
            limit = 200

        latest, events = self._runtime.events_since(thread_id, cursor, limit=limit)
        return web.json_response(
            {
                "threadId": thread_id,
                "cursor": latest,
                "events": events,
                "state": self._runtime.snapshot(thread_id),
            },
            headers=self._cors_headers(request),
        )

    def _build_command_line(self, payload: dict[str, Any]) -> str | None:
        raw_command = payload.get("command")
        if isinstance(raw_command, str) and raw_command.strip():
            command = raw_command.strip()
            if not command.startswith("/"):
                command = f"/{command}"
            return command

        name = str(payload.get("name", "") or "").strip()
        if not name:
            return None
        if not name.startswith("/"):
            name = f"/{name}"

        args_raw = payload.get("args", [])
        args: list[str] = []
        if isinstance(args_raw, str):
            args = [args_raw.strip()] if args_raw.strip() else []
        elif isinstance(args_raw, list):
            args = [str(item) for item in args_raw if str(item).strip()]

        if not args:
            return name
        return " ".join([name, *[shlex.quote(arg) for arg in args]])

    def _resolve_command_agent(self) -> Any:
        try:
            from ...cli import channel as cli_channel

            return getattr(cli_channel, "_cli_agent", None)
        except Exception:
            return None

    async def _resolve_command_workspace(self, thread_id: str) -> str | None:
        try:
            from ...sessions import get_thread_metadata

            metadata = await get_thread_metadata(thread_id)
        except Exception:
            metadata = None
        workspace = (metadata or {}).get("workspace_dir")
        if isinstance(workspace, str) and workspace.strip():
            return workspace
        return None

    async def _handle_commands_catalog(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        return web.json_response(
            {"commands": build_command_catalog()},
            headers=self._cors_headers(request),
        )

    async def _handle_commands_execute(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"error": "invalid json"},
                status=400,
                headers=self._cors_headers(request),
            )

        if not isinstance(payload, dict):
            return web.json_response(
                {"error": "payload must be a JSON object"},
                status=400,
                headers=self._cors_headers(request),
            )

        thread_id = str(
            request.headers.get("X-Thread-Id", "")
            or payload.get("threadId")
            or payload.get("clientThreadId")
            or ""
        ).strip()
        if not thread_id:
            return web.json_response(
                {"error": "threadId is required"},
                status=400,
                headers=self._cors_headers(request),
            )

        command_line = self._build_command_line(payload)
        if not command_line:
            return web.json_response(
                {"error": "command is required"},
                status=400,
                headers=self._cors_headers(request),
            )

        workspace_dir = await self._resolve_command_workspace(thread_id)
        result = await execute_command_line(
            command_line,
            thread_id=thread_id,
            agent=self._resolve_command_agent(),
            workspace_dir=workspace_dir,
        )
        result_payload = result.to_dict()
        result_payload["runtimeState"] = self._runtime.snapshot(
            result.resolved_thread_id
        )
        if not result.executed:
            result_payload["ok"] = False
            result_payload["error"] = f"unknown command: {command_line}"
            return web.json_response(
                result_payload,
                status=404,
                headers=self._cors_headers(request),
            )

        return web.json_response(result_payload, headers=self._cors_headers(request))

    async def _handle_ui_skills(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        payload = await get_skills_overview(
            tag=str(request.query.get("tag", "") or ""),
            search=str(request.query.get("search", "") or ""),
        )
        return web.json_response(payload, headers=self._cors_headers(request))

    async def _handle_ui_skills_install(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"error": "invalid json"},
                status=400,
                headers=self._cors_headers(request),
            )

        if not isinstance(payload, dict):
            return web.json_response(
                {"error": "payload must be an object"},
                status=400,
                headers=self._cors_headers(request),
            )

        sources_raw = payload.get("sources", [])
        sources: list[str] = []
        if isinstance(sources_raw, list):
            sources.extend(str(item) for item in sources_raw if str(item).strip())
        source = str(payload.get("source", "") or "").strip()
        if source:
            sources.append(source)

        result = await install_skills(
            sources=sources,
            local=bool(payload.get("local", False)),
        )
        status = 200 if result.get("ok", False) else 400
        return web.json_response(
            result,
            status=status,
            headers=self._cors_headers(request),
        )

    async def _handle_ui_skills_uninstall(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"error": "invalid json"},
                status=400,
                headers=self._cors_headers(request),
            )

        if not isinstance(payload, dict):
            return web.json_response(
                {"error": "payload must be an object"},
                status=400,
                headers=self._cors_headers(request),
            )

        result = await uninstall_skill(str(payload.get("name", "") or ""))
        status = 200 if result.get("ok", False) else 400
        return web.json_response(
            result,
            status=status,
            headers=self._cors_headers(request),
        )

    async def _handle_ui_mcp(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        payload = await get_mcp_overview(
            tag=str(request.query.get("tag", "") or ""),
            search=str(request.query.get("search", "") or ""),
        )
        return web.json_response(payload, headers=self._cors_headers(request))

    async def _handle_ui_mcp_install(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"error": "invalid json"},
                status=400,
                headers=self._cors_headers(request),
            )

        if not isinstance(payload, dict):
            return web.json_response(
                {"error": "payload must be an object"},
                status=400,
                headers=self._cors_headers(request),
            )

        names_raw = payload.get("names", [])
        names: list[str] = []
        if isinstance(names_raw, list):
            names.extend(str(item) for item in names_raw if str(item).strip())
        name = str(payload.get("name", "") or "").strip()
        if name:
            names.append(name)

        result = await install_mcp_servers(names)
        status = 200 if result.get("ok", False) else 400
        return web.json_response(
            result,
            status=status,
            headers=self._cors_headers(request),
        )

    async def _handle_ui_mcp_remove(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"error": "invalid json"},
                status=400,
                headers=self._cors_headers(request),
            )

        if not isinstance(payload, dict):
            return web.json_response(
                {"error": "payload must be an object"},
                status=400,
                headers=self._cors_headers(request),
            )

        result = await remove_mcp(str(payload.get("name", "") or ""))
        status = 200 if result.get("ok", False) else 400
        return web.json_response(
            result,
            status=status,
            headers=self._cors_headers(request),
        )

    async def _handle_ui_channels(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        payload = await asyncio.to_thread(get_channels_overview)
        return web.json_response(payload, headers=self._cors_headers(request))

    async def _handle_ui_channels_start(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"error": "invalid json"},
                status=400,
                headers=self._cors_headers(request),
            )

        if not isinstance(payload, dict):
            return web.json_response(
                {"error": "payload must be an object"},
                status=400,
                headers=self._cors_headers(request),
            )

        channel_types_raw = payload.get("channelTypes", [])
        channel_types: list[str] = []
        if isinstance(channel_types_raw, list):
            channel_types.extend(
                str(item) for item in channel_types_raw if str(item).strip()
            )
        channel_type = str(payload.get("channelType", "") or "").strip()
        if channel_type:
            channel_types.append(channel_type)

        thread_id = str(
            payload.get("threadId") or request.headers.get("X-Thread-Id", "") or ""
        ).strip()
        result = await asyncio.to_thread(
            start_channels,
            channel_types,
            fallback_thread_id=thread_id,
            persist=bool(payload.get("persist", False)),
        )
        status = 200 if result.get("ok", False) else 400
        return web.json_response(
            result,
            status=status,
            headers=self._cors_headers(request),
        )

    async def _handle_ui_channels_stop(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"error": "invalid json"},
                status=400,
                headers=self._cors_headers(request),
            )

        if not isinstance(payload, dict):
            return web.json_response(
                {"error": "payload must be an object"},
                status=400,
                headers=self._cors_headers(request),
            )

        result = await asyncio.to_thread(
            stop_channels,
            channel_type=str(payload.get("channelType", "") or ""),
            persist=bool(payload.get("persist", False)),
        )
        status = 200 if result.get("ok", False) else 400
        return web.json_response(
            result,
            status=status,
            headers=self._cors_headers(request),
        )

    async def _handle_ui_session_shutdown(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"error": "invalid json"},
                status=400,
                headers=self._cors_headers(request),
            )

        if not isinstance(payload, dict) or not payload.get("confirm", False):
            return web.json_response(
                {"error": "confirm=true is required"},
                status=400,
                headers=self._cors_headers(request),
            )

        schedule_shutdown()
        return web.json_response(
            {"ok": True, "message": "shutdown scheduled"},
            headers=self._cors_headers(request),
        )

    async def _handle_run_stop(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        payload: dict[str, Any] = {}
        try:
            parsed = await request.json()
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = {}

        thread_id = str(
            request.headers.get("X-Thread-Id", "")
            or payload.get("threadId")
            or payload.get("clientThreadId")
            or request.query.get("threadId", "")
            or ""
        ).strip()
        if not thread_id:
            return web.json_response(
                {"error": "threadId is required"},
                status=400,
                headers=self._cors_headers(request),
            )

        self._runtime.request_cancel(thread_id)

        with self._approval_lock:
            approval_waiter = self._approval_waiters.get(thread_id)
            if approval_waiter is not None:
                approval_waiter.decision = "reject"
                approval_waiter.event.set()

        with self._ask_user_lock:
            ask_user_waiter = self._ask_user_waiters.get(thread_id)
            if ask_user_waiter is not None:
                ask_user_waiter.reply = "cancel"
                ask_user_waiter.event.set()

        pending = await self._get_pending_run(thread_id)
        if pending is not None:
            pending.state["approval"] = None
            pending.state["askUser"] = None
            pending.state["status"] = "Stopping..."
            pending.state["isRunning"] = True
            await pending.queue.put(
                (
                    "ops",
                    [
                        {"type": "set", "path": ["approval"], "value": None},
                        {"type": "set", "path": ["askUser"], "value": None},
                        {"type": "set", "path": ["status"], "value": "Stopping..."},
                        {"type": "set", "path": ["isRunning"], "value": True},
                    ],
                )
            )

        return web.json_response(
            {"ok": True, "threadId": thread_id},
            headers=self._cors_headers(request),
        )

    async def _handle_assistant_options(self, request):
        from aiohttp import web

        return web.Response(status=204, headers=self._cors_headers(request))

    async def _handle_approval_options(self, request):
        from aiohttp import web

        return web.Response(status=204, headers=self._cors_headers(request))

    async def _handle_ask_user_options(self, request):
        from aiohttp import web

        return web.Response(status=204, headers=self._cors_headers(request))

    def _set_approval_state_nowait(
        self,
        thread_id: str,
        approval: dict[str, Any] | None,
    ) -> bool:
        self._runtime.set_prompt(thread_id, approval=approval)

        def _build_ops(pending: _PendingRun) -> list[dict[str, Any]]:
            pending.state["approval"] = approval
            return [{"type": "set", "path": ["approval"], "value": approval}]

        return self._enqueue_pending_ops_nowait(thread_id, _build_ops)

    def _set_ask_user_state_nowait(
        self,
        thread_id: str,
        ask_user: dict[str, Any] | None,
    ) -> bool:
        self._runtime.set_prompt(thread_id, ask_user=ask_user)

        def _build_ops(pending: _PendingRun) -> list[dict[str, Any]]:
            pending.state["askUser"] = ask_user
            return [{"type": "set", "path": ["askUser"], "value": ask_user}]

        return self._enqueue_pending_ops_nowait(thread_id, _build_ops)

    def prompt_approval(
        self,
        *,
        thread_id: str,
        action_requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        waiter = _PendingApproval(
            event=threading.Event(),
            count=max(1, len(action_requests)),
        )
        with self._approval_lock:
            self._approval_waiters[thread_id] = waiter

        approval_payload = {
            "kind": "tool_approval",
            "title": "Approval Required",
            "options": [
                {"id": "approve", "label": "Approve"},
                {"id": "reject", "label": "Reject"},
                {"id": "auto", "label": "Approve All"},
            ],
            "actions": [
                {
                    "name": (
                        req.get("name", "")
                        if isinstance(req, dict)
                        else getattr(req, "name", "")
                    ),
                    "command": (
                        (req.get("args", {}) or {}).get("command", "")
                        if isinstance(req, dict)
                        else getattr(req, "args", {}).get("command", "")
                    ),
                }
                for req in action_requests
            ],
        }
        self._set_approval_state_nowait(thread_id, approval_payload)

        try:
            approved = waiter.event.wait(timeout=120.0)
            decision = waiter.decision if approved else "reject"
        finally:
            with self._approval_lock:
                self._approval_waiters.pop(thread_id, None)
            self._set_approval_state_nowait(thread_id, None)

        if decision == "auto":
            return [{"type": "approve"} for _ in range(waiter.count)]
        if decision == "approve":
            return [{"type": "approve"} for _ in range(waiter.count)]
        return None

    async def _handle_approval(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"error": "invalid json"},
                status=400,
                headers=self._cors_headers(request),
            )

        thread_id = str(
            request.headers.get("X-Thread-Id", "")
            or payload.get("threadId")
            or payload.get("clientThreadId")
            or ""
        ).strip()
        decision = str(payload.get("decision", "")).strip().lower()
        if not thread_id or decision not in {"approve", "reject", "auto"}:
            return web.json_response(
                {"error": "threadId and valid decision are required"},
                status=400,
                headers=self._cors_headers(request),
            )

        with self._approval_lock:
            waiter = self._approval_waiters.get(thread_id)
            if waiter is None:
                self._runtime.submit_approval_decision(thread_id, decision)
                self._runtime.set_prompt(thread_id, approval=None)
                return web.json_response(
                    {"ok": True, "queued": True},
                    headers=self._cors_headers(request),
                )
            waiter.decision = decision
            waiter.event.set()

        return web.json_response({"ok": True}, headers=self._cors_headers(request))

    def prompt_ask_user(
        self,
        *,
        thread_id: str,
        ask_user_data: dict[str, Any],
    ) -> dict[str, Any]:
        questions = ask_user_data.get("questions", [])
        if not questions:
            return {"answers": [], "status": "answered"}

        total = len(questions)
        answers: list[str] = []

        for index, question in enumerate(questions):
            waiter = _PendingAskUser(event=threading.Event())
            with self._ask_user_lock:
                self._ask_user_waiters[thread_id] = waiter

            q_type = str(question.get("type", "text") or "text")
            choices = question.get("choices", [])
            ask_user_payload = {
                "title": (
                    "Quick check-in from EvoScientist"
                    if total == 1
                    else f"Question {index + 1}/{total}"
                ),
                "question": str(question.get("question", "")),
                "required": bool(question.get("required", True)),
                "type": q_type,
                "choices": [
                    {"value": str(choice.get("value", str(choice)))}
                    for choice in choices
                    if isinstance(choice, dict) or isinstance(choice, str)
                ],
            }
            self._set_ask_user_state_nowait(thread_id, ask_user_payload)

            try:
                replied = waiter.event.wait(timeout=300.0)
                raw = (waiter.reply or "").strip() if replied else ""
            finally:
                with self._ask_user_lock:
                    self._ask_user_waiters.pop(thread_id, None)
                self._set_ask_user_state_nowait(thread_id, None)

            if not raw:
                return {"status": "cancelled"}
            if raw.lower() == "cancel":
                return {"status": "cancelled"}
            answers.append(raw)

        return {"answers": answers, "status": "answered"}

    async def _handle_ask_user(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"error": "invalid json"},
                status=400,
                headers=self._cors_headers(request),
            )

        thread_id = str(
            request.headers.get("X-Thread-Id", "")
            or payload.get("threadId")
            or payload.get("clientThreadId")
            or ""
        ).strip()
        reply = str(payload.get("reply", "")).strip()
        if not thread_id:
            return web.json_response(
                {"error": "threadId is required"},
                status=400,
                headers=self._cors_headers(request),
            )

        with self._ask_user_lock:
            waiter = self._ask_user_waiters.get(thread_id)
            if waiter is None:
                self._runtime.submit_ask_user_reply(thread_id, reply)
                self._runtime.set_prompt(thread_id, ask_user=None)
                return web.json_response(
                    {"ok": True, "queued": True},
                    headers=self._cors_headers(request),
                )
            waiter.reply = reply
            waiter.event.set()

        return web.json_response({"ok": True}, headers=self._cors_headers(request))

    def _check_auth(self, request) -> bool:
        if not self.config.api_key:
            return True
        auth_header = request.headers.get("Authorization", "")
        if auth_header == f"Bearer {self.config.api_key}":
            return True
        return request.headers.get("X-API-Key", "") == self.config.api_key

    def _normalize_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = copy.deepcopy(payload.get("state") or {})
        if not isinstance(state, dict):
            state = {}
        messages = state.get("messages")
        if not isinstance(messages, list):
            state["messages"] = []
        if not isinstance(state.get("subagents"), list):
            state["subagents"] = []
        if not isinstance(state.get("activity"), list):
            state["activity"] = []
        if not isinstance(state.get("approval"), dict | type(None)):
            state["approval"] = None
        if not isinstance(state.get("askUser"), dict | type(None)):
            state["askUser"] = None
        state.setdefault("status", None)
        state.setdefault("isRunning", False)
        state.setdefault("approval", None)
        state.setdefault("askUser", None)
        return state

    def _apply_commands(
        self,
        state: dict[str, Any],
        commands: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        messages = state.setdefault("messages", [])
        latest_text = ""

        for command in commands:
            if not isinstance(command, dict):
                continue
            if command.get("type") != "add-message":
                continue
            # Keep webui transcript append-only. assistant-ui may include parentId
            # for branching/edit flows, but the current EvoScientist webui expects
            # linear chat history and should not truncate prior turns.

            message = command.get("message") or {}
            if not isinstance(message, dict):
                continue
            message_id = message.get("id") or f"user_{uuid.uuid4().hex}"
            message_text = _text_from_transport_message(message)
            latest_text = message_text or latest_text
            messages.append(
                {
                    "id": message_id,
                    "role": "user",
                    "content": message_text,
                    "createdAt": _utc_now_iso(),
                }
            )

        assistant_message_id = f"assistant_{uuid.uuid4().hex}"
        messages.append(
            {
                "id": assistant_message_id,
                "role": "assistant",
                "content": "",
                "createdAt": _utc_now_iso(),
            }
        )
        state["status"] = None
        state["isRunning"] = True
        return state, latest_text

    async def _register_pending_run(self, pending: _PendingRun) -> bool:
        async with self._pending_lock:
            if pending.thread_id in self._pending_runs:
                return False
            self._pending_runs[pending.thread_id] = pending
            return True

    async def _get_pending_run(self, thread_id: str) -> _PendingRun | None:
        async with self._pending_lock:
            return self._pending_runs.get(thread_id)

    async def _pop_pending_run(self, thread_id: str) -> _PendingRun | None:
        async with self._pending_lock:
            return self._pending_runs.pop(thread_id, None)

    def _enqueue_pending_ops_nowait(
        self,
        thread_id: str,
        builder: Any,
    ) -> bool:
        """Schedule a synchronous pending-run update onto the channel loop."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return False

        def _apply() -> None:
            pending = self._pending_runs.get(thread_id)
            if pending is None:
                return
            try:
                ops = builder(pending)
            except Exception:
                logger.exception("Failed to build webui pending-run ops")
                return
            if not ops:
                return
            pending.queue.put_nowait(("ops", ops))

        loop.call_soon_threadsafe(_apply)
        return True

    def send_thinking_message_nowait(
        self,
        sender: str,
        thinking: str,
        metadata: dict | None = None,
    ) -> bool:
        thread_id = str((metadata or {}).get("chat_id", sender) or sender)
        self._runtime.set_status(thread_id, status=thinking, is_running=True)

        def _build_ops(pending: _PendingRun) -> list[dict[str, Any]]:
            pending.state["status"] = thinking
            return [{"type": "set", "path": ["status"], "value": thinking}]

        return self._enqueue_pending_ops_nowait(thread_id, _build_ops)

    def send_stream_state_nowait(
        self,
        sender: str,
        stream_state: Any,
        metadata: dict | None = None,
    ) -> bool:
        thread_id = str((metadata or {}).get("chat_id", sender) or sender)
        self._runtime.update_subagents_from_stream_state(thread_id, stream_state)
        thinking_text = str(getattr(stream_state, "thinking_text", "") or "").rstrip()
        is_thinking = bool(getattr(stream_state, "is_thinking", False))
        status_value = thinking_text if is_thinking and thinking_text else None

        next_subagents: list[dict[str, Any]] = []
        for subagent in getattr(stream_state, "subagents", []) or []:
            name = str(getattr(subagent, "name", "sub-agent") or "sub-agent")
            description = str(getattr(subagent, "description", "") or "")
            is_active = bool(getattr(subagent, "is_active", False))
            tool_calls = list(getattr(subagent, "tool_calls", []) or [])
            detail = ""
            if tool_calls:
                last_tool = tool_calls[-1]
                detail = f"Tool: {last_tool.get('name', 'unknown')}"

            next_subagents.append(
                {
                    "id": name,
                    "name": name,
                    "description": description,
                    "status": "running" if is_active else "complete",
                    "detail": detail,
                }
            )

        def _build_ops(pending: _PendingRun) -> list[dict[str, Any]]:
            pending.state["status"] = status_value
            pending.state["subagents"] = next_subagents
            return [
                {"type": "set", "path": ["status"], "value": status_value},
                {"type": "set", "path": ["subagents"], "value": next_subagents},
            ]

        return self._enqueue_pending_ops_nowait(thread_id, _build_ops)

    async def _publish_inbound(
        self,
        *,
        thread_id: str,
        content: str,
        request_id: str,
    ) -> None:
        if not self._bus:
            raise ChannelError("webui channel is not attached to a MessageBus")
        inbound = InboundMessage(
            channel=self.name,
            sender_id=thread_id,
            chat_id=thread_id,
            content=content,
            metadata={
                "chat_id": thread_id,
                "thread_id": thread_id,
                "webui_request_id": request_id,
            },
        )
        await self._bus.publish_inbound(inbound)

    async def _stream_pending_run(self, request, pending: _PendingRun):
        from aiohttp import web

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                **self._cors_headers(request),
            },
        )
        await response.prepare(request)

        try:
            while True:
                item_type, payload = await pending.queue.get()
                if item_type == "ops":
                    await response.write(
                        _sse_data(
                            {
                                "type": "update-state",
                                "path": [],
                                "operations": payload,
                            }
                        )
                    )
                    await response.drain()
                    continue
                if item_type == "error":
                    await response.write(
                        _sse_data(
                            {
                                "type": "error",
                                "path": [],
                                "error": str(payload),
                            }
                        )
                    )
                    await response.drain()
                    break
                if item_type == "close":
                    break
        finally:
            await self._pop_pending_run(pending.thread_id)
            if response.prepared:
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
        return response

    async def _handle_assistant(self, request):
        from aiohttp import web

        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers=self._cors_headers(request),
            )

        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"error": "invalid json"},
                status=400,
                headers=self._cors_headers(request),
            )

        commands = payload.get("commands")
        if not isinstance(commands, list):
            return web.json_response(
                {"error": "commands must be a list"},
                status=400,
                headers=self._cors_headers(request),
            )

        header_thread_id = request.headers.get("X-Thread-Id", "")
        thread_id = (
            header_thread_id or payload.get("threadId") or payload.get("clientThreadId")
        )
        if not isinstance(thread_id, str) or not thread_id.strip():
            thread_id = f"thread_{uuid.uuid4().hex}"
        else:
            thread_id = thread_id.strip()

        state = self._normalize_state(payload)
        state, latest_text = self._apply_commands(state, commands)
        if not latest_text:
            return web.json_response(
                {"error": "no user text found in add-message command"},
                status=400,
                headers=self._cors_headers(request),
            )

        assistant_message = state["messages"][-1]
        pending = _PendingRun(
            thread_id=thread_id,
            state=state,
            assistant_message_id=assistant_message["id"],
        )
        if not await self._register_pending_run(pending):
            return web.json_response(
                {"error": f"thread '{thread_id}' already has an active run"},
                status=409,
                headers=self._cors_headers(request),
            )

        self._runtime.begin_run(thread_id)
        await pending.queue.put(("ops", [{"type": "set", "path": [], "value": state}]))

        try:
            await self._publish_inbound(
                thread_id=thread_id,
                content=latest_text,
                request_id=str(uuid.uuid4()),
            )
        except Exception as e:
            self._runtime.set_status(thread_id, status=None, is_running=False)
            await pending.queue.put(("error", f"failed to queue message: {e}"))
            await pending.queue.put(("close", None))

        return await self._stream_pending_run(request, pending)
