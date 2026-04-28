import asyncio
from types import SimpleNamespace

from EvoScientist.channels.bus import MessageBus
from EvoScientist.channels.bus.events import InboundMessage, OutboundMessage
from EvoScientist.channels.channel_manager import available_channels
from EvoScientist.channels.consumer import InboundConsumer
from EvoScientist.channels.webui import create_from_config
from EvoScientist.channels.webui.channel import WebUIChannel, WebUIConfig, _PendingRun
from EvoScientist.runtime.command_runtime import (
    build_command_catalog,
    execute_command_line,
)
from EvoScientist.runtime.native_ui import get_workspace_tree
from EvoScientist.runtime.thread_registry import ThreadRuntimeRegistry


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def test_webui_registered():
    assert "webui" in available_channels()


def test_webui_create_from_config():
    config = SimpleNamespace(
        webui_port=8123,
        webui_api_key="secret",
        webui_base_path="/assistant",
    )
    channel = create_from_config(config)
    assert isinstance(channel, WebUIChannel)
    assert channel.config.webhook_port == 8123
    assert channel.config.api_key == "secret"
    assert channel.config.base_path == "/assistant"


def test_webui_apply_commands_is_append_only():
    channel = WebUIChannel(WebUIConfig())
    state = {
        "messages": [
            {"id": "m1", "role": "user", "content": "old"},
            {"id": "m2", "role": "assistant", "content": "reply"},
            {"id": "m3", "role": "user", "content": "stale"},
        ]
    }
    updated, latest_text = channel._apply_commands(
        state,
        [
            {
                "type": "add-message",
                "parentId": "m1",
                "message": {
                    "id": "u2",
                    "parts": [{"type": "text", "text": "new prompt"}],
                },
            }
        ],
    )

    assert latest_text == "new prompt"
    assert [m["id"] for m in updated["messages"][:4]] == ["m1", "m2", "m3", "u2"]
    assert updated["messages"][-1]["role"] == "assistant"
    assert updated["isRunning"] is True


def test_webui_check_auth():
    channel = WebUIChannel(WebUIConfig(api_key="secret"))
    ok_request = SimpleNamespace(
        headers={"Authorization": "Bearer secret", "X-API-Key": ""},
    )
    bad_request = SimpleNamespace(headers={"Authorization": "Bearer nope"})

    assert channel._check_auth(ok_request) is True
    assert channel._check_auth(bad_request) is False


def test_extract_thread_id_prefers_query_over_header():
    channel = WebUIChannel(WebUIConfig())
    request = SimpleNamespace(
        headers={"X-Thread-Id": "cli-thread"},
        query={"threadId": "webui-thread"},
    )

    thread_id = channel._extract_thread_id(request, {"threadId": "payload-thread"})
    assert thread_id == "webui-thread"


def test_handle_assistant_prefers_payload_thread_over_header():
    async def _case():
        channel = WebUIChannel(WebUIConfig())
        payload = {
            "threadId": "payload-thread",
            "commands": [
                {
                    "type": "add-message",
                    "message": {
                        "id": "u1",
                        "parts": [{"type": "text", "text": "hello"}],
                    },
                }
            ],
        }

        class _Request:
            def __init__(self) -> None:
                self.headers = {"X-Thread-Id": "header-thread"}
                self.query: dict[str, str] = {}

            async def json(self):
                return payload

        captured: dict[str, str] = {}

        async def _register(pending):
            captured["pending_thread_id"] = pending.thread_id
            return True

        async def _resolve_workspace(thread_id: str):
            captured["workspace_thread_id"] = thread_id
            return "/tmp/workspace-payload"

        async def _publish_inbound(**kwargs):
            captured["inbound_thread_id"] = str(kwargs.get("thread_id", ""))

        async def _stream_pending(_request, pending):
            return pending.thread_id

        channel._register_pending_run = _register
        channel._resolve_or_create_workspace = _resolve_workspace
        channel._publish_inbound = _publish_inbound
        channel._stream_pending_run = _stream_pending

        result = await channel._handle_assistant(_Request())
        assert result == "payload-thread"
        assert captured["pending_thread_id"] == "payload-thread"
        assert captured["workspace_thread_id"] == "payload-thread"
        assert captured["inbound_thread_id"] == "payload-thread"

    _run(_case())


def test_workspace_name_for_thread_is_sanitized():
    name = WebUIChannel._workspace_name_for_thread("webui:thread/with spaces")
    assert name.startswith("webui_")
    assert ":" not in name
    assert "/" not in name
    assert " " not in name
    assert (
        WebUIChannel._workspace_name_for_thread(
            "webui_fe51316f-3f2c-4910-af37-33a1fb8b173b"
        )
        == "webui_fe51316f-3f2c-4910-af37-33a1fb8b173b"
    )


def test_workspace_tree_accepts_explicit_workspace_without_thread_metadata(tmp_path):
    (tmp_path / "notes.md").write_text("# Notes\n", encoding="utf-8")

    async def _case():
        payload = await get_workspace_tree(
            thread_id="new-webui-thread",
            workspace_dir=str(tmp_path),
        )
        assert payload["threadId"] == "new-webui-thread"
        assert payload["workspaceDir"] == str(tmp_path)
        assert [entry["name"] for entry in payload["entries"]] == ["notes.md"]

    _run(_case())


def test_webui_files_tree_creates_workspace_for_new_thread(monkeypatch):
    async def _case():
        channel = WebUIChannel(WebUIConfig())
        captured: dict[str, str] = {}

        class _Request:
            def __init__(self) -> None:
                self.headers = {"Origin": "http://localhost:3000"}
                self.query = {"threadId": "new-thread", "path": ""}

        async def _resolve_workspace(thread_id: str):
            captured["resolved_thread_id"] = thread_id
            return "/tmp/new-thread-workspace"

        async def _get_workspace_tree(**kwargs):
            captured["tree_thread_id"] = kwargs["thread_id"]
            captured["tree_workspace_dir"] = kwargs["workspace_dir"]
            return {
                "ok": True,
                "threadId": kwargs["thread_id"],
                "workspaceDir": kwargs["workspace_dir"],
                "path": kwargs["relative_path"],
                "entries": [],
            }

        monkeypatch.setattr(channel, "_resolve_or_create_workspace", _resolve_workspace)
        monkeypatch.setattr(
            "EvoScientist.channels.webui.channel.get_workspace_tree",
            _get_workspace_tree,
        )

        response = await channel._handle_ui_files_tree(_Request())

        assert response.status == 200
        assert captured["resolved_thread_id"] == "new-thread"
        assert captured["tree_thread_id"] == "new-thread"
        assert captured["tree_workspace_dir"] == "/tmp/new-thread-workspace"

    _run(_case())


def test_webui_send_updates_pending_run():
    async def _case():
        channel = WebUIChannel(WebUIConfig())
        pending = _PendingRun(
            thread_id="thread-1",
            state={
                "messages": [
                    {
                        "id": "assistant-1",
                        "role": "assistant",
                        "content": "",
                    }
                ],
                "status": "thinking",
                "isRunning": True,
            },
            assistant_message_id="assistant-1",
        )
        await channel._register_pending_run(pending)

        result = await channel.send(
            OutboundMessage(
                channel="webui",
                chat_id="thread-1",
                content="hello world",
            )
        )
        assert result is True

        item1 = await pending.queue.get()
        item2 = await pending.queue.get()
        item3 = await pending.queue.get()

        assert item1[0] == "ops"
        assert item2 == (
            "ops",
            [{"type": "set", "path": ["isRunning"], "value": False}],
        )
        assert item3 == ("close", None)
        assert pending.state["messages"][0]["content"] == "hello world"
        assert pending.state["isRunning"] is False

    _run(_case())


def test_publish_inbound_includes_workspace_dir():
    async def _case():
        channel = WebUIChannel(WebUIConfig())

        class _CaptureBus:
            captured = None

            async def publish_inbound(self, msg):
                self.captured = msg

        bus = _CaptureBus()
        channel.set_bus(bus)
        await channel._publish_inbound(
            thread_id="thread-1",
            content="hello",
            request_id="req-1",
            workspace_dir="/tmp/workspace-one",
        )

        assert bus.captured is not None
        assert bus.captured.metadata["thread_id"] == "thread-1"
        assert bus.captured.metadata["workspace_dir"] == "/tmp/workspace-one"

    _run(_case())


def test_consumer_prefers_explicit_thread_id():
    consumer = InboundConsumer(
        bus=MessageBus(),
        manager=SimpleNamespace(get_channel=lambda _name: None),
        agent=object(),
        thread_id="fallback",
    )
    msg = InboundMessage(
        channel="webui",
        sender_id="sender-1",
        chat_id="chat-1",
        content="hello",
        metadata={"thread_id": "thread-123"},
    )

    assert consumer._resolve_thread_id(msg) == "thread-123"


def test_consumer_passes_inbound_metadata_to_stream(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_stream_agent_events(
        agent,
        message,
        thread_id,
        metadata=None,
        media=None,
    ):
        captured["thread_id"] = thread_id
        captured["metadata"] = metadata
        yield {"type": "done", "content": "ok"}

    monkeypatch.setattr(
        "EvoScientist.stream.events.stream_agent_events",
        _fake_stream_agent_events,
    )

    async def _case():
        bus = MessageBus()
        consumer = InboundConsumer(
            bus=bus,
            manager=SimpleNamespace(get_channel=lambda _name: None),
            agent=object(),
            thread_id="fallback",
        )
        msg = InboundMessage(
            channel="webui",
            sender_id="sender-1",
            chat_id="chat-1",
            content="hello",
            metadata={"thread_id": "thread-123", "workspace_dir": "/tmp/ws"},
        )
        await consumer._stream_with_hitl(msg, None, "thread-123", msg.session_key)
        outbound = await bus.consume_outbound()
        assert outbound.content == "ok"

    _run(_case())
    assert captured["thread_id"] == "thread-123"
    assert isinstance(captured["metadata"], dict)
    assert captured["metadata"]["workspace_dir"] == "/tmp/ws"


def test_command_catalog_exposes_help():
    catalog = build_command_catalog()
    names = {entry["name"] for entry in catalog}
    assert "/help" in names


def test_command_catalog_exposes_native_ui_actions():
    catalog = build_command_catalog()
    by_name = {entry["name"]: entry for entry in catalog}

    assert by_name["/threads"]["nativeAction"] == "show_threads"
    assert by_name["/evoskills"]["nativeAction"] == "browse_skills"
    assert by_name["/install-mcp"]["nativeAction"] == "browse_mcp"
    assert by_name["/channel"]["nativeAction"] == "manage_channels"
    assert by_name["/exit"]["nativeAction"] == "exit_session"


def test_webui_routes_include_native_ui_endpoints():
    channel = WebUIChannel(WebUIConfig())
    routes = {(method, path) for method, path, _handler in channel._webhook_routes()}

    assert ("GET", "/webui/ui/skills") in routes
    assert ("POST", "/webui/ui/skills/install") in routes
    assert ("GET", "/webui/ui/mcp") in routes
    assert ("POST", "/webui/ui/mcp/install") in routes
    assert ("GET", "/webui/ui/channels") in routes
    assert ("POST", "/webui/ui/channels/start") in routes
    assert ("GET", "/webui/ui/files/tree") in routes
    assert ("GET", "/webui/ui/files/read") in routes
    assert ("GET", "/webui/ui/files/download-all") in routes
    assert ("POST", "/webui/ui/session/shutdown") in routes


def test_execute_command_line_help():
    async def _case():
        result = await execute_command_line(
            "/help",
            thread_id="thread-1",
            agent=object(),
            workspace_dir=None,
        )
        assert result.executed is True
        assert result.command == "/help"
        assert result.outputs

    _run(_case())


def test_runtime_registry_tracks_file_tool_calls():
    registry = ThreadRuntimeRegistry()
    thread_id = "thread-write-file"

    registry.begin_run(thread_id)
    registry.apply_tool_event(
        thread_id,
        "tool_call",
        {
            "id": "tool-1",
            "name": "write_file",
            "args": {"path": "reports/summary.md"},
        },
    )
    registry.apply_tool_event(
        thread_id,
        "tool_call",
        {
            "id": "tool-1",
            "name": "write_file",
            "args": {"path": "reports/summary.md"},
        },
    )
    registry.apply_tool_event(
        thread_id,
        "tool_call",
        {
            "id": "tool-2",
            "name": "edit_file",
            "args": {"path": "reports/summary.md"},
        },
    )

    state = registry.snapshot(thread_id)
    file_events = [
        item
        for item in state.get("activity", [])
        if isinstance(item, dict) and item.get("type") in {"write_file", "edit_file"}
    ]

    assert len(file_events) == 2
    assert all(item["filePath"] == "reports/summary.md" for item in file_events)
