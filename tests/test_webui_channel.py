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
