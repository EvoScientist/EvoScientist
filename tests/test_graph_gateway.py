"""Tests for the graph/thread gateway abstraction."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from EvoScientist.gateway import (
    GraphEvent,
    GraphGateway,
    LocalGraphGateway,
    RunRequest,
    ThreadResolution,
    ThreadStore,
)
from EvoScientist.stream import display as display_mod
from tests.conftest import run_async


def test_local_gateway_streams_from_injected_streamer():
    seen: dict[str, Any] = {}

    async def _streamer(agent, message, thread_id, **kwargs):
        seen.update(
            {
                "agent": agent,
                "message": message,
                "thread_id": thread_id,
                "metadata": kwargs.get("metadata"),
                "media": kwargs.get("media"),
            }
        )
        yield {"type": "text", "content": "hi"}
        yield {"type": "done", "response": "hi"}

    agent = MagicMock()
    gateway = LocalGraphGateway(agent)

    async def _collect():
        request = RunRequest(
            message="hello",
            thread_id="t1",
            metadata={"workspace_dir": "/tmp/ws"},
            media=["plot.png"],
        )
        return [event async for event in gateway.stream_events(request)]

    with patch("EvoScientist.stream.events.stream_agent_events", new=_streamer):
        events = run_async(_collect())

    assert events == [
        {"type": "text", "content": "hi"},
        {"type": "done", "response": "hi"},
    ]
    assert seen == {
        "agent": agent,
        "message": "hello",
        "thread_id": "t1",
        "metadata": {"workspace_dir": "/tmp/ws"},
        "media": ["plot.png"],
    }


def test_local_graph_gateway_delegates_thread_operations():
    class _ThreadStore(ThreadStore):
        def __init__(self):
            self.calls: list[tuple[str, Any]] = []

        def generate_thread_id(self) -> str:
            self.calls.append(("generate_thread_id", None))
            return "new12345"

        async def list_threads(
            self,
            *,
            limit: int = 20,
            include_message_count: bool = False,
            include_preview: bool = False,
        ) -> list[dict[str, Any]]:
            self.calls.append(
                (
                    "list_threads",
                    {
                        "limit": limit,
                        "include_message_count": include_message_count,
                        "include_preview": include_preview,
                    },
                )
            )
            return [{"thread_id": "abc12345"}]

        async def resolve_thread_id_prefix(
            self,
            thread_id_or_prefix: str,
        ) -> tuple[str | None, list[str]]:
            self.calls.append(("resolve_thread_id_prefix", thread_id_or_prefix))
            return "abc12345", []

        async def get_thread_metadata(self, thread_id: str) -> dict[str, Any] | None:
            self.calls.append(("get_thread_metadata", thread_id))
            return {"workspace_dir": "/tmp/ws"}

        async def get_thread_messages(self, thread_id: str) -> list[Any]:
            self.calls.append(("get_thread_messages", thread_id))
            return ["message"]

        async def thread_exists(self, thread_id: str) -> bool:
            self.calls.append(("thread_exists", thread_id))
            return True

        async def delete_thread(self, thread_id: str) -> bool:
            self.calls.append(("delete_thread", thread_id))
            return True

    thread_store = _ThreadStore()
    agent = MagicMock()

    async def _run():
        gateway = LocalGraphGateway(agent, thread_store=thread_store)
        resolution = await gateway.resolve_thread("abc")
        return {
            "created": await gateway.create_thread(),
            "threads": await gateway.list_threads(
                limit=3,
                include_message_count=True,
            ),
            "resolution": resolution,
            "metadata": await gateway.get_thread_metadata("abc12345"),
            "messages": await gateway.get_thread_messages("abc12345"),
            "exists": await gateway.thread_exists("abc12345"),
            "deleted": await gateway.delete_thread("abc12345"),
        }

    result = run_async(_run())

    assert result["created"] == "new12345"
    assert result["threads"] == [{"thread_id": "abc12345"}]
    assert result["resolution"].thread_id == "abc12345"
    assert result["resolution"].matches == ()
    assert result["resolution"].found
    assert not result["resolution"].ambiguous
    assert result["metadata"] == {"workspace_dir": "/tmp/ws"}
    assert result["messages"] == ["message"]
    assert result["exists"] is True
    assert result["deleted"] is True
    assert thread_store.calls == [
        ("resolve_thread_id_prefix", "abc"),
        ("generate_thread_id", None),
        (
            "list_threads",
            {
                "limit": 3,
                "include_message_count": True,
                "include_preview": False,
            },
        ),
        ("get_thread_metadata", "abc12345"),
        ("get_thread_messages", "abc12345"),
        ("thread_exists", "abc12345"),
        ("delete_thread", "abc12345"),
    ]


def test_run_streaming_can_consume_injected_gateway():
    class _Gateway(GraphGateway):
        def __init__(self):
            self.requests: list[RunRequest] = []

        async def create_thread(self) -> str:
            return "unused"

        async def list_threads(
            self,
            *,
            limit: int = 20,
            include_message_count: bool = False,
            include_preview: bool = False,
        ) -> list[dict[str, Any]]:
            return []

        async def resolve_thread(self, thread_id_or_prefix: str) -> ThreadResolution:
            raise AssertionError("resolve_thread should not be called")

        async def get_thread_metadata(self, thread_id: str) -> dict[str, Any] | None:
            raise AssertionError("get_thread_metadata should not be called")

        async def get_thread_messages(self, thread_id: str) -> list[Any]:
            raise AssertionError("get_thread_messages should not be called")

        async def thread_exists(self, thread_id: str) -> bool:
            raise AssertionError("thread_exists should not be called")

        async def delete_thread(self, thread_id: str) -> bool:
            raise AssertionError("delete_thread should not be called")

        def stream_events(self, request: RunRequest) -> AsyncIterator[GraphEvent]:
            self.requests.append(request)

            async def _events():
                yield {"type": "text", "content": "gateway-ok"}
                yield {"type": "done", "response": "gateway-ok"}

            return _events()

    gateway: GraphGateway = _Gateway()

    with patch("EvoScientist.stream.display.Live"):
        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            metadata={"workspace_dir": "/tmp/ws"},
            gateway=gateway,
        )

    assert result == "gateway-ok"
    assert gateway.requests == [
        RunRequest(
            message="hello",
            thread_id="t1",
            metadata={"workspace_dir": "/tmp/ws"},
        )
    ]


def test_resume_command_consumes_context_gateway():
    from EvoScientist.commands.base import CommandContext
    from EvoScientist.commands.implementation.session import ResumeCommand

    class _ThreadStore(ThreadStore):
        def generate_thread_id(self) -> str:
            return "unused"

        async def list_threads(
            self,
            *,
            limit: int = 20,
            include_message_count: bool = False,
            include_preview: bool = False,
        ) -> list[dict[str, Any]]:
            return []

        async def resolve_thread_id_prefix(
            self,
            thread_id_or_prefix: str,
        ) -> tuple[str | None, list[str]]:
            assert thread_id_or_prefix == "abc"
            return "abc12345", []

        async def get_thread_metadata(self, thread_id: str) -> dict[str, Any] | None:
            assert thread_id == "abc12345"
            return {"workspace_dir": "/restored"}

        async def get_thread_messages(self, thread_id: str) -> list[Any]:
            return []

        async def thread_exists(self, thread_id: str) -> bool:
            return False

        async def delete_thread(self, thread_id: str) -> bool:
            return False

    ui = MagicMock()
    ui.handle_session_resume = AsyncMock()
    thread_store: ThreadStore = _ThreadStore()
    ctx = CommandContext(
        agent=None,
        thread_id="current",
        ui=ui,
        workspace_dir="/old",
        thread_store=thread_store,
    )

    run_async(ResumeCommand().execute(ctx, ["abc"]))

    assert ctx.thread_id == "abc12345"
    assert ctx.workspace_dir == "/restored"
    ui.handle_session_resume.assert_awaited_once_with("abc12345", "/restored")
