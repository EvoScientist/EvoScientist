"""Ordering and shutdown coverage for non-blocking channel callback sends."""

from __future__ import annotations

import asyncio
import threading
import time

from EvoScientist.cli import channel as channel_cli
from EvoScientist.cli.channel import ChannelMessage


class _BusLoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        self.loop.run_forever()

    def __enter__(self) -> asyncio.AbstractEventLoop:
        self.thread.start()
        assert self.ready.wait(2)
        return self.loop

    def __exit__(self, *_exc) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(2)
        self.loop.close()


def _message() -> ChannelMessage:
    return ChannelMessage(
        msg_id="relay-message",
        content="hello",
        sender="alice",
        channel_type="telegram",
        chat_id="chat-1",
    )


def test_channel_sends_return_immediately_and_run_in_submission_order(monkeypatch):
    events: list[str] = []
    first_started = threading.Event()
    release_first: asyncio.Event | None = None

    async def _first() -> None:
        nonlocal release_first
        release_first = asyncio.Event()
        events.append("first-start")
        first_started.set()
        await release_first.wait()
        events.append("first-end")

    async def _second() -> None:
        events.append("second")

    with _BusLoopThread() as loop:
        monkeypatch.setattr(channel_cli, "_bus_loop", loop)
        started = time.monotonic()
        assert channel_cli.schedule_channel_send(_message(), _first(), label="first")
        assert channel_cli.schedule_channel_send(_message(), _second(), label="second")
        assert time.monotonic() - started < 0.2
        assert first_started.wait(2)
        assert events == ["first-start"]

        assert release_first is not None
        loop.call_soon_threadsafe(release_first.set)
        assert channel_cli._drain_channel_send_relay(loop, timeout=2.0)

    assert events == ["first-start", "first-end", "second"]


def test_channel_send_shutdown_drain_waits_for_pending_delivery(monkeypatch):
    delivered = threading.Event()

    async def _send() -> None:
        await asyncio.sleep(0.1)
        delivered.set()

    with _BusLoopThread() as loop:
        monkeypatch.setattr(channel_cli, "_bus_loop", loop)
        assert channel_cli.schedule_channel_send(_message(), _send(), label="send")

        assert channel_cli._drain_channel_send_relay(loop, timeout=2.0)
        assert delivered.is_set()


def test_channel_send_shutdown_drain_rejects_late_submission(monkeypatch):
    class _ClosableAwaitable:
        def __init__(self) -> None:
            self.closed = False

        def __await__(self):
            return iter(())

        def close(self) -> None:
            self.closed = True

    late_send = _ClosableAwaitable()

    with _BusLoopThread() as loop:
        monkeypatch.setattr(channel_cli, "_bus_loop", loop)
        assert channel_cli._drain_channel_send_relay(loop, timeout=2.0)

        assert not channel_cli.schedule_channel_send(
            _message(), late_send, label="late"
        )
        assert late_send.closed


def test_final_response_barrier_waits_for_chat_relay(monkeypatch):
    events: list[str] = []

    async def _status() -> None:
        await asyncio.sleep(0.05)
        events.append("status")

    async def _wait_then_record() -> None:
        await channel_cli._wait_for_channel_sends("telegram", "chat-1")
        events.append("response")

    with _BusLoopThread() as loop:
        monkeypatch.setattr(channel_cli, "_bus_loop", loop)
        assert channel_cli.schedule_channel_send(_message(), _status(), label="status")
        asyncio.run_coroutine_threadsafe(_wait_then_record(), loop).result(timeout=2)
        assert channel_cli._drain_channel_send_relay(loop, timeout=2.0)

    assert events == ["status", "response"]
