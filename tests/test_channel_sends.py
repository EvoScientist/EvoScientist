"""Behavioral tests for channel sends crossing frontend event loops."""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from EvoScientist.cli.channel_sends import PendingChannelSends
from EvoScientist.runtime import AsyncRuntime


@pytest.mark.asyncio
async def test_pending_send_does_not_stall_owned_runtime() -> None:
    """A blocked channel transport must not block unrelated runtime work."""
    bus_loop = asyncio.get_running_loop()
    send_started = asyncio.Event()
    release_send = asyncio.Event()
    runtime_progressed = threading.Event()
    sends = PendingChannelSends(bus_loop, logging.getLogger(__name__))

    async def _blocked_send() -> None:
        send_started.set()
        await release_send.wait()

    async def _stream_callback_and_probe() -> None:
        sends.submit(_blocked_send(), "Thinking")
        await asyncio.sleep(0)
        runtime_progressed.set()

    with AsyncRuntime(thread_name="test-channel-send-runtime") as runtime:
        callback = runtime.submit(_stream_callback_and_probe)
        await asyncio.wait_for(send_started.wait(), timeout=1)
        assert runtime_progressed.wait(timeout=1)
        callback.result(timeout=1)

        settle = asyncio.create_task(sends.settle_async())
        await asyncio.sleep(0)
        assert not settle.done()

        release_send.set()
        await asyncio.wait_for(settle, timeout=1)


@pytest.mark.asyncio
async def test_async_settlement_waits_for_every_scheduled_send() -> None:
    """The channel response can wait for all callback delivery off-loop."""
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    events: list[str] = []
    sends = PendingChannelSends(asyncio.get_running_loop(), logging.getLogger(__name__))

    async def _first() -> None:
        events.append("first-started")
        first_started.set()
        await first_release.wait()
        events.append("first-finished")

    async def _second() -> None:
        events.append("second-finished")

    sends.submit(_first(), "First")
    sends.submit(_second(), "Second")

    settle = asyncio.create_task(sends.settle_async())
    await asyncio.wait_for(first_started.wait(), timeout=1)
    assert not settle.done()
    assert events == ["first-started"]

    first_release.set()
    await asyncio.wait_for(settle, timeout=1)

    assert events == ["first-started", "first-finished", "second-finished"]
