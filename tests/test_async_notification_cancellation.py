"""Cancellation before notification delivery must not lose a completion."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from EvoScientist.cli import async_notifier as notifier


@pytest.fixture(autouse=True)
def _empty_notifications():
    notifier.drain_notifications()
    yield
    notifier.drain_notifications()


@pytest.mark.parametrize("phase", ["batch", "state"])
@pytest.mark.parametrize("thread_id", [None, "thread-a"])
@pytest.mark.parametrize("kind", ["agent", "bg-process"])
async def test_cancelled_preparation_preserves_notifications(
    monkeypatch, phase, thread_id, kind
):
    paused = asyncio.Event()
    never = asyncio.Event()
    first = notifier.AsyncTaskNotification(
        "first", "worker", "success", "", kind=kind, origin_cli_thread_id=thread_id
    )
    second = notifier.AsyncTaskNotification(
        "second", "worker", "error", "", kind=kind, origin_cli_thread_id=thread_id
    )
    other = notifier.AsyncTaskNotification(
        "other", "worker", "success", "", origin_cli_thread_id="thread-b"
    )
    notifier.enqueue_task_notification(first)
    if thread_id is not None:
        notifier.enqueue_task_notification(other)
    original_drain = notifier.drain_notifications
    first_drain = True

    def drain(current_thread_id=None):
        nonlocal first_drain
        items = original_drain(current_thread_id)
        if first_drain:
            first_drain = False
            # A second completion arrives during the batching window.
            notifier.enqueue_task_notification(second)
            if phase == "batch":
                paused.set()
        return items

    async def read_state():
        paused.set()
        await never.wait()

    monkeypatch.setattr(notifier, "drain_notifications", drain)
    monkeypatch.setattr(
        notifier, "NOTIFICATION_BATCH_GRACE_SECONDS", 3600 if phase == "batch" else 0
    )
    runner = AsyncMock()
    consumer = asyncio.create_task(
        notifier.consume_notifications(runner, read_state, thread_id)
    )
    try:
        await asyncio.wait_for(paused.wait(), timeout=5)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)

    runner.assert_not_awaited()
    assert notifier.has_pending_notifications(thread_id)
    monkeypatch.setattr(notifier, "NOTIFICATION_BATCH_GRACE_SECONDS", 0)
    delivered = []

    async def receive(_text, notifications):
        delivered.extend(notifications)

    await notifier.consume_notifications(receive, AsyncMock(return_value={}), thread_id)
    await notifier.consume_notifications(receive, AsyncMock(return_value={}), thread_id)

    assert sorted(delivered, key=lambda item: item.task_id) == [first, second]
    assert not notifier.has_pending_notifications(thread_id)
    if thread_id is not None:
        assert original_drain("thread-b") == [other]


async def test_cancellation_after_delivery_does_not_requeue(monkeypatch):
    monkeypatch.setattr(notifier, "NOTIFICATION_BATCH_GRACE_SECONDS", 0)
    notification = notifier.AsyncTaskNotification("done", "worker", "success", "")
    notifier.enqueue_task_notification(notification)
    started = asyncio.Event()
    never = asyncio.Event()

    async def receive(_text, notifications):
        assert notifications == [notification]
        started.set()
        await never.wait()

    consumer = asyncio.create_task(
        notifier.consume_notifications(receive, AsyncMock(return_value={}))
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)

    assert notifier.drain_notifications() == []
