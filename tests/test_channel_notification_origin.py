"""Tests for the per-thread channel-origin registry used by the async-notifier
push-back path."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from EvoScientist.cli import channel as channel_cli
from EvoScientist.cli.channel import ChannelMessage


@pytest.fixture(autouse=True)
def _restore_channel_globals():
    """Restore bus globals + the origin registry between tests."""
    original = {
        "_manager": channel_cli._manager,
        "_bus_loop": channel_cli._bus_loop,
        "_bus_thread": channel_cli._bus_thread,
    }
    with channel_cli._thread_channel_origins_lock:
        original_origins = dict(channel_cli._thread_channel_origins)
        channel_cli._thread_channel_origins.clear()
    yield
    channel_cli._manager = original["_manager"]
    channel_cli._bus_loop = original["_bus_loop"]
    channel_cli._bus_thread = original["_bus_thread"]
    with channel_cli._thread_channel_origins_lock:
        channel_cli._thread_channel_origins.clear()
        channel_cli._thread_channel_origins.update(original_origins)


def _make_msg(
    *,
    channel_type: str = "imessage",
    chat_id: str = "+15551234567",
    metadata: dict | None = None,
) -> ChannelMessage:
    return ChannelMessage(
        msg_id="msg-1",
        content="hi",
        sender="alice",
        channel_type=channel_type,
        metadata=metadata if metadata is not None else {"foo": "bar"},
        chat_id=chat_id,
    )


def _install_fake_bus(
    monkeypatch,
) -> tuple[asyncio.AbstractEventLoop, MagicMock, threading.Thread]:
    """Spin up a real asyncio loop on a background thread + a stub bus.

    Returns (loop, publish_outbound_mock, thread). The caller is responsible
    for stopping the loop at the end of the test.
    """
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _runner():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    ready.wait(timeout=2)

    publish_outbound = MagicMock()

    async def _publish_outbound(msg):
        publish_outbound(msg)

    bus = MagicMock()
    bus.publish_outbound = _publish_outbound
    manager = MagicMock()
    manager.bus = bus

    monkeypatch.setattr(channel_cli, "_bus_loop", loop)
    monkeypatch.setattr(channel_cli, "_manager", manager)
    return loop, publish_outbound, thread


def _stop_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


def _wait_for_publish(
    mock: MagicMock, *, expected: int = 1, timeout: float = 2.0
) -> None:
    """Block until ``mock`` has been called ``expected`` times (or timeout)."""
    import time

    deadline = time.time() + timeout
    while mock.call_count < expected and time.time() < deadline:
        time.sleep(0.01)


def test_remember_and_publish_roundtrip(monkeypatch):
    loop, publish_outbound, thread = _install_fake_bus(monkeypatch)
    try:
        msg = _make_msg(metadata={"thread_root": "root-1"})
        channel_cli.remember_channel_origin("tid-1", msg)

        assert channel_cli.publish_to_channel_origin("tid-1", "hello") is True
        _wait_for_publish(publish_outbound)

        publish_outbound.assert_called_once()
        sent = publish_outbound.call_args.args[0]
        assert sent.channel == "imessage"
        assert sent.chat_id == "+15551234567"
        assert sent.content == "hello"
        assert sent.metadata == {"thread_root": "root-1"}
        assert sent.reply_to is None
    finally:
        _stop_loop(loop, thread)


def test_origin_remembers_sender_distinct_from_chat_id():
    """The origin stores the human-readable sender separately from chat_id, so
    the notifier closer can show the same handle a normal turn shows.

    Regression: iMessage exposes an internal chat id (e.g. "1") as chat_id
    while the handle (phone number) lives on sender. The closer must display
    sender, not chat_id, or the user sees a confusing "Replied to 1".
    """
    channel_cli.remember_channel_origin(
        "tid-sender",
        _make_msg(chat_id="1"),  # iMessage-style internal chat id
    )
    origin = channel_cli.get_channel_origin("tid-sender")
    assert origin is not None
    assert origin.chat_id == "1"  # routing id preserved
    assert origin.sender == "alice"  # human handle preserved for display


def test_publish_records_sent_metric(monkeypatch):
    """A forwarded notification records a "sent" metric, mirroring the normal
    channel-reply path (``manager.record_message``)."""
    loop, _publish_outbound, thread = _install_fake_bus(monkeypatch)
    try:
        channel_cli.remember_channel_origin(
            "tid-metric", _make_msg(channel_type="telegram")
        )
        assert channel_cli.publish_to_channel_origin("tid-metric", "done") is True
        # record_message runs right after publish_outbound inside the same
        # coroutine; sync on it before asserting.
        _wait_for_publish(channel_cli._manager.record_message)
        channel_cli._manager.record_message.assert_called_once_with("telegram", "sent")
    finally:
        _stop_loop(loop, thread)


def test_publish_returns_false_without_origin(monkeypatch):
    loop, publish_outbound, thread = _install_fake_bus(monkeypatch)
    try:
        assert channel_cli.publish_to_channel_origin("unknown-tid", "hi") is False
        publish_outbound.assert_not_called()
    finally:
        _stop_loop(loop, thread)


def test_publish_returns_false_when_bus_down(monkeypatch):
    monkeypatch.setattr(channel_cli, "_bus_loop", None)
    monkeypatch.setattr(channel_cli, "_manager", None)

    channel_cli.remember_channel_origin("tid-2", _make_msg())
    assert channel_cli.publish_to_channel_origin("tid-2", "hi") is False


def test_publish_returns_false_for_empty_content(monkeypatch):
    loop, publish_outbound, thread = _install_fake_bus(monkeypatch)
    try:
        channel_cli.remember_channel_origin("tid-3", _make_msg())
        assert channel_cli.publish_to_channel_origin("tid-3", "") is False
        assert channel_cli.publish_to_channel_origin("tid-3", "   \n  ") is False
        publish_outbound.assert_not_called()
    finally:
        _stop_loop(loop, thread)


def test_forget_origin(monkeypatch):
    loop, publish_outbound, thread = _install_fake_bus(monkeypatch)
    try:
        channel_cli.remember_channel_origin("tid-4", _make_msg())
        channel_cli.forget_channel_origin("tid-4")
        assert channel_cli.publish_to_channel_origin("tid-4", "hi") is False
        publish_outbound.assert_not_called()
    finally:
        _stop_loop(loop, thread)


def test_remember_overwrites_same_thread(monkeypatch):
    loop, publish_outbound, thread = _install_fake_bus(monkeypatch)
    try:
        channel_cli.remember_channel_origin(
            "tid-5", _make_msg(channel_type="telegram", chat_id="111")
        )
        channel_cli.remember_channel_origin(
            "tid-5", _make_msg(channel_type="imessage", chat_id="222")
        )
        assert channel_cli.publish_to_channel_origin("tid-5", "hi") is True
        _wait_for_publish(publish_outbound)
        sent = publish_outbound.call_args.args[0]
        assert sent.channel == "imessage"
        assert sent.chat_id == "222"
    finally:
        _stop_loop(loop, thread)


def test_publish_swallows_bus_error(monkeypatch, caplog):
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _runner():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    ready.wait(timeout=2)

    async def _boom(_msg):
        raise RuntimeError("bus dead")

    bus = MagicMock()
    bus.publish_outbound = _boom
    manager = MagicMock()
    manager.bus = bus
    monkeypatch.setattr(channel_cli, "_bus_loop", loop)
    monkeypatch.setattr(channel_cli, "_manager", manager)

    channel_cli.remember_channel_origin("tid-6", _make_msg())
    try:
        with caplog.at_level("WARNING", logger="EvoScientist.cli.channel"):
            # The publish is scheduled (returns True); the coroutine raises
            # asynchronously and the done-callback logs the failure.
            assert channel_cli.publish_to_channel_origin("tid-6", "hi") is True
            # Give the bus thread a moment to run the coroutine + callback.
            import time

            deadline = time.time() + 2.0
            while (
                not any(
                    "Async notification publish" in r.message for r in caplog.records
                )
                and time.time() < deadline
            ):
                time.sleep(0.01)
        assert any("Async notification publish" in r.message for r in caplog.records)
    finally:
        _stop_loop(loop, thread)


def test_remember_with_falsy_thread_id_noop(monkeypatch):
    """Defensive: a falsy thread_id must not pollute the registry."""
    msg = _make_msg()
    channel_cli.remember_channel_origin(None, msg)
    channel_cli.remember_channel_origin("", msg)
    with channel_cli._thread_channel_origins_lock:
        assert channel_cli._thread_channel_origins == {}


def test_get_channel_origin_returns_none_for_unknown():
    assert channel_cli.get_channel_origin("never-registered") is None
    assert channel_cli.get_channel_origin(None) is None
