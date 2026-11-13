"""Runtime warnings reach the TUI without leaking logging state on exit."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("textual")

from EvoScientist.cli.commands import _configure_logging
from EvoScientist.cli.tui_logging import TUIWarningHandler, tui_warning_notifications
from tests.fakes.tui import capture_app


@pytest.fixture
def cli_logging(monkeypatch):
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    noisy = logging.getLogger("langchain_google_genai._function_utils")
    noisy_level = noisy.level
    monkeypatch.setattr(
        "EvoScientist.config.get_effective_config",
        lambda: SimpleNamespace(log_level="WARNING"),
    )
    _configure_logging()
    configured = root.handlers[:]
    try:
        yield configured
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
        noisy.setLevel(noisy_level)


async def _app(monkeypatch, workspace):
    app = await capture_app(monkeypatch, workspace)
    monkeypatch.setattr(app, "_check_for_updates", AsyncMock())
    return app


def _shown(app, message):
    return [n for n in app._notifications if n.message == message]


def _aborted(abort):
    return (
        pytest.raises(ValueError, match="interrupted")
        if abort
        else contextlib.nullcontext()
    )


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize(
    ("level", "severity"),
    [
        (logging.WARNING, "warning"),
        (logging.ERROR, "error"),
        (logging.CRITICAL, "error"),
    ],
)
async def test_runtime_warning_notifies(
    monkeypatch, workspace, cli_logging, worker, level, severity
):
    app = await _app(monkeypatch, workspace)
    emit = Mock()
    monkeypatch.setattr(cli_logging[0], "emit", emit)
    message = "MCP config: env var $X [unset] is not set"
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        logger = logging.getLogger("EvoScientist.mcp")
        if worker:
            await asyncio.to_thread(logger.log, level, message)
        else:
            logger.log(level, message)
        await pilot.pause()
        (shown,) = _shown(app, message)
        assert shown.severity == severity
        assert shown.markup is False
        emit.assert_not_called()


async def test_python_warning_is_shown(monkeypatch, workspace, cli_logging):
    app = await _app(monkeypatch, workspace)
    message = "unknown reasoning_effort 'turbo'"
    async with app.run_test(size=(80, 24)) as pilot:
        warnings.warn(message, UserWarning, stacklevel=1)
        await pilot.pause()
        (shown,) = _shown(app, message)
        assert shown.severity == "warning"
        assert shown.markup is False


@pytest.mark.parametrize("abort", [False, True])
async def test_cli_handlers_restored_on_exit(
    monkeypatch, workspace, cli_logging, abort
):
    app = await _app(monkeypatch, workspace)
    root = logging.getLogger()
    showwarning = warnings.showwarning
    before = root.handlers[:]
    emit = Mock()
    monkeypatch.setattr(cli_logging[0], "emit", emit)

    with _aborted(abort):
        async with app.run_test(size=(80, 24)) as pilot:
            assert all(h not in root.handlers for h in cli_logging)
            logging.warning("inside TUI")
            await pilot.pause()
            emit.assert_not_called()
            if abort:
                raise ValueError("interrupted session")
    assert root.handlers == before
    assert warnings.showwarning is showwarning
    logging.warning("back in CLI")
    emit.assert_called_once()
    assert emit.call_args.args[0].getMessage() == "back in CLI"


@pytest.mark.parametrize("headless", [False, True])
async def test_warning_after_exit_reaches_cli(
    monkeypatch, workspace, cli_logging, headless
):
    app = await _app(monkeypatch, workspace)
    emit = Mock()
    monkeypatch.setattr(cli_logging[0], "emit", emit)
    async with app.run_test(size=(80, 24), headless=headless) as pilot:
        await pilot.pause()
        app.exit()
        assert app.is_running
        logging.getLogger("EvoScientist.channels").warning("late warning")
        emit.assert_not_called()
    emit.assert_called_once()
    assert emit.call_args.args[0].getMessage() == "late warning"


@pytest.mark.parametrize(
    "level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
)
async def test_records_during_channel_stop_replayed_after_exit(
    monkeypatch, workspace, cli_logging, level
):
    app = await _app(monkeypatch, workspace)
    emit = Mock()
    monkeypatch.setattr(cli_logging[0], "emit", emit)
    monkeypatch.setattr(
        logging.getLogger("EvoScientist.channels"), "level", logging.DEBUG
    )
    monkeypatch.setattr(cli_logging[0], "level", logging.DEBUG)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        # _do_exit sets this, then waits on the channels while the bus thread logs.
        app._exiting = True
        logger = logging.getLogger("EvoScientist.channels")
        await asyncio.to_thread(logger.log, level, "bus warning during stop")
        app._exiting = False
        await pilot.pause()
        emit.assert_not_called()
        assert not _shown(app, "bus warning during stop")
    records = [
        c.args[0]
        for c in emit.call_args_list
        if c.args[0].getMessage() == "bus warning during stop"
    ]
    assert len(records) == 1
    assert records[0].levelno == level


@pytest.mark.parametrize("raises", [False, True])
async def test_warning_refused_by_closing_app_reaches_cli(
    monkeypatch, workspace, cli_logging, raises
):
    app = await _app(monkeypatch, workspace)
    emit = Mock()
    monkeypatch.setattr(cli_logging[0], "emit", emit)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        # The pump starts closing after the shutdown check: post_message refuses.
        with monkeypatch.context() as patch:
            if raises:
                patch.setattr(
                    app, "post_message", Mock(side_effect=RuntimeError("closed loop"))
                )
            else:
                patch.setattr(app, "_closing", True)
            logging.getLogger("EvoScientist.channels").warning("refused warning")
        await pilot.pause()
        emit.assert_not_called()
        assert not _shown(app, "refused warning")
    emit.assert_called_once()
    assert emit.call_args.args[0].getMessage() == "refused warning"


async def test_worker_warning_does_not_wait_for_ui(monkeypatch, workspace, cli_logging):
    app = await _app(monkeypatch, workspace)
    message = "warning during channel shutdown"
    logged = threading.Event()

    def log_warning():
        logger = logging.getLogger("EvoScientist.channels")
        logger.warning(message)
        logger.warning(message)
        logged.set()

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        # Two at once, while the UI thread is blocked as in a channel shutdown.
        workers = [threading.Thread(target=log_warning, daemon=True) for _ in "ab"]
        for worker in workers:
            worker.start()
        returned_while_ui_blocked = logged.wait(timeout=0.5)
        await pilot.pause()
        for worker in workers:
            await asyncio.to_thread(worker.join, timeout=1)
            assert not worker.is_alive()
        assert returned_while_ui_blocked, "logging waited for the blocked UI loop"
        assert len(_shown(app, message)) == 1


async def test_warning_memory_is_bounded(monkeypatch, workspace, cli_logging):
    app = await _app(monkeypatch, workspace)
    async with app.run_test(size=(80, 24)) as pilot:
        for i in range(257):
            logging.warning("bounded warning %s", i)
        logging.warning("bounded warning 256")
        logging.warning("bounded warning 0")
        await pilot.pause()
        assert len(_shown(app, "bounded warning 256")) == 1
        assert len(_shown(app, "bounded warning 0")) == 2


async def test_retry_throttle(monkeypatch, workspace, cli_logging):
    app = await _app(monkeypatch, workspace)
    emit = Mock()
    monkeypatch.setattr(cli_logging[0], "emit", emit)
    async with app.run_test(size=(80, 24)) as pilot:
        handler = next(
            h for h in logging.getLogger().handlers if isinstance(h, TUIWarningHandler)
        )
        now = 0
        handler.clock = lambda: now
        logger = logging.getLogger("EvoScientist.channels.middleware")
        retry = {"notify_throttle": "slack.send"}
        logger.warning("retry %s in %.1fs", 1, 0.5, extra=retry)
        now = 19.9
        logger.warning("retry %s in %.1fs", 2, 1.1, extra=retry)
        logger.warning(
            "retry %s in %.1fs", 2, 1.1, extra={"notify_throttle": "feishu.send"}
        )
        # Not opted in: distinct warnings from one logger are all shown.
        logger.warning("env var $%s is not set", "FIRST_KEY")
        logger.warning("env var $%s is not set", "SECOND_KEY")
        logger.warning("gateway error")
        logger.error("gateway error")  # a warning never hides the same error
        now = 20
        logger.warning("retry %s in %.1fs", 3, 2.2, extra=retry)
        now = 40
        logger.warning("retry %s in %.1fs", 3, 2.2, extra=retry)
        logger.warning("retry %s in %.1fs", 4, 4.4, extra=retry)
        await pilot.pause()
        assert len(_shown(app, "retry 1 in 0.5s")) == 1
        assert len(_shown(app, "retry 2 in 1.1s")) == 1  # the feishu one
        assert len(_shown(app, "env var $FIRST_KEY is not set")) == 1
        assert len(_shown(app, "env var $SECOND_KEY is not set")) == 1
        assert [n.severity for n in _shown(app, "gateway error")] == [
            "warning",
            "error",
        ]
        assert len(_shown(app, "retry 3 in 2.2s")) == 1
        assert len(_shown(app, "retry 4 in 4.4s")) == 1
        emit.assert_not_called()


async def test_channel_send_retry_opts_into_the_throttle(monkeypatch, caplog):
    from EvoScientist.channels.base import Channel
    from EvoScientist.channels.retry import RetryConfig

    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("down")
        return "ok"

    channel = SimpleNamespace(
        name="slack",
        _trace_event=lambda *a, **k: None,
        _retry_config=RetryConfig(attempts=2, min_delay_s=0.1, max_delay_s=1, jitter=0),
        _extract_retry_after=lambda exc: 1.0,
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    with caplog.at_level(logging.WARNING, "EvoScientist.channels"):
        assert await Channel._send_with_retry(channel, flaky) == "ok"
    (record,) = [r for r in caplog.records if "retry" in r.getMessage()]
    assert record.notify_throttle == "slack.send"


@pytest.mark.parametrize("abort", [False, True])
def test_worker_logging_during_finally_replay(monkeypatch, cli_logging, abort):
    root = logging.getLogger()
    before = root.handlers[:]
    showwarning = warnings.showwarning
    messages = []
    worker = None

    def emit(record):
        nonlocal worker
        assert root.handlers == before
        assert warnings.showwarning is showwarning
        assert not handler.active
        messages.append(record.getMessage())
        if record.getMessage() == "buffered":
            # A bus thread may already hold a reference to the removed handler.
            late = logging.makeLogRecord({"msg": "during replay", "levelno": 30})
            worker = threading.Thread(target=handler.handle, args=(late,), daemon=True)
            worker.start()

    monkeypatch.setattr(cli_logging[0], "emit", emit)

    with _aborted(abort):
        with tui_warning_notifications(SimpleNamespace(is_running=False)):
            handler = next(h for h in root.handlers if isinstance(h, TUIWarningHandler))
            logging.warning("buffered")
            assert not messages
            if abort:
                raise ValueError("interrupted session")
    assert worker is not None
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert messages == ["buffered", "during replay"]


async def test_second_session_can_show_the_same_warning(
    monkeypatch, workspace, cli_logging
):
    for _ in range(2):
        app = await _app(monkeypatch, workspace)
        before = logging.getLogger().handlers[:]
        async with app.run_test(size=(80, 24)) as pilot:
            logging.warning("same warning in a new session")
            await pilot.pause()
            assert len(_shown(app, "same warning in a new session")) == 1
        assert logging.getLogger().handlers == before
