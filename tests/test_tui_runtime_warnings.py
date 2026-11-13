"""Runtime warnings reach the TUI without leaking logging state on exit."""

import asyncio
import logging
import threading
import warnings
from contextlib import asynccontextmanager, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("textual")

from EvoScientist.cli.commands import _configure_logging
from EvoScientist.cli.tui_logging import TUIWarningHandler, tui_warning_notifications
from tests.fakes.tui import capture_app


@pytest.fixture
def console(monkeypatch):
    root = logging.getLogger()
    noisy = logging.getLogger("langchain_google_genai._function_utils")
    monkeypatch.setattr(root, "handlers", root.handlers[:])
    monkeypatch.setattr(root, "level", root.level)
    monkeypatch.setattr(noisy, "level", noisy.level)
    config = SimpleNamespace(log_level="WARNING")
    monkeypatch.setattr("EvoScientist.config.get_effective_config", lambda: config)
    _configure_logging()
    handler = root.handlers[0]
    monkeypatch.setattr(handler, "emit", Mock())
    return handler


@pytest.fixture
def session(monkeypatch, workspace, console):
    @asynccontextmanager
    async def run():
        app = await capture_app(monkeypatch, workspace)
        monkeypatch.setattr(app, "_check_for_updates", AsyncMock())
        with tui_warning_notifications(app):
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                yield app, pilot

    return run


def _shown(app, message):
    return [n for n in app._notifications if n.message == message]


def _aborted(abort):
    return pytest.raises(ValueError, match="interrupted") if abort else nullcontext()


async def test_tui_session_runs_inside_the_context(monkeypatch, workspace, console):
    from EvoScientist.cli import tui_logging

    ran_inside = []
    context = tui_logging.tui_warning_notifications

    @contextmanager
    def spy(app):
        run = AsyncMock(wraps=app.run_async)
        with context(app), monkeypatch.context() as patch:
            patch.setattr(app, "run_async", run)
            yield
            run.assert_awaited_once()
            ran_inside.append(app)

    monkeypatch.setattr(tui_logging, "tui_warning_notifications", spy)
    app = await capture_app(monkeypatch, workspace)
    assert ran_inside == [app]


@pytest.mark.parametrize(
    ("source", "level", "worker"),
    [("log", level, worker) for level in (30, 40, 50) for worker in (False, True)]
    + [(s, 40, False) for s in ("exception", "broken")]
    + [("python", 30, False), ("hint", 30, False)],
)
async def test_runtime_warning_notifies(
    session, console, monkeypatch, source, level, worker
):
    message = "MCP config: env var $X [unset] is not set"
    if source == "hint":
        from EvoScientist import agent_shell

        monkeypatch.setattr(agent_shell, "setup_hint", lambda: message)
    async with session() as (app, pilot):
        logger = logging.getLogger("EvoScientist.mcp")
        if source == "exception":
            try:
                raise ConnectionError("gateway unreachable")
            except ConnectionError:
                logger.exception("dispatch failed")
            message = "dispatch failed: ConnectionError: gateway unreachable"
        elif source == "broken":
            broken = type("BrokenError", (Exception,), {"__str__": lambda _: 1 / 0})
            logger.error("dispatch failed", exc_info=broken())
            message = "dispatch failed: BrokenError"
        elif source == "hint":
            await app.workers.wait_for_complete()
            agent_shell.logger.warning(message)
        elif source == "python":
            warnings.warn(message, UserWarning, stacklevel=1)
        elif worker:
            await asyncio.to_thread(logger.log, level, message)
        else:
            logger.log(level, message)
        await pilot.pause()
        (shown,) = _shown(app, message)
        assert shown.severity == ("error" if level >= 40 else "warning")
        assert shown.markup is False
        console.emit.assert_not_called()


@pytest.mark.parametrize(("abort", "count"), [(False, 1), (True, 1), (False, 2)])
async def test_cli_handlers_restored_on_exit(session, console, abort, count):
    root = logging.getLogger()
    showwarning = warnings.showwarning
    before = root.handlers[:]
    for _ in range(count):
        with _aborted(abort):
            async with session() as (app, pilot):
                assert console not in root.handlers
                logging.warning("inside TUI")
                await pilot.pause()
                assert len(_shown(app, "inside TUI")) == 1
                console.emit.assert_not_called()
                if abort:
                    raise ValueError("interrupted session")
        assert root.handlers == before
        assert warnings.showwarning is showwarning
    logging.warning("back in CLI")
    console.emit.assert_called_once()
    assert console.emit.call_args.args[0].getMessage() == "back in CLI"


@pytest.mark.parametrize(
    ("state", "level", "replay"),
    [("_exiting", level, True) for level in (10, 20, 30, 40)]
    + [(state, 30, True) for state in ("exit", "_closing", "post_message")]
    + [(None, 20, False), (None, 10, False), ("_exiting", 10, False)],
)
async def test_console_routing(session, monkeypatch, console, state, level, replay):
    logger = logging.getLogger("EvoScientist.channels")
    monkeypatch.setattr(logger, "level", logging.DEBUG)
    if replay and state == "_exiting":
        monkeypatch.setattr(console, "level", logging.DEBUG)
    message = "warning during shutdown"
    async with session() as (app, pilot):
        with monkeypatch.context() as patch:
            if state == "exit":
                app.exit()
                assert app.is_running
            elif state == "post_message":
                patch.setattr(app, state, Mock(side_effect=RuntimeError("closed loop")))
            elif state:
                patch.setattr(app, state, True)
            if state == "_exiting" and replay:
                await asyncio.to_thread(logger.log, level, message)
            else:
                logger.log(level, message)
        if state != "exit":
            await pilot.pause()
            assert not _shown(app, message)
        console.emit.assert_not_called()
    if replay:
        console.emit.assert_called_once()
        record = console.emit.call_args.args[0]
        assert record.getMessage() == message
        assert record.levelno == level
    else:
        console.emit.assert_not_called()


async def test_worker_warning_does_not_wait_for_ui(session):
    async with session() as (app, pilot):
        message = "warning during channel shutdown"
        logged = threading.Event()

        def log_warning():
            logger = logging.getLogger("EvoScientist.channels")
            logger.warning(message)
            logger.warning(message)
            logged.set()

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


async def test_warning_memory_is_bounded(session):
    async with session() as (app, pilot):
        for i in range(257):
            logging.warning("bounded warning %s", i)
        logging.warning("bounded warning 256")
        logging.warning("bounded warning 0")
        await pilot.pause()
        assert len(_shown(app, "bounded warning 256")) == 1
        assert len(_shown(app, "bounded warning 0")) == 2


async def test_retry_throttle(session, console):
    async with session() as (app, pilot):
        handler = next(
            h for h in logging.getLogger().handlers if isinstance(h, TUIWarningHandler)
        )
        handler.clock = lambda: now
        logger = logging.getLogger("EvoScientist.channels.middleware")
        retry = {"notify_throttle": "slack.send"}
        attempts = (
            (0, 1, 0.5),
            (19.9, 2, 1.1),
            (20, 3, 2.2),
            (40, 3, 2.2),
            (40, 4, 4.4),
        )
        for now, attempt, delay in attempts:
            logger.warning("retry %s in %.1fs", attempt, delay, extra=retry)
            if now == 19.9:
                other = {"notify_throttle": "feishu.send"}
                logger.warning("retry %s in %.1fs", 2, 1.2, extra=other)
                logger.error("retry %s failed", 2, extra=retry)
                for key in ("FIRST_KEY", "SECOND_KEY"):
                    logger.warning("env var $%s is not set", key)
                logger.warning("gateway error")
                logger.error("gateway error")
            await pilot.pause()
            message = f"retry {attempt} in {delay:.1f}s"
            assert len(_shown(app, message)) == (attempt != 2)
        for message in (
            "retry 2 in 1.2s",
            "retry 2 failed",
            "env var $FIRST_KEY is not set",
            "env var $SECOND_KEY is not set",
        ):
            assert len(_shown(app, message)) == 1
        gateway = _shown(app, "gateway error")
        assert [n.severity for n in gateway] == ["warning", "error"]
    console.emit.assert_not_called()


async def test_channel_send_retry_opts_into_the_throttle(monkeypatch, caplog):
    from EvoScientist.channels.base import Channel
    from EvoScientist.channels.retry import RetryConfig

    flaky = AsyncMock(side_effect=[ConnectionError("down"), "ok"])
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


@pytest.mark.parametrize(
    ("abort", "bad"), [(False, False), (True, False), (False, True)]
)
def test_finally_replay(monkeypatch, console, abort, bad):
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
            late = logging.makeLogRecord({"msg": "during replay", "levelno": 30})
            worker = threading.Thread(target=handler.handle, args=(late,), daemon=True)
            worker.start()

    monkeypatch.setattr(console, "emit", emit)
    monkeypatch.setattr(logging, "raiseExceptions", False)
    with _aborted(abort), tui_warning_notifications(SimpleNamespace(is_running=False)):
        handler = next(h for h in root.handlers if isinstance(h, TUIWarningHandler))
        if bad:
            logging.warning("bad %d", "not a number")
        logging.warning("buffered")
        assert not messages
        if abort:
            raise ValueError("interrupted session")
    assert worker is not None
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert messages == ["buffered", "during replay"]
