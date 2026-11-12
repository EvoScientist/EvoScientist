"""Runtime warnings reach the TUI without leaking logging state on exit."""

from __future__ import annotations

import asyncio
import logging
import threading
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("textual")

from EvoScientist.cli.commands import _configure_logging
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


@pytest.mark.parametrize("worker", [False, True])
async def test_runtime_warning_uses_app_thread(
    monkeypatch, workspace, cli_logging, worker
):
    app = await _app(monkeypatch, workspace)
    message = "MCP config: env var $X [unset] is not set"
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        ui_thread = threading.get_ident()
        notify = app.notify
        notified_on = []

        def checked_notify(*args, **kwargs):
            notified_on.append(threading.get_ident())
            return notify(*args, **kwargs)

        monkeypatch.setattr(app, "notify", checked_notify)
        hop = Mock(wraps=app.call_from_thread)
        monkeypatch.setattr(app, "call_from_thread", hop)
        logger = logging.getLogger("EvoScientist.mcp")
        if worker:
            await asyncio.to_thread(logger.warning, message)
        else:
            logger.warning(message)
        await pilot.pause()
        (shown,) = _shown(app, message)
        assert shown.severity == "warning"
        assert shown.markup is False
        assert notified_on == [ui_thread]
        assert hop.call_count == int(worker)


async def test_repeated_warning_is_shown_once(monkeypatch, workspace, cli_logging):
    app = await _app(monkeypatch, workspace)
    message = "unknown reasoning_effort 'turbo'"
    async with app.run_test(size=(80, 24)) as pilot:
        for _ in range(3):
            logging.getLogger("EvoScientist").warning(message)
        await pilot.pause()
        assert len(_shown(app, message)) == 1


async def test_python_warning_is_shown(monkeypatch, workspace, cli_logging):
    app = await _app(monkeypatch, workspace)
    message = "unknown reasoning_effort 'turbo'"
    showwarning = warnings.showwarning
    async with app.run_test(size=(80, 24)) as pilot:
        warnings.warn(message, UserWarning, stacklevel=1)
        await pilot.pause()
        (shown,) = _shown(app, message)
        assert shown.severity == "warning"
        assert shown.markup is False
    assert warnings.showwarning is showwarning


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

    async def run():
        async with app.run_test(size=(80, 24)) as pilot:
            assert all(h not in root.handlers for h in cli_logging)
            logging.warning("inside TUI")
            await pilot.pause()
            emit.assert_not_called()
            if abort:
                raise ValueError("interrupted session")

    if abort:
        with pytest.raises(ValueError, match="interrupted session"):
            await run()
    else:
        await run()
    assert root.handlers == before
    assert warnings.showwarning is showwarning
    logging.warning("back in CLI")
    emit.assert_called_once()
    assert emit.call_args.args[0].getMessage() == "back in CLI"


@pytest.mark.parametrize("level", [logging.ERROR, logging.CRITICAL])
async def test_higher_levels_keep_cli_handling(
    monkeypatch, workspace, cli_logging, level
):
    app = await _app(monkeypatch, workspace)
    emit = Mock()
    monkeypatch.setattr(cli_logging[0], "emit", emit)
    async with app.run_test(size=(80, 24)) as pilot:
        logging.log(level, "serious failure")
        await pilot.pause()
        emit.assert_called_once()
        assert emit.call_args.args[0].levelno == level


async def test_concurrent_warnings_do_not_deadlock(monkeypatch, workspace, cli_logging):
    app = await _app(monkeypatch, workspace)
    async with app.run_test(size=(80, 24)) as pilot:
        logger = logging.getLogger("EvoScientist.mcp")
        await asyncio.gather(
            asyncio.to_thread(logger.warning, "concurrent warning"),
            asyncio.to_thread(logger.warning, "concurrent warning"),
        )
        logger.warning("concurrent warning")
        await pilot.pause()
        assert len(_shown(app, "concurrent warning")) == 1


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
