"""Regression tests for TUI runtime warning routing and lifecycle (issue #538)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("textual")

from textual.app import App

from EvoScientist.cli.tui_interactive import (
    TUIWarningLogHandler,
)


async def _no_update(self):
    return None


async def _capture_app(
    monkeypatch, *, startup_warnings=None, gateway_backend=None
) -> object:
    """Build an ``EvoTextualInteractiveApp`` without entering its main loop."""
    from EvoScientist.cli import tui_interactive as tui_mod

    if gateway_backend is not None:
        import EvoScientist.config as config_mod

        monkeypatch.setattr(
            config_mod,
            "resolve_gateway_backend",
            lambda *_args, **_kwargs: gateway_backend,
        )

    captured: dict = {}

    async def _no_run(self, *args, **kwargs):
        captured["app"] = self

    monkeypatch.setattr(App, "run_async", _no_run)

    fake_load_agent = AsyncMock(return_value=None)

    @asynccontextmanager
    async def _fake_checkpointer(*_a, **_k):
        yield None

    monkeypatch.setattr(
        "EvoScientist.cli.tui_interactive.get_checkpointer",
        _fake_checkpointer,
        raising=False,
    )

    class _FakeSuggester:
        def __init__(self, *_a, **_k):
            pass

    monkeypatch.setattr(
        "EvoScientist.cli.history_suggester.HistorySuggester", _FakeSuggester
    )

    monkeypatch.setattr(
        "EvoScientist.cli.tui_interactive._auto_start_channel",
        lambda *_a, **_k: None,
        raising=False,
    )

    monkeypatch.setattr("EvoScientist.cli.tui_interactive.mode", "dev", raising=False)

    try:
        await asyncio.to_thread(
            tui_mod.run_textual_interactive,
            show_thinking=False,
            channel_send_thinking=False,
            workspace_dir=None,
            workspace_fixed=False,
            mode="dev",
            model=None,
            provider=None,
            run_name="test-run",
            thread_id=None,
            load_agent=fake_load_agent,
            create_session_workspace=lambda *_a, **_k: str(Path.cwd()),
            config=None,
            startup_warnings=startup_warnings,
        )
    except SystemExit:
        pass

    app = captured.get("app")
    if app is None:
        raise RuntimeError("Failed to capture EvoTextualInteractiveApp")
    return app


async def test_runtime_warning_surfaces_as_tui_notification(monkeypatch):
    """A WARNING log record emitted while the TUI is active reaches TUI notifications."""
    notifications: list[tuple[str, dict[str, object]]] = []

    def capture_notify(self, message, *args, **kwargs):
        notifications.append((str(message), dict(kwargs)))

    monkeypatch.setattr(App, "notify", capture_notify)
    app = await _capture_app(monkeypatch)
    monkeypatch.setattr(type(app), "_check_for_updates", _no_update)

    async with app.run_test(size=(80, 24)) as pilot:
        test_logger = logging.getLogger("test.runtime.worker")
        test_logger.warning("Background worker encountered an issue: code=42")
        await pilot.pause()

    warnings = [
        (msg, opts) for msg, opts in notifications if opts.get("severity") == "warning"
    ]
    assert any(
        "Background worker encountered an issue: code=42" in msg for msg, _ in warnings
    )


async def test_below_warning_filtering(monkeypatch):
    """INFO and DEBUG log records are not converted into TUI notifications."""
    notifications: list[tuple[str, dict[str, object]]] = []

    def capture_notify(self, message, *args, **kwargs):
        notifications.append((str(message), dict(kwargs)))

    monkeypatch.setattr(App, "notify", capture_notify)
    app = await _capture_app(monkeypatch)
    monkeypatch.setattr(type(app), "_check_for_updates", _no_update)

    async with app.run_test(size=(80, 24)) as pilot:
        test_logger = logging.getLogger("test.verbose")
        test_logger.info("Routine informational message")
        test_logger.debug("Routine debugging trace")
        await pilot.pause()

    all_messages = [msg for msg, _ in notifications]
    assert not any("Routine informational message" in msg for msg in all_messages)
    assert not any("Routine debugging trace" in msg for msg in all_messages)


async def test_async_notifier_failure_path(monkeypatch):
    """Async-notifier failure warnings reach TUI notifications while active."""
    notifications: list[tuple[str, dict[str, object]]] = []

    def capture_notify(self, message, *args, **kwargs):
        notifications.append((str(message), dict(kwargs)))

    monkeypatch.setattr(App, "notify", capture_notify)
    app = await _capture_app(monkeypatch)
    monkeypatch.setattr(type(app), "_check_for_updates", _no_update)

    async with app.run_test(size=(80, 24)) as pilot:
        notifier_logger = logging.getLogger("EvoScientist.cli.tui_interactive")
        notifier_logger.warning("async-notifier consume failed (TUI)", exc_info=True)
        notifier_logger.warning(
            "async-notifier idle reader failed (TUI)", exc_info=True
        )
        await pilot.pause()

    warnings = [msg for msg, opts in notifications if opts.get("severity") == "warning"]
    assert any("async-notifier consume failed (TUI)" in msg for msg in warnings)
    assert any("async-notifier idle reader failed (TUI)" in msg for msg in warnings)


async def test_npx_node_install_failure_path(monkeypatch):
    """On-demand Node.js/npx installation failure warning reaches TUI notifications."""
    notifications: list[tuple[str, dict[str, object]]] = []

    def capture_notify(self, message, *args, **kwargs):
        notifications.append((str(message), dict(kwargs)))

    monkeypatch.setattr(App, "notify", capture_notify)
    app = await _capture_app(monkeypatch)
    monkeypatch.setattr(type(app), "_check_for_updates", _no_update)

    async with app.run_test(size=(80, 24)) as pilot:
        from EvoScientist.mcp import client as mcp_client
        from EvoScientist.setup.protocol import StageError

        def fake_ensure_node(*_args, **_kwargs):
            raise StageError("download_failed", "simulated offline network timeout")

        monkeypatch.setattr("EvoScientist.setup.node.ensure_node", fake_ensure_node)
        monkeypatch.delenv("EVOSCIENTIST_DEPLOY_MODE", raising=False)

        await asyncio.to_thread(
            mcp_client._ensure_node_for_stdio,
            {
                "test-server": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "test-server"],
                }
            },
        )
        await pilot.pause()

    warnings = [msg for msg, opts in notifications if opts.get("severity") == "warning"]
    assert any(
        "MCP servers test-server need Node.js, and installing it failed: simulated offline network timeout. Run 'EvoSci setup' to retry."
        in msg
        for msg in warnings
    )


async def test_runtime_warning_from_background_thread(monkeypatch):
    """WARNING emitted from arbitrary background OS thread reaches TUI safely."""
    notifications: list[str] = []

    def capture_notify(self, message, *args, **kwargs):
        notifications.append(str(message))

    monkeypatch.setattr(App, "notify", capture_notify)
    app = await _capture_app(monkeypatch)
    monkeypatch.setattr(type(app), "_check_for_updates", _no_update)

    async with app.run_test(size=(80, 24)) as pilot:
        import threading

        def bg_worker():
            logging.getLogger("test.thread").warning("Warning from raw OS thread")

        thread = threading.Thread(target=bg_worker)
        thread.start()
        thread.join()
        await pilot.pause()

    assert "Warning from raw OS thread" in notifications


def test_non_tui_behavior_does_not_install_tui_handler():
    """Rich CLI / one-shot / serve modes do not install TUIWarningLogHandler."""
    from EvoScientist.cli.commands import _configure_logging

    _configure_logging()
    root_handlers = logging.getLogger().handlers
    assert not any(isinstance(h, TUIWarningLogHandler) for h in root_handlers)


async def test_handler_cleanup_and_subsequent_runs(monkeypatch):
    """Handler is removed on TUI exit, and re-entry does not duplicate notifications."""
    root = logging.getLogger()
    assert not any(isinstance(h, TUIWarningLogHandler) for h in root.handlers)

    notifications_1: list[str] = []
    notifications_2: list[str] = []

    current_notifications = notifications_1

    def capture_notify(self, message, *args, **kwargs):
        current_notifications.append(str(message))

    monkeypatch.setattr(App, "notify", capture_notify)

    # First session
    app1 = await _capture_app(monkeypatch)
    monkeypatch.setattr(type(app1), "_check_for_updates", _no_update)
    async with app1.run_test(size=(80, 24)) as pilot:
        assert any(isinstance(h, TUIWarningLogHandler) for h in root.handlers)
        logging.getLogger("test").warning("warning from session 1")
        await pilot.pause()

    # After exit 1
    assert not any(isinstance(h, TUIWarningLogHandler) for h in root.handlers)
    assert notifications_1.count("warning from session 1") == 1

    # Second session
    current_notifications = notifications_2
    app2 = await _capture_app(monkeypatch)
    monkeypatch.setattr(type(app2), "_check_for_updates", _no_update)
    async with app2.run_test(size=(80, 24)) as pilot:
        assert any(isinstance(h, TUIWarningLogHandler) for h in root.handlers)
        logging.getLogger("test").warning("warning from session 2")
        await pilot.pause()

    # After exit 2
    assert not any(isinstance(h, TUIWarningLogHandler) for h in root.handlers)
    assert notifications_2.count("warning from session 2") == 1


async def test_no_startup_warning_duplication_when_logged(monkeypatch):
    """Startup warnings appear once and are not re-notified if logged at runtime."""
    notifications: list[tuple[str, dict[str, object]]] = []

    def capture_notify(self, message, *args, **kwargs):
        notifications.append((str(message), dict(kwargs)))

    monkeypatch.setattr(App, "notify", capture_notify)
    startup_msg = "⚠ Config changed since the background agent server was launched"
    app = await _capture_app(monkeypatch, startup_warnings=[startup_msg])
    monkeypatch.setattr(type(app), "_check_for_updates", _no_update)

    async with app.run_test(size=(80, 24)) as pilot:
        # Attempt to log the identical message at runtime
        logging.getLogger("test").warning(startup_msg)
        await pilot.pause()

    warnings = [msg for msg, opts in notifications if opts.get("severity") == "warning"]
    assert warnings.count(startup_msg) == 1
