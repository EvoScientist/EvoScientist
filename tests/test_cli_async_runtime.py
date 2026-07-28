"""The first bounded CLI adoption of the owned async runtime."""

import asyncio
import threading

import pytest
from typer.testing import CliRunner

import EvoScientist.cli.commands  # noqa: F401 - registers commands on app
from EvoScientist.cli import commands
from EvoScientist.cli._app import app


@pytest.mark.parametrize("args", [["sessions"], ["sessions", "stats"]])
def test_sessions_stats_uses_and_closes_cli_owned_runtime(monkeypatch, args):
    execution: dict[str, object] = {}

    async def fake_db_stats():
        execution["thread"] = threading.current_thread().name
        execution["loop"] = asyncio.get_running_loop()
        return {
            "db_path": "/tmp/sessions.db",
            "size_bytes": 0,
            "thread_count": 0,
            "checkpoint_count": 0,
            "write_count": 0,
            "top_threads": [],
        }

    monkeypatch.setattr("EvoScientist.sessions.db_stats", fake_db_stats)

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.exception
    assert execution["thread"] == "evosci-async-runtime"
    assert isinstance(execution["loop"], asyncio.AbstractEventLoop)
    assert not any(
        thread.name == "evosci-async-runtime" and thread.is_alive()
        for thread in threading.enumerate()
    )


@pytest.mark.parametrize(
    ("args", "patch_target"),
    [
        (["onboard"], "EvoScientist.config.onboard.run_onboard"),
        (["configure", "channels"], "EvoScientist.cli.commands._run_onboard_cli"),
    ],
)
def test_onboarding_commands_share_and_close_cli_runtime(
    monkeypatch, args, patch_target
):
    execution: dict[str, object] = {}

    async def record_execution():
        execution["thread"] = threading.current_thread().name
        execution["loop"] = asyncio.get_running_loop()

    def fake_onboard(**kwargs):
        runtime = kwargs["runtime"]
        execution["runtime"] = runtime
        runtime.run_sync(record_execution)
        return True

    monkeypatch.setattr(patch_target, fake_onboard)

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.exception
    assert execution["thread"] == "evosci-async-runtime"
    assert isinstance(execution["loop"], asyncio.AbstractEventLoop)
    assert not any(
        thread.name == "evosci-async-runtime" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_channel_setup_shares_and_closes_cli_runtime(monkeypatch):
    execution: dict[str, object] = {}

    async def record_execution():
        execution["thread"] = threading.current_thread().name
        execution["loop"] = asyncio.get_running_loop()

    def fake_step_channels(_config, *, runtime):
        runtime.run_sync(record_execution)
        return {}

    monkeypatch.setattr("EvoScientist.config.load_config", object)
    monkeypatch.setattr(
        "EvoScientist.config.onboard.channels._step_channels", fake_step_channels
    )

    result = CliRunner().invoke(app, ["channel", "setup"])

    assert result.exit_code == 0, result.exception
    assert execution["thread"] == "evosci-async-runtime"
    assert isinstance(execution["loop"], asyncio.AbstractEventLoop)
    assert not any(
        thread.name == "evosci-async-runtime" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_cli_reports_runtime_close_timeout_without_raw_exception(monkeypatch):
    class _TimeoutRuntime:
        def run_sync(self, factory):
            return asyncio.run(factory())

        def close(self):
            raise TimeoutError("executor work still active")

    async def fake_db_stats():
        return {
            "db_path": "/tmp/sessions.db",
            "size_bytes": 0,
            "thread_count": 0,
            "checkpoint_count": 0,
            "write_count": 0,
            "top_threads": [],
        }

    monkeypatch.setattr(commands, "AsyncRuntime", _TimeoutRuntime)
    monkeypatch.setattr("EvoScientist.sessions.db_stats", fake_db_stats)

    result = CliRunner().invoke(app, ["sessions", "stats"])

    assert result.exit_code == 1
    assert "Async runtime shutdown did not complete" in result.output
    assert "executor work still active" in result.output
    assert not isinstance(result.exception, TimeoutError)
