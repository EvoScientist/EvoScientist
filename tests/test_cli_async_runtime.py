"""The first bounded CLI adoption of the owned async runtime."""

import asyncio
import threading

import pytest
from typer.testing import CliRunner

import EvoScientist.cli.commands  # noqa: F401 - registers commands on app
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
