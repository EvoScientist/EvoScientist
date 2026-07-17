"""Cancellation behavior for ``/install-mcp``.

A cancel (Ctrl+C) reaches the command coroutine as ``CancelledError`` while the
blocking installer runs in a worker thread that cannot be preempted. These tests
pin the required behavior: no config mutation lands after the cancel, and the
user-visible message is honest about what actually happened.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from EvoScientist.commands.base import CommandContext
from EvoScientist.commands.implementation.mcp_install import InstallMCPCommand
from EvoScientist.mcp import client, registry
from EvoScientist.mcp.registry import MCPServerEntry


def _ctx():
    ui = MagicMock()
    ui.supports_interactive = True
    ui.flush = AsyncMock()
    return CommandContext(agent=None, thread_id="tid", ui=ui), ui


def _messages(ui) -> list[str]:
    return [c.args[0] for c in ui.append_system.call_args_list]


def _temp_config(tmp_path):
    """Point the user MCP config at a temp file for real add_mcp_server writes."""
    cfg = tmp_path / "mcp.yaml"
    return (
        patch.object(client, "USER_MCP_CONFIG", cfg),
        patch.object(client, "USER_CONFIG_DIR", tmp_path),
        cfg,
    )


async def test_cancel_gate_discards_uncommitted_install(tmp_path):
    """Worker blocked mid-fetch + cancel → no config write, worker completion
    still does not commit, and the message says nothing was applied."""
    ctx, ui = _ctx()
    entry = MCPServerEntry(name="slow-srv", command="slow-srv", pip_package="pkg")

    started = threading.Event()
    release = threading.Event()
    fetch_done = threading.Event()

    def fake_install_cli_tool(_pkg, *, verify_command=None):
        started.set()
        release.wait(2)
        try:
            return True
        finally:
            fetch_done.set()

    cfg_patch, dir_patch, cfg_file = _temp_config(tmp_path)

    real_install = registry.install_mcp_server
    worker_done = threading.Event()

    def tracked_install(*args, **kwargs):
        worker_done.clear()
        try:
            return real_install(*args, **kwargs)
        finally:
            worker_done.set()

    with (
        cfg_patch,
        dir_patch,
        patch.object(registry, "fetch_marketplace_index", return_value=[entry]),
        patch.object(registry, "find_server_by_name", return_value=entry),
        patch.object(registry, "get_installed_names", return_value=set()),
        patch.object(registry, "install_cli_tool", fake_install_cli_tool),
        patch.object(registry, "install_mcp_server", tracked_install),
        patch.object(registry, "_resolve_command_path", side_effect=lambda c: c),
    ):
        task = asyncio.create_task(InstallMCPCommand().execute(ctx, ["slow-srv"]))
        assert await asyncio.to_thread(started.wait, 2)
        # Cancel while the (unpreemptable) fetch is in flight.
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Let the worker run to completion; the gate must still discard.
        release.set()
        assert await asyncio.to_thread(fetch_done.wait, 2)
        # fetch_done fires inside the fetch, before install_mcp_server reaches
        # its commit gate; wait for the whole call to return before the temp
        # config patch unwinds so an abandoned write can never reach real config.
        assert await asyncio.to_thread(worker_done.wait, 2)

    assert not cfg_file.exists() or cfg_file.read_text().strip() == ""
    msgs = " ".join(_messages(ui)).lower()
    assert "nothing will be written" in msgs
    assert "background" in msgs
    assert "retry" in msgs


async def test_late_cancel_reports_completed(tmp_path):
    """Commit already happened, then cancel → message says it was applied and
    the config retains the install."""
    ctx, ui = _ctx()
    entry = MCPServerEntry(name="fast-srv", command="fast-srv")

    committed = threading.Event()
    release = threading.Event()
    worker_done = threading.Event()

    real_add = client.add_mcp_server

    def fake_install(srv, *, print_fn=None, cancel_event=None, commit_gate=None):
        # Model the real installer: the gate runs the config write and records
        # the ledger under one lock, THEN the worker stays alive so the awaiting
        # coroutine is still suspended when the cancel arrives — i.e. the commit
        # genuinely happened before the cancel.
        def _write() -> bool:
            real_add(srv.name, "stdio", command=srv.command)
            return True

        ok = commit_gate(srv.name, _write) if commit_gate is not None else _write()
        committed.set()
        release.wait(2)
        try:
            return ok
        finally:
            worker_done.set()

    cfg_patch, dir_patch, cfg_file = _temp_config(tmp_path)

    with (
        cfg_patch,
        dir_patch,
        patch.object(registry, "fetch_marketplace_index", return_value=[entry]),
        patch.object(registry, "find_server_by_name", return_value=entry),
        patch.object(registry, "get_installed_names", return_value=set()),
        patch.object(registry, "install_mcp_server", fake_install),
    ):
        task = asyncio.create_task(InstallMCPCommand().execute(ctx, ["fast-srv"]))
        assert await asyncio.to_thread(committed.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        # Wait for the abandoned worker to return before the temp config patch
        # unwinds, so no late frame can touch real config after teardown.
        assert await asyncio.to_thread(worker_done.wait, 2)

    assert "fast-srv" in cfg_file.read_text()
    msgs = " ".join(_messages(ui)).lower()
    assert "before the cancel" in msgs
    assert "applied" in msgs
    assert "nothing will be written" not in msgs


async def test_write_failure_records_no_ledger_entry(tmp_path):
    """If the config write fails, no ledger entry is recorded — a concurrent
    cancel must report nothing-applied, never a phantom success."""
    ctx, ui = _ctx()
    entry = MCPServerEntry(name="bad-srv", command="bad-srv")

    gate_ran = threading.Event()
    release = threading.Event()
    worker_done = threading.Event()

    def fake_install(srv, *, print_fn=None, cancel_event=None, commit_gate=None):
        def _write() -> bool:
            print_fn(f"  Failed to add {srv.name}: disk full", "red")
            return False  # write fails — gate must not record

        ok = commit_gate(srv.name, _write) if commit_gate is not None else _write()
        gate_ran.set()
        release.wait(2)
        try:
            return ok
        finally:
            worker_done.set()

    cfg_patch, dir_patch, cfg_file = _temp_config(tmp_path)

    with (
        cfg_patch,
        dir_patch,
        patch.object(registry, "fetch_marketplace_index", return_value=[entry]),
        patch.object(registry, "find_server_by_name", return_value=entry),
        patch.object(registry, "get_installed_names", return_value=set()),
        patch.object(registry, "install_mcp_server", fake_install),
    ):
        task = asyncio.create_task(InstallMCPCommand().execute(ctx, ["bad-srv"]))
        assert await asyncio.to_thread(gate_ran.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        # Wait for the abandoned worker to return before the temp config patch
        # unwinds, so no late frame can touch real config after teardown.
        assert await asyncio.to_thread(worker_done.wait, 2)

    assert not cfg_file.exists() or cfg_file.read_text().strip() == ""
    msgs = " ".join(_messages(ui)).lower()
    assert "nothing will be written" in msgs
    assert "before the cancel" not in msgs  # never claim it was applied


async def test_multi_server_cancel_prints_ledger(tmp_path):
    """Cancel after server 1 commits → server 1 present, server 2 absent, and
    the ledger names both."""
    ctx, ui = _ctx()
    srv1 = MCPServerEntry(name="srv1", command="srv1")
    srv2 = MCPServerEntry(name="srv2", command="srv2", pip_package="pkg2")

    started2 = threading.Event()
    release2 = threading.Event()
    fetch2_done = threading.Event()

    def fake_install_cli_tool(_pkg, *, verify_command=None):
        started2.set()
        release2.wait(2)
        try:
            return True
        finally:
            fetch2_done.set()

    ui.wait_for_mcp_browse = AsyncMock(return_value=[srv1, srv2])
    cfg_patch, dir_patch, cfg_file = _temp_config(tmp_path)

    real_install = registry.install_mcp_server
    worker_done = threading.Event()

    def tracked_install(*args, **kwargs):
        worker_done.clear()
        try:
            return real_install(*args, **kwargs)
        finally:
            worker_done.set()

    with (
        cfg_patch,
        dir_patch,
        patch.object(registry, "fetch_marketplace_index", return_value=[srv1, srv2]),
        patch.object(registry, "get_installed_names", return_value=set()),
        patch.object(registry, "install_cli_tool", fake_install_cli_tool),
        patch.object(registry, "install_mcp_server", tracked_install),
        patch.object(registry, "_resolve_command_path", side_effect=lambda c: c),
    ):
        task = asyncio.create_task(InstallMCPCommand().execute(ctx, []))
        # srv1 commits instantly; srv2 blocks in its fetch.
        assert await asyncio.to_thread(started2.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release2.set()
        assert await asyncio.to_thread(fetch2_done.wait, 2)
        # fetch2_done fires inside srv2's fetch, before its commit gate; wait for
        # srv2's install_mcp_server to return before the temp config patch unwinds
        # so an abandoned write can never reach real config.
        assert await asyncio.to_thread(worker_done.wait, 2)

    written = cfg_file.read_text()
    assert "srv1" in written
    assert "srv2" not in written
    msgs = " ".join(_messages(ui))
    assert "srv1" in msgs
    assert "srv2" in msgs
    assert "Not installed" in msgs
