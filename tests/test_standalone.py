"""Tests for the headless standalone channel runner's dev-server startup."""

from __future__ import annotations

import os
from types import SimpleNamespace

import EvoScientist.channels.standalone as standalone


def _patch_backend_deps(monkeypatch, *, gateway_backend, default_workdir):
    """Stub the lazy imports in ``_ensure_standalone_dev_server`` and record calls.

    Returns the list that ``ensure_langgraph_dev`` calls append to, so a test can
    assert whether (and with what workspace) the dev server was ensured.
    """
    import EvoScientist.config as config_mod
    import EvoScientist.langgraph_dev.manager as manager_mod
    import EvoScientist.paths as paths_mod

    ensure_calls: list[dict] = []

    config = SimpleNamespace(
        gateway_backend=gateway_backend,
        default_workdir=default_workdir,
    )
    monkeypatch.setattr(config_mod, "get_effective_config", lambda: config)
    monkeypatch.setattr(
        manager_mod,
        "ensure_langgraph_dev",
        lambda cfg, *, workspace_dir: ensure_calls.append(
            {"config": cfg, "workspace_dir": workspace_dir}
        ),
    )
    # Avoid mutating the real process-global workspace / creating dirs.
    monkeypatch.setattr(paths_mod, "set_workspace_root", lambda path: None)
    monkeypatch.setattr(paths_mod, "ensure_dirs", lambda: None)
    return ensure_calls


def test_ensure_dev_server_spawns_on_server_backend(monkeypatch, tmp_path):
    ensure_calls = _patch_backend_deps(
        monkeypatch,
        gateway_backend="langgraph_server",
        default_workdir=str(tmp_path),
    )

    standalone._ensure_standalone_dev_server()

    assert len(ensure_calls) == 1
    assert ensure_calls[0]["workspace_dir"] == os.path.abspath(str(tmp_path))


def test_ensure_dev_server_noop_on_local_backend(monkeypatch, tmp_path):
    ensure_calls = _patch_backend_deps(
        monkeypatch,
        gateway_backend="local",
        default_workdir=str(tmp_path),
    )

    standalone._ensure_standalone_dev_server()

    assert ensure_calls == []


def test_ensure_dev_server_falls_back_to_cwd(monkeypatch):
    ensure_calls = _patch_backend_deps(
        monkeypatch,
        gateway_backend="langgraph_server",
        default_workdir="",
    )

    standalone._ensure_standalone_dev_server()

    assert len(ensure_calls) == 1
    assert ensure_calls[0]["workspace_dir"] == os.getcwd()


def test_run_standalone_ensures_dev_server_only_with_agent(monkeypatch):
    """``run_standalone`` ensures the dev server iff the agent is loaded."""
    ensure_flags: list[bool] = []
    monkeypatch.setattr(
        standalone,
        "_ensure_standalone_dev_server",
        lambda: ensure_flags.append(True),
    )

    def _fake_run(coro):
        coro.close()  # never await; just drop the coroutine

    monkeypatch.setattr(standalone.asyncio, "run", _fake_run)

    standalone.run_standalone(channel=None, bus=None, use_agent=True)
    assert ensure_flags == [True]

    ensure_flags.clear()
    standalone.run_standalone(channel=None, bus=None, use_agent=False)
    assert ensure_flags == []
