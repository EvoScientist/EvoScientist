"""Tests for the environment ``start_langgraph_dev`` gives the server.

The server always runs with ``EVOSCIENTIST_SERVER_PROCESS`` set; the parent
process and a plain ``import EvoScientist`` leave it unset.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest

from EvoScientist.langgraph_dev import manager


class _PopenAbort(Exception):
    """Raised by the fake ``Popen`` to short-circuit ``start_langgraph_dev``
    after the env dict is constructed but before health-polling runs."""


def _patch_start_prereqs(monkeypatch, tmp_path: Path, runtime_paths) -> dict:
    """Mock everything ``start_langgraph_dev`` does before ``subprocess.Popen``
    so we can run it end-to-end up to the point where the env dict is captured.
    Returns a ``captured`` dict that the test populates from the fake Popen."""
    captured: dict = {}

    from EvoScientist.mcp import client as mcp_client

    monkeypatch.setattr(mcp_client, "load_mcp_config", lambda: {})
    monkeypatch.setattr(manager, "_langgraph_exe", lambda: "/usr/bin/langgraph")

    fake_config = tmp_path / "langgraph.json"
    fake_config.write_text("{}")
    monkeypatch.setattr(manager, "_packaged_langgraph_config", lambda: fake_config)

    # No conflicts, no stale process — straight to spawn. The ``**_kw`` tails
    # absorb the ``host`` argument these probes now take.
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_: False)
    monkeypatch.setattr(manager, "_is_port_occupied", lambda _port, *_a, **_kw: False)
    monkeypatch.setattr(
        manager, "_wait_for_port_bindable", lambda _port, *_a, **_kw: True
    )
    monkeypatch.setattr(manager, "_kill_owned_stale_process", lambda _port: False)
    monkeypatch.setattr(
        manager, "_wait_for_port_release", lambda _port, *_a, **_kw: True
    )

    # Redirect the log file — pid_dir already rooted under tmp via the fixture.
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, log_file=tmp_path / "langgraph_dev.log"),
    )

    def _fake_popen(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs.get("env", {})
        captured["cwd"] = kwargs.get("cwd")
        raise _PopenAbort("env captured")

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    return captured


@pytest.mark.parametrize("inherited", [None, "", "0"])
def test_start_sets_server_process_env(monkeypatch, tmp_path, runtime_paths, inherited):
    """Whatever the caller's shell exports, the server process is marked."""
    if inherited is None:
        monkeypatch.delenv("EVOSCIENTIST_SERVER_PROCESS", raising=False)
    else:
        monkeypatch.setenv("EVOSCIENTIST_SERVER_PROCESS", inherited)
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16174)

    assert captured["env"]["EVOSCIENTIST_SERVER_PROCESS"] == "1"


def test_start_sets_workspace_dir_env(monkeypatch, tmp_path, runtime_paths):
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16178)
    assert captured["env"].get("EVOSCIENTIST_WORKSPACE_DIR") == str(tmp_path)


# =============================================================================
# Bind host — argv flag + env propagation
# =============================================================================


def test_host_defaults_to_loopback_in_argv(monkeypatch, tmp_path, runtime_paths):
    """``--host`` must always be emitted rather than left to the langgraph
    CLI's own default, so the bind stays pinned to _DEFAULT_HOST even if that
    default moves."""
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16178)

    args = captured["args"]
    assert args[args.index("--host") + 1] == "127.0.0.1"


def test_explicit_wildcard_host_reaches_argv(monkeypatch, tmp_path, runtime_paths):
    """The opt-in to a public bind has to survive all the way into argv."""
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16178, host="0.0.0.0")

    args = captured["args"]
    assert args[args.index("--host") + 1] == "0.0.0.0"


def test_env_carries_explicit_bind_host(monkeypatch, tmp_path, runtime_paths):
    """The subprocess resolves its self-dispatch URL from config, so the
    caller-resolved host must be injected — mirrors the port propagation."""
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path, port=16178, host="192.168.1.5"
        )

    assert captured["env"].get("EVOSCIENTIST_LANGGRAPH_DEV_HOST") == "192.168.1.5"


def test_env_host_replaces_inherited(monkeypatch, tmp_path, runtime_paths):
    """A stray export in the user's shell must not override the host this
    caller resolved — otherwise the bind and the dispatch URL desync."""
    monkeypatch.setenv("EVOSCIENTIST_LANGGRAPH_DEV_HOST", "10.0.0.9")
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16178, host="0.0.0.0")

    assert captured["env"].get("EVOSCIENTIST_LANGGRAPH_DEV_HOST") == "0.0.0.0"


# =============================================================================
# Module-load behavior — _ASYNC_SUBAGENTS_AVAILABLE reads env var on import
# =============================================================================


def test_async_subagents_available_init_true_in_server(monkeypatch):
    """When ``EVOSCIENTIST_SERVER_PROCESS`` is set in the env at module
    import time, ``_ASYNC_SUBAGENTS_AVAILABLE`` initializes to True so the
    deployed main agent's ``_maybe_swap_async_subagents`` swaps eagerly
    without waiting for ``start_langgraph_dev`` to flip the flag (which it
    can't — the deploy subprocess never calls that function on itself)."""
    monkeypatch.setenv("EVOSCIENTIST_SERVER_PROCESS", "1")
    # Re-import the module to re-run the module-level initialization.
    import importlib

    import EvoScientist.langgraph_dev.manager as mgr

    reloaded = importlib.reload(mgr)
    try:
        assert reloaded._ASYNC_SUBAGENTS_AVAILABLE is True
        assert reloaded.is_async_subagents_available() is True
    finally:
        # Restore: reload again without the env var so subsequent tests
        # see the normal initialization.
        monkeypatch.delenv("EVOSCIENTIST_SERVER_PROCESS", raising=False)
        importlib.reload(mgr)


def test_async_subagents_available_init_false_without_env(monkeypatch):
    """When the env var is unset, ``_ASYNC_SUBAGENTS_AVAILABLE`` initializes
    to False — the pre-existing safety behavior (fall back to sync if
    langgraph dev isn't reachable)."""
    monkeypatch.delenv("EVOSCIENTIST_SERVER_PROCESS", raising=False)
    import importlib

    import EvoScientist.langgraph_dev.manager as mgr

    reloaded = importlib.reload(mgr)
    assert reloaded._ASYNC_SUBAGENTS_AVAILABLE is False


def test_tunnel_true_appends_flag(monkeypatch, tmp_path, runtime_paths):
    """``tunnel=True`` must add ``--tunnel`` to the langgraph dev argv."""
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path,
            port=16190,
            tunnel=True,
        )

    assert "--tunnel" in captured["args"]


def test_tunnel_false_default_omits_flag(monkeypatch, tmp_path, runtime_paths):
    """``tunnel`` defaults to False — no ``--tunnel`` in the argv."""
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path,
            port=16191,
        )

    assert "--tunnel" not in captured["args"]


def test_node_for_mcp_servers_is_installed_before_the_spawn(
    monkeypatch, tmp_path, runtime_paths
):
    from EvoScientist.mcp import client as mcp_client

    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    servers = {"fs": {"transport": "stdio", "command": "npx"}}
    monkeypatch.setattr(mcp_client, "load_mcp_config", lambda: servers)

    def install(config):
        assert config == servers
        monkeypatch.setenv("PATH", "private-node-bin")

    monkeypatch.setattr(mcp_client, "_ensure_node_for_stdio", install)
    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16176)
    assert captured["env"]["PATH"] == "private-node-bin"


def test_a_failing_node_check_does_not_block_the_spawn(
    monkeypatch, tmp_path, runtime_paths
):
    from EvoScientist.mcp import client as mcp_client

    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    monkeypatch.setattr(mcp_client, "load_mcp_config", lambda: {"fs": None})
    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16177)
    assert "env" in captured


def test_start_records_the_agent_python_in_the_sidecar(
    monkeypatch, tmp_path, runtime_paths
):
    """Reuse compares this record; without it the python warning never fires."""
    import json
    from types import SimpleNamespace

    from EvoScientist.setup import research_env

    _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(
            manager.RUNTIME, workspace_sidecar=tmp_path / "workspace.json"
        ),
    )
    monkeypatch.setattr(research_env, "agent_python", lambda: "/env/bin/python")

    def _poll():
        # Called by the health loop, after the sidecar is written.
        raise _PopenAbort("sidecar written")

    monkeypatch.setattr(
        subprocess, "Popen", lambda *_a, **_kw: SimpleNamespace(pid=4242, poll=_poll)
    )
    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16179)
    sidecar = json.loads((tmp_path / "workspace.json").read_text())
    assert sidecar["agent_python"] == "/env/bin/python"
