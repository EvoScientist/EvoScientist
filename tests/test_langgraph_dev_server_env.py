"""Tests for the environment ``start_langgraph_dev`` gives the server.

The server always runs with ``EVOSCIENTIST_SERVER_PROCESS`` set; the parent
process and a plain ``import EvoScientist`` leave it unset.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from EvoScientist.langgraph_dev import boundary, manager
from tests.fakes.langgraph_dev import PopenAbort, patch_start_prereqs


@pytest.mark.parametrize("inherited", [None, "", "0"])
def test_start_sets_server_process_env(monkeypatch, tmp_path, runtime_paths, inherited):
    """Whatever the caller's shell exports, the server process is marked."""
    if inherited is None:
        monkeypatch.delenv("EVOSCIENTIST_SERVER_PROCESS", raising=False)
    else:
        monkeypatch.setenv("EVOSCIENTIST_SERVER_PROCESS", inherited)
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16174)

    assert captured["env"]["EVOSCIENTIST_SERVER_PROCESS"] == "1"


def test_start_sets_workspace_dir_env(monkeypatch, tmp_path, runtime_paths):
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16178)
    assert captured["env"].get("EVOSCIENTIST_WORKSPACE_DIR") == str(tmp_path)


def test_start_sets_the_browser_boundary(monkeypatch, tmp_path, runtime_paths):
    """Set on every start, so nothing inherited (a workspace ``.env``) widens
    it; cloudflared inherits the server's environment."""
    monkeypatch.setenv("CORS_CONFIG", '{"allow_origins": ["*"]}')
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")
    monkeypatch.setenv("TUNNEL_HTTP_HOST_HEADER", "localhost")
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path,
            port=16178,
            allowed_origins="https://smith.langchain.com",
        )
    env = captured["env"]
    cors = json.loads(env["CORS_CONFIG"])
    assert cors["allow_origins"] == ["https://smith.langchain.com"]
    assert json.loads(env[boundary.BOUNDARY_ENV])["origins"] == cors["allow_origins"]
    assert "CORS_ALLOW_ORIGINS" not in env
    assert "TUNNEL_HTTP_HOST_HEADER" not in env


def test_start_passes_this_process_data_folder(monkeypatch, tmp_path, runtime_paths):
    """The server runs in the workspace and loads its ``.env``; a relative or
    later-changed variable must not give it another folder."""
    from EvoScientist import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv("EVOSCIENTIST_DATA_DIR", "rel-data")
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16178)
    assert captured["env"]["EVOSCIENTIST_DATA_DIR"] == str(tmp_path / "data")


def test_start_trusts_the_bind_host(monkeypatch, tmp_path, runtime_paths):
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path, port=16178, host="myhost.local"
        )
    names = json.loads(captured["env"][boundary.BOUNDARY_ENV])["names"]
    assert "myhost.local" in names


def test_ensure_passes_backend_allowed_origins(monkeypatch, tmp_path, runtime_paths):
    from EvoScientist.config import EvoScientistConfig

    started: dict = {}
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: False)
    monkeypatch.setattr(manager, "_is_port_occupied", lambda *_a, **_kw: False)
    monkeypatch.setattr(manager.atexit, "register", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        manager, "start_langgraph_dev", lambda *_a, **kw: started.update(kw)
    )
    config = EvoScientistConfig(
        langgraph_dev_port=16178, backend_allowed_origins="https://smith.langchain.com"
    )
    manager.ensure_langgraph_dev(config, workspace_dir=tmp_path)
    assert started["allowed_origins"] == "https://smith.langchain.com"


# =============================================================================
# Bind host — argv flag + env propagation
# =============================================================================


def test_host_defaults_to_loopback_in_argv(monkeypatch, tmp_path, runtime_paths):
    """``--host`` must always be emitted rather than left to the langgraph
    CLI's own default, so the bind stays pinned to _DEFAULT_HOST even if that
    default moves."""
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16178)

    args = captured["args"]
    assert args[args.index("--host") + 1] == "127.0.0.1"


def test_explicit_wildcard_host_reaches_argv(monkeypatch, tmp_path, runtime_paths):
    """The opt-in to a public bind has to survive all the way into argv."""
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16178, host="0.0.0.0")

    args = captured["args"]
    assert args[args.index("--host") + 1] == "0.0.0.0"


def test_env_carries_explicit_bind_host(monkeypatch, tmp_path, runtime_paths):
    """The subprocess resolves its self-dispatch URL from config, so the
    caller-resolved host must be injected — mirrors the port propagation."""
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path, port=16178, host="192.168.1.5"
        )

    assert captured["env"].get("EVOSCIENTIST_LANGGRAPH_DEV_HOST") == "192.168.1.5"


def test_env_host_replaces_inherited(monkeypatch, tmp_path, runtime_paths):
    """A stray export in the user's shell must not override the host this
    caller resolved — otherwise the bind and the dispatch URL desync."""
    monkeypatch.setenv("EVOSCIENTIST_LANGGRAPH_DEV_HOST", "10.0.0.9")
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
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
    can't — the server never calls that function on itself)."""
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
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path,
            port=16190,
            tunnel=True,
        )

    assert "--tunnel" in captured["args"]


def test_tunnel_false_default_omits_flag(monkeypatch, tmp_path, runtime_paths):
    """``tunnel`` defaults to False — no ``--tunnel`` in the argv."""
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=tmp_path,
            port=16191,
        )

    assert "--tunnel" not in captured["args"]


def test_node_for_mcp_servers_is_installed_before_the_spawn(
    monkeypatch, tmp_path, runtime_paths
):
    from EvoScientist.mcp import client as mcp_client

    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    servers = {"fs": {"transport": "stdio", "command": "npx"}}
    monkeypatch.setattr(mcp_client, "load_mcp_config", lambda: servers)

    def install(config):
        assert config == servers
        monkeypatch.setenv("PATH", "private-node-bin")

    monkeypatch.setattr(mcp_client, "_ensure_node_for_stdio", install)
    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16176)
    assert captured["env"]["PATH"] == "private-node-bin"


def test_a_failing_node_check_does_not_block_the_spawn(
    monkeypatch, tmp_path, runtime_paths
):
    from EvoScientist.mcp import client as mcp_client

    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    monkeypatch.setattr(mcp_client, "load_mcp_config", lambda: {"fs": None})
    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16177)
    assert "env" in captured


def test_start_records_the_agent_python_in_the_sidecar(
    monkeypatch, tmp_path, runtime_paths
):
    """Reuse compares this record; without it the python warning never fires."""
    import json
    from types import SimpleNamespace

    from EvoScientist.setup import research_env

    patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
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
        raise PopenAbort("sidecar written")

    monkeypatch.setattr(
        subprocess, "Popen", lambda *_a, **_kw: SimpleNamespace(pid=4242, poll=_poll)
    )
    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16179)
    sidecar = json.loads((tmp_path / "workspace.json").read_text())
    assert sidecar["agent_python"] == "/env/bin/python"


def test_start_puts_the_server_in_the_callers_job(monkeypatch, tmp_path, runtime_paths):
    """The server is spawned with the job's flags and joins the job before the
    PID file is written (``started`` stops the run here)."""
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    monkeypatch.setattr(manager, "_PROCESS", None)
    proc = SimpleNamespace(pid=4242)

    def _popen(args, **kwargs):
        captured["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(subprocess, "Popen", _popen)

    class _Job:
        creationflags = 0x4

        def started(self, process):
            captured["started"] = process
            # Already recorded, so a caller's stop reaches it if this fails.
            captured["recorded"] = manager._PROCESS
            raise PopenAbort("joined")

    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16179, job=_Job())
    assert captured["kwargs"]["creationflags"] & 0x4
    assert captured["started"] is proc
    assert captured["recorded"] is proc
    assert not manager.RUNTIME.pid_file.exists()


def test_start_without_a_job_keeps_the_spawn_flags(
    monkeypatch, tmp_path, runtime_paths
):
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)

    def _popen(args, **kwargs):
        captured["kwargs"] = kwargs
        raise PopenAbort("spawned")

    monkeypatch.setattr(subprocess, "Popen", _popen)
    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16179)
    if manager.os.name == "nt":
        # The process-group flag plus the suspended start the server sweep
        # job resumes (#604): the server joins the job before its first
        # worker can escape it.
        assert captured["kwargs"]["creationflags"] == 0x00000200 | 0x00000004
    else:
        assert captured["kwargs"]["start_new_session"] is True
        assert "creationflags" not in captured["kwargs"]


def test_start_records_the_bind_host(monkeypatch, tmp_path, runtime_paths):
    patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(
        subprocess, "Popen", lambda *_a, **_k: SimpleNamespace(pid=4242)
    )
    recorded: dict = {}

    def _write(**kwargs):
        recorded.update(kwargs)
        raise PopenAbort("recorded")

    monkeypatch.setattr(manager, "_write_workspace_sidecar", _write)
    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16180, host="0.0.0.0")
    assert recorded["host"] == "0.0.0.0"


def test_start_passes_env_origins_to_the_server_only(
    monkeypatch, tmp_path, runtime_paths
):
    """The record of where config variables came from reaches the server's
    environment, never this process's (so no other child inherits it)."""
    from EvoScientist.config import settings

    monkeypatch.setattr(
        settings, "_ENV_ORIGINS", {"OPENAI_API_KEY": {"source": "file"}}
    )
    monkeypatch.delenv("EVOSCIENTIST_LANGGRAPH_DEV_PORT", raising=False)
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16190)

    origins = json.loads(captured["env"][settings.ENV_ORIGINS_VAR])
    assert origins["OPENAI_API_KEY"] == {"source": "file"}
    # Not in the file, not the default port: a --port flag.
    assert origins["EVOSCIENTIST_LANGGRAPH_DEV_PORT"] == {"source": "cli"}
    assert settings.ENV_ORIGINS_VAR not in os.environ


def test_env_origins_leave_the_users_own_export_alone(
    monkeypatch, tmp_path, runtime_paths
):
    from EvoScientist.config import settings

    monkeypatch.setattr(settings, "_ENV_ORIGINS", {})
    monkeypatch.setenv("EVOSCIENTIST_LANGGRAPH_DEV_PORT", "16191")
    captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(PopenAbort):
        manager.start_langgraph_dev(workspace_dir=tmp_path, port=16191)

    origins = json.loads(captured["env"][settings.ENV_ORIGINS_VAR])
    assert "EVOSCIENTIST_LANGGRAPH_DEV_PORT" not in origins
