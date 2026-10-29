"""Tests for the server's ownership record (the workspace sidecar) and the
reuse of a running server.

The server serves every registered project, so a client reuses any
EvoScientist server on the port and registers its own project; the record
says what the server was launched with, for the drift warnings.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time

import psutil
import pytest

from EvoScientist import paths
from EvoScientist.langgraph_dev import manager


def _wait_for_child_argv(pid: int, needle: str, timeout: float = 5.0) -> bool:
    """Wait until a just-started child shows its own argv.

    ``Popen`` can return while the child is still inside ``exec``, before
    Linux has recorded the new argv, so a read right then comes back empty.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            if needle in psutil.Process(pid).cmdline():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


def test_sidecar_path_is_next_to_pid_file():
    """Sidecar JSON lives at ``pid_dir / 'langgraph_dev.workspace.json'``."""
    assert (
        manager.RUNTIME.workspace_sidecar
        == manager.RUNTIME.pid_dir / "langgraph_dev.workspace.json"
    )


def test_write_workspace_sidecar_records_workspace_and_pid(
    tmp_path, monkeypatch, runtime_paths
):
    """``_write_workspace_sidecar`` writes JSON with workspace + pid."""
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    workspace = tmp_path / "some" / "ws"
    manager._write_workspace_sidecar(workspace_dir=workspace, pid=12345)
    data = json.loads((tmp_path / "ws.json").read_text())
    assert data["workspace"] == str(workspace)
    assert data["pid"] == 12345


def test_read_workspace_sidecar_returns_none_when_missing(
    tmp_path, monkeypatch, runtime_paths
):
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "absent.json"),
    )
    assert manager._read_workspace_sidecar() is None


def test_read_workspace_sidecar_returns_none_on_corrupt_json(
    tmp_path, monkeypatch, runtime_paths
):
    sidecar = tmp_path / "bad.json"
    sidecar.write_text("not json at all")
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=sidecar),
    )
    assert manager._read_workspace_sidecar() is None


@pytest.mark.parametrize(
    "payload",
    [
        "[]",  # valid JSON, wrong top-level type
        "{}",  # valid JSON dict but missing "workspace"
        '{"pid": 12345}',  # valid JSON dict but missing "workspace"
        '"just a string"',  # valid JSON scalar
        "null",  # valid JSON null
        '{"workspace": null}',  # workspace present but null → Path(None) TypeError
        '{"workspace": []}',  # workspace present but list → Path([]) TypeError
        '{"workspace": 12345}',  # workspace present but int → Path(int) TypeError
        '{"workspace": ""}',  # workspace present but empty string → resolves to cwd
    ],
)
def test_read_workspace_sidecar_returns_none_on_wrong_schema(
    payload, tmp_path, monkeypatch, runtime_paths
):
    """JSON that parses but doesn't match the expected schema degrades to
    None: readers show ``workspace`` as a path."""
    sidecar = tmp_path / "schema.json"
    sidecar.write_text(payload)
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=sidecar),
    )
    assert manager._read_workspace_sidecar() is None


def test_read_workspace_sidecar_round_trip(tmp_path, monkeypatch, runtime_paths):
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "rt.json"),
    )
    workspace = tmp_path / "x" / "y"
    manager._write_workspace_sidecar(workspace_dir=workspace, pid=42)
    data = manager._read_workspace_sidecar()
    assert data == {
        "workspace": str(workspace),
        "pid": 42,
        "data_dir": str(paths.DATA_DIR),
        "config_dir": str(paths.CONFIG_DIR),
    }


def test_write_workspace_sidecar_records_the_bind_host(
    tmp_path, monkeypatch, runtime_paths
):
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    manager._write_workspace_sidecar(workspace_dir=tmp_path, pid=7, host="0.0.0.0")
    assert manager._read_workspace_sidecar()["host"] == "0.0.0.0"


def test_write_workspace_sidecar_records_the_data_and_config_folders(
    tmp_path, monkeypatch, runtime_paths
):
    """The server inherits this process's folders, so the record names them
    for a client that attaches from another environment."""
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data dir é")
    monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path / "config" / "evoscientist")
    manager._write_workspace_sidecar(workspace_dir=tmp_path / "ws", pid=7)
    data = manager._read_workspace_sidecar()
    assert data["data_dir"] == str(tmp_path / "data dir é")
    assert data["config_dir"] == str(tmp_path / "config" / "evoscientist")


def test_ensure_langgraph_dev_serves_another_project_from_running_server(
    tmp_path, monkeypatch, runtime_paths
):
    """A server started from A serves B once B is registered."""
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    manager._write_workspace_sidecar(workspace_dir=tmp_path / "A", pid=99999)
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", None)

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    assert manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path / "B") is True

    assert manager.is_async_subagents_available()


def test_ensure_langgraph_dev_reuses_when_sidecar_missing(
    tmp_path, monkeypatch, runtime_paths
):
    """A server without an ownership record is reused without the drift
    checks; the registration tells an older release apart."""
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "absent.json"),
    )
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", None)

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path / "B")

    assert manager.CONFIG_DRIFT_SINCE_LAUNCH is False


def test_stop_langgraph_dev_removes_sidecar(tmp_path, monkeypatch, runtime_paths):
    """``stop_langgraph_dev`` should unlink the sidecar alongside the PID file."""
    sidecar = tmp_path / "ws.json"
    pid_file = tmp_path / "pid.txt"
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(
            runtime_paths, workspace_sidecar=sidecar, pid_file=pid_file
        ),
    )
    manager._write_workspace_sidecar(workspace_dir=tmp_path / "x", pid=42)
    assert sidecar.exists()

    # _PROCESS is None so stop_langgraph_dev shouldn't try to kill anything;
    # we're only verifying the sidecar cleanup path here.
    monkeypatch.setattr(manager, "_PROCESS", None)
    manager.stop_langgraph_dev()
    assert not sidecar.exists()


def test_started_server_outlives_session(tmp_path, monkeypatch, runtime_paths):
    """Every session shares it until ``EvoSci server stop``."""
    import atexit

    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: False)
    monkeypatch.setattr(manager, "_PROCESS", None)
    started: list = []
    monkeypatch.setattr(
        manager, "start_langgraph_dev", lambda **kw: started.append(kw) or object()
    )
    registered = []
    monkeypatch.setattr(atexit, "register", lambda *a, **kw: registered.append(a))

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    assert manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path / "A")
    assert registered == []
    assert started[0].get("lasting", True) is True


def test_sidecar_records_whether_server_lasts(tmp_path, monkeypatch, runtime_paths):
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    manager._write_workspace_sidecar(workspace_dir=tmp_path, pid=1, lasting=False)
    assert manager._read_workspace_sidecar()["lasting"] is False


# ---------------------------------------------------------------------------
# Config fingerprint + drift detection + explicit server stop
# ---------------------------------------------------------------------------


def _dead_pid() -> int:
    """Spawn-and-reap a process so its pid is reliably dead."""
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_sidecar_records_config_fingerprint_when_passed(
    tmp_path, monkeypatch, runtime_paths
):
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    manager._write_workspace_sidecar(
        workspace_dir=tmp_path / "ws", pid=1, config_fingerprint="abc123"
    )
    assert json.loads((tmp_path / "ws.json").read_text())["config_fingerprint"] == (
        "abc123"
    )


def test_server_config_fingerprint_tracks_relevant_fields():
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    assert base == manager._server_config_fingerprint(cfg), "must be deterministic"
    cfg.model = "some-other-model"
    assert manager._server_config_fingerprint(cfg) != base


def _reuse_setup(tmp_path, monkeypatch, runtime_paths, fingerprint):
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    manager._write_workspace_sidecar(
        workspace_dir=tmp_path / "A", pid=99999, config_fingerprint=fingerprint
    )
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", None)
    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    return cfg


def test_reuse_serves_registered_project(tmp_path, monkeypatch, runtime_paths):
    cfg = _reuse_setup(tmp_path, monkeypatch, runtime_paths, "fp")
    manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path / "A")


def test_reuse_sets_drift_flag_on_fingerprint_mismatch(
    tmp_path, monkeypatch, runtime_paths
):
    cfg = _reuse_setup(tmp_path, monkeypatch, runtime_paths, "stale-fingerprint")
    manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path / "A")
    assert manager.CONFIG_DRIFT_SINCE_LAUNCH is True


def test_reuse_sets_drift_flag_for_older_stripped_server(
    tmp_path, monkeypatch, runtime_paths
):
    """A server an older version started without MCP (``deploy_mode: false``)
    is reused with the drift warning, also by a server-backend session."""
    cfg = _reuse_setup(tmp_path, monkeypatch, runtime_paths, "old-version-fingerprint")
    sidecar = tmp_path / "ws.json"
    sidecar.write_text(
        json.dumps({**json.loads(sidecar.read_text()), "deploy_mode": False})
    )

    reused = manager.ensure_langgraph_dev(
        cfg, workspace_dir=tmp_path / "A", backend="langgraph_server"
    )

    assert reused is True
    assert manager.CONFIG_DRIFT_SINCE_LAUNCH is True
    assert manager.is_async_subagents_available() is True


def test_reuse_sets_drift_flag_on_version_change(tmp_path, monkeypatch, runtime_paths):
    """A server left running across an upgrade counts as drift: the session
    reuses it and warns, and the server is not stopped."""
    import importlib.metadata

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.3.4")
    old_fp = manager._server_config_fingerprint(cfg)
    cfg2 = _reuse_setup(tmp_path, monkeypatch, runtime_paths, old_fp)
    stopped = []
    monkeypatch.setattr(manager, "stop_langgraph_dev", lambda *a: stopped.append(a))

    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.3.5")
    assert manager.ensure_langgraph_dev(cfg2, workspace_dir=tmp_path / "A") is True

    assert manager.CONFIG_DRIFT_SINCE_LAUNCH is True
    assert manager.is_async_subagents_available() is True
    assert stopped == []


def test_started_server_recorded_for_next_version(tmp_path, monkeypatch, runtime_paths):
    """A server the CLI starts records what the next version compares: the
    marker, the PID file for ``EvoSci server stop`` and the fingerprint."""
    import importlib.metadata
    import subprocess
    from types import SimpleNamespace

    from EvoScientist.mcp import client as mcp_client

    up = {"running": False}
    spawned = []

    def _popen(*_a, **kw):
        spawned.append(kw["env"])
        up["running"] = True
        return SimpleNamespace(pid=4242, poll=lambda: None)

    monkeypatch.setattr(mcp_client, "load_mcp_config", lambda: {})
    monkeypatch.setattr(manager, "_langgraph_exe", lambda: "/usr/bin/langgraph")
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_: up["running"])
    monkeypatch.setattr(manager, "_is_port_occupied", lambda *_a, **_kw: False)
    monkeypatch.setattr(manager, "_wait_for_port_bindable", lambda *_a, **_kw: True)
    monkeypatch.setattr(subprocess, "Popen", _popen)
    monkeypatch.setattr(manager, "_PROCESS", None)
    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True

    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.3.5")
    manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path / "A")
    assert spawned[0]["EVOSCIENTIST_SERVER_PROCESS"] == "1"
    assert runtime_paths.pid_file.read_text() == "4242"

    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.3.6")
    assert manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path / "A") is True
    assert manager.CONFIG_DRIFT_SINCE_LAUNCH is True
    assert len(spawned) == 1


def test_reuse_clears_drift_flag_on_matching_fingerprint(
    tmp_path, monkeypatch, runtime_paths
):
    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    fp = manager._server_config_fingerprint(cfg)
    cfg2 = _reuse_setup(tmp_path, monkeypatch, runtime_paths, fp)
    manager.ensure_langgraph_dev(cfg2, workspace_dir=tmp_path / "A")
    assert manager.CONFIG_DRIFT_SINCE_LAUNCH is False


def test_server_config_fingerprint_ignores_the_agent_python(monkeypatch):
    """The fingerprint covers launch settings only; a PATH difference is
    reported on its own (``AGENT_SHELL_DRIFT``)."""
    from EvoScientist.setup import research_env

    cfg = manager.EvoScientistConfig()
    monkeypatch.setattr(research_env, "agent_python", lambda: None)
    base = manager._server_config_fingerprint(cfg)
    monkeypatch.setattr(research_env, "agent_python", lambda: "/env/bin/python")
    assert manager._server_config_fingerprint(cfg) == base


def test_sidecar_records_the_agent_python_only_when_passed(
    tmp_path, monkeypatch, runtime_paths
):
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    manager._write_workspace_sidecar(workspace_dir=tmp_path, pid=1, agent_python=None)
    assert json.loads((tmp_path / "ws.json").read_text())["agent_python"] is None
    manager._write_workspace_sidecar(workspace_dir=tmp_path, pid=1)
    assert "agent_python" not in json.loads((tmp_path / "ws.json").read_text())


def test_sidecar_records_the_agent_bash_only_when_passed(
    tmp_path, monkeypatch, runtime_paths
):
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    bash = r"C:\Program Files\Git\bin\bash.exe"
    manager._write_workspace_sidecar(workspace_dir=tmp_path, pid=1, agent_bash=bash)
    assert json.loads((tmp_path / "ws.json").read_text())["agent_bash"] == bash
    manager._write_workspace_sidecar(workspace_dir=tmp_path, pid=1)
    assert "agent_bash" not in json.loads((tmp_path / "ws.json").read_text())


def test_reuse_warns_when_the_server_shell_differs(
    tmp_path, monkeypatch, runtime_paths
):
    """A server started before `EvoSci setup` recorded Git Bash keeps cmd.exe
    for its agents; this session runs bash."""
    from types import SimpleNamespace

    from EvoScientist import agent_shell
    from EvoScientist.setup import research_env

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    fp = manager._server_config_fingerprint(cfg)
    cfg2 = _reuse_setup(tmp_path, monkeypatch, runtime_paths, fp)
    manager._write_workspace_sidecar(
        workspace_dir=tmp_path / "A",
        pid=99999,
        config_fingerprint=fp,
        agent_python="/env/bin/python",
        agent_bash=None,
    )
    monkeypatch.setattr(research_env, "agent_python", lambda: "/env/bin/python")
    bash = r"C:\Git\bin\bash.exe"
    monkeypatch.setattr(agent_shell, "agent_bash", lambda: SimpleNamespace(bash=bash))
    manager.ensure_langgraph_dev(cfg2, workspace_dir=tmp_path / "A")
    assert "cmd.exe" in manager.AGENT_SHELL_DRIFT
    assert bash in manager.AGENT_SHELL_DRIFT
    assert "python" not in manager.AGENT_SHELL_DRIFT


def _reuse_with_agent_python(tmp_path, monkeypatch, runtime_paths, recorded):
    """Reuse a server whose sidecar records ``recorded`` (or nothing, when
    ``recorded`` is ``...``) from a session that resolves the research env."""
    from EvoScientist.setup import research_env

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    fp = manager._server_config_fingerprint(cfg)
    cfg2 = _reuse_setup(tmp_path, monkeypatch, runtime_paths, fp)
    if recorded is not ...:
        manager._write_workspace_sidecar(
            workspace_dir=tmp_path / "A",
            pid=99999,
            config_fingerprint=fp,
            agent_python=recorded,
        )
    monkeypatch.setattr(research_env, "agent_python", lambda: "/env/bin/python")
    manager.ensure_langgraph_dev(cfg2, workspace_dir=tmp_path / "A")


def test_reuse_warns_when_the_server_python_differs(
    tmp_path, monkeypatch, runtime_paths
):
    """A server launched from a shell with conda active, reused from one
    without: its agents keep conda's python."""
    _reuse_with_agent_python(tmp_path, monkeypatch, runtime_paths, "/conda/bin/python")
    assert "/conda/bin/python" in manager.AGENT_SHELL_DRIFT
    assert "/env/bin/python" in manager.AGENT_SHELL_DRIFT
    assert manager.CONFIG_DRIFT_SINCE_LAUNCH is False


def test_reuse_does_not_warn_when_the_server_python_matches(
    tmp_path, monkeypatch, runtime_paths
):
    _reuse_with_agent_python(tmp_path, monkeypatch, runtime_paths, "/env/bin/python")
    assert manager.AGENT_SHELL_DRIFT is None


def test_reuse_does_not_warn_without_a_recorded_python(
    tmp_path, monkeypatch, runtime_paths
):
    """A server started by an older version has no record."""
    _reuse_with_agent_python(tmp_path, monkeypatch, runtime_paths, ...)
    assert manager.AGENT_SHELL_DRIFT is None


def test_stop_recorded_server_none_when_no_pid_file(
    tmp_path, monkeypatch, runtime_paths
):
    monkeypatch.setattr(manager, "RUNTIME", runtime_paths)
    monkeypatch.setattr(manager, "_PROCESS", None)
    assert manager.stop_recorded_server() is None


def test_stop_recorded_server_cleans_stale_files_for_dead_pid(
    tmp_path, monkeypatch, runtime_paths
):
    pid_file = tmp_path / "pid.txt"
    sidecar = tmp_path / "ws.json"
    pid_file.write_text(str(_dead_pid()))
    sidecar.write_text(json.dumps({"workspace": "/a", "pid": 1}))
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(
            runtime_paths, pid_file=pid_file, workspace_sidecar=sidecar
        ),
    )
    monkeypatch.setattr(manager, "_PROCESS", None)
    assert manager.stop_recorded_server() is None
    assert not pid_file.exists()
    assert not sidecar.exists()


def test_stop_recorded_server_refuses_foreign_process(
    tmp_path, monkeypatch, runtime_paths
):
    import subprocess
    import sys

    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        pid_file = tmp_path / "pid.txt"
        pid_file.write_text(str(victim.pid))
        monkeypatch.setattr(
            manager,
            "RUNTIME",
            dataclasses.replace(runtime_paths, pid_file=pid_file),
        )
        monkeypatch.setattr(manager, "_PROCESS", None)
        assert manager.stop_recorded_server() is None
        assert victim.poll() is None, "foreign process must not be killed"
    finally:
        victim.kill()
        victim.wait()


def test_stop_recorded_server_kills_owned_langgraph_process(
    tmp_path, monkeypatch, runtime_paths
):
    import subprocess
    import sys

    victim = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "langgraph"]
    )
    try:
        assert _wait_for_child_argv(victim.pid, "langgraph"), (
            "child never reported its own argv"
        )
        pid_file = tmp_path / "pid.txt"
        sidecar = tmp_path / "ws.json"
        pid_file.write_text(str(victim.pid))
        sidecar.write_text(json.dumps({"workspace": "/a", "pid": victim.pid}))
        monkeypatch.setattr(
            manager,
            "RUNTIME",
            dataclasses.replace(
                runtime_paths, pid_file=pid_file, workspace_sidecar=sidecar
            ),
        )
        monkeypatch.setattr(manager, "_PROCESS", None)
        assert manager.stop_recorded_server() == victim.pid
        assert victim.poll() is not None, "owned langgraph process must be stopped"
        assert not pid_file.exists()
        assert not sidecar.exists()
    finally:
        if victim.poll() is None:
            victim.kill()
        victim.wait()


def test_stop_inflight_owned_server_ignores_disk_record(
    tmp_path, monkeypatch, runtime_paths
):
    """The launcher's mid-start fallback stops only the in-memory owned process.
    With nothing spawned yet (pre-Popen), the on-disk PID file still names a
    different session's backend — it must be left untouched, not killed."""
    import subprocess
    import sys

    victim = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "langgraph"]
    )
    try:
        pid_file = tmp_path / "pid.txt"
        pid_file.write_text(str(victim.pid))
        monkeypatch.setattr(
            manager, "RUNTIME", dataclasses.replace(runtime_paths, pid_file=pid_file)
        )
        monkeypatch.setattr(manager, "_PROCESS", None)  # nothing spawned yet
        assert manager.stop_inflight_owned_server() is None
        assert victim.poll() is None, "disk-recorded server must not be killed"
    finally:
        victim.kill()
        victim.wait()


def test_stop_inflight_owned_server_stops_the_owned_process(monkeypatch, runtime_paths):
    """Once our own child is spawned (``_PROCESS`` set), a mid-start stop still
    tears it down, so the close-during-boot orphan is cleaned up."""
    import subprocess
    import sys

    owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        monkeypatch.setattr(manager, "RUNTIME", runtime_paths)
        monkeypatch.setattr(manager, "_PROCESS", owned)
        assert manager.stop_inflight_owned_server() == owned.pid
        owned.wait(timeout=5)
        assert owned.poll() is not None, "owned in-flight process must be stopped"
    finally:
        if owned.poll() is None:
            owned.kill()
            owned.wait()


def test_stop_recorded_server_cleans_corrupt_pid_file(
    tmp_path, monkeypatch, runtime_paths
):
    """A corrupt PID file is cleaned up, matching the docstring's promise."""
    pid_file = tmp_path / "pid.txt"
    pid_file.write_text("not-a-pid")
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, pid_file=pid_file),
    )
    monkeypatch.setattr(manager, "_PROCESS", None)
    assert manager.stop_recorded_server() is None
    assert not pid_file.exists()


# ---------------------------------------------------------------------------
# Windows: Ctrl+Break before the hard kill, so a cut run is run again
# ---------------------------------------------------------------------------


def _sleeper(*extra: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", *extra]
    )


@pytest.fixture
def stop_steps(monkeypatch):
    """Records, in order, the clean-stop requests and ``terminate()`` calls.
    The clean stop is faked by the test (``clean_stop``); terminate is real."""
    steps: list[tuple] = []
    real_terminate = psutil.Process.terminate

    def _terminate(self):
        steps.append(("terminate", self.pid))
        return real_terminate(self)

    monkeypatch.setattr(psutil.Process, "terminate", _terminate)
    return steps


def _clean_stop(monkeypatch, steps, *, sent: bool = True, then=None):
    def _request(pid):
        steps.append(("clean", pid))
        if then is not None:
            then()
        return sent

    monkeypatch.setattr(manager, "_request_clean_stop", _request)


def test_clean_stop_that_exits_in_time_is_never_terminated(monkeypatch, stop_steps):
    proc = _sleeper()
    try:
        # The server shuts down on the Ctrl+Break.
        _clean_stop(monkeypatch, stop_steps, then=proc.kill)
        manager._end_server_tree(proc.pid, proc, break_group=True)
        assert stop_steps == [("clean", proc.pid)]
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_clean_stop_that_does_not_exit_ends_hard_with_a_warning(
    monkeypatch, stop_steps, caplog
):
    proc = _sleeper()
    try:
        monkeypatch.setattr(manager, "CLEAN_STOP_TIMEOUT", 0.2)
        _clean_stop(monkeypatch, stop_steps)  # sent, but ignored
        with caplog.at_level("WARNING", logger=manager.__name__):
            manager._end_server_tree(proc.pid, proc, break_group=True)
        assert stop_steps == [("clean", proc.pid), ("terminate", proc.pid)]
        assert proc.poll() is not None
        assert "did not exit within 0.2 s of Ctrl+Break" in caplog.text
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_failed_clean_stop_ends_hard_at_once(monkeypatch, stop_steps):
    proc = _sleeper()
    try:
        monkeypatch.setattr(manager, "CLEAN_STOP_TIMEOUT", 30)
        _clean_stop(monkeypatch, stop_steps, sent=False)
        started = time.monotonic()
        manager._end_server_tree(proc.pid, proc, break_group=True)
        assert time.monotonic() - started < 10, "waited for a Ctrl+Break never sent"
        assert stop_steps == [("clean", proc.pid), ("terminate", proc.pid)]
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_no_clean_stop_without_a_known_process_group(monkeypatch, stop_steps):
    proc = _sleeper()
    try:
        _clean_stop(monkeypatch, stop_steps)
        manager._end_server_tree(proc.pid, proc)
        assert stop_steps == [("terminate", proc.pid)]
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_clean_stop_kills_children_left_after_the_server_exits(monkeypatch, stop_steps):
    """The children are listed before the Ctrl+Break: after the server exits
    they can no longer be found from it."""
    code = (
        "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "print(c.pid, flush=True)\n"
        "time.sleep(30)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    grandchild = None
    try:
        grandchild = psutil.Process(int(proc.stdout.readline()))
        _clean_stop(monkeypatch, stop_steps, then=proc.kill)  # only the server
        manager._end_server_tree(proc.pid, proc, break_group=True)
        _, alive = psutil.wait_procs([grandchild], timeout=5)
        assert alive == []
    finally:
        for p in (proc, grandchild):
            try:
                if p is not None:
                    p.kill()
            except psutil.NoSuchProcess:
                pass
        proc.wait()
        proc.stdout.close()


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM can be ignored on POSIX only")
def test_server_that_ignores_terminate_is_killed(monkeypatch):
    """The escalation both stop paths had before sharing one helper."""
    code = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(30)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    try:
        assert proc.stdout.readline().strip() == b"ready"
        monkeypatch.setattr(manager, "_TERMINATE_TIMEOUT", 0.2)
        manager._end_server_tree(proc.pid, proc)
        assert proc.returncode == -9
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()


_SERVER_WITH_CHILD = (
    "import subprocess, sys, time\n"
    "child = (\n"
    "    'import signal, time\\n'\n"
    "    'signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n'\n"
    "    'print(\"ready\", flush=True)\\n'\n"
    "    'time.sleep(30)\\n'\n"
    ")\n"
    "c = subprocess.Popen([sys.executable, '-c', child], stdout=subprocess.PIPE)\n"
    "c.stdout.readline()\n"
    "print(c.pid, flush=True)\n"
    "time.sleep(30)\n"
)


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM can be ignored on POSIX only")
def test_hard_stop_kills_a_child_that_outlives_its_server(monkeypatch):
    """The server exits on SIGTERM, its child ignores it: after the grace the
    sweep kills the child."""
    monkeypatch.setattr(manager, "_SURVIVOR_GRACE", 0.2)
    proc = subprocess.Popen(
        [sys.executable, "-c", _SERVER_WITH_CHILD], stdout=subprocess.PIPE
    )
    child = None
    try:
        child = psutil.Process(int(proc.stdout.readline()))
        manager._end_server_tree(proc.pid, proc)
        assert proc.poll() is not None
        _, alive = psutil.wait_procs([child], timeout=5)
        assert alive == []
    finally:
        for p in (proc, child):
            try:
                if p is not None:
                    p.kill()
            except psutil.NoSuchProcess:
                pass
        proc.wait()
        proc.stdout.close()


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM can be ignored on POSIX only")
def test_hard_stop_kills_a_child_started_after_the_first_listing(monkeypatch):
    """A server that ignores SIGTERM and starts a child meanwhile: the kill
    lists the children again, so that child goes too."""
    code = (
        "import signal, subprocess, sys, time\n"
        "def late(*_):\n"
        "    c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "    print(c.pid, flush=True)\n"
        "signal.signal(signal.SIGTERM, late)\n"
        "print('ready', flush=True)\n"
        "while True:\n"
        "    time.sleep(0.05)\n"
    )
    monkeypatch.setattr(manager, "_TERMINATE_TIMEOUT", 1.0)
    monkeypatch.setattr(manager, "_SURVIVOR_GRACE", 0.2)
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    late = None
    try:
        assert proc.stdout.readline().strip() == b"ready"
        manager._end_server_tree(proc.pid, proc)
        assert proc.returncode == -9
        late = psutil.Process(int(proc.stdout.readline()))
        _, alive = psutil.wait_procs([late], timeout=5)
        assert alive == []
    except psutil.NoSuchProcess:
        pass  # already gone, as it should be
    finally:
        for p in (proc, late):
            try:
                if p is not None:
                    p.kill()
            except psutil.NoSuchProcess:
                pass
        proc.wait()
        proc.stdout.close()


@pytest.mark.skipif(os.name == "nt", reason="the POSIX stop")
def test_request_clean_stop_runs_no_helper_on_posix(monkeypatch):
    monkeypatch.setattr(
        manager.subprocess,
        "run",
        lambda *a, **k: pytest.fail("the Ctrl+Break helper ran on POSIX"),
    )
    assert manager._request_clean_stop(4242) is False


def test_request_clean_stop_runs_the_helper_without_a_window(monkeypatch):
    seen: dict = {}

    def _run(args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(manager.os, "name", "nt")
    monkeypatch.setattr(manager.subprocess, "run", _run)
    assert manager._request_clean_stop(4242) is True
    assert seen["args"] == [
        sys.executable,
        "-I",  # a project's signal.py in the cwd must not shadow the stdlib
        "-m",
        "EvoScientist.langgraph_dev.console_break",
        "4242",
    ]
    assert seen["kwargs"]["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert seen["kwargs"]["timeout"] == manager._CLEAN_STOP_HELPER_TIMEOUT


@pytest.mark.parametrize(
    ("outcome", "logged"),
    [
        (
            subprocess.CompletedProcess(
                [], 3, "", "AttachConsole(4242) failed: error 6\n"
            ),
            "exited with 3 (AttachConsole(4242) failed: error 6)",
        ),
        (OSError("no python"), "(no python)"),
        (subprocess.TimeoutExpired("python", 5), "timed out"),
    ],
)
def test_request_clean_stop_warns_when_the_helper_fails(
    monkeypatch, caplog, outcome, logged
):
    def _run(*_a, **_k):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(manager.os, "name", "nt")
    monkeypatch.setattr(manager.subprocess, "run", _run)
    with caplog.at_level("WARNING", logger=manager.__name__):
        assert manager._request_clean_stop(4242) is False
    assert logged in caplog.text
    assert "ending it hard" in caplog.text


@pytest.mark.parametrize(
    ("spawned_here", "clean", "expected"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_stop_langgraph_dev_sends_ctrl_break_only_to_its_own_group(
    monkeypatch, runtime_paths, spawned_here, clean, expected
):
    monkeypatch.setattr(manager, "RUNTIME", runtime_paths)
    monkeypatch.setattr(manager, "_PROCESS", None)
    proc = _sleeper()
    seen: list = []

    def _end(pid, popen=None, *, break_group=False):
        seen.append(break_group)
        popen.kill()
        popen.wait()

    monkeypatch.setattr(manager, "_end_server_tree", _end)
    if spawned_here:
        monkeypatch.setattr(manager, "_BREAK_GROUP_PIDS", {proc.pid})
    try:
        manager.stop_langgraph_dev(proc, clean=clean)
        assert seen == [expected]
        assert proc.pid not in manager._BREAK_GROUP_PIDS
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_stop_inflight_owned_server_skips_the_clean_stop(
    monkeypatch, runtime_paths, stop_steps
):
    """A server still starting has no run to put back in the queue."""
    owned = _sleeper()
    try:
        monkeypatch.setattr(manager, "RUNTIME", runtime_paths)
        monkeypatch.setattr(manager, "_PROCESS", owned)
        monkeypatch.setattr(manager, "_BREAK_GROUP_PIDS", {owned.pid})
        _clean_stop(monkeypatch, stop_steps)
        assert manager.stop_inflight_owned_server() == owned.pid
        assert ("clean", owned.pid) not in stop_steps
        assert owned.poll() is not None
    finally:
        if owned.poll() is None:
            owned.kill()
            owned.wait()


@pytest.mark.parametrize(
    ("record", "created_offset", "expected"),
    [
        ({"ctrl_break_group": True}, 0.0, True),
        ({}, None, False),  # a record from before this release
        ({"ctrl_break_group": True}, None, False),  # no creation time recorded
        ({"ctrl_break_group": True, "pid": 1}, 0.0, False),  # another server
        # The PID was reused by a process started later: no Ctrl+Break, which
        # would reach every process on that process's console.
        ({"ctrl_break_group": True}, -60.0, False),
    ],
)
def test_stop_recorded_server_trusts_the_record_for_the_process_group(
    tmp_path, monkeypatch, runtime_paths, record, created_offset, expected
):
    victim = _sleeper("langgraph")
    try:
        assert _wait_for_child_argv(victim.pid, "langgraph")
        pid_file, sidecar = tmp_path / "pid.txt", tmp_path / "ws.json"
        pid_file.write_text(str(victim.pid))
        if created_offset is not None:
            created = psutil.Process(victim.pid).create_time() + created_offset
            record = {**record, "create_time": created}
        sidecar.write_text(json.dumps({"workspace": "/a", "pid": victim.pid, **record}))
        monkeypatch.setattr(
            manager,
            "RUNTIME",
            dataclasses.replace(
                runtime_paths, pid_file=pid_file, workspace_sidecar=sidecar
            ),
        )
        monkeypatch.setattr(manager, "_PROCESS", None)
        seen: list = []

        def _end(pid, popen=None, *, break_group=False):
            seen.append((pid, break_group))
            victim.kill()
            victim.wait()

        monkeypatch.setattr(manager, "_end_server_tree", _end)
        assert manager.stop_recorded_server() == victim.pid
        assert seen == [(victim.pid, expected)]
    finally:
        if victim.poll() is None:
            victim.kill()
            victim.wait()


def test_sidecar_records_the_process_group_only_when_asked(
    tmp_path, monkeypatch, runtime_paths
):
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    manager._write_workspace_sidecar(workspace_dir=tmp_path, pid=1)
    record = manager._read_workspace_sidecar()
    assert "ctrl_break_group" not in record
    assert "create_time" not in record
    manager._write_workspace_sidecar(
        workspace_dir=tmp_path, pid=1, break_group_created=1234.5
    )
    record = manager._read_workspace_sidecar()
    assert record["ctrl_break_group"] is True
    assert record["create_time"] == 1234.5


def test_server_config_fingerprint_tracks_dangerous_mode():
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    cfg.dangerous_mode = True
    assert manager._server_config_fingerprint(cfg) != base


def test_pid_serves_port_matrix():
    import subprocess
    import sys

    assert manager._pid_serves_port(None, 6174) is False
    assert manager._pid_serves_port(True, 6174) is False
    assert manager._pid_serves_port(-1, 6174) is False

    victim = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "langgraph", "6174"]
    )
    try:
        assert _wait_for_child_argv(victim.pid, "6174"), (
            "child never reported its own argv"
        )
        assert manager._pid_serves_port(victim.pid, 6174) is True
        assert manager._pid_serves_port(victim.pid, 9999) is False, (
            "must not attribute a server bound to a different port"
        )
    finally:
        victim.kill()
        victim.wait()
    assert manager._pid_serves_port(victim.pid, 6174) is False, "dead pid"


def test_server_config_fingerprint_covers_new_fields_by_default():
    """The fingerprint iterates the dataclass minus an exclusion set, so
    server-affecting values din0s called out — reasoning effort, provider
    keys/base URLs — count without per-field registration."""
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    cfg.reasoning_effort = "definitely-different"
    after_effort = manager._server_config_fingerprint(cfg)
    assert after_effort != base
    cfg.anthropic_base_url = "https://proxy.example"
    assert manager._server_config_fingerprint(cfg) != after_effort


def test_server_config_fingerprint_ignores_cli_only_fields():
    """Channel/UI-side fields never reach the server — no spurious drift."""
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    cfg.show_thinking = not cfg.show_thinking
    cfg.telegram_bot_token = "tg-token"
    cfg.default_mode = "run"
    assert manager._server_config_fingerprint(cfg) == base


def test_server_config_fingerprint_ignores_shell_allow_list():
    """shell_allow_list is read only by CLI/channel approval resolvers —
    it never reaches the subprocess, so it must not cause drift."""
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    cfg.shell_allow_list = "git status,ls"
    assert manager._server_config_fingerprint(cfg) == base


def test_server_config_fingerprint_ignores_gateway_backend():
    """The gateway settings only decide, in the CLI, whether a server is
    needed; nothing in the server reads them, so they must not cause drift."""
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    cfg.gateway_backend_serve = "langgraph_server"
    cfg.gateway_backend_tui = "langgraph_server"
    assert manager._server_config_fingerprint(cfg) == base


def test_server_config_fingerprint_tracks_mcp_yaml_content(tmp_path, monkeypatch):
    from EvoScientist.config import settings as settings_mod

    monkeypatch.setattr(settings_mod, "get_config_dir", lambda: tmp_path)
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    (tmp_path / "mcp.yaml").write_text("server-a:\n  transport: stdio\n")
    changed = manager._server_config_fingerprint(cfg)
    assert changed != base
    (tmp_path / "mcp.yaml").write_text("server-b:\n  transport: http\n")
    assert manager._server_config_fingerprint(cfg) != changed


def test_server_config_fingerprint_tracks_subagent_yaml_content(tmp_path, monkeypatch):
    fake_dir = tmp_path / "subagents"
    fake_dir.mkdir()
    (fake_dir / "research.yaml").write_text("name: research\nprompt: v1\n")
    monkeypatch.setattr(manager, "_SUBAGENTS_DIR", fake_dir)
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    (fake_dir / "research.yaml").write_text("name: research\nprompt: v2\n")
    assert manager._server_config_fingerprint(cfg) != base


def test_server_config_fingerprint_reads_flagged_field_from_file():
    """A lasting server does not get this session's --dangerous: the session
    sees no drift for it, and a change in config.yaml still is one."""
    from EvoScientist.config import settings

    plain = manager.EvoScientistConfig()
    flagged = manager.EvoScientistConfig(dangerous_mode=True)
    assert manager._server_config_fingerprint(flagged) != (
        manager._server_config_fingerprint(plain)
    )

    settings.record_env_origin("EVOSCIENTIST_DANGEROUS_MODE", "cli")
    assert manager._server_config_fingerprint(flagged) == (
        manager._server_config_fingerprint(plain)
    )


@pytest.mark.parametrize("editable", [True, False])
def test_server_config_fingerprint_tracks_source_changes(
    tmp_path, monkeypatch, editable
):
    """A source checkout changes its code without a version bump: an edited
    file and a deleted one both count. Two installs of one release do not."""
    import os

    from EvoScientist import ccproxy_manager

    monkeypatch.setattr(ccproxy_manager, "_is_editable_install", lambda: editable)
    package = tmp_path / "pkg"
    package.mkdir()
    old, edited = package / "old.py", package / "edited.py"
    for module in (old, edited):
        module.write_text("x = 1\n", encoding="utf-8")
    os.utime(old, ns=(1_000_000_000, 1_000_000_000))
    monkeypatch.setattr(manager, "_PACKAGE_DIR", package)
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)

    later = edited.stat().st_mtime_ns + 1_000_000_000
    os.utime(edited, ns=(later, later))
    after_edit = manager._server_config_fingerprint(cfg)
    old.unlink()

    seen = {base, after_edit, manager._server_config_fingerprint(cfg)}
    assert len(seen) == (3 if editable else 1)


def test_server_config_fingerprint_keeps_flagged_auth_mode():
    """A lasting server takes its starter's --auth-mode: the fingerprint
    hashes that one, so a later session with another one sees the drift."""
    from EvoScientist.config import settings

    plain = manager.EvoScientistConfig()
    flagged = manager.EvoScientistConfig(openai_auth_mode="oauth")
    settings.record_env_origin("EVOSCIENTIST_OPENAI_AUTH_MODE", "cli")
    assert manager._server_config_fingerprint(flagged) != (
        manager._server_config_fingerprint(plain)
    )
