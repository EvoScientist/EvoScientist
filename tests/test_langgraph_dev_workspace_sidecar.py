"""Tests for the workspace-fingerprint sidecar protocol.

When langgraph dev is reused across processes (e.g., TUI / serve detects
a deploy-started instance on the configured port), the workspace recorded
in the sidecar JSON must match the workspace requested by the caller. On
mismatch we raise ``WorkspaceMismatchError`` so callers can surface a
clear refuse-with-hint error rather than silently operating on the wrong
project's files.

Background: ``EvoSci deploy --workdir /A`` running + ``EvoSci`` (TUI) in
/B previously took the "reuse externally-managed langgraph dev" branch
in ``ensure_langgraph_dev`` and only logged a warning. The deployed
sub-agents stayed pinned to /A while the TUI's main agent ran in /B,
breaking ``task()`` delegations.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from EvoScientist.langgraph_dev import manager


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
    """JSON that parses but doesn't match the expected schema must degrade
    to None — otherwise the reuse branch's ``Path(sidecar["workspace"]).resolve()``
    would raise KeyError/TypeError or silently resolve to cwd, surfacing as
    an unhandled exception or producing a misleading match check."""
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
    import os

    assert data == {
        "workspace": str(workspace),
        "pid": 42,
        "owner_pids": [os.getpid()],
    }


def test_workspace_mismatch_error_is_runtime_error_subclass():
    assert issubclass(manager.WorkspaceMismatchError, RuntimeError)


def test_ensure_langgraph_dev_refuses_on_workspace_mismatch(
    tmp_path, monkeypatch, runtime_paths
):
    """Cross-process reuse with sidecar workspace ≠ requested → raises."""
    ws_a = tmp_path / "A"
    ws_b = tmp_path / "B"
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    manager._write_workspace_sidecar(workspace_dir=ws_a, pid=99999)

    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    with pytest.raises(manager.WorkspaceMismatchError) as exc:
        manager.ensure_langgraph_dev(cfg, workspace_dir=ws_b)
    assert str(ws_a.resolve()) in str(exc.value)
    assert str(ws_b) in str(exc.value)


def test_ensure_langgraph_dev_refuses_on_mismatch_with_stale_process(
    tmp_path, monkeypatch, runtime_paths
):
    """A non-None but dead ``_PROCESS`` handle must NOT short-circuit the
    sidecar check. Regression for the case where our subprocess exited and a
    different langgraph dev rebound the port."""
    ws_a = tmp_path / "A"
    ws_b = tmp_path / "B"
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    manager._write_workspace_sidecar(workspace_dir=ws_a, pid=99999)

    class _DeadProc:
        def poll(self):
            return 1  # non-None → process has exited

    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", _DeadProc())
    # _PROCESS_WORKSPACE matches ws_b so the earlier owned-restart branch (which
    # also gates on _PROCESS.poll() is None) doesn't fire on this dead handle.
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", ws_b)

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    with pytest.raises(manager.WorkspaceMismatchError):
        manager.ensure_langgraph_dev(cfg, workspace_dir=ws_b)


def test_ensure_langgraph_dev_reuses_when_workspace_matches(
    tmp_path, monkeypatch, runtime_paths
):
    """Cross-process reuse with matching sidecar workspace → no raise."""
    ws_a = tmp_path / "A"
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    manager._write_workspace_sidecar(workspace_dir=ws_a, pid=99999)

    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    # Should NOT raise.
    manager.ensure_langgraph_dev(cfg, workspace_dir=ws_a)


def test_ensure_langgraph_dev_reuses_when_sidecar_missing(
    tmp_path, monkeypatch, runtime_paths
):
    """Backward compat: pre-feature langgraph dev (no sidecar) falls back to
    the existing log-warning behavior rather than refusing."""
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "absent.json"),
    )
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    # Should NOT raise — degrades to the prior reuse-with-warning branch.
    manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path / "B")


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


# ---------------------------------------------------------------------------
# Owner tracking + keepalive reclaim
# ---------------------------------------------------------------------------


def _dead_pid() -> int:
    """Spawn-and-reap a process so its pid is reliably dead."""
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_write_workspace_sidecar_records_current_owner_by_default(
    tmp_path, monkeypatch, runtime_paths
):
    import os

    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    manager._write_workspace_sidecar(workspace_dir=tmp_path / "ws", pid=12345)
    data = json.loads((tmp_path / "ws.json").read_text())
    assert data["owner_pids"] == [os.getpid()]


def test_sidecar_live_owners_none_for_pre_feature_sidecar():
    assert manager._sidecar_live_owners({"workspace": "/a", "pid": 1}) is None


def test_sidecar_live_owners_prunes_dead_and_keeps_alive():
    import os

    dead = _dead_pid()
    live = manager._sidecar_live_owners(
        {"workspace": "/a", "pid": 1, "owner_pids": [dead, os.getpid()]}
    )
    assert live == [os.getpid()]


def test_ensure_reclaims_leftover_when_all_owners_dead(
    tmp_path, monkeypatch, runtime_paths
):
    """Workspace mismatch + every owner CLI exited + ownership confirmed →
    the leftover is killed and a fresh server starts for the new workspace."""
    ws_a = tmp_path / "A"
    ws_b = tmp_path / "B"
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    manager._write_workspace_sidecar(
        workspace_dir=ws_a, pid=99999, owner_pids=[_dead_pid()]
    )

    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)
    killed = []
    monkeypatch.setattr(
        manager,
        "_kill_owned_leftover_server",
        lambda port, expected_pid=None: killed.append(port) or True,
    )
    monkeypatch.setattr(manager, "_wait_for_port_release", lambda port: True)
    started = []
    fake_proc = object()
    monkeypatch.setattr(
        manager,
        "start_langgraph_dev",
        lambda **kw: started.append(kw) or fake_proc,
    )
    registered = []
    monkeypatch.setattr(
        manager.atexit, "register", lambda *a, **kw: registered.append(a)
    )

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    result = manager.ensure_langgraph_dev(cfg, workspace_dir=ws_b)

    assert killed, "leftover was not reclaimed"
    assert started
    assert started[0]["workspace_dir"] == ws_b
    assert result is fake_proc
    assert registered, "non-keepalive start must register atexit cleanup"


def test_ensure_refuses_on_mismatch_with_live_owner(
    tmp_path, monkeypatch, runtime_paths
):
    """A live owner CLI (this test process) keeps the hard refusal."""
    import os

    ws_a = tmp_path / "A"
    ws_b = tmp_path / "B"
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    manager._write_workspace_sidecar(
        workspace_dir=ws_a, pid=99999, owner_pids=[os.getpid()]
    )
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)
    monkeypatch.setattr(
        manager,
        "_kill_owned_leftover_server",
        lambda *a, **kw: pytest.fail("must not kill a server with a live owner"),
    )

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    with pytest.raises(manager.WorkspaceMismatchError):
        manager.ensure_langgraph_dev(cfg, workspace_dir=ws_b)


def test_ensure_refuses_on_mismatch_with_pre_feature_sidecar(
    tmp_path, monkeypatch, runtime_paths
):
    """Sidecar without owner record → owners unknown → conservative refusal."""
    sidecar = tmp_path / "ws.json"
    sidecar.write_text(json.dumps({"workspace": str(tmp_path / "A"), "pid": 99999}))
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=sidecar),
    )
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)
    monkeypatch.setattr(
        manager,
        "_kill_owned_leftover_server",
        lambda *a, **kw: pytest.fail("must not kill when owners are unknown"),
    )

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    with pytest.raises(manager.WorkspaceMismatchError):
        manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path / "B")


def test_ensure_reuse_registers_owner_and_prunes_dead(
    tmp_path, monkeypatch, runtime_paths
):
    """Cross-process reuse adds this CLI to owner_pids and drops dead ones."""
    import os

    ws_a = tmp_path / "A"
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    manager._write_workspace_sidecar(
        workspace_dir=ws_a, pid=99999, owner_pids=[_dead_pid()]
    )
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    manager.ensure_langgraph_dev(cfg, workspace_dir=ws_a)

    data = json.loads((tmp_path / "ws.json").read_text())
    assert data["owner_pids"] == [os.getpid()]


def test_keepalive_skips_atexit_registration(tmp_path, monkeypatch, runtime_paths):
    """keepalive=True leaves the server running on exit; False registers stop."""
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: False)
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)
    fake_proc = object()
    monkeypatch.setattr(manager, "start_langgraph_dev", lambda **kw: fake_proc)
    registered = []
    monkeypatch.setattr(
        manager.atexit, "register", lambda *a, **kw: registered.append(a)
    )

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    cfg.langgraph_dev_keepalive = True
    assert manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path / "A") is fake_proc
    assert registered == []

    cfg.langgraph_dev_keepalive = False
    assert manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path / "A") is fake_proc
    assert registered
    assert registered[0][0] is manager.stop_langgraph_dev


def test_kill_owned_leftover_server_refuses_non_langgraph_pid(
    tmp_path, monkeypatch, runtime_paths
):
    """PID recycled to a foreign process → refuse to kill, return False."""
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
        assert manager._kill_owned_leftover_server(6174) is False
        assert victim.poll() is None, "foreign process must not be killed"
    finally:
        victim.kill()
        victim.wait()


def test_kill_owned_leftover_server_false_when_pid_dead(
    tmp_path, monkeypatch, runtime_paths
):
    pid_file = tmp_path / "pid.txt"
    pid_file.write_text(str(_dead_pid()))
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, pid_file=pid_file),
    )
    assert manager._kill_owned_leftover_server(6174) is False


def test_kill_owned_leftover_server_refuses_on_pid_file_sidecar_mismatch(
    tmp_path, monkeypatch, runtime_paths
):
    """PID file and sidecar recording different server pids → refuse (mixed
    process generations, e.g. an unlocked deploy start overwrote one file)."""
    import os

    pid_file = tmp_path / "pid.txt"
    pid_file.write_text(str(os.getpid()))
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, pid_file=pid_file),
    )
    assert (
        manager._kill_owned_leftover_server(6174, expected_pid=os.getpid() + 1) is False
    )


def test_kill_owned_leftover_server_refuses_on_port_mismatch(
    tmp_path, monkeypatch, runtime_paths
):
    """A langgraph process serving a DIFFERENT port must not be reclaimed."""
    import subprocess
    import sys

    victim = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "langgraph", "9999"]
    )
    try:
        pid_file = tmp_path / "pid.txt"
        pid_file.write_text(str(victim.pid))
        monkeypatch.setattr(
            manager,
            "RUNTIME",
            dataclasses.replace(runtime_paths, pid_file=pid_file),
        )
        assert manager._kill_owned_leftover_server(6174) is False
        assert victim.poll() is None, "different-port process must not be killed"
    finally:
        victim.kill()
        victim.wait()


@pytest.mark.parametrize(
    "owners",
    [
        ["123"],  # string entry
        [None],  # null entry
        [True],  # bool entry (bool is an int subclass — must not pass)
        [0],  # non-positive
        [12345, "junk"],  # one valid + one malformed
    ],
)
def test_sidecar_live_owners_malformed_entries_mean_unknown(owners):
    """Corrupt owner lists must read as 'ownership unknown' (None), never as
    'all owners dead' ([]) — an empty result authorizes reclaim."""
    assert (
        manager._sidecar_live_owners(
            {"workspace": "/a", "pid": 1, "owner_pids": owners}
        )
        is None
    )


def test_register_owner_preserves_unknown_ownership(
    tmp_path, monkeypatch, runtime_paths
):
    """Pre-feature sidecar (no owner record) must stay untouched: stamping this
    CLI as sole owner would later read as 'all owners dead' and authorize a
    reclaim while unknown pre-feature sessions may still use the server."""
    sidecar = tmp_path / "ws.json"
    sidecar.write_text(json.dumps({"workspace": str(tmp_path / "A"), "pid": 99999}))
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=sidecar),
    )
    manager._sidecar_register_owner()
    data = json.loads(sidecar.read_text())
    assert "owner_pids" not in data
