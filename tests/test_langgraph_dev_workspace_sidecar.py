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
    assert data == {"workspace": str(workspace), "pid": 42}


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
