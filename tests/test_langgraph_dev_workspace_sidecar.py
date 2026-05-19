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

import json
from pathlib import Path

import pytest

from EvoScientist.langgraph_dev import manager


def test_sidecar_path_is_next_to_pid_file():
    """Sidecar JSON lives at _PID_DIR / 'langgraph_dev.workspace.json'."""
    assert (
        manager._WORKSPACE_SIDECAR == manager._PID_DIR / "langgraph_dev.workspace.json"
    )


def test_write_workspace_sidecar_records_workspace_and_pid(tmp_path, monkeypatch):
    """``_write_workspace_sidecar`` writes JSON with workspace + pid."""
    monkeypatch.setattr(manager, "_WORKSPACE_SIDECAR", tmp_path / "ws.json")
    manager._write_workspace_sidecar(workspace_dir=Path("/some/ws"), pid=12345)
    data = json.loads((tmp_path / "ws.json").read_text())
    assert data["workspace"] == "/some/ws"
    assert data["pid"] == 12345


def test_read_workspace_sidecar_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "_WORKSPACE_SIDECAR", tmp_path / "absent.json")
    assert manager._read_workspace_sidecar() is None


def test_read_workspace_sidecar_returns_none_on_corrupt_json(tmp_path, monkeypatch):
    sidecar = tmp_path / "bad.json"
    sidecar.write_text("not json at all")
    monkeypatch.setattr(manager, "_WORKSPACE_SIDECAR", sidecar)
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
    payload, tmp_path, monkeypatch
):
    """JSON that parses but doesn't match the expected schema must degrade
    to None — otherwise the reuse branch's ``Path(sidecar["workspace"]).resolve()``
    would raise KeyError/TypeError or silently resolve to cwd, surfacing as
    an unhandled exception or producing a misleading match check."""
    sidecar = tmp_path / "schema.json"
    sidecar.write_text(payload)
    monkeypatch.setattr(manager, "_WORKSPACE_SIDECAR", sidecar)
    assert manager._read_workspace_sidecar() is None


def test_read_workspace_sidecar_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "_WORKSPACE_SIDECAR", tmp_path / "rt.json")
    manager._write_workspace_sidecar(workspace_dir=Path("/x/y"), pid=42)
    data = manager._read_workspace_sidecar()
    assert data == {"workspace": "/x/y", "pid": 42}


def test_workspace_mismatch_error_is_runtime_error_subclass():
    assert issubclass(manager.WorkspaceMismatchError, RuntimeError)


def test_ensure_langgraph_dev_refuses_on_workspace_mismatch(tmp_path, monkeypatch):
    """Cross-process reuse with sidecar workspace ≠ requested → raises."""
    monkeypatch.setattr(manager, "_WORKSPACE_SIDECAR", tmp_path / "ws.json")
    manager._write_workspace_sidecar(workspace_dir=Path("/workspace/A"), pid=99999)

    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    with pytest.raises(manager.WorkspaceMismatchError) as exc:
        manager.ensure_langgraph_dev(cfg, workspace_dir=Path("/workspace/B"))
    assert "/workspace/A" in str(exc.value)
    assert "/workspace/B" in str(exc.value)


def test_ensure_langgraph_dev_reuses_when_workspace_matches(tmp_path, monkeypatch):
    """Cross-process reuse with matching sidecar workspace → no raise."""
    monkeypatch.setattr(manager, "_WORKSPACE_SIDECAR", tmp_path / "ws.json")
    manager._write_workspace_sidecar(workspace_dir=Path("/workspace/A"), pid=99999)

    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    # Should NOT raise.
    manager.ensure_langgraph_dev(cfg, workspace_dir=Path("/workspace/A"))


def test_ensure_langgraph_dev_reuses_when_sidecar_missing(tmp_path, monkeypatch):
    """Backward compat: pre-feature langgraph dev (no sidecar) falls back to
    the existing log-warning behavior rather than refusing."""
    monkeypatch.setattr(manager, "_WORKSPACE_SIDECAR", tmp_path / "absent.json")
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    # Should NOT raise — degrades to the prior reuse-with-warning branch.
    manager.ensure_langgraph_dev(cfg, workspace_dir=Path("/workspace/B"))


def test_stop_langgraph_dev_removes_sidecar(tmp_path, monkeypatch):
    """``stop_langgraph_dev`` should unlink the sidecar alongside the PID file."""
    sidecar = tmp_path / "ws.json"
    pid_file = tmp_path / "pid.txt"
    monkeypatch.setattr(manager, "_WORKSPACE_SIDECAR", sidecar)
    monkeypatch.setattr(manager, "_PID_FILE", pid_file)
    manager._write_workspace_sidecar(workspace_dir=Path("/x"), pid=42)
    assert sidecar.exists()

    # _PROCESS is None so stop_langgraph_dev shouldn't try to kill anything;
    # we're only verifying the sidecar cleanup path here.
    monkeypatch.setattr(manager, "_PROCESS", None)
    manager.stop_langgraph_dev()
    assert not sidecar.exists()
