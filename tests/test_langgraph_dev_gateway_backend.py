"""Tests for ``gateway_backend``-keyed deploy mode in ``ensure_langgraph_dev``.

With ``gateway_backend = "langgraph_server"`` the auto-started langgraph dev
must spawn in full deploy mode (MCP + async sub-agents loaded server-side),
and a cross-process reuse of a server recorded as stripped must be refused
(``DeployModeMismatchError``) instead of silently serving a degraded main
graph. The ``local`` default keeps the historical stripped spawn.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from EvoScientist.langgraph_dev import manager


def _prepare_reuse(monkeypatch, tmp_path, runtime_paths, *, deploy_mode):
    """Stage an externally-managed running server with a recorded sidecar."""
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(runtime_paths, workspace_sidecar=tmp_path / "ws.json"),
    )
    manager._write_workspace_sidecar(
        workspace_dir=tmp_path, pid=99999, deploy_mode=deploy_mode
    )

    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)
    monkeypatch.setattr(manager, "_PROCESS_DEPLOY_MODE", None)

    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    return cfg


class TestSpawnMode:
    """``gateway_backend`` decides the deploy_mode passed to the spawn."""

    def _capture_spawn(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: False)
        monkeypatch.setattr(manager, "_PROCESS", None)
        monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)
        monkeypatch.setattr(manager, "_PROCESS_DEPLOY_MODE", None)
        # The fake spawn returns a bare object(); a real atexit registration
        # would blow up at interpreter exit calling stop_langgraph_dev on it.
        monkeypatch.setattr(manager.atexit, "register", lambda *a, **kw: None)

        def _fake_start(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(manager, "start_langgraph_dev", _fake_start)
        return captured

    def test_langgraph_server_backend_spawns_full(self, monkeypatch):
        captured = self._capture_spawn(monkeypatch)

        cfg = manager.EvoScientistConfig()
        cfg.enable_async_subagents = True
        cfg.gateway_backend = "langgraph_server"
        manager.ensure_langgraph_dev(cfg, workspace_dir=None)

        assert captured.get("deploy_mode") is True

    def test_local_backend_spawns_stripped(self, monkeypatch):
        captured = self._capture_spawn(monkeypatch)

        cfg = manager.EvoScientistConfig()
        cfg.enable_async_subagents = True
        manager.ensure_langgraph_dev(cfg, workspace_dir=None)

        assert captured.get("deploy_mode") is False


class TestReuseRefusal:
    """A full-mode session must not reuse a stripped-mode server."""

    def test_full_mode_refuses_stripped_leftover(
        self, tmp_path, monkeypatch, runtime_paths
    ):
        cfg = _prepare_reuse(monkeypatch, tmp_path, runtime_paths, deploy_mode=False)
        cfg.gateway_backend = "langgraph_server"

        with pytest.raises(manager.DeployModeMismatchError) as exc:
            manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path)
        assert "stripped mode" in str(exc.value)
        assert "EvoSci server stop" in str(exc.value)

    def test_deploy_mode_mismatch_is_workspace_mismatch_subclass(self):
        """Every existing print-and-exit handler catches the parent class."""
        assert issubclass(
            manager.DeployModeMismatchError, manager.WorkspaceMismatchError
        )

    def test_full_mode_reuses_full_server(self, tmp_path, monkeypatch, runtime_paths):
        cfg = _prepare_reuse(monkeypatch, tmp_path, runtime_paths, deploy_mode=True)
        cfg.gateway_backend = "langgraph_server"

        # Must not raise.
        manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path)

    def test_full_mode_reuses_legacy_server_without_deploy_mode_record(
        self, tmp_path, monkeypatch, runtime_paths, caplog
    ):
        """Sidecars predating the deploy_mode record have unknown mode —
        warn and reuse rather than bricking pre-existing servers."""
        sidecar_path = tmp_path / "ws.json"
        sidecar_path.write_text(json.dumps({"workspace": str(tmp_path), "pid": 99999}))
        monkeypatch.setattr(
            manager,
            "RUNTIME",
            dataclasses.replace(runtime_paths, workspace_sidecar=sidecar_path),
        )
        monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
        monkeypatch.setattr(manager, "_PROCESS", None)
        monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)
        monkeypatch.setattr(manager, "_PROCESS_DEPLOY_MODE", None)

        cfg = manager.EvoScientistConfig()
        cfg.enable_async_subagents = True
        cfg.gateway_backend = "langgraph_server"

        # Must not raise.
        manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path)

    def test_local_session_still_reuses_stripped_server(
        self, tmp_path, monkeypatch, runtime_paths
    ):
        """The local default keeps today's behavior: reuse is fine."""
        cfg = _prepare_reuse(monkeypatch, tmp_path, runtime_paths, deploy_mode=False)

        # Must not raise.
        manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path)


class TestOwnedRestart:
    """A stripped subprocess we own is restarted, not reused, for full mode."""

    def test_owned_stripped_process_restarts_full(self, monkeypatch, tmp_path):
        class _AliveProc:
            def poll(self):
                return None

        monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: False)
        monkeypatch.setattr(manager, "_PROCESS", _AliveProc())
        monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", tmp_path)
        monkeypatch.setattr(manager, "_PROCESS_DEPLOY_MODE", False)

        calls = {"stop": 0, "deploy_mode": None}

        def _fake_stop(proc=None):
            calls["stop"] += 1

        def _fake_start(**kwargs):
            calls["deploy_mode"] = kwargs.get("deploy_mode")
            return object()

        monkeypatch.setattr(manager, "stop_langgraph_dev", _fake_stop)
        monkeypatch.setattr(manager, "start_langgraph_dev", _fake_start)
        monkeypatch.setattr(manager, "_wait_for_port_release", lambda *_a, **_kw: None)
        monkeypatch.setattr(manager.atexit, "register", lambda *a, **kw: None)

        cfg = manager.EvoScientistConfig()
        cfg.enable_async_subagents = True
        cfg.gateway_backend = "langgraph_server"
        manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path)

        assert calls["stop"] == 1
        assert calls["deploy_mode"] is True

    def test_owned_stripped_process_kept_for_local_session(self, monkeypatch, tmp_path):
        """local backend never mode-restarts — reuse stays the behavior."""
        manager_runs = []

        class _AliveProc:
            def poll(self):
                return None

        monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
        monkeypatch.setattr(manager, "_PROCESS", _AliveProc())
        monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", tmp_path)
        monkeypatch.setattr(manager, "_PROCESS_DEPLOY_MODE", False)
        monkeypatch.setattr(
            manager, "stop_langgraph_dev", lambda proc=None: manager_runs.append("stop")
        )

        cfg = manager.EvoScientistConfig()
        cfg.enable_async_subagents = True
        manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path)

        assert manager_runs == []
