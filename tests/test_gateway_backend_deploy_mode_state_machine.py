"""State-machine tests for gateway_backend-keyed deploy mode.

NOT an end-to-end test: every OS boundary (the ``langgraph dev`` subprocess,
the ``/ok`` health probe, psutil process-kill, port binding) is faked. What
this pins is the manager's cross-process DECISION logic and the on-disk
sidecar contract — not that a real server boots or serves. Booting a real
langgraph dev and driving a live run is left to a manual, out-of-CI probe.

Simulates, within one process, the cross-process sequence the manual runs
exercised — a stripped-mode session leaving a keepalive server behind, then
a ``gateway_backend = "langgraph_server"`` session meeting that leftover:

1. ``ensure_langgraph_dev`` (local backend) spawns a langgraph dev and the
   workspace sidecar records ``deploy_mode: false``;
2. the owning session exits — with keepalive the server survives, so a new
   process finds the port occupied and a sidecar pointing at the old server;
3. ``ensure_langgraph_dev`` (langgraph_server backend) must REFUSE the reuse
   (``DeployModeMismatchError``), never silently serve a degraded graph;
4. after ``stop_recorded_server`` clears the leftover, the same session
   spawns cleanly in FULL deploy mode.

Process boundaries are faked where the OS would sit between them: spawning
goes through a fake Popen (real env-var construction is asserted by the
deploy-mode test module), while the sidecar file, PID file, and port-health
check run against the real manager logic. What this pins is the state
machine that spans those boundaries.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from EvoScientist.langgraph_dev import manager


class _FakeLanggraphProcess:
    """Popen stand-in for a live langgraph dev: alive until stopped."""

    def __init__(self) -> None:
        self.pid = 424_242
        self._alive = True

    def poll(self) -> int | None:
        return None if self._alive else 0

    def wait(self, timeout=None) -> int:
        self._alive = False
        return 0

    def terminate(self) -> None:
        self._alive = False


class _FakePsutilProcess:
    """psutil.Process stand-in — the OS process-kill boundary.

    ``stop_langgraph_dev`` reaps the server through ``psutil.Process(pid)``,
    not the Popen handle, so a fake pid would raise ``NoSuchProcess`` and the
    fake server would never actually stop. Terminating this fake flips the
    matching ``_FakeLanggraphProcess`` dead, mirroring the real tree-kill.
    """

    def __init__(self, pid: int, registry: _ProcRegistry) -> None:
        self._pid = pid
        self._registry = registry

    def children(self, recursive: bool = False) -> list:
        return []

    def terminate(self) -> None:
        for proc in self._registry.processes:
            if proc.pid == self._pid and proc.poll() is None:
                proc.terminate()

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout=None) -> int:
        return 0

    def is_running(self) -> bool:
        return any(
            p.pid == self._pid and p.poll() is None for p in self._registry.processes
        )

    def cmdline(self) -> list[str]:
        return ["langgraph", "dev"]


class _ProcRegistry:
    """Cross-'process' state: spawned servers + the fake OS hooks."""

    def __init__(self) -> None:
        self.spawn_envs: list[dict[str, str]] = []
        self.processes: list[_FakeLanggraphProcess] = []

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(manager, "_langgraph_exe", lambda: "/usr/bin/langgraph")
        fake_config = manager.Path(manager.RUNTIME.pid_dir) / "langgraph.json"
        fake_config.parent.mkdir(parents=True, exist_ok=True)
        fake_config.write_text("{}")
        monkeypatch.setattr(manager, "_packaged_langgraph_config", lambda: fake_config)
        monkeypatch.setattr(manager, "_is_port_occupied", lambda *_a, **_kw: False)
        monkeypatch.setattr(manager, "_wait_for_port_bindable", lambda *_a, **_kw: True)
        monkeypatch.setattr(manager, "_kill_owned_stale_process", lambda *_a: False)
        monkeypatch.setattr(manager, "_wait_for_port_release", lambda *_a, **_kw: None)
        registry = self

        # ``is_langgraph_dev_running`` serves two seams: the pre-spawn
        # "already running?" gate and the post-spawn health poll inside
        # ``start_langgraph_dev``. A per-session constant breaks the health
        # poll (it never turns True after Popen, so the spawn burns its 60s
        # budget and fails). Model the real thing instead — server liveness —
        # so both seams read the true state: False before any spawn, True
        # while a fake server is alive, False again once it is stopped.
        monkeypatch.setattr(
            manager,
            "is_langgraph_dev_running",
            lambda *_a, **_kw: any(p.poll() is None for p in registry.processes),
        )
        # ``stop_langgraph_dev`` reaps via psutil, not the Popen handle.
        monkeypatch.setattr(
            manager.psutil, "Process", lambda pid: _FakePsutilProcess(pid, registry)
        )

        def _fake_popen(args, **kwargs):
            proc = _FakeLanggraphProcess()
            registry.spawn_envs.append(dict(kwargs.get("env") or {}))
            registry.processes.append(proc)
            return proc

        import subprocess as _subprocess

        monkeypatch.setattr(_subprocess, "Popen", _fake_popen)


def _spawned_deploy_mode(registry: _ProcRegistry) -> list[str | None]:
    """Deploy-mode env value of every spawned server, in spawn order."""
    return [env.get("EVOSCIENTIST_DEPLOY_MODE") for env in registry.spawn_envs]


def _reset_process_state(monkeypatch) -> None:
    """Simulate a NEW process: fresh manager globals, no atexit handlers."""
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)
    monkeypatch.setattr(manager, "_PROCESS_DEPLOY_MODE", None)
    monkeypatch.setattr(manager.atexit, "register", lambda *a, **kw: None)


def _local_backend_cfg(**kw):
    cfg = manager.EvoScientistConfig(**kw)
    cfg.enable_async_subagents = True
    cfg.langgraph_dev_keepalive = True
    return cfg


def _server_backend_cfg(**kw):
    cfg = _local_backend_cfg(**kw)
    cfg.gateway_backend = "langgraph_server"
    return cfg


class TestGatewayBackendLifecycleE2E:
    async def test_stripped_leftover_is_refused_then_fresh_full_spawn(
        self, tmp_path, monkeypatch, runtime_paths
    ):
        """The full manual-e2e sequence, simulated in-process."""
        registry = _ProcRegistry()
        registry.install(monkeypatch)
        monkeypatch.setattr(
            manager,
            "RUNTIME",
            dataclasses.replace(runtime_paths, log_file=tmp_path / "langgraph_dev.log"),
        )
        workspace = tmp_path / "ws"

        # --- Session A: local backend spawns a keepalive stripped server ---
        _reset_process_state(monkeypatch)
        manager.ensure_langgraph_dev(_local_backend_cfg(), workspace_dir=workspace)
        assert _spawned_deploy_mode(registry) == ["stripped"]

        # Keepalive leftovers: sidecar written, server still alive on disk.
        sidecar = json.loads(manager.RUNTIME.workspace_sidecar.read_text())
        assert sidecar["deploy_mode"] is False
        assert manager.RUNTIME.pid_file.exists()

        # --- Session B (new process): full-mode caller meets the leftover ---
        # The stateful liveness fake already reports the Session-A server as
        # alive, so the full-mode caller sees the leftover and must refuse it.
        _reset_process_state(monkeypatch)
        with pytest.raises(manager.DeployModeMismatchError) as exc:
            manager.ensure_langgraph_dev(_server_backend_cfg(), workspace_dir=workspace)
        assert "stripped mode" in str(exc.value)
        assert "EvoSci server stop" in str(exc.value)
        # Refusal must NOT kill the server or spawn a replacement.
        assert len(registry.processes) == 1
        assert registry.processes[0].poll() is None

        # --- Cleanup path: `EvoSci server stop` clears the leftover ---
        # stop_recorded_server must see our fake process as langgraph's.
        monkeypatch.setattr(manager, "_PROCESS", registry.processes[0])
        monkeypatch.setattr(manager, "_PROCESS_DEPLOY_MODE", False)
        monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", workspace)
        stopped = manager.stop_recorded_server()
        assert stopped == registry.processes[0].pid
        assert not manager.RUNTIME.pid_file.exists()
        assert not manager.RUNTIME.workspace_sidecar.exists()
        assert registry.processes[0].poll() is not None

        # --- Session C: with the leftover gone, full-mode spawns cleanly ---
        _reset_process_state(monkeypatch)
        manager.ensure_langgraph_dev(_server_backend_cfg(), workspace_dir=workspace)
        assert _spawned_deploy_mode(registry) == ["stripped", "full"]
        sidecar = json.loads(manager.RUNTIME.workspace_sidecar.read_text())
        assert sidecar["deploy_mode"] is True

    async def test_full_mode_session_reuses_full_leftover(
        self, tmp_path, monkeypatch, runtime_paths
    ):
        """Two consecutive full-mode sessions share one server — the normal
        keepalive path must not refuse its own leftovers."""
        registry = _ProcRegistry()
        registry.install(monkeypatch)
        monkeypatch.setattr(
            manager,
            "RUNTIME",
            dataclasses.replace(runtime_paths, log_file=tmp_path / "langgraph_dev.log"),
        )
        workspace = tmp_path / "ws"

        _reset_process_state(monkeypatch)
        manager.ensure_langgraph_dev(_server_backend_cfg(), workspace_dir=workspace)
        assert _spawned_deploy_mode(registry) == ["full"]

        _reset_process_state(monkeypatch)
        # The stateful liveness fake reports the first server as alive; the
        # second full-mode session must reuse it, not refuse or respawn.
        # Must not raise; no second spawn.
        manager.ensure_langgraph_dev(_server_backend_cfg(), workspace_dir=workspace)
        assert len(registry.processes) == 1
