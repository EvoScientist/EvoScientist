"""Stand-ins for the langgraph dev server's start: its lock and its spawn."""

from __future__ import annotations

import dataclasses
import subprocess
import threading
from pathlib import Path

from EvoScientist.langgraph_dev import manager


class LockHolder:
    """Another process's start: holds the start lock until released."""

    def __init__(self) -> None:
        from filelock import FileLock

        manager.RUNTIME.pid_dir.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(manager.RUNTIME.lock_file))
        self.held = threading.Event()
        self.release = threading.Event()
        self.thread = threading.Thread(target=self._hold, daemon=True)
        self.thread.start()
        assert self.held.wait(5)

    def _hold(self) -> None:
        with self.lock:
            self.held.set()
            self.release.wait(10)

    def done(self) -> None:
        self.release.set()
        self.thread.join(5)


class PopenAbort(Exception):
    """Raised by the fake ``Popen`` to short-circuit ``start_langgraph_dev``
    after the env dict is constructed but before health-polling runs."""


def patch_start_prereqs(monkeypatch, tmp_path: Path, runtime_paths) -> dict:
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
        raise PopenAbort("env captured")

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    return captured


def start_lock_is_held() -> bool:
    from filelock import FileLock, Timeout

    probe = FileLock(str(manager.RUNTIME.lock_file))
    try:
        probe.acquire(timeout=0)
    except Timeout:
        return True
    probe.release()
    return False


def break_start_lock(monkeypatch, tmp_path, step: str) -> None:
    """Make the server start lock fail with an OSError, as a read-only or
    full config dir does: at the mkdir of its folder, or at the acquire."""
    import filelock

    if step == "mkdir":
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")
        monkeypatch.setattr(
            manager,
            "RUNTIME",
            manager.LanggraphRuntimePaths.for_directory(blocker / "rt"),
        )
    else:

        def _denied(self, *_a, **_k):
            raise PermissionError(13, "Permission denied", str(self.lock_file))

        monkeypatch.setattr(filelock.FileLock, "acquire", _denied)
