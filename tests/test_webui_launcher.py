"""Tests for ``run_webui`` bind-host wiring.

The front-end is an external npm package (``@evoscientist/webui``) with no
``--host`` flag: its bin launcher does
``HOSTNAME: process.env.HOSTNAME || "127.0.0.1"`` and hands that to the Next
standalone server. Setting ``HOSTNAME`` on the npx env is therefore the *only*
supported way to widen the front-end's interface — these tests pin that
contract so a refactor can't quietly drop it and silently re-narrow the bind.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from EvoScientist.deploy import webui as webui_mod


def _make_config(
    *,
    default_workdir: str = "",
    langgraph_dev_port: int = 6174,
    langgraph_dev_host: str = "127.0.0.1",
    webui_port: int = 4716,
    webui_host: str = "127.0.0.1",
):
    return SimpleNamespace(
        default_workdir=default_workdir,
        langgraph_dev_port=langgraph_dev_port,
        langgraph_dev_host=langgraph_dev_host,
        webui_port=webui_port,
        webui_host=webui_host,
        langgraph_dev_jobs_per_worker=10,
        langgraph_dev_file_persistence=True,
    )


class _RecordingConsole:
    """A real Rich console rendering to a buffer.

    Rendering for real (rather than stringifying the arguments) matters here:
    the remote-backend hint lives *inside* a ``Panel``, so a naive ``str(arg)``
    would only ever see ``<rich.panel.Panel object at ...>`` and the assertion
    would pass or fail for the wrong reason. Width is pinned wide so the
    strings under test don't wrap mid-token.
    """

    def __init__(self, sink: list):
        import io

        from rich.console import Console

        self._sink = sink
        self._buf = io.StringIO()
        self._console = Console(file=self._buf, width=200, no_color=True)

    def print(self, *args, **kwargs):
        self._buf.seek(0)
        self._buf.truncate()
        self._console.print(*args, **kwargs)
        self._sink.append(self._buf.getvalue())

    def status(self, *args, **kwargs):
        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

        return _Ctx()


class _ImmediateEvent:
    """Exits ``run_webui``'s block loop after a single iteration."""

    def __init__(self):
        self._called = 0

    def is_set(self) -> bool:
        self._called += 1
        return self._called > 1

    def wait(self, timeout: float | None = None):
        return None

    def set(self):
        self._called = 99


def _run_webui_once(monkeypatch, config, *, backend_port_occupied: bool = False):
    """Run ``run_webui`` with every external dependency mocked."""
    import atexit
    import os
    import shutil
    import signal
    import threading

    import EvoScientist.config as config_mod
    from EvoScientist.langgraph_dev import manager as lgm

    captured: dict[str, Any] = {"printed": [], "npx_env": {}, "npx_args": []}

    monkeypatch.setattr(config_mod, "apply_config_to_env", lambda _cfg: None)
    monkeypatch.setattr(webui_mod, "console", _RecordingConsole(captured["printed"]))
    monkeypatch.setattr(os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/npx")

    monkeypatch.setattr(
        lgm, "_is_port_occupied", lambda _p, *_a, **_kw: backend_port_occupied
    )
    monkeypatch.setattr(lgm, "is_langgraph_dev_running", lambda **_kw: False)
    monkeypatch.setattr(lgm, "_read_workspace_sidecar", lambda: None)

    def _fake_start_langgraph_dev(workspace_dir=None, *, port=None, host=None, **_kw):
        captured["backend_port"] = port
        captured["backend_host"] = host
        return SimpleNamespace(pid=99999)

    monkeypatch.setattr(lgm, "start_langgraph_dev", _fake_start_langgraph_dev)
    monkeypatch.setattr(lgm, "stop_langgraph_dev", lambda *_a, **_kw: None)

    class _FakeProc:
        pid = 12345

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

        def terminate(self):
            return None

    def _fake_popen(args, **kwargs):
        captured["npx_args"] = args
        captured["npx_env"] = kwargs.get("env", {})
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    # _stop_webui shells out to taskkill on Windows — neutralize it.
    monkeypatch.setattr(webui_mod, "_stop_webui", lambda _proc: None)
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: fn)
    monkeypatch.setattr(signal, "signal", lambda _sig, _handler: lambda *a: None)
    monkeypatch.setattr(threading, "Event", _ImmediateEvent)

    webui_mod.run_webui(config, workspace_dir="/tmp/ws")
    return captured


# =============================================================================
# Front-end bind interface (HOSTNAME)
# =============================================================================


def test_hostname_env_carries_webui_host(monkeypatch):
    config = _make_config(webui_host="0.0.0.0")
    captured = _run_webui_once(monkeypatch, config)

    assert captured["npx_env"].get("HOSTNAME") == "0.0.0.0", (
        "HOSTNAME is the package's only bind knob — without it the front-end "
        "falls back to its own 127.0.0.1 default"
    )


def test_hostname_env_defaults_to_loopback(monkeypatch):
    """The front-end serves the workspace file/upload and skill-install
    endpoints, so it stays off the network until ``webui_host`` opts in."""
    config = _make_config()
    captured = _run_webui_once(monkeypatch, config)

    assert captured["npx_env"].get("HOSTNAME") == "127.0.0.1"


def test_port_env_and_flag_still_set(monkeypatch):
    config = _make_config(webui_port=4800)
    captured = _run_webui_once(monkeypatch, config)

    assert captured["npx_env"].get("PORT") == "4800"
    assert "--port" in captured["npx_args"]
    assert captured["npx_args"][captured["npx_args"].index("--port") + 1] == "4800"


def test_no_host_flag_passed_to_npx(monkeypatch):
    """The package's arg parser only knows ``--port``; a stray ``--host`` is at
    best ignored and at worst breaks startup, so we must not emit one."""
    config = _make_config()
    captured = _run_webui_once(monkeypatch, config)

    assert "--host" not in captured["npx_args"]


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_webui_host_falls_back_to_loopback(monkeypatch, blank):
    config = _make_config(webui_host=blank)
    captured = _run_webui_once(monkeypatch, config)

    assert captured["npx_env"].get("HOSTNAME") == "127.0.0.1"


# =============================================================================
# Backend bind interface + security warning
# =============================================================================


def test_backend_host_reaches_start_langgraph_dev(monkeypatch):
    config = _make_config(langgraph_dev_host="0.0.0.0")
    captured = _run_webui_once(monkeypatch, config)

    assert captured["backend_host"] == "0.0.0.0"


def test_backend_defaults_to_loopback(monkeypatch):
    """The backend is an unauthenticated API whose agent can run shell, so it
    stays off the network unless ``langgraph_dev_host`` opts in."""
    config = _make_config()
    captured = _run_webui_once(monkeypatch, config)

    assert captured["backend_host"] == "127.0.0.1"


def test_public_bind_warning_when_backend_exposed(monkeypatch):
    """Users who widen the backend get told every time it is reachable
    off-box — an unauthenticated, shell-capable API deserves a standing
    reminder, not a one-time opt-in prompt."""
    config = _make_config(langgraph_dev_host="0.0.0.0")
    captured = _run_webui_once(monkeypatch, config)

    assert any("PUBLIC BIND" in line for line in captured["printed"])


def test_no_public_bind_warning_when_backend_on_loopback(monkeypatch):
    """The warning must be silenceable, or it degrades into background noise
    that users learn to skip past."""
    config = _make_config(langgraph_dev_host="127.0.0.1")
    captured = _run_webui_once(monkeypatch, config)

    assert not any("PUBLIC BIND" in line for line in captured["printed"])


def test_public_bind_warning_when_frontend_exposed(monkeypatch):
    """The front-end earns its own banner: it is not a passive app shell — its
    API reads, writes and uploads workspace files and installs skills."""
    config = _make_config(webui_host="0.0.0.0", langgraph_dev_host="127.0.0.1")
    captured = _run_webui_once(monkeypatch, config)

    banner = "\n".join(captured["printed"])
    assert "WebUI listening on 0.0.0.0" in banner
    assert "Backend listening" not in banner


def test_remote_backend_hint_when_frontend_exposed_but_backend_is_not(monkeypatch):
    """The UI talks to the backend from the browser, so a remote visitor
    cannot reach a loopback backend — say so instead of letting every request
    fail silently."""
    config = _make_config(webui_host="0.0.0.0", langgraph_dev_host="127.0.0.1")
    captured = _run_webui_once(monkeypatch, config)

    banner = "\n".join(captured["printed"])
    assert "Remote visitors cannot reach" in banner
    assert "langgraph_dev_host" in banner


def test_no_remote_hint_when_both_exposed(monkeypatch):
    config = _make_config(webui_host="0.0.0.0", langgraph_dev_host="0.0.0.0")
    captured = _run_webui_once(monkeypatch, config)

    banner = "\n".join(captured["printed"])
    assert "Remote visitors cannot reach" not in banner
