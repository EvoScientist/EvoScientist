"""Tests for ``run_webui`` bind-host wiring.

The front-end is the ``@evoscientist/webui`` Next.js standalone server, started
as ``node dist/server.js``. It takes no flags: it reads ``PORT`` and
``HOSTNAME`` from its environment, and without ``HOSTNAME`` it binds every
interface. Setting ``HOSTNAME`` on the node env is therefore the *only* way to
choose the front-end's interface — these tests pin that contract so a refactor
can't quietly drop it and silently widen or re-narrow the bind.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from EvoScientist.deploy import webui as webui_mod

NODE = "/opt/node/bin/node"
APP = "/data/tools/webui/0.3.1"


@pytest.fixture(autouse=True)
def _isolated_runtime(runtime_paths):
    """``webui.log`` lands next to the isolated ``langgraph_dev.log``."""


def _make_config(
    *,
    default_workdir: str = "",
    langgraph_dev_port: int = 6174,
    langgraph_dev_host: str = "127.0.0.1",
    webui_port: int = 4716,
    webui_host: str = "127.0.0.1",
    langgraph_dev_keepalive: bool = False,
):
    return SimpleNamespace(
        default_workdir=default_workdir,
        langgraph_dev_port=langgraph_dev_port,
        langgraph_dev_host=langgraph_dev_host,
        webui_port=webui_port,
        webui_host=webui_host,
        langgraph_dev_jobs_per_worker=10,
        langgraph_dev_file_persistence=True,
        langgraph_dev_keepalive=langgraph_dev_keepalive,
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

            def update(self_inner, *a, **k):
                self._sink.append(f"status: {a[0] if a else ''}")

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


class _InterruptingConsole(_RecordingConsole):
    """Raises ``KeyboardInterrupt`` when a line containing ``trigger`` prints,
    simulating a Ctrl+C that lands while ``run_webui`` renders its output."""

    def __init__(self, sink: list, trigger: str):
        super().__init__(sink)
        self._trigger = trigger

    def print(self, *args, **kwargs):
        super().print(*args, **kwargs)
        if self._trigger in self._sink[-1]:
            raise KeyboardInterrupt


def _run_webui_once(
    monkeypatch,
    config,
    *,
    backend_port_occupied: bool = False,
    interrupt_on: str | None = None,
):
    """Run ``run_webui`` with every external dependency mocked.

    ``interrupt_on``: raise ``KeyboardInterrupt`` from the console when a line
    containing this text prints; ``run_webui`` then propagates it."""
    import atexit
    import os
    import signal
    import threading

    import EvoScientist.config as config_mod
    from EvoScientist.langgraph_dev import manager as lgm

    captured: dict[str, Any] = {"printed": [], "node_env": {}, "node_args": []}

    monkeypatch.setattr(config_mod, "apply_config_to_env", lambda _cfg: None)
    console = (
        _InterruptingConsole(captured["printed"], interrupt_on)
        if interrupt_on
        else _RecordingConsole(captured["printed"])
    )
    monkeypatch.setattr(webui_mod, "console", console)
    monkeypatch.setattr(os, "makedirs", lambda *a, **k: None)
    # The installed front-end: Node and WebUI provisioning are stubbed; the
    # launcher's readiness wait is covered in test_launcher.py.
    from pathlib import Path

    from EvoScientist.deploy import launcher as launcher_mod
    from EvoScientist.setup import node as node_mod
    from EvoScientist.setup import webui as setup_webui

    monkeypatch.setattr(
        node_mod,
        "ensure_node",
        lambda **_kw: node_mod.NodeInfo("system", "22.0.0", Path(NODE)),
    )
    monkeypatch.setattr(node_mod, "activate_runtime", lambda: None)
    monkeypatch.setattr(
        setup_webui,
        "ensure_webui",
        lambda *_a, **_kw: setup_webui.WebUIInfo("0.3.1", Path(APP)),
    )
    monkeypatch.setattr(setup_webui, "mark_in_use", lambda _info: None)
    monkeypatch.setattr(setup_webui, "cleanup_old_versions", lambda: [])
    captured["update_checks"] = 0

    def _fake_update_check(_self):
        captured["update_checks"] += 1

    monkeypatch.setattr(
        launcher_mod.InstalledWebUIRunner, "start_update_check", _fake_update_check
    )
    monkeypatch.setattr(
        launcher_mod.BundledWebUIRunner, "preflight", lambda _self, _cfg: None
    )
    monkeypatch.setattr(
        launcher_mod.WebUILauncher, "wait_ready", lambda self, *a, **k: self._result()
    )

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

    captured["stop_calls"] = 0

    def _fake_stop(*_a, **_kw):
        captured["stop_calls"] += 1

    monkeypatch.setattr(lgm, "stop_langgraph_dev", _fake_stop)

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
        captured["node_args"] = args
        captured["node_env"] = kwargs.get("env", {})
        captured["node_stdout"] = kwargs.get("stdout")
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    # The front-end runner's stop shells out to taskkill on Windows — neutralize.
    monkeypatch.setattr(launcher_mod, "_stop_process_tree", lambda _proc: None)
    captured["atexit_fns"] = []
    monkeypatch.setattr(
        atexit, "register", lambda fn, *a, **k: captured["atexit_fns"].append(fn) or fn
    )
    monkeypatch.setattr(signal, "signal", lambda _sig, _handler: lambda *a: None)
    monkeypatch.setattr(threading, "Event", _ImmediateEvent)

    if interrupt_on:
        with pytest.raises(KeyboardInterrupt):
            webui_mod.run_webui(config, workspace_dir="/tmp/ws")
    else:
        webui_mod.run_webui(config, workspace_dir="/tmp/ws")
    return captured


# =============================================================================
# Front-end bind interface (HOSTNAME)
# =============================================================================


def test_hostname_env_carries_webui_host(monkeypatch):
    config = _make_config(webui_host="0.0.0.0")
    captured = _run_webui_once(monkeypatch, config)

    assert captured["node_env"].get("HOSTNAME") == "0.0.0.0", (
        "HOSTNAME is the package's only bind knob — without it the front-end "
        "falls back to its own 127.0.0.1 default"
    )


def test_hostname_env_defaults_to_loopback(monkeypatch):
    """The front-end serves the workspace file/upload and skill-install
    endpoints, so it stays off the network until ``webui_host`` opts in."""
    config = _make_config()
    captured = _run_webui_once(monkeypatch, config)

    assert captured["node_env"].get("HOSTNAME") == "127.0.0.1"


def test_port_env_set(monkeypatch):
    config = _make_config(webui_port=4800)
    captured = _run_webui_once(monkeypatch, config)

    assert captured["node_env"].get("PORT") == "4800"


def test_installed_server_runs_without_npx_or_flags(monkeypatch):
    """The installed copy runs as ``node dist/server.js``: no ``npx``, and no
    flags, since the standalone server reads only its environment."""
    captured = _run_webui_once(monkeypatch, _make_config())

    from pathlib import Path

    assert captured["node_args"] == [
        str(Path(NODE)),
        str(Path(APP) / "dist" / "server.js"),
    ]
    assert captured["node_env"].get("NODE_ENV") == "production"


def test_node_output_goes_to_webui_log_next_to_the_backend_log(
    monkeypatch, runtime_paths
):
    captured = _run_webui_once(monkeypatch, _make_config())
    log = runtime_paths.log_file.parent / "webui.log"
    assert captured["node_stdout"].name == str(log)
    assert any("webui.log" in line for line in captured["printed"])


def test_update_check_starts_once_the_ui_is_up(monkeypatch):
    captured = _run_webui_once(monkeypatch, _make_config())
    assert captured["update_checks"] == 1


def test_a_crashed_webui_points_at_webui_log(monkeypatch):
    import typer

    class _CrashingLauncher(webui_mod.WebUILauncher):
        def wait_ready(self, *a, **k):
            raise webui_mod.LauncherError(
                "webui_start_failed",
                "WebUI process exited before it became ready.",
                "[Errno 54] Connection reset by peer",
            )

    monkeypatch.setattr(webui_mod, "WebUILauncher", _CrashingLauncher)
    with pytest.raises(typer.Exit):
        _run_webui_once(monkeypatch, _make_config())
    panel = webui_mod.console._sink[-1]
    assert "Connection reset by peer" in panel
    assert "webui.log" in panel


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_webui_host_falls_back_to_loopback(monkeypatch, blank):
    config = _make_config(webui_host=blank)
    captured = _run_webui_once(monkeypatch, config)

    assert captured["node_env"].get("HOSTNAME") == "127.0.0.1"


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


# =============================================================================
# Backend keepalive
# =============================================================================


def test_backend_default_stops_backend_on_exit(monkeypatch):
    """Without keepalive the WebUI-started backend dies with the session.

    Teardown moved into ``launcher.stop`` (registered via atexit and also run in
    the finally block): the contract is now "stop_langgraph_dev is called", not
    "which callable was handed to atexit"."""
    captured = _run_webui_once(monkeypatch, _make_config())
    assert captured["stop_calls"] >= 1
    assert captured["atexit_fns"], "launcher.stop must be registered for teardown"


def test_backend_keepalive_leaves_backend_running(monkeypatch):
    """With keepalive the backend outlives the WebUI session, so the next
    same-workspace launch reuses it instead of paying the cold boot."""
    captured = _run_webui_once(monkeypatch, _make_config(langgraph_dev_keepalive=True))
    assert captured["stop_calls"] == 0


def test_teardown_registered_before_output_is_rendered(monkeypatch):
    """``start()`` has already spawned the backend and the front-end, so a
    Ctrl+C while the ready lines print must still find the teardown registered;
    otherwise both processes are left running."""
    captured = _run_webui_once(
        monkeypatch, _make_config(), interrupt_on="langgraph dev ready"
    )
    assert len(captured["atexit_fns"]) == 1
    assert captured["atexit_fns"][0].__name__ == "stop"


def test_missing_python_hint_is_printed_by_the_launcher(monkeypatch):
    """The agent is built inside the server, so the launching process shows
    the hint (conftest pins the agent's python as missing)."""
    from EvoScientist.setup import research_env

    captured = _run_webui_once(monkeypatch, _make_config())
    assert any(research_env.MISSING_PYTHON_HINT in line for line in captured["printed"])


def test_warnings_with_brackets_print_as_they_are(monkeypatch):
    """Warnings can carry paths; Rich markup must not eat or choke on them."""
    from EvoScientist import agent_shell

    warning = "gives its agents /home/u/[lab]/[/x]/bin/python"
    monkeypatch.setattr(agent_shell, "server_setup_hint", lambda _s: warning)
    captured = _run_webui_once(monkeypatch, _make_config())
    assert any(warning in line for line in captured["printed"])
