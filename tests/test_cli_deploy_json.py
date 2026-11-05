"""Tests for ``EvoSci deploy --json`` (``EvoScientist.deploy.json_mode``).

The server's spawn, its record, the health check and the port probes are
faked; ``ensure_server`` runs for real on top of them.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
import typer

from EvoScientist import paths
from EvoScientist.deploy import json_mode
from EvoScientist.deploy import server as deploy_server
from EvoScientist.langgraph_dev import registration
from EvoScientist.langgraph_dev.assistants import workspace_assistant_id
from EvoScientist.paths import Workspace
from tests.fakes.langgraph_dev import LockHolder


def _config(**overrides):
    base = {
        "default_workdir": "",
        "langgraph_dev_port": 6174,
        "langgraph_dev_host": "127.0.0.1",
        "anthropic_auth_mode": "api_key",
        "openai_auth_mode": "api_key",
        "log_level": "warning",
        "langgraph_dev_jobs_per_worker": 10,
        "langgraph_dev_file_persistence": True,
        "dangerous_mode": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _Proc:
    def __init__(self) -> None:
        self.pid = 4242
        self.returncode: int | None = None

    def poll(self):
        return self.returncode


class Harness:
    """Fakes everything ``run_json`` touches outside this module and records
    what happened, in order, in :attr:`log`."""

    def __init__(self, monkeypatch, tmp_path) -> None:
        import EvoScientist.config as config_mod
        from EvoScientist import agent_shell
        from EvoScientist.deploy import backend_job
        from EvoScientist.langgraph_dev import manager as lgm

        self.mp = monkeypatch
        self.tmp_path = tmp_path
        self.workspace = str(tmp_path / "ws")
        self.config = _config(default_workdir=self.workspace)
        self.log: list[Any] = []
        self.clean_stops: list[bool] = []
        self.started: dict[str, Any] | None = None
        self.proc = _Proc()
        self.occupied = False
        # The port once this run's backend started: True, False or a callable.
        self.listening: Any = True
        self.running = True
        # This data folder's running server, as its record names it.
        self.record: Any = None
        # Whose server answers on the port without a record.
        self.whose = "own"
        self.tunnel_url: str | None = None
        # What a reused backend answers the registration.
        self.exposure = registration.ServerExposure()
        self.registered: list[Workspace] = []
        self.job: Any = None
        # "eof": a pipe closed right after ``ready``; "eof-now": already
        # closed; "devnull"; or a file object.
        self.stdin_kind = "eof"
        self._stdin_write: int | None = None
        self._stdin: Any = None

        monkeypatch.setattr(
            config_mod, "get_effective_config", lambda _o=None: self.config
        )
        monkeypatch.setattr(config_mod, "apply_config_to_env", lambda _c: None)
        monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path / "config" / "evoscientist")

        from EvoScientist.langgraph_dev import server_home

        monkeypatch.setattr(lgm, "_is_port_occupied", self._occupied)
        monkeypatch.setattr(lgm, "is_langgraph_dev_running", self._running)
        monkeypatch.setattr(lgm, "read_record", lambda **_k: self.record)
        monkeypatch.setattr(lgm, "ask", self._ask)
        monkeypatch.setattr(server_home, "lock_state", lambda wait=0: server_home.FREE)
        monkeypatch.setattr(lgm, "_server_config_fingerprint", lambda _c: "fp-now")
        monkeypatch.setattr(lgm, "_install_node_for_mcp", lambda waits: None)
        # Another program on the port keeps it from being bound.
        monkeypatch.setattr(
            lgm, "_wait_for_port_bindable", lambda *a, **k: not self.occupied
        )
        monkeypatch.setattr(lgm, "_spawn_server", self._start)
        monkeypatch.setattr(lgm, "stop_langgraph_dev", self._stop)
        monkeypatch.setattr(lgm, "read_tunnel_url", lambda *a, **k: self.tunnel_url)
        monkeypatch.setattr(lgm, "_POLL_INTERVAL", 0.001)
        monkeypatch.setattr(lgm, "_START_BUDGET", 0.3)
        monkeypatch.setattr(registration, "register_folder", self._register)
        monkeypatch.setattr(lgm, "_PROCESS", None)

        monkeypatch.setattr(agent_shell, "setup_hint", lambda: None)
        monkeypatch.setattr(agent_shell, "server_setup_hint", lambda _s: None)
        monkeypatch.setattr(agent_shell, "shell_drift_message", lambda _s: None)
        monkeypatch.setattr(backend_job, "backend_job", lambda: self.job)
        emit = json_mode._Events.__call__

        def _emit_then_close(events, event, **fields):
            emit(events, event, **fields)
            self.log.append(("event", event))
            if event == "ready":
                self.close_stdin()

        monkeypatch.setattr(json_mode._Events, "__call__", _emit_then_close)
        monkeypatch.setattr(json_mode, "_TICK", 0.01)
        monkeypatch.setattr(json_mode, "_TUNNEL_WAIT", 0.2)
        monkeypatch.setattr(json_mode, "HEALTH_INTERVAL", 0.0)

    # -- fakes ------------------------------------------------------------ #
    def _running(self, *_a, **_k):
        if self.started is None and self.record is None and not self.occupied:
            return False
        return self.running() if callable(self.running) else self.running

    def _ask(self, _url):
        if self.started is None and self.record is None and not self.occupied:
            return "refused"
        if self._running():
            return self.whose
        return "foreign" if self.started is None and self.occupied else "silent"

    def _occupied(self, *_a, **_k):
        if self.started is None:
            return self.occupied
        return self.listening() if callable(self.listening) else self.listening

    def _start(self, **kwargs):
        """The spawn: the server joins the job and records itself."""
        from EvoScientist.langgraph_dev import manager as lgm

        self.started = kwargs
        self.log.append("backend start")
        lgm._PROCESS = self.proc
        if kwargs.get("job") is not None:
            kwargs["job"].started(self.proc)
        self.record = lgm.ServerRecord(
            pid=self.proc.pid,
            port=kwargs["port"],
            host=kwargs["host"],
            lasting=kwargs["lasting"],
            fields={
                "data_dir": str(paths.DATA_DIR),
                "config_dir": str(paths.CONFIG_DIR),
            },
        )
        return self.proc

    def _stop(self, proc=None, *, clean=True):
        from EvoScientist.langgraph_dev import manager as lgm

        self.log.append(("backend stop", proc))
        self.clean_stops.append(clean)
        if proc is lgm._PROCESS:
            lgm._PROCESS = None

    def _register(self, workspace, *, url):
        """The backend's answer: one this run started reports its options."""
        self.registered.append(workspace)
        if self.started is None:
            return self.exposure
        return registration.ServerExposure.of(
            self.started["host"], self.started.get("tunnel")
        )

    # -- running ---------------------------------------------------------- #
    def close_stdin(self) -> None:
        if self._stdin_write is not None:
            os.close(self._stdin_write)
            self._stdin_write = None

    def _set_stdin(self) -> None:
        if self.stdin_kind in ("eof", "eof-now"):
            read, write = os.pipe()
            self._stdin_write = write
            if self.stdin_kind == "eof-now":
                self.close_stdin()
            stdin = os.fdopen(read, "r")
        elif self.stdin_kind == "devnull":
            stdin = open(os.devnull)
        else:
            stdin = self.stdin_kind
        self._stdin = stdin
        self.mp.setattr(sys, "stdin", stdin)

    def run(self, capsys, **kwargs) -> tuple[int, list[dict[str, Any]]]:
        """Run with the real JSON emitter; returns the exit code and the
        parsed stdout lines (every line must parse)."""
        self._set_stdin()
        code = json_mode.run_json(
            workdir=kwargs.get("workdir"),
            port=kwargs.get("port"),
            host=kwargs.get("host"),
            tunnel=kwargs.get("tunnel", False),
            debug=False,
        )
        out = capsys.readouterr().out
        lines = [json.loads(line) for line in out.splitlines()]
        return code, lines


@pytest.fixture
def h(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)
    yield harness
    harness.close_stdin()  # lets a watcher left by an error run end
    # The run's watcher reads its own copy of the descriptor.
    if harness._stdin is not None:
        harness._stdin.close()


def _wait_for_stdin_watcher():
    """Until the stdin watcher has seen the end of stdin and set the stop."""
    deadline = time.monotonic() + 5
    while any(
        t.name == json_mode._STDIN_THREAD and t.is_alive()
        for t in threading.enumerate()
    ):
        assert time.monotonic() < deadline, "stdin watcher still running"
        time.sleep(0.01)


def _events(lines):
    return [line["event"] for line in lines]


def _by_event(lines, event):
    (found,) = [line for line in lines if line["event"] == event]
    return found


# --------------------------------------------------------------------------- #
# Owned backend
# --------------------------------------------------------------------------- #
def test_owned_run_prints_starting_ready_stopped(h, capsys):
    code, lines = h.run(capsys)
    assert code == 0
    assert _events(lines) == ["starting", "ready", "stopped"]
    assert all(line["protocol"] == 1 for line in lines)
    assert _by_event(lines, "starting") == {
        "protocol": 1,
        "event": "starting",
        "workspace": h.workspace,
        "host": "127.0.0.1",
        "port": 6174,
    }
    ready = _by_event(lines, "ready")
    assert ready["url"] == "http://127.0.0.1:6174"
    assert ready["owned"] is True
    assert ready["workspace"] == h.workspace
    assert ready["assistant_id"] == workspace_assistant_id(
        "EvoScientist", Workspace(h.workspace)
    )
    assert ready["warnings"] == []
    assert "mode" not in ready
    assert "public_url" not in ready


def test_ready_carries_this_process_folders(h, capsys):
    _, lines = h.run(capsys)
    ready = _by_event(lines, "ready")
    assert ready["data_dir"] == str(h.tmp_path / "data")
    assert ready["config_dir"] == str(h.tmp_path / "config" / "evoscientist")
    assert os.path.isabs(ready["data_dir"])
    assert os.path.isabs(ready["config_dir"])


def test_end_of_stdin_stops_the_owned_backend_before_stopped(h, capsys):
    h.run(capsys)
    stop = h.log.index(("backend stop", h.proc))
    assert h.log.index(("event", "stopped")) > stop
    assert h.clean_stops == [True]


def test_stdout_carries_only_json(h, capsys, monkeypatch):
    """Rich output printed while the command runs goes to stderr."""
    from EvoScientist.langgraph_dev import manager as lgm
    from EvoScientist.stream.console import console

    def _start(**kwargs):
        console.print("a Rich line")
        return h._start(**kwargs)

    monkeypatch.setattr(lgm, "_spawn_server", _start)
    monkeypatch.setattr(
        "EvoScientist.agent_shell.setup_hint", lambda: "Run `EvoSci setup`."
    )
    h._set_stdin()
    code = json_mode.run_json(
        workdir=None, port=None, host=None, tunnel=False, debug=False
    )
    captured = capsys.readouterr()
    lines = [json.loads(line) for line in captured.out.splitlines()]
    assert code == 0
    assert "a Rich line" in captured.err
    assert "Run `EvoSci setup`." in _by_event(lines, "ready")["warnings"]


def test_non_ascii_paths_are_escaped(h, capsys):
    h.config.default_workdir = str(h.tmp_path / "wś é")
    h._set_stdin()
    json_mode.run_json(workdir=None, port=None, host=None, tunnel=False, debug=False)
    out = capsys.readouterr().out
    assert out.isascii()
    assert json.loads(out.splitlines()[0])["workspace"].endswith("wś é")


def test_start_gets_the_resolved_options(h, capsys):
    h.config.backend_allowed_origins = "https://smith.langchain.com"
    h.run(capsys, port=7001, host=" 0.0.0.0 ")
    assert h.started["port"] == 7001
    assert h.started["host"] == "0.0.0.0"
    assert h.started["allowed_origins"] == "https://smith.langchain.com"
    assert h.registered == [Workspace(h.workspace)]
    assert h.started["job"] is None  # no job off Windows


def test_tunnel_adds_public_url(h, capsys):
    h.tunnel_url = "https://demo.trycloudflare.com"
    _, lines = h.run(capsys, tunnel=True)
    assert _by_event(lines, "ready")["public_url"] == "https://demo.trycloudflare.com"
    assert h.started["tunnel"] is True


def test_a_stop_during_the_tunnel_wait_ends_it(h, capsys, monkeypatch):
    """The app stops while cloudflared has not reported a URL yet: no
    ``ready``, and no wait for the rest of the 35 s."""
    from EvoScientist.langgraph_dev import manager as lgm

    reads: list[float] = []

    def _read(timeout=35.0, **_k):
        reads.append(timeout)
        if len(reads) == 1:
            h.close_stdin()
            _wait_for_stdin_watcher()
        return None

    monkeypatch.setattr(lgm, "read_tunnel_url", _read)
    code, lines = h.run(capsys, tunnel=True)
    assert code == 0
    assert _events(lines) == ["starting", "stopped"]
    assert reads == [json_mode._TICK]
    assert ("backend stop", h.proc) in h.log


def test_tunnel_url_is_read_in_short_waits(h, capsys, monkeypatch):
    from EvoScientist.langgraph_dev import manager as lgm

    answers = iter([None, None, "https://late.trycloudflare.com"])
    monkeypatch.setattr(
        lgm, "read_tunnel_url", lambda timeout=35.0, **_k: next(answers)
    )
    _, lines = h.run(capsys, tunnel=True)
    assert _by_event(lines, "ready")["public_url"] == "https://late.trycloudflare.com"


def test_tunnel_without_url_reports_null(h, capsys):
    _, lines = h.run(capsys, tunnel=True)
    assert _by_event(lines, "ready")["public_url"] is None


def test_tunnel_failure_reports_null_and_the_reason(h, capsys, monkeypatch):
    """cloudflared could not create the tunnel: ``ready`` follows the first
    read, with no URL and cloudflared's reason among the warnings."""
    from EvoScientist.langgraph_dev import manager as lgm

    reads: list[float] = []

    def _read(timeout=35.0, **_k):
        reads.append(timeout)
        raise lgm.TunnelFailedError("failed to request quick Tunnel: i/o timeout")

    monkeypatch.setattr(lgm, "read_tunnel_url", _read)
    _, lines = h.run(capsys, tunnel=True)
    ready = _by_event(lines, "ready")
    assert ready["public_url"] is None
    assert (
        "Cloudflare quick tunnel failed: failed to request quick Tunnel: i/o timeout"
        in ready["warnings"]
    )
    assert reads == [json_mode._TICK]


@pytest.mark.parametrize("exposed", ["0.0.0.0", "192.168.1.5"])
def test_public_bind_warns_in_ready(h, capsys, exposed):
    _, lines = h.run(capsys, host=exposed)
    expected = registration.exposure_messages(
        registration.ServerExposure.of(exposed, False)
    )
    assert set(expected) <= set(_by_event(lines, "ready")["warnings"])


def test_dangerous_mode_and_tunnel_warn_in_ready(h, capsys):
    h.config.dangerous_mode = True
    _, lines = h.run(capsys, tunnel=True)
    warnings = _by_event(lines, "ready")["warnings"]
    assert any("DANGEROUS MODE" in w for w in warnings)
    tunnel = registration.ServerExposure.of("127.0.0.1", True)
    assert set(registration.exposure_messages(tunnel)) <= set(warnings)


class _FakeJob:
    def __init__(self, log, warning=None):
        self.log = log
        self.warning = warning

    def adopt(self, process):
        self.log.append(("job adopt", process))

    def started(self, process):
        self.log.append(("job started", process))

    def close(self):
        self.log.append("job close")


def test_job_is_closed_only_after_the_backend_stopped(h, capsys):
    h.job = _FakeJob(h.log)
    h.run(capsys)
    assert h.started["job"] is h.job
    assert h.log.index("backend start") < h.log.index(("job started", h.proc))
    assert h.log.index(("backend stop", h.proc)) < h.log.index("job close")


def test_job_warning_lands_in_ready(h, capsys):
    h.job = _FakeJob(h.log, warning="Could not put the backend in a job.")
    _, lines = h.run(capsys)
    assert (
        "Could not put the backend in a job." in _by_event(lines, "ready")["warnings"]
    )


def test_a_failed_backend_stop_still_closes_the_job(h, capsys, monkeypatch):
    """``stop_langgraph_dev()`` lets ``psutil.AccessDenied`` through; the
    later steps still run and the run ends with ``internal_error``."""
    import psutil

    from EvoScientist.langgraph_dev import manager as lgm

    def _stop(proc=None, **_kw):
        h.log.append(("backend stop", proc))
        raise psutil.AccessDenied(proc.pid)

    monkeypatch.setattr(lgm, "stop_langgraph_dev", _stop)
    h.job = _FakeJob(h.log)
    code, lines = h.run(capsys)
    assert code == 1
    assert lines[-1]["code"] == "internal_error"
    assert h.log.index(("backend stop", h.proc)) < h.log.index("job close")


def test_job_is_closed_after_a_failed_start(h, capsys):
    def _fail(**_kw):
        raise RuntimeError("boom")

    h.job = _FakeJob(h.log)
    h.mp.setattr("EvoScientist.langgraph_dev.manager._spawn_server", _fail)
    code, _ = h.run(capsys)
    assert code == 1
    assert "job close" in h.log


# --------------------------------------------------------------------------- #
# Attached backend
# --------------------------------------------------------------------------- #
def _attach(h, *, port=6174, lasting=True, **fields):
    """This data folder's server runs, recorded with *fields*."""
    from EvoScientist.langgraph_dev import manager as lgm

    h.occupied = True
    h.record = lgm.ServerRecord(
        pid=1, port=port, host="127.0.0.1", lasting=lasting, fields=fields
    )


def test_attached_run_reports_the_running_backend_and_leaves_it(h, capsys):
    _attach(
        h,
        data_dir="/other/data",
        config_dir="/other/config/evoscientist",
        config_fingerprint="fp-now",
    )
    code, lines = h.run(capsys)
    assert code == 0
    assert _events(lines) == ["starting", "ready", "stopped"]
    ready = _by_event(lines, "ready")
    assert ready["owned"] is False
    assert ready["data_dir"] == "/other/data"
    assert ready["config_dir"] == "/other/config/evoscientist"
    assert h.started is None
    assert not any(isinstance(e, tuple) and e[0] == "backend stop" for e in h.log)


def test_attached_run_makes_no_job(h, capsys, monkeypatch):
    from EvoScientist.deploy import backend_job

    made: list = []
    monkeypatch.setattr(backend_job, "backend_job", lambda: made.append(1))
    _attach(h)
    h.run(capsys)
    assert made == []


def test_attached_record_without_folders_falls_back_to_own(h, capsys):
    """A backend an older release started may have recorded no folders."""
    _attach(h)
    _, lines = h.run(capsys)
    ready = _by_event(lines, "ready")
    assert ready["data_dir"] == str(h.tmp_path / "data")
    assert ready["config_dir"] == str(h.tmp_path / "config" / "evoscientist")


def test_attached_backend_of_another_version_gets_drift_warning(h, capsys):
    _attach(h, config_fingerprint="fp-old")
    code, lines = h.run(capsys)
    assert code == 0
    ready = _by_event(lines, "ready")
    assert ready["owned"] is False
    assert any("EvoSci server stop" in w for w in ready["warnings"])


@pytest.mark.parametrize(
    ("answered", "requested", "warns"),
    [
        ("0.0.0.0", "127.0.0.1", True),  # the backend is public, the request not
        ("127.0.0.1", "0.0.0.0", False),  # the request is public, the backend not
    ],
)
def test_attached_public_bind_warning_follows_backend(
    h, capsys, answered, requested, warns
):
    """The backend says how it is bound when it registers the workspace."""
    _attach(h)
    h.exposure = registration.ServerExposure.of(answered, False)
    _, lines = h.run(capsys, host=requested)
    (bind,) = registration.exposure_messages(
        registration.ServerExposure.of("0.0.0.0", False)
    )
    assert (bind in _by_event(lines, "ready")["warnings"]) is warns


def test_attached_tunneled_backend_warns(h, capsys):
    _attach(h)
    h.exposure = registration.ServerExposure.of("127.0.0.1", True)
    _, lines = h.run(capsys)
    expected = registration.exposure_messages(h.exposure)
    assert set(expected) <= set(_by_event(lines, "ready")["warnings"])


def test_attached_tunnel_warns_and_reports_null(h, capsys):
    _attach(h)
    h.tunnel_url = "https://stale.trycloudflare.com"
    _, lines = h.run(capsys, tunnel=True)
    ready = _by_event(lines, "ready")
    assert ready["public_url"] is None
    assert any("--tunnel" in w for w in ready["warnings"])


def test_attached_backend_going_away_is_backend_exited(h, capsys):
    _attach(h)
    h.stdin_kind = "devnull"  # nothing else stops the run
    checks = iter([True, False, False, False])
    h.running = lambda: next(checks, False)
    code, lines = h.run(capsys)
    assert code == 1
    assert _events(lines) == ["starting", "ready", "error"]
    assert lines[-1]["code"] == "backend_exited"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_port", [0, -1, 70000])
def test_invalid_port(h, capsys, bad_port):
    code, lines = h.run(capsys, port=bad_port)
    assert code == 1
    assert _events(lines) == ["error"]
    assert lines[0]["code"] == "invalid_port"
    assert lines[0]["detail"] == {"port": bad_port}


def test_bind_flags_are_config_overrides(h, capsys, monkeypatch):
    """The fingerprint compared with a running server's sees ``--host`` and
    ``--port`` as ``EvoSci --host`` does."""
    import EvoScientist.config as config_mod

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(
        config_mod,
        "get_effective_config",
        lambda o=None: seen.append(dict(o or {})) or h.config,
    )
    h.run(capsys, host=" 0.0.0.0 ", port=6190)
    assert seen == [{"langgraph_dev_host": "0.0.0.0", "langgraph_dev_port": 6190}]


def test_foreign_process_is_port_conflict(h, capsys):
    h.occupied = True
    h.running = False
    code, lines = h.run(capsys)
    assert code == 1
    assert _events(lines) == ["starting", "error"]
    assert lines[-1]["code"] == "port_conflict"
    assert lines[-1]["detail"]["reason"] == "foreign"
    assert "hint" in lines[-1]["detail"]
    assert h.started is None


def test_unrecorded_server_of_this_data_folder_is_attached(h, capsys):
    h.occupied = True
    code, lines = h.run(capsys)
    assert code == 0
    ready = _by_event(lines, "ready")
    assert ready["owned"] is False
    assert ready["data_dir"] == str(h.tmp_path / "data")
    assert h.started is None


@pytest.mark.parametrize(
    ("whose", "reason"),
    [("other", "other_data_folder"), ("older", "older_server")],
)
def test_another_servers_backend_is_port_conflict(h, capsys, whose, reason):
    h.occupied = True
    h.whose = whose
    code, lines = h.run(capsys)
    assert code == 1
    assert lines[-1]["code"] == "port_conflict"
    assert lines[-1]["detail"]["reason"] == reason
    assert "hint" in lines[-1]["detail"]
    assert h.started is None


def test_this_data_folders_backend_on_another_port_is_attached(h, capsys):
    """ready has the URL it answers on, and a warning says so."""
    _attach(h, port=7000)
    code, lines = h.run(capsys)
    assert code == 0
    assert _by_event(lines, "starting")["port"] == 6174
    ready = _by_event(lines, "ready")
    assert ready["url"] == "http://127.0.0.1:7000"
    assert ready["owned"] is False
    assert any("7000" in w for w in ready["warnings"])
    assert h.started is None


def test_attached_backend_older_release_started_says_to_restart_it(h, capsys):
    from EvoScientist.langgraph_dev import manager as lgm

    _attach(h)
    h.record = lgm.ServerRecord(pid=1, port=6174, host="127.0.0.1", older=True)
    _, lines = h.run(capsys)
    ready = _by_event(lines, "ready")
    assert lgm.OLDER_SERVER_MESSAGE in ready["warnings"]
    assert ready["log_file"] == str(lgm.OLDER_RUNTIME_DIR / "langgraph_dev.log")


def test_attached_backend_registers_this_project(h, capsys):
    """The backend serves every registered project: this run registers its
    own."""
    _attach(h)
    code, lines = h.run(capsys)
    assert code == 0
    ready = _by_event(lines, "ready")
    assert ready["owned"] is False
    assert ready["workspace"] == h.workspace
    assert h.registered == [Workspace(h.workspace)]
    assert h.started is None


@pytest.mark.parametrize("problem", [None, "The saved sign-in is corrupt."])
def test_missing_chatgpt_sign_in_is_reported_before_starting(
    h, capsys, monkeypatch, problem
):
    from EvoScientist.llm import chatgpt_oauth
    from EvoScientist.llm.chatgpt_oauth import ChatGPTLogin

    monkeypatch.setattr(
        chatgpt_oauth,
        "status",
        lambda: ChatGPTLogin(signed_in=False, problem=problem),
    )
    h.config.openai_auth_mode = "oauth"
    code, lines = h.run(capsys)
    assert code == 1
    assert _events(lines) == ["error"]
    assert lines[0]["code"] == "sign_in_required"
    assert lines[0]["message"] == chatgpt_oauth.SIGN_IN_HINT
    expected = {"provider": "openai"}
    if problem:
        expected["problem"] = problem
    assert lines[0]["detail"] == expected
    assert not (h.tmp_path / "ws").exists()


def test_backend_that_cannot_start_ccproxy_is_ccproxy_failed(h, capsys):
    """The backend exits with its ccproxy code, leaving the reason."""
    from EvoScientist.langgraph_dev import manager as lgm
    from EvoScientist.langgraph_dev.server_ccproxy import CCPROXY_FAILED_EXIT

    h.proc.returncode = CCPROXY_FAILED_EXIT
    start = h._start

    def _exits(**kwargs):
        proc = start(**kwargs)
        lgm.RUNTIME.ccproxy_failure.write_text(
            "ccproxy is not signed in.\nRun: ccproxy auth login claude_api",
            encoding="utf-8",
        )
        h.record = None  # it exited before it answered
        return proc

    h.mp.setattr(lgm, "_spawn_server", _exits)
    code, lines = h.run(capsys)
    assert code == 1
    assert lines[-1]["code"] == "ccproxy_failed"
    assert "ccproxy auth login" in lines[-1]["message"]
    assert lines[-1]["detail"] == {"log_file": str(lgm.RUNTIME.log_file)}


def test_start_failure_before_launch_is_backend_start_failed_without_file(
    h, capsys, monkeypatch
):
    from EvoScientist.langgraph_dev import manager as lgm

    def _fail(**_kw):
        raise FileNotFoundError("langgraph CLI not found.")

    monkeypatch.setattr(lgm, "_spawn_server", _fail)
    code, lines = h.run(capsys)
    assert code == 1
    assert _events(lines) == ["starting", "error"]
    assert lines[-1]["code"] == "backend_start_failed"
    assert lines[-1]["detail"] == {}


def test_backend_that_exits_while_it_starts_reports_its_code_and_log(h, capsys):
    from EvoScientist.langgraph_dev import manager as lgm

    h.proc.returncode = 1
    start = h._start

    def _exits(**kwargs):
        proc = start(**kwargs)
        h.record = None  # it exited before it answered
        return proc

    h.mp.setattr(lgm, "_spawn_server", _exits)
    code, lines = h.run(capsys)
    assert code == 1
    assert lines[-1]["code"] == "backend_start_failed"
    assert lines[-1]["detail"] == {
        "log_file": str(lgm.RUNTIME.log_file),
        "exit_code": 1,
    }


def test_unexpected_exception_is_internal_error_with_traceback(h, capsys, monkeypatch):
    import EvoScientist.config as config_mod

    def _broken(_o=None):
        raise ValueError("config.yaml: bad indentation")

    monkeypatch.setattr(config_mod, "get_effective_config", _broken)
    h._set_stdin()
    code = json_mode.run_json(
        workdir=None, port=None, host=None, tunnel=False, debug=False
    )
    captured = capsys.readouterr()
    (line,) = [json.loads(x) for x in captured.out.splitlines()]
    assert code == 1
    assert line["code"] == "internal_error"
    assert line["message"] == "config.yaml: bad indentation"
    assert line["detail"] == {"type": "ValueError"}
    assert "Traceback" in captured.err


def test_internal_error_after_start_stops_the_backend_first(h, capsys, monkeypatch):
    def _broken():
        raise OSError("hint failed")

    monkeypatch.setattr("EvoScientist.agent_shell.setup_hint", _broken)
    code, lines = h.run(capsys)
    assert code == 1
    assert lines[-1]["code"] == "internal_error"
    assert h.log.index(("backend stop", h.proc)) < h.log.index(("event", "error"))


def test_owned_backend_dying_is_backend_exited(h, capsys, monkeypatch):
    h.stdin_kind = "devnull"
    emit = json_mode._Events.__call__

    def _dies_once_ready(events, event, **fields):
        emit(events, event, **fields)
        if event == "ready":
            h.proc.returncode = 3

    monkeypatch.setattr(json_mode._Events, "__call__", _dies_once_ready)
    code, lines = h.run(capsys)
    assert code == 1
    assert lines[-1]["code"] == "backend_exited"
    assert lines[-1]["detail"]["exit_code"] == 3
    assert ("backend stop", h.proc) in h.log


def _then_interrupt(values):
    """A probe answering ``values`` in turn, then Ctrl+C (ends the run)."""
    answers = iter(values)

    def _probe():
        value = next(answers, None)
        if value is None:
            raise KeyboardInterrupt
        return value

    return _probe


def test_owned_backend_that_stops_answering_keeps_running(h, capsys):
    """A stalled owned backend keeps its port: it is not reported as exited,
    and not stopped while the run waits."""
    h.stdin_kind = "devnull"
    answers = iter([True])  # /ok answers the start, then never again
    h.running = lambda: next(answers, False)
    h.listening = _then_interrupt([True] * 6)
    code, lines = h.run(capsys)
    assert code == 0
    assert _events(lines) == ["starting", "ready", "stopped"]


def test_owned_backend_that_stops_listening_is_backend_exited(h, capsys):
    h.stdin_kind = "devnull"
    h.listening = False
    code, lines = h.run(capsys)
    assert code == 1
    assert lines[-1]["code"] == "backend_exited"
    assert "stopped listening" in lines[-1]["message"]
    assert "log_file" in lines[-1]["detail"]
    assert ("backend stop", h.proc) in h.log


def test_attached_backend_that_stops_answering_is_backend_exited(h, capsys):
    _attach(h)
    h.stdin_kind = "devnull"
    checks = iter([True])
    h.running = lambda: next(checks, False)
    code, lines = h.run(capsys)
    assert code == 1
    assert lines[-1]["code"] == "backend_exited"
    assert "stopped answering" in lines[-1]["message"]
    assert "log_file" in lines[-1]["detail"]
    assert not any(isinstance(e, tuple) and e[0] == "backend stop" for e in h.log)


def test_one_failed_health_check_is_not_an_exit(h, capsys):
    h.stdin_kind = "devnull"
    h.listening = _then_interrupt([False, True, False, True, False, True])
    code, lines = h.run(capsys)
    assert code == 0
    assert lines[-1]["event"] == "stopped"


def test_ctrl_c_stops_cleanly(h, capsys):
    h.stdin_kind = "devnull"
    h.listening = _then_interrupt([])
    code, lines = h.run(capsys)
    assert code == 0
    assert _events(lines) == ["starting", "ready", "stopped"]
    assert ("backend stop", h.proc) in h.log


@pytest.mark.skipif(
    sys.platform == "win32", reason="Node's kill() sends SIGTERM on POSIX only"
)
def test_sigterm_stops_the_owned_backend_before_stopped(h, capsys):
    import signal

    def _fallback(signum, _frame):
        missed.append(signum)
        raise KeyboardInterrupt

    def _terminate():
        signal.raise_signal(signal.SIGTERM)
        return True

    missed: list[int] = []
    previous = signal.signal(signal.SIGTERM, _fallback)
    try:
        h.stdin_kind = "devnull"
        h.listening = _terminate
        code, lines = h.run(capsys)
        assert signal.getsignal(signal.SIGTERM) is _fallback
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert missed == []
    assert code == 0
    assert _events(lines) == ["starting", "ready", "stopped"]
    assert h.log.index(("backend stop", h.proc)) < h.log.index(("event", "stopped"))


@pytest.mark.parametrize(
    ("raised", "terminal"),
    [(KeyboardInterrupt(), "stopped"), (RuntimeError("health check"), "error")],
)
def test_a_start_cut_after_the_spawn_stops_the_backend(
    h, capsys, monkeypatch, raised, terminal
):
    """No handle came back, so the stop uses the one the manager recorded."""
    from EvoScientist.langgraph_dev import manager as lgm

    def _spawned_then_cut(**kwargs):
        h._start(**kwargs)
        raise raised

    monkeypatch.setattr(lgm, "_spawn_server", _spawned_then_cut)
    _, lines = h.run(capsys)
    assert _events(lines) == ["starting", terminal]
    assert h.log.index(("backend stop", h.proc)) < h.log.index(("event", terminal))
    assert h.clean_stops == [False]


def test_stdin_closed_during_the_start_stops_without_ready(h, capsys, monkeypatch):
    """The app went away while the backend started: it is stopped as soon as
    the start returns, without the tunnel wait or ``ready``."""
    from EvoScientist.langgraph_dev import manager as lgm

    def _start(**kwargs):
        proc = h._start(**kwargs)
        h.close_stdin()
        _wait_for_stdin_watcher()
        return proc

    reads: list[int] = []
    monkeypatch.setattr(lgm, "_spawn_server", _start)
    monkeypatch.setattr(lgm, "read_tunnel_url", lambda *a, **k: reads.append(1))
    code, lines = h.run(capsys, tunnel=True)
    assert code == 0
    assert _events(lines) == ["starting", "stopped"]
    assert ("backend stop", h.proc) in h.log
    assert reads == []


def test_stdin_closed_before_start_starts_nothing(h, capsys, monkeypatch):
    from EvoScientist.langgraph_dev import manager as lgm

    def _bindable(*_a, **_k):
        _wait_for_stdin_watcher()
        return True

    monkeypatch.setattr(lgm, "_wait_for_port_bindable", _bindable)
    h.stdin_kind = "eof-now"
    code, lines = h.run(capsys)
    assert code == 0
    assert _events(lines) == ["starting", "stopped"]
    assert h.started is None


def test_a_long_stdin_line_without_newline_is_dropped(h, capsys, monkeypatch):
    monkeypatch.setattr(json_mode, "_MAX_LINE", 8)
    read, write = os.pipe()
    h.stdin_kind = os.fdopen(read, "r")
    os.write(write, b"x" * 20 + b'"command": "stop"}\n{"command": "stop"}\n')
    try:
        code, lines = h.run(capsys)
    finally:
        os.close(write)
    assert code == 0
    assert lines[-1]["event"] == "stopped"


def test_a_pipe_stdin_is_not_handed_on(h, capsys):
    """While the run watches the pipe, stdin itself is the null device, so a
    child the run starts does not inherit the pipe."""
    from EvoScientist.setup.protocol import stdin_is_pipe

    code, _ = h.run(capsys)
    assert code == 0
    assert stdin_is_pipe() is False


def test_second_start_waits_for_lock(h, capsys):
    """The start waits until another start released the lock."""
    holder = LockHolder()
    threading.Timer(0.05, holder.done).start()
    code, _ = h.run(capsys)
    assert code == 0
    assert h.started is not None


def test_stop_while_waiting_for_lock_starts_nothing(h, capsys, monkeypatch):
    from EvoScientist.langgraph_dev import manager as lgm

    monkeypatch.setattr(lgm, "_START_BUDGET", 5.0)
    holder = LockHolder()
    threading.Timer(0.3, h.close_stdin).start()
    try:
        code, lines = h.run(capsys)
    finally:
        holder.done()
    assert code == 0
    assert _events(lines) == ["starting", "stopped"]
    assert h.started is None


def test_a_lock_held_too_long_is_backend_start_failed(h, capsys, monkeypatch):
    from EvoScientist.langgraph_dev import manager as lgm

    holder = LockHolder()
    try:
        code, lines = h.run(capsys)
    finally:
        holder.done()
    assert code == 1
    assert lines[-1]["code"] == "backend_start_failed"
    assert lines[-1]["detail"] == {"lock_file": str(lgm.RUNTIME.lock_file)}
    assert h.started is None


def test_the_lock_is_released_after_the_start(h, capsys):
    from filelock import FileLock

    from EvoScientist.langgraph_dev import manager as lgm

    h.run(capsys)
    lock = FileLock(str(lgm.RUNTIME.lock_file))
    lock.acquire(timeout=0.5)
    lock.release()


def test_closed_stdout_stops_the_run(h, monkeypatch):
    """The reader is gone: the run stops its backend and writes nothing more.
    The exit code of a real process is checked in a child below."""

    class _Closed:
        """The reader leaves after ``starting``."""

        lines = 0

        def write(self, _s):
            self.lines += 1
            if self.lines > 1:
                raise BrokenPipeError

        def flush(self):
            pass

    h.stdin_kind = "devnull"
    h._set_stdin()
    monkeypatch.setattr(sys, "stdout", _Closed())
    code = json_mode.run_json(
        workdir=None, port=None, host=None, tunnel=False, debug=False
    )
    assert code == 0
    assert ("backend stop", h.proc) in h.log


def test_deploy_json_exits_non_zero_on_error(h, capsys):
    h.occupied = True
    h.running = False
    h._set_stdin()
    with pytest.raises(typer.Exit) as exc:
        deploy_server.deploy(
            workdir=None,
            port=None,
            host=None,
            tunnel=False,
            debug=False,
            json_output=True,
        )
    assert exc.value.exit_code == 1
    assert (
        json.loads(capsys.readouterr().out.splitlines()[-1])["code"] == "port_conflict"
    )


# --------------------------------------------------------------------------- #
# stdin
# --------------------------------------------------------------------------- #
def test_stop_command_stops_while_stdin_stays_open(h, capsys):
    read, write = os.pipe()
    h.stdin_kind = os.fdopen(read, "r")
    os.write(write, b'not json\n{"command": "other"}\n{"command": "stop"}\n')
    try:
        code, lines = h.run(capsys)
    finally:
        os.close(write)
    assert code == 0
    assert lines[-1]["event"] == "stopped"


@pytest.mark.skipif(sys.platform == "win32", reason="UNIX socketpair as stdin")
def test_end_of_a_socket_stdin_stops(h, capsys):
    """Node's ``stdio: "pipe"`` is a socketpair on macOS and Linux."""
    ours, theirs = socket.socketpair()
    h.stdin_kind = os.fdopen(ours.detach(), "r")
    timer = threading.Timer(0.2, theirs.close)
    timer.start()
    code, lines = h.run(capsys)
    timer.join()
    assert code == 0
    assert lines[-1]["event"] == "stopped"


def test_stdin_kinds(tmp_path, monkeypatch):
    from EvoScientist.setup.protocol import stdin_is_pipe

    read, write = os.pipe()
    regular = tmp_path / "input.txt"
    regular.write_text("x")
    cases = {
        "pipe": (os.fdopen(read, "r"), True),
        "devnull": (open(os.devnull), False),
        "file": (open(regular), False),
    }
    try:
        for name, (stdin, expected) in cases.items():
            monkeypatch.setattr(sys, "stdin", stdin)
            assert stdin_is_pipe() is expected, name
        if sys.platform != "win32":
            ours, theirs = socket.socketpair()
            with os.fdopen(ours.detach(), "r") as stdin:
                monkeypatch.setattr(sys, "stdin", stdin)
                assert stdin_is_pipe() is True
            theirs.close()
    finally:
        os.close(write)
        for stdin, _ in cases.values():
            stdin.close()


def test_devnull_stdin_does_not_stop_the_run(h, capsys):
    """Only the backend going away ends this run, not stdin."""
    h.stdin_kind = "devnull"
    checks = iter([True] * 5 + [False] * 3)
    h.listening = lambda: next(checks, False)
    _, lines = h.run(capsys)
    assert lines[-1]["code"] == "backend_exited"


_GRANDCHILD = (
    "def start(**k):\n"
    "    import subprocess\n"
    "    code = ('import os, stat; m = os.fstat(0).st_mode; '\n"
    '            \'print("pipe" if stat.S_ISFIFO(m) or stat.S_ISSOCK(m) else "other")\')\n'
    "    try:\n"
    "        out = subprocess.run([sys.executable, '-c', code], capture_output=True,\n"
    "                             text=True, timeout=10).stdout.strip()\n"
    "    except subprocess.TimeoutExpired:\n"
    "        out = 'hung'\n"
    "    print('grandchild stdin: ' + out, file=sys.stderr, flush=True)\n"
    "    return spawn(**k)\n"
    "m._spawn_server = start\n"
)


def _spawn_child(tmp_path, *, grandchild: bool = False):
    """The real command in a child, with the backend and the probes faked.
    With ``grandchild``, the faked start runs a Python child that inherits
    stdin, as the backend does, and reports what its stdin is."""
    script = tmp_path / "child.py"
    script.write_text(
        "import sys\n"
        "from types import SimpleNamespace\n"
        "import EvoScientist.config as c\n"
        "from EvoScientist import agent_shell\n"
        "from EvoScientist.langgraph_dev import manager as m\n"
        "from pathlib import Path\n"
        "from EvoScientist import paths\n"
        "paths.SERVER_HOME = Path(sys.argv[1]).parent / 'rt'\n"
        "cfg = SimpleNamespace(default_workdir=sys.argv[1], langgraph_dev_port=6174,\n"
        "    langgraph_dev_host='127.0.0.1', anthropic_auth_mode='api_key',\n"
        "    openai_auth_mode='api_key')\n"
        "c.get_effective_config = lambda _o=None: cfg\n"
        "c.apply_config_to_env = lambda _c: None\n"
        "agent_shell.setup_hint = lambda: None\n"
        "class P:\n"
        "    pid = 1; returncode = None\n"
        "    def poll(self): return None\n"
        "# The started backend records itself, then answers on its port.\n"
        "rec = []\n"
        "m.read_record = lambda **k: rec[0] if rec else None\n"
        "m.ask = lambda url: 'own' if rec else 'refused'\n"
        "m._is_port_occupied = lambda *a, **k: bool(rec)\n"
        "m.is_langgraph_dev_running = lambda *a, **k: bool(rec)\n"
        "m._install_node_for_mcp = lambda waits: None\n"
        "m._wait_for_port_bindable = lambda *a, **k: True\n"
        "m._server_config_fingerprint = lambda _c: 'fp'\n"
        "def spawn(**k):\n"
        "    rec.append(m.ServerRecord(pid=1, port=k['port'], host=k['host']))\n"
        "    return P()\n"
        "m._spawn_server = spawn\n"
        "m.stop_langgraph_dev = lambda p=None, **_k: print('stopped backend', file=sys.stderr)\n"
        "from EvoScientist.langgraph_dev import registration as r\n"
        "r.register_folder = lambda *a, **k: None\n"
        + (_GRANDCHILD if grandchild else "")
        + "from EvoScientist.deploy import json_mode\n"
        "sys.exit(json_mode.run_json(workdir=None, port=None, host=None,\n"
        "    tunnel=False, debug=False))\n"
    )
    return subprocess.Popen(
        [sys.executable, str(script), str(tmp_path / "ws")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


@pytest.mark.timeout(60)
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell and pipes")
def test_subprocess_closing_stdin_prints_stopped(tmp_path):
    """End to end in a child: a parent closes the stdin pipe, the child stops."""
    proc = _spawn_child(tmp_path)
    assert proc.stdout is not None
    assert proc.stdin is not None
    starting = json.loads(proc.stdout.readline())
    ready = json.loads(proc.stdout.readline())
    assert (starting["event"], ready["event"]) == ("starting", "ready")
    out, err = proc.communicate(timeout=30)  # closes stdin
    assert proc.returncode == 0
    assert [json.loads(x)["event"] for x in out.splitlines()] == ["stopped"]
    assert b"stopped backend" in err


@pytest.mark.timeout(60)
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell and pipes")
def test_subprocess_with_closed_stdout_exits_zero(tmp_path):
    """The reader closes stdout, then stdin: the `stopped` line cannot be
    written, and the process still exits 0 without a shutdown error."""
    proc = _spawn_child(tmp_path)
    assert proc.stdout is not None
    assert proc.stdin is not None
    assert json.loads(proc.stdout.readline())["event"] == "starting"
    assert json.loads(proc.stdout.readline())["event"] == "ready"
    proc.stdout.close()
    proc.stdin.close()
    assert proc.wait(timeout=30) == 0
    err = proc.stderr.read()
    proc.stderr.close()
    assert b"stopped backend" in err
    assert b"Exception ignored" not in err
    assert b"BrokenPipeError" not in err


@pytest.mark.timeout(60)
def test_a_child_of_the_run_inherits_no_stdin_pipe(tmp_path):
    """On Windows a child that inherits a pipe another thread is reading hangs
    in its start-up; the backend gets the null device instead (every
    platform), and still starts while the run watches stdin."""
    proc = _spawn_child(tmp_path, grandchild=True)
    assert proc.stdout is not None
    assert json.loads(proc.stdout.readline())["event"] == "starting"
    assert json.loads(proc.stdout.readline())["event"] == "ready"
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 0
    assert b"grandchild stdin: other" in err
    assert [json.loads(x)["event"] for x in out.splitlines()] == ["stopped"]
