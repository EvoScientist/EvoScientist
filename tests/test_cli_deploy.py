"""Tests for ``EvoSci deploy`` command flow.

Verifies the orchestration:
- workspace resolution (CLI > config > cwd)
- port resolution (CLI > config > default)
- port collision pre-flight
- ccproxy lifecycle (only if OAuth configured)
- ``start_langgraph_dev(deploy_mode=True)`` invocation
- clean shutdown via signal handler / KeyboardInterrupt
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import typer

from EvoScientist.deploy import server as deploy_server


def _make_config(
    *,
    default_workdir: str = "",
    langgraph_dev_port: int = 6174,
    anthropic_auth_mode: str = "api_key",
    openai_auth_mode: str = "api_key",
    log_level: str = "warning",
    langgraph_dev_jobs_per_worker: int = 10,
    langgraph_dev_file_persistence: bool = True,
    dangerous_mode: bool = False,
):
    return SimpleNamespace(
        default_workdir=default_workdir,
        langgraph_dev_port=langgraph_dev_port,
        anthropic_auth_mode=anthropic_auth_mode,
        openai_auth_mode=openai_auth_mode,
        log_level=log_level,
        langgraph_dev_jobs_per_worker=langgraph_dev_jobs_per_worker,
        langgraph_dev_file_persistence=langgraph_dev_file_persistence,
        dangerous_mode=dangerous_mode,
    )


class _ImmediateEvent:
    """Fake ``threading.Event``: first ``is_set()`` returns False so the
    while-loop body runs once; subsequent calls return True, exiting the
    loop. ``wait`` is a no-op (no real blocking)."""

    def __init__(self):
        self._called = 0
        self.set_was_called = False

    def is_set(self) -> bool:
        self._called += 1
        return self._called > 1

    def wait(self, timeout: float | None = None):
        return None

    def set(self):
        self.set_was_called = True
        self._called = 99


def _run_deploy_once(
    monkeypatch,
    config,
    *,
    workdir: str | None = None,
    port: int | None = None,
    debug: bool = False,
    cwd: str | None = None,
    port_occupied: bool = False,
    langgraph_dev_running: bool = True,  # health-check passes after start
    tunnel: bool = False,
    tunnel_url: str | None = None,
):
    """Run ``deploy()`` end-to-end with all external dependencies mocked.
    Returns a ``captured`` dict with observation points."""
    import EvoScientist.config as config_mod

    captured: dict[str, Any] = {
        "ccproxy_started": False,
        "ccproxy_stopped": False,
        "langgraph_dev_started": False,
        "langgraph_dev_stopped": False,
        "deploy_mode_passed": None,
        "workspace_passed": None,
        "port_passed": None,
        "atexit_callbacks": [],
    }

    def _fake_get_effective_config(cli_overrides=None):
        captured["cli_overrides"] = dict(cli_overrides or {})
        merged = vars(config).copy()
        merged.update(cli_overrides or {})
        return SimpleNamespace(**merged)

    monkeypatch.setattr(config_mod, "get_effective_config", _fake_get_effective_config)
    monkeypatch.setattr(config_mod, "apply_config_to_env", lambda _cfg: None)

    monkeypatch.setattr(deploy_server, "console", _SilentConsole())

    # Workspace setup mocks
    from EvoScientist import paths as paths_mod

    monkeypatch.setattr(paths_mod, "set_workspace_root", lambda _p: None)
    monkeypatch.setattr(paths_mod, "ensure_dirs", lambda: None)

    # langgraph_dev.manager mocks
    from EvoScientist.langgraph_dev import manager as lgm

    monkeypatch.setattr(lgm, "_is_port_occupied", lambda _p: port_occupied)
    monkeypatch.setattr(
        lgm,
        "is_langgraph_dev_running",
        lambda **_kw: langgraph_dev_running,
    )

    def _fake_start_langgraph_dev(
        workspace_dir=None,
        *,
        port=None,
        file_persistence=True,
        jobs_per_worker=10,
        deploy_mode=False,
        tunnel=False,
    ):
        captured["langgraph_dev_started"] = True
        captured["workspace_passed"] = str(workspace_dir) if workspace_dir else None
        captured["port_passed"] = port
        captured["deploy_mode_passed"] = deploy_mode
        captured["jobs_per_worker_passed"] = jobs_per_worker
        captured["file_persistence_passed"] = file_persistence
        captured["tunnel_passed"] = tunnel
        return SimpleNamespace(pid=99999)

    def _fake_stop_langgraph_dev(_proc=None):
        captured["langgraph_dev_stopped"] = True

    monkeypatch.setattr(lgm, "start_langgraph_dev", _fake_start_langgraph_dev)
    monkeypatch.setattr(lgm, "stop_langgraph_dev", _fake_stop_langgraph_dev)
    # Never poll a real log for the tunnel URL in tests.
    monkeypatch.setattr(
        lgm, "read_tunnel_url", lambda *a, **k: tunnel_url, raising=False
    )

    # ccproxy mocks
    from EvoScientist import ccproxy_manager as ccp

    def _fake_maybe_start_ccproxy(_cfg):
        captured["ccproxy_started"] = True
        return SimpleNamespace(pid=88888)

    def _fake_stop_ccproxy(_proc):
        captured["ccproxy_stopped"] = True

    monkeypatch.setattr(ccp, "maybe_start_ccproxy", _fake_maybe_start_ccproxy)
    monkeypatch.setattr(ccp, "stop_ccproxy", _fake_stop_ccproxy)

    # atexit mock — capture without executing (don't pollute test process)
    import atexit

    def _fake_atexit_register(fn, *args, **kwargs):
        captured["atexit_callbacks"].append((fn.__name__, args, kwargs))
        return fn

    monkeypatch.setattr(atexit, "register", _fake_atexit_register)

    # signal mock — capture handlers so tests can exercise _handle_shutdown.
    # Returning a no-op original means deploy()'s finally block restores
    # something harmless onto the real signal module.
    import signal

    # deploy() calls signal.signal twice per signum: first to install
    # _handle_shutdown, then in finally to restore the original. We want the
    # first (real) handler — keep first-write-wins semantics.
    captured["signal_handlers"] = {}

    def _capture_signal(signum, handler):
        if signum not in captured["signal_handlers"]:
            captured["signal_handlers"][signum] = handler
        return lambda *_a, **_kw: None

    monkeypatch.setattr(signal, "signal", _capture_signal)

    # threading.Event mock — exits the wait loop after one iteration;
    # factory captures the instance so tests can inspect set() calls.
    import threading

    def _make_event():
        ev = _ImmediateEvent()
        captured["event_instance"] = ev
        return ev

    monkeypatch.setattr(threading, "Event", _make_event)

    # os.makedirs / os.getcwd
    import os

    monkeypatch.setattr(os, "makedirs", lambda *a, **k: None)
    if cwd is not None:
        monkeypatch.setattr(os, "getcwd", lambda: cwd)

    deploy_server.deploy(workdir=workdir, port=port, debug=debug, tunnel=tunnel)
    return captured


class _SilentConsole:
    """Stand-in for the Rich console — swallows all output so test runs
    don't spew ANSI to the captured pytest output (but doesn't break the
    code paths that call ``console.print`` / ``console.status``)."""

    def print(self, *args, **kwargs):
        pass

    def status(self, *args, **kwargs):
        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

        return _Ctx()


# =============================================================================
# Tests
# =============================================================================


def test_deploy_starts_langgraph_dev_with_deploy_mode_true(monkeypatch, tmp_path):
    config = _make_config(default_workdir=str(tmp_path))
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["langgraph_dev_started"] is True
    assert captured["deploy_mode_passed"] is True, (
        "deploy command MUST call start_langgraph_dev with deploy_mode=True"
    )


def test_deploy_tunnel_default_off(monkeypatch, tmp_path):
    """Without ``--tunnel``, start_langgraph_dev is called with tunnel=False."""
    config = _make_config(default_workdir=str(tmp_path))
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["tunnel_passed"] is False


def test_deploy_tunnel_flag_passed_through(monkeypatch, tmp_path):
    """``--tunnel`` propagates to start_langgraph_dev(tunnel=True)."""
    config = _make_config(default_workdir=str(tmp_path))
    captured = _run_deploy_once(
        monkeypatch,
        config,
        tunnel=True,
        tunnel_url="https://demo-xyz.trycloudflare.com",
    )

    assert captured["tunnel_passed"] is True


def test_deploy_workdir_cli_arg_beats_config(monkeypatch, tmp_path):
    cli_ws = tmp_path / "cli_ws"
    cfg_ws = tmp_path / "cfg_ws"
    config = _make_config(default_workdir=str(cfg_ws))
    captured = _run_deploy_once(monkeypatch, config, workdir=str(cli_ws))

    assert captured["workspace_passed"] == str(cli_ws)


def test_deploy_workdir_config_beats_cwd(monkeypatch, tmp_path):
    cfg_ws = tmp_path / "cfg_ws"
    config = _make_config(default_workdir=str(cfg_ws))
    captured = _run_deploy_once(monkeypatch, config, cwd="/tmp/should_not_be_used")

    assert captured["workspace_passed"] == str(cfg_ws)


def test_deploy_workdir_falls_back_to_cwd(monkeypatch, tmp_path):
    config = _make_config(default_workdir="")
    cwd = str(tmp_path / "cwd")
    captured = _run_deploy_once(monkeypatch, config, cwd=cwd)

    assert captured["workspace_passed"] == cwd


def test_deploy_port_cli_arg_beats_config(monkeypatch, tmp_path):
    config = _make_config(default_workdir=str(tmp_path), langgraph_dev_port=6174)
    captured = _run_deploy_once(monkeypatch, config, port=7000)

    assert captured["port_passed"] == 7000


def test_deploy_port_defaults_to_config(monkeypatch, tmp_path):
    config = _make_config(default_workdir=str(tmp_path), langgraph_dev_port=6543)
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["port_passed"] == 6543


@pytest.mark.parametrize("bad_port", [0, -1, 70000])
def test_deploy_refuses_invalid_port(monkeypatch, tmp_path, bad_port):
    """CLI must reject out-of-range ports (port=0 was the original silent-fail
    case: ``port or default`` treated 0 as falsy)."""
    config = _make_config(default_workdir=str(tmp_path))
    with pytest.raises(typer.Exit) as exc:
        _run_deploy_once(monkeypatch, config, port=bad_port)
    assert exc.value.exit_code == 1


def test_deploy_no_ccproxy_when_api_key_auth(monkeypatch, tmp_path):
    config = _make_config(
        default_workdir=str(tmp_path),
        anthropic_auth_mode="api_key",
        openai_auth_mode="api_key",
    )
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["ccproxy_started"] is False


def test_deploy_starts_ccproxy_when_anthropic_oauth(monkeypatch, tmp_path):
    config = _make_config(default_workdir=str(tmp_path), anthropic_auth_mode="oauth")
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["ccproxy_started"] is True


def test_deploy_starts_ccproxy_when_openai_oauth(monkeypatch, tmp_path):
    config = _make_config(default_workdir=str(tmp_path), openai_auth_mode="oauth")
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["ccproxy_started"] is True


def test_deploy_refuses_when_port_occupied_by_evosci(monkeypatch, tmp_path):
    """If port is occupied AND it's serving an EvoSci langgraph dev,
    refuse with exit code 1."""
    config = _make_config(default_workdir=str(tmp_path))
    with pytest.raises(typer.Exit) as exc:
        _run_deploy_once(
            monkeypatch,
            config,
            port_occupied=True,
            langgraph_dev_running=True,  # /ok responds → existing EvoSci instance
        )
    assert exc.value.exit_code == 1


def test_deploy_refuses_when_port_occupied_by_foreign(monkeypatch, tmp_path):
    """If port is occupied but /ok doesn't respond, treat as foreign process
    and refuse."""
    config = _make_config(default_workdir=str(tmp_path))
    with pytest.raises(typer.Exit) as exc:
        _run_deploy_once(
            monkeypatch,
            config,
            port_occupied=True,
            langgraph_dev_running=False,  # foreign process holds the port
        )
    assert exc.value.exit_code == 1


def test_deploy_registers_cleanup_for_langgraph_dev(monkeypatch, tmp_path):
    config = _make_config(default_workdir=str(tmp_path))
    captured = _run_deploy_once(monkeypatch, config)

    names = [name for (name, _args, _kw) in captured["atexit_callbacks"]]
    assert "_fake_stop_langgraph_dev" in names, (
        "deploy must register stop_langgraph_dev via atexit for clean shutdown"
    )


def test_deploy_registers_cleanup_for_ccproxy_when_oauth(monkeypatch, tmp_path):
    config = _make_config(default_workdir=str(tmp_path), anthropic_auth_mode="oauth")
    captured = _run_deploy_once(monkeypatch, config)

    names = [name for (name, _args, _kw) in captured["atexit_callbacks"]]
    assert "_fake_stop_ccproxy" in names


def test_deploy_registers_signal_handlers_for_shutdown(monkeypatch, tmp_path):
    """deploy() must register SIGINT and SIGTERM handlers so external signals
    trigger clean shutdown via the shutdown_event wait loop."""
    import signal

    config = _make_config(default_workdir=str(tmp_path))
    captured = _run_deploy_once(monkeypatch, config)

    handlers = captured["signal_handlers"]
    assert signal.SIGINT in handlers, "deploy must register a SIGINT handler"
    assert signal.SIGTERM in handlers, "deploy must register a SIGTERM handler"
    assert callable(handlers[signal.SIGINT])
    assert callable(handlers[signal.SIGTERM])


def test_handle_shutdown_sigterm_sets_shutdown_event(monkeypatch, tmp_path):
    """Invoking the captured SIGTERM handler must set deploy()'s shutdown_event.

    SIGTERM is used (not SIGINT) because the SIGINT branch of _handle_shutdown
    calls signal.default_int_handler which raises KeyboardInterrupt — that
    would terminate the test process rather than exercise the event path.
    """
    import signal

    config = _make_config(default_workdir=str(tmp_path))
    captured = _run_deploy_once(monkeypatch, config)

    event = captured["event_instance"]
    assert event is not None, "deploy() must have constructed a threading.Event"
    # During the normal helper run the wait loop exits via is_set() flipping
    # to True (not via set()), so set_was_called should still be False here.
    assert event.set_was_called is False, (
        "Sanity check: helper's _ImmediateEvent should exit naturally without "
        "set() being called; if this fails, the helper changed behavior."
    )

    sigterm_handler = captured["signal_handlers"][signal.SIGTERM]
    sigterm_handler(signal.SIGTERM, None)

    assert event.set_was_called is True, (
        "_handle_shutdown(SIGTERM, None) must call shutdown_event.set()"
    )
