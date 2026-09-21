"""Tests for the shell-agnostic launcher core (``EvoScientist.deploy.launcher``).

These pin the pieces that moved out of ``run_webui`` so they can be reused by a
desktop shell: the backend reuse/start decision and its error-code taxonomy,
the secret-scrubbing env, the front-end runners' preflight, readiness polling,
and the JSON ready/error signal from the standalone entrypoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from EvoScientist.deploy import launcher as lm
from EvoScientist.langgraph_dev import manager as lgm


def _cfg(workspace_dir: str = "/tmp/wsA", **kw):
    base = {
        "workspace_dir": workspace_dir,
        "backend_host": "127.0.0.1",
        "backend_port": 6174,
        "webui_host": "127.0.0.1",
        "webui_port": 4716,
        **kw,
    }
    return lm.LauncherConfig(**base)


# --------------------------------------------------------------------------- #
# _resolve_backend — decision + error-code taxonomy
# --------------------------------------------------------------------------- #
def _patch_backend_probes(
    monkeypatch, *, occupied: bool, running: bool, sidecar=None, fingerprint="fp-now"
):
    monkeypatch.setattr(lgm, "_is_port_occupied", lambda *_a, **_k: occupied)
    monkeypatch.setattr(lgm, "is_langgraph_dev_running", lambda **_k: running)
    monkeypatch.setattr(lgm, "_read_workspace_sidecar", lambda: sidecar)
    monkeypatch.setattr(lgm, "_server_config_fingerprint", lambda _c: fingerprint)


def test_resolve_backend_free_port_starts(monkeypatch):
    _patch_backend_probes(monkeypatch, occupied=False, running=False)
    decision = lm._resolve_backend(_cfg(), object())
    assert decision.action == "start"
    assert decision.warnings == []


def test_resolve_backend_foreign_occupant_is_port_conflict(monkeypatch):
    _patch_backend_probes(monkeypatch, occupied=True, running=False)
    with pytest.raises(lm.LauncherError) as ei:
        lm._resolve_backend(_cfg(), object())
    assert ei.value.code == "port_conflict"


def test_resolve_backend_other_workspace_is_mismatch(monkeypatch):
    _patch_backend_probes(
        monkeypatch,
        occupied=True,
        running=True,
        sidecar={"workspace": "/tmp/wsB", "deploy_mode": True},
    )
    with pytest.raises(lm.LauncherError) as ei:
        lm._resolve_backend(_cfg(workspace_dir="/tmp/wsA"), object())
    assert ei.value.code == "workspace_mismatch"


def test_resolve_backend_stripped_server_is_refused(monkeypatch):
    _patch_backend_probes(
        monkeypatch,
        occupied=True,
        running=True,
        sidecar={"workspace": "/tmp/wsA", "deploy_mode": False},
    )
    with pytest.raises(lm.LauncherError) as ei:
        lm._resolve_backend(_cfg(workspace_dir="/tmp/wsA"), object())
    assert ei.value.code == "stripped_backend"


def test_resolve_backend_same_workspace_reuses(monkeypatch):
    _patch_backend_probes(
        monkeypatch,
        occupied=True,
        running=True,
        sidecar={
            "workspace": "/tmp/wsA",
            "deploy_mode": True,
            "config_fingerprint": "fp-now",
        },
    )
    decision = lm._resolve_backend(_cfg(workspace_dir="/tmp/wsA"), object())
    assert decision.action == "reuse"
    assert decision.warnings == []


def test_resolve_backend_fingerprint_drift_reuses_with_warning(monkeypatch):
    _patch_backend_probes(
        monkeypatch,
        occupied=True,
        running=True,
        sidecar={
            "workspace": "/tmp/wsA",
            "deploy_mode": True,
            "config_fingerprint": "fp-old",
        },
        fingerprint="fp-now",
    )
    decision = lm._resolve_backend(_cfg(workspace_dir="/tmp/wsA"), object())
    assert decision.action == "reuse"
    assert any("Config changed" in w for w in decision.warnings)


def test_resolve_backend_no_sidecar_reuses(monkeypatch):
    """An older subprocess with no sidecar is reused, as before."""
    _patch_backend_probes(monkeypatch, occupied=True, running=True, sidecar=None)
    decision = lm._resolve_backend(_cfg(), object())
    assert decision.action == "reuse"


# --------------------------------------------------------------------------- #
# Auto-port — collision recovery for GUI shells (no terminal to act on it)
# --------------------------------------------------------------------------- #
def test_find_free_port_scans_upward(monkeypatch):
    occupied = {6174, 6175}
    monkeypatch.setattr(
        lgm, "_is_port_occupied", lambda port, *a, **k: port in occupied
    )
    assert lm._find_free_port(6174, "127.0.0.1") == 6176


def test_find_free_port_exhausted_raises(monkeypatch):
    monkeypatch.setattr(lgm, "_is_port_occupied", lambda *a, **k: True)
    with pytest.raises(lm.LauncherError) as ei:
        lm._find_free_port(6174, "127.0.0.1", limit=3)
    assert ei.value.code == "port_conflict"


class _FakeProc:
    def poll(self):
        return None


class _FakeRunner:
    handles_browser_open = False

    def preflight(self, cfg):
        pass

    def start(self, cfg, env):
        return _FakeProc()

    def stop(self, proc):
        pass


def _patch_for_start(monkeypatch, occupied_ports):
    """Foreign occupant on ``occupied_ports``; every other port free."""
    occ = set(occupied_ports)
    monkeypatch.setattr(lgm, "_is_port_occupied", lambda port, *a, **k: port in occ)
    monkeypatch.setattr(lgm, "is_langgraph_dev_running", lambda **k: False)
    monkeypatch.setattr(lgm, "_read_workspace_sidecar", lambda: None)
    monkeypatch.setattr(lgm, "_server_config_fingerprint", lambda _c: "fp")
    monkeypatch.setattr(lgm, "start_langgraph_dev", lambda **k: _FakeProc())


def test_start_auto_ports_off_occupied_backend(monkeypatch):
    _patch_for_start(monkeypatch, {6174})  # default backend port taken (foreign)
    launcher = lm.WebUILauncher(object(), _cfg(auto_port=True), _FakeRunner())
    result = launcher.start()
    assert launcher._cfg.backend_port == 6175  # moved to next free
    assert launcher._cfg.webui_port == 4716  # webui untouched (was free)
    assert result.backend_started is True
    assert "6175" in result.backend_url
    assert any("port_conflict" in w and "6175" in w for w in result.warnings)


def test_start_without_auto_port_raises_on_conflict(monkeypatch):
    """CLI (auto_port off) still gets the explicit conflict error."""
    _patch_for_start(monkeypatch, {6174})
    launcher = lm.WebUILauncher(object(), _cfg(auto_port=False), _FakeRunner())
    with pytest.raises(lm.LauncherError) as ei:
        launcher.start()
    assert ei.value.code == "port_conflict"


def _patch_for_evosci_occupant(monkeypatch, sidecar, occupied=(6174,)):
    """An EvoSci langgraph dev occupies ``occupied`` with ``sidecar``.

    Any other port is free, so an auto-port fallback resolves to ``start``.
    """
    occ = set(occupied)
    monkeypatch.setattr(lgm, "_is_port_occupied", lambda port, *a, **k: port in occ)
    monkeypatch.setattr(lgm, "is_langgraph_dev_running", lambda **k: True)
    monkeypatch.setattr(lgm, "_read_workspace_sidecar", lambda: sidecar)
    monkeypatch.setattr(lgm, "_server_config_fingerprint", lambda _c: "fp")
    monkeypatch.setattr(lgm, "start_langgraph_dev", lambda **k: _FakeProc())


def test_start_auto_ports_off_workspace_mismatch(monkeypatch):
    """A GUI shell starts its own backend when the port serves another workspace."""
    _patch_for_evosci_occupant(
        monkeypatch, sidecar={"workspace": "/tmp/wsB", "deploy_mode": True}
    )
    launcher = lm.WebUILauncher(
        object(), _cfg(workspace_dir="/tmp/wsA", auto_port=True), _FakeRunner()
    )
    result = launcher.start()
    assert launcher._cfg.backend_port == 6175  # own backend on a free port
    assert result.backend_started is True
    assert any("workspace_mismatch" in w for w in result.warnings)


def test_start_auto_ports_off_stripped_backend(monkeypatch):
    """A GUI shell starts its own deploy-mode backend past a stripped server."""
    _patch_for_evosci_occupant(
        monkeypatch, sidecar={"workspace": "/tmp/wsA", "deploy_mode": False}
    )
    launcher = lm.WebUILauncher(
        object(), _cfg(workspace_dir="/tmp/wsA", auto_port=True), _FakeRunner()
    )
    result = launcher.start()
    assert launcher._cfg.backend_port == 6175
    assert any("stripped_backend" in w for w in result.warnings)


def test_start_without_auto_port_raises_on_workspace_mismatch(monkeypatch):
    """CLI (auto_port off) still refuses a different-workspace server."""
    _patch_for_evosci_occupant(
        monkeypatch, sidecar={"workspace": "/tmp/wsB", "deploy_mode": True}
    )
    launcher = lm.WebUILauncher(
        object(), _cfg(workspace_dir="/tmp/wsA", auto_port=False), _FakeRunner()
    )
    with pytest.raises(lm.LauncherError) as ei:
        launcher.start()
    assert ei.value.code == "workspace_mismatch"


def test_start_auto_ports_occupied_webui(monkeypatch):
    _patch_for_start(monkeypatch, {4716})  # webui port taken, backend free
    launcher = lm.WebUILauncher(object(), _cfg(auto_port=True), _FakeRunner())
    result = launcher.start()
    assert launcher._cfg.backend_port == 6174  # backend untouched
    assert launcher._cfg.webui_port == 4717  # moved to next free
    assert any("WebUI" in w for w in result.warnings)


def test_stop_process_tree_taskkill_no_console_window(monkeypatch):
    """taskkill runs with CREATE_NO_WINDOW so exit doesn't flash a console."""
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

    monkeypatch.setattr(lm.os, "name", "nt")
    monkeypatch.setattr(lm.subprocess, "run", _fake_run)
    # CREATE_NO_WINDOW is Windows-only; provide it on the (Linux) test host.
    monkeypatch.setattr(lm.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    class _Proc:
        pid = 4321

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    lm._stop_process_tree(_Proc())

    assert captured["cmd"][0] == "taskkill"
    assert captured["kwargs"]["creationflags"] == lm.subprocess.CREATE_NO_WINDOW


# --------------------------------------------------------------------------- #
# _scrubbed_env
# --------------------------------------------------------------------------- #
def test_scrubbed_env_strips_secrets_keeps_essentials(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    monkeypatch.setenv("SOME_TOKEN", "t")
    monkeypatch.setenv("DB_PASSWORD", "p")
    monkeypatch.setenv("ANTHROPIC_KEY", "k")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("NODE_OPTIONS", "--max-old-space-size=4096")

    env = lm._scrubbed_env({"PORT": "4716"})

    assert "OPENROUTER_API_KEY" not in env
    assert "SOME_TOKEN" not in env
    assert "DB_PASSWORD" not in env
    assert "ANTHROPIC_KEY" not in env  # matched by the *_KEY suffix rule
    assert env["PATH"] == "/usr/bin"
    assert env["NODE_OPTIONS"] == "--max-old-space-size=4096"
    assert env["PORT"] == "4716"


# --------------------------------------------------------------------------- #
# Runner preflight
# --------------------------------------------------------------------------- #
def test_npx_runner_preflight_reports_node_missing(monkeypatch):
    monkeypatch.setattr(lm.shutil, "which", lambda _n: None)
    with pytest.raises(lm.LauncherError) as ei:
        lm.NpxWebUIRunner().preflight(_cfg())
    assert ei.value.code == "node_missing"


def test_bundled_runner_preflight_missing_node(tmp_path):
    runner = lm.BundledWebUIRunner(app_dir=tmp_path, node_exe=tmp_path / "node")
    with pytest.raises(lm.LauncherError) as ei:
        runner.preflight(_cfg())
    assert ei.value.code == "node_missing"


def test_bundled_runner_preflight_missing_server(tmp_path):
    node = tmp_path / "node"
    node.write_text("#!/bin/sh\n")
    runner = lm.BundledWebUIRunner(app_dir=tmp_path, node_exe=node)
    with pytest.raises(lm.LauncherError) as ei:
        runner.preflight(_cfg())
    assert ei.value.code == "node_missing"
    assert "server" in ei.value.message.lower()


def test_bundled_runner_preflight_ok(tmp_path):
    node = tmp_path / "node"
    node.write_text("#!/bin/sh\n")
    server = tmp_path / "dist" / "server.js"
    server.parent.mkdir(parents=True)
    server.write_text("// server")
    lm.BundledWebUIRunner(app_dir=tmp_path, node_exe=node).preflight(_cfg())


def _bundled_runner_files(tmp_path):
    node = tmp_path / "node"
    node.write_text("#!/bin/sh\n")
    server = tmp_path / "dist" / "server.js"
    server.parent.mkdir(parents=True)
    server.write_text("// server")
    return node


class _DoneProc:
    """A process already exited, so ``_stop_process_tree`` returns early."""

    def poll(self):
        return 0


def test_bundled_runner_redirects_node_output_to_log(monkeypatch, tmp_path):
    node = _bundled_runner_files(tmp_path)
    log = tmp_path / "logs" / "webui.log"
    captured = {}

    def _fake_popen(cmd, **kw):
        captured["kw"] = kw
        return _DoneProc()

    monkeypatch.setattr(lm.subprocess, "Popen", _fake_popen)
    runner = lm.BundledWebUIRunner(app_dir=tmp_path, node_exe=node, log_path=log)
    proc = runner.start(_cfg(), {"PORT": "4716"})

    assert captured["kw"]["stdout"] is runner._log_fh
    assert captured["kw"]["stderr"] == lm.subprocess.STDOUT
    assert log.exists()  # parent dir created + file opened for the node output

    runner.stop(proc)
    assert runner._log_fh is None  # handle closed on stop


def test_bundled_runner_without_log_path_does_not_redirect(monkeypatch, tmp_path):
    node = _bundled_runner_files(tmp_path)
    captured = {}

    def _fake_popen(cmd, **kw):
        captured["kw"] = kw
        return _DoneProc()

    monkeypatch.setattr(lm.subprocess, "Popen", _fake_popen)
    runner = lm.BundledWebUIRunner(app_dir=tmp_path, node_exe=node)
    runner.start(_cfg(), {"PORT": "4716"})
    assert "stdout" not in captured["kw"]  # output left to inherit, as before


# --------------------------------------------------------------------------- #
# Readiness polling
# --------------------------------------------------------------------------- #
def test_poll_ready_times_out(monkeypatch):
    class _DeadOpener:
        def open(self, *_a, **_k):
            raise ConnectionRefusedError("nope")

    monkeypatch.setattr(lm.urllib.request, "build_opener", lambda *_h: _DeadOpener())
    with pytest.raises(lm.LauncherError) as ei:
        lm._poll_ready("http://127.0.0.1:4716", timeout=0.05, interval=0.01)
    assert ei.value.code == "not_ready"


def test_poll_ready_bypasses_proxy(monkeypatch):
    # Must probe loopback with proxies disabled — on Windows the default opener
    # honours the system (registry) proxy and never reaches 127.0.0.1.
    captured = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Opener:
        def open(self, *_a, **_k):
            return _Resp()

    def _fake_build_opener(*handlers):
        captured["handlers"] = handlers
        return _Opener()

    monkeypatch.setattr(lm.urllib.request, "build_opener", _fake_build_opener)
    lm._poll_ready("http://127.0.0.1:4716", timeout=1)
    assert any(
        isinstance(h, lm.urllib.request.ProxyHandler) and h.proxies == {}
        for h in captured["handlers"]
    )


def test_wait_ready_times_out_when_backend_never_up(monkeypatch):
    monkeypatch.setattr(lgm, "is_langgraph_dev_running", lambda **_k: False)
    launcher = lm.WebUILauncher(object(), _cfg(), lm.NpxWebUIRunner())
    with pytest.raises(lm.LauncherError) as ei:
        launcher.wait_ready(timeout=0.05)
    assert ei.value.code == "not_ready"


# --------------------------------------------------------------------------- #
# Standalone JSON entrypoint
# --------------------------------------------------------------------------- #
def test_main_emits_error_json_on_launcher_error(monkeypatch, capsys):
    import EvoScientist.config as config_mod

    monkeypatch.setattr(
        config_mod, "get_effective_config", lambda: SimpleNamespace(default_workdir="")
    )
    monkeypatch.setattr(config_mod, "apply_config_to_env", lambda _c: None)
    monkeypatch.setattr(lm.os, "makedirs", lambda *a, **k: None)

    class _FailingLauncher:
        def __init__(self, *_a, **_k):
            pass

        def start(self):
            raise lm.LauncherError("stripped_backend", "boom", "do X")

        def stop(self):
            pass

    monkeypatch.setattr(lm, "WebUILauncher", _FailingLauncher)

    rc = lm.main(["--workspace", "/tmp/ws"])
    assert rc == 1

    out = capsys.readouterr()
    payload = json.loads(out.out.strip())
    assert payload["status"] == "error"
    assert payload["code"] == "stripped_backend"
    assert payload["message"] == "boom"
    assert payload["detail"] == "do X"
    # stderr carries the error for logs; stdout stays machine-parseable.
    assert "stripped_backend" in out.err


# --------------------------------------------------------------------------- #
# build_launcher_config resolution
# --------------------------------------------------------------------------- #
def test_build_launcher_config_resolves_ports_and_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(lm.os, "makedirs", lambda *a, **k: None)
    config = SimpleNamespace(
        default_workdir=str(tmp_path),
        langgraph_dev_port=6000,
        langgraph_dev_host="127.0.0.1",
        webui_port=4000,
        webui_host="0.0.0.0",
        langgraph_dev_keepalive=True,
    )
    cfg = lm.build_launcher_config(config, workspace_dir=None)
    assert cfg.workspace_dir == str(Path(tmp_path).resolve())
    assert cfg.backend_port == 6000
    assert cfg.webui_port == 4000
    assert cfg.webui_host == "0.0.0.0"
    assert cfg.keepalive is True
    assert cfg.open_browser is False


def test_build_launcher_config_blank_host_falls_back_to_loopback(monkeypatch):
    monkeypatch.setattr(lm.os, "makedirs", lambda *a, **k: None)
    config = SimpleNamespace(default_workdir="/tmp/x", webui_host="   ")
    cfg = lm.build_launcher_config(config, workspace_dir=None)
    assert cfg.webui_host == "127.0.0.1"
