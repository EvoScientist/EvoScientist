"""Tests for the shell-agnostic launcher core (``EvoScientist.deploy.launcher``).

These pin the pieces that moved out of ``run_webui`` so they can be reused by
other front-ends: the backend reuse/start decision and its error-code taxonomy,
the secret-scrubbing env, the front-end runners' preflight and readiness
polling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from EvoScientist.deploy import launcher as lm
from EvoScientist.langgraph_dev import manager as lgm
from tests.fakes.langgraph_dev import break_start_lock, start_lock_is_held


@pytest.fixture(autouse=True)
def _isolated_runtime(runtime_paths):
    """The start lock, pid file and sidecar under tmp_path."""


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
    monkeypatch,
    *,
    occupied: bool,
    running: bool,
    sidecar=None,
    fingerprint="fp-now",
    pid_serves=True,
):
    monkeypatch.setattr(lgm, "_is_port_occupied", lambda *_a, **_k: occupied)
    monkeypatch.setattr(lgm, "is_langgraph_dev_running", lambda **_k: running)
    monkeypatch.setattr(lgm, "_read_workspace_sidecar", lambda: sidecar)
    monkeypatch.setattr(lgm, "_server_config_fingerprint", lambda _c: fingerprint)
    monkeypatch.setattr(lgm, "_pid_serves_port", lambda *_a, **_k: pid_serves)


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
    assert ei.value.fields == {"reason": "foreign"}


def test_resolve_backend_other_project_is_mismatch(monkeypatch):
    """The front-ends launched here get the backend's default project."""
    _patch_backend_probes(
        monkeypatch, occupied=True, running=True, sidecar={"workspace": "/tmp/wsB"}
    )
    with pytest.raises(lm.LauncherError) as ei:
        lm._resolve_backend(_cfg(workspace_dir="/tmp/wsA"), object())
    assert ei.value.code == "workspace_mismatch"
    assert ei.value.fields == {"workspace": "/tmp/wsB"}
    # The way out, next to what it costs the other sessions.
    assert "EvoSci server stop" in ei.value.detail


def test_resolve_backend_older_run_folder_record_reuses(monkeypatch):
    """A record from before the shared backend may name a run folder of the
    same project."""
    _patch_backend_probes(
        monkeypatch,
        occupied=True,
        running=True,
        sidecar={
            "workspace": "/tmp/wsA",
            "run_dir": "/tmp/wsA/runs/x",
            "config_fingerprint": "fp-now",
        },
    )
    assert (
        lm._resolve_backend(_cfg(workspace_dir="/tmp/wsA"), object()).action == "reuse"
    )


def test_resolve_backend_same_workspace_reuses(monkeypatch):
    _patch_backend_probes(
        monkeypatch,
        occupied=True,
        running=True,
        sidecar={
            "workspace": "/tmp/wsA",
            "config_fingerprint": "fp-now",
        },
    )
    decision = lm._resolve_backend(_cfg(workspace_dir="/tmp/wsA"), object())
    assert decision.action == "reuse"
    assert decision.warnings == []


@pytest.mark.parametrize(
    "legacy",
    [{}, {"deploy_mode": False}],
    ids=["current", "stripped_from_older_version"],
)
def test_resolve_backend_fingerprint_drift_reuses_with_warning(monkeypatch, legacy):
    """A server from another version is reused with the drift warning, also
    one an older version started without MCP (``deploy_mode: false``)."""
    _patch_backend_probes(
        monkeypatch,
        occupied=True,
        running=True,
        sidecar={
            "workspace": "/tmp/wsA",
            "config_fingerprint": "fp-old",
            **legacy,
        },
        fingerprint="fp-now",
    )
    decision = lm._resolve_backend(_cfg(workspace_dir="/tmp/wsA"), object())
    assert decision.action == "reuse"
    assert any("EvoSci server stop" in w for w in decision.warnings)


def test_resolve_backend_reuse_returns_the_record_it_read(monkeypatch):
    sidecar = {"workspace": "/tmp/wsA", "config_fingerprint": "fp-now"}
    _patch_backend_probes(monkeypatch, occupied=True, running=True, sidecar=sidecar)
    decision = lm._resolve_backend(_cfg(workspace_dir="/tmp/wsA"), object())
    assert decision.sidecar is sidecar


def test_resolve_backend_record_less_server_reuses_by_default(monkeypatch):
    _patch_backend_probes(monkeypatch, occupied=True, running=True, sidecar=None)
    decision = lm._resolve_backend(_cfg(), object())
    assert decision.action == "reuse"
    assert decision.sidecar is None


def test_resolve_backend_record_less_server_is_port_conflict_when_required(
    monkeypatch,
):
    _patch_backend_probes(monkeypatch, occupied=True, running=True, sidecar=None)
    with pytest.raises(lm.LauncherError) as ei:
        lm._resolve_backend(_cfg(), object(), require_record=True)
    assert ei.value.code == "port_conflict"
    assert "no ownership record" in ei.value.message
    assert ei.value.fields == {"reason": "no_record"}


@pytest.mark.parametrize(
    ("recorded", "action"), [("/tmp/wsA", "reuse"), ("/tmp/wsB", None)]
)
def test_resolve_backend_takes_a_backend_target(monkeypatch, recorded, action):
    """A caller without a front-end passes only the backend fields."""
    _patch_backend_probes(
        monkeypatch,
        occupied=True,
        running=True,
        sidecar={"workspace": recorded, "config_fingerprint": "fp-now"},
    )
    target = lm.BackendTarget(
        workspace_dir="/tmp/wsA", backend_host="127.0.0.1", backend_port=6174
    )
    if action is None:
        with pytest.raises(lm.LauncherError) as ei:
            lm._resolve_backend(target, object())
        assert ei.value.code == "workspace_mismatch"
    else:
        assert lm._resolve_backend(target, object()).action == action


def test_launcher_error_fields_default_to_empty():
    assert lm.LauncherError("port_conflict", "m").fields == {}


def test_resolve_backend_no_drift_for_a_session_started_with_auto_flags(monkeypatch):
    """``EvoSci --auto-approve`` / ``--auto-mode`` override fields the server
    reads from config.yaml; a second launcher with the file's values must not
    report drift (and advise ``EvoSci server stop``)."""
    from EvoScientist.config.settings import EvoScientistConfig

    fingerprint = lgm._server_config_fingerprint
    recorded = fingerprint(
        EvoScientistConfig(auto_approve=True, auto_mode=True, enable_ask_user=False)
    )
    _patch_backend_probes(
        monkeypatch,
        occupied=True,
        running=True,
        sidecar={"workspace": "/tmp/wsA", "config_fingerprint": recorded},
    )
    monkeypatch.setattr(lgm, "_server_config_fingerprint", fingerprint)
    decision = lm._resolve_backend(_cfg(workspace_dir="/tmp/wsA"), EvoScientistConfig())
    assert decision.action == "reuse"
    assert decision.warnings == []


def test_resolve_backend_python_drift_reuses_with_warning(monkeypatch):
    from EvoScientist.setup import research_env

    _patch_backend_probes(
        monkeypatch,
        occupied=True,
        running=True,
        sidecar={
            "workspace": "/tmp/wsA",
            "config_fingerprint": "fp-now",
            "agent_python": "/conda/bin/python",
        },
    )
    monkeypatch.setattr(research_env, "agent_python", lambda: "/env/bin/python")
    decision = lm._resolve_backend(_cfg(workspace_dir="/tmp/wsA"), object())
    assert decision.action == "reuse"
    assert len(decision.warnings) == 1
    assert "/conda/bin/python" in decision.warnings[0]


@pytest.mark.parametrize(
    ("recorded", "own", "hinted"),
    [
        ("/conda/bin/python", None, False),
        (None, None, True),
        (None, "/usr/bin/python", False),
    ],
)
def test_start_hint_follows_the_reused_server(monkeypatch, recorded, own, hinted):
    from EvoScientist.setup import research_env

    sidecar = {"workspace": "/tmp/wsA", "pid": 1}
    _patch_for_evosci_occupant(monkeypatch, {**sidecar, "agent_python": recorded})
    monkeypatch.setattr(research_env, "agent_python", lambda: own)
    result = lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    assert (research_env.MISSING_PYTHON_HINT in result.warnings) is hinted


def test_resolve_backend_no_sidecar_reuses(monkeypatch):
    """An older subprocess with no sidecar is reused, as before —
    backward-compat for pre-sidecar / externally-managed servers."""
    _patch_backend_probes(monkeypatch, occupied=True, running=True, sidecar=None)
    decision = lm._resolve_backend(_cfg(), object())
    assert decision.action == "reuse"


def test_resolve_backend_sidecar_pid_not_serving_port_is_refused(monkeypatch):
    """A sidecar whose recorded PID does not serve this port (a launch on another
    port overwrote the global record) is rejected before its workspace is trusted."""
    _patch_backend_probes(
        monkeypatch,
        occupied=True,
        running=True,
        sidecar={"workspace": "/tmp/wsA", "pid": 12345},
        pid_serves=False,
    )
    with pytest.raises(lm.LauncherError) as ei:
        lm._resolve_backend(_cfg(workspace_dir="/tmp/wsA"), object())
    assert ei.value.code == "sidecar_port_mismatch"
    assert ei.value.fields == {"reason": "record_mismatch"}
    # The remedy must not misdirect to `EvoSci server stop` (which stops the
    # recorded server on another port, not the one occupying this one).
    assert "not this one" in (ei.value.detail or "")


@pytest.mark.parametrize(
    ("recorded", "requested", "reused"),
    [
        (None, "192.168.1.5", True),  # an older record: not checked
        ("127.0.0.1", "127.0.0.1", True),
        ("localhost", "127.0.0.1", True),
        ("::1", "localhost", True),
        ("127.0.0.1", "::1", False),  # two servers can hold the port on these
        ("::1", "127.0.0.1", False),
        ("0.0.0.0", "192.168.1.5", True),
        ("0.0.0.0", "::1", False),  # an IPv4 wildcard is not on ::1
        ("::", "::1", True),
        ("127.0.0.1", "0.0.0.0", True),  # probed at 127.0.0.1
        ("192.168.1.5", "127.0.0.1", False),
        ("127.0.0.1", "192.168.1.5", False),
        ("192.168.1.5", "0.0.0.0", False),
    ],
)
def test_resolve_backend_checks_the_recorded_bind_host(
    monkeypatch, recorded, requested, reused
):
    """Two servers can share a port on different interfaces: a record for the
    one the probe did not reach is refused before its workspace is trusted."""
    sidecar = {"workspace": "/tmp/wsA", "pid": 12345, "config_fingerprint": "fp-now"}
    if recorded is not None:
        sidecar["host"] = recorded
    _patch_backend_probes(monkeypatch, occupied=True, running=True, sidecar=sidecar)
    cfg = _cfg(workspace_dir="/tmp/wsA", backend_host=requested)
    if reused:
        assert lm._resolve_backend(cfg, object()).action == "reuse"
        return
    with pytest.raises(lm.LauncherError) as ei:
        lm._resolve_backend(cfg, object())
    assert ei.value.fields == {"reason": "record_mismatch"}


# --------------------------------------------------------------------------- #
# WebUILauncher.start — conflicts, teardown, concurrent stop
# --------------------------------------------------------------------------- #
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


def test_start_raises_on_foreign_occupant(monkeypatch):
    """A foreign process on the backend port is an explicit error."""
    _patch_for_start(monkeypatch, {6174})
    launcher = lm.WebUILauncher(object(), _cfg(), _FakeRunner())
    with pytest.raises(lm.LauncherError) as ei:
        launcher.start()
    assert ei.value.code == "port_conflict"


def test_start_passes_backend_allowed_origins(monkeypatch):
    _patch_for_start(monkeypatch, set())
    started: dict = {}
    monkeypatch.setattr(
        lgm, "start_langgraph_dev", lambda **kw: started.update(kw) or _FakeProc()
    )
    config = SimpleNamespace(backend_allowed_origins="https://smith.langchain.com")
    lm.WebUILauncher(config, _cfg(), _FakeRunner()).start()
    assert started["allowed_origins"] == "https://smith.langchain.com"


def test_start_registers_workspace_before_front_end(monkeypatch):
    from EvoScientist.langgraph_dev import registration

    _patch_for_start(monkeypatch, set())
    events: list = []
    monkeypatch.setattr(
        registration,
        "register_folder",
        lambda workspace, *, url: events.append(("register", workspace.root, url)),
    )

    class _RecordingRunner(_FakeRunner):
        def start(self, cfg, env):
            events.append(("front end",))
            return _FakeProc()

    lm.WebUILauncher(object(), _cfg(), _RecordingRunner()).start()

    assert events[0][:2] == ("register", Path("/tmp/wsA").resolve())
    assert events[1] == ("front end",)


def test_start_registration_failure_tears_the_backend_down(monkeypatch):
    from EvoScientist.langgraph_dev import registration

    _patch_for_start(monkeypatch, set())
    stopped: list = []
    monkeypatch.setattr(
        lgm, "stop_langgraph_dev", lambda proc=None: stopped.append(proc)
    )

    def fail(_workspace, *, url):
        raise registration.RegistrationError("setup failed")

    monkeypatch.setattr(registration, "register_folder", fail)
    launcher = lm.WebUILauncher(object(), _cfg(), _FakeRunner())

    with pytest.raises(lm.LauncherError) as ei:
        launcher.start()
    assert ei.value.code == "registration_failed"
    assert len(stopped) == 1


def test_start_failure_after_backend_start_tears_the_backend_down(monkeypatch):
    """A front-end failure after the backend was spawned must not leave that
    backend running: start()'s error path tears down what it started."""
    _patch_for_start(monkeypatch, set())
    stopped: list = []
    monkeypatch.setattr(
        lgm, "stop_langgraph_dev", lambda proc=None: stopped.append(proc)
    )

    class _FailingRunner(_FakeRunner):
        def start(self, cfg, env):
            raise RuntimeError("node failed to spawn")

    launcher = lm.WebUILauncher(object(), _cfg(), _FailingRunner())
    with pytest.raises(RuntimeError, match="node failed"):
        launcher.start()
    assert len(stopped) == 1
    assert isinstance(stopped[0], _FakeProc)
    assert launcher._backend_proc is None


def test_stop_during_blocked_backend_start_cancels_and_tears_down(monkeypatch):
    """A real concurrent stop() while start_langgraph_dev is still blocking: the
    in-flight fallback fires, and once the call returns start() aborts with
    ``cancelled`` and stops the late-wired backend instead of carrying on."""
    import threading

    _patch_for_start(monkeypatch, set())
    entered, release = threading.Event(), threading.Event()
    spawned = _FakeProc()

    def _blocking_start(**_k):
        entered.set()
        release.wait(5)
        return spawned

    monkeypatch.setattr(lgm, "start_langgraph_dev", _blocking_start)
    inflight: list[int] = []
    monkeypatch.setattr(
        lgm, "stop_inflight_owned_server", lambda: (inflight.append(1), None)[1]
    )
    stopped: list = []
    monkeypatch.setattr(
        lgm, "stop_langgraph_dev", lambda proc=None: stopped.append(proc)
    )

    launcher = lm.WebUILauncher(object(), _cfg(), _FakeRunner())
    errors: list[BaseException] = []

    def _boot():
        try:
            launcher.start()
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=_boot)
    t.start()
    assert entered.wait(5)
    launcher.stop()  # close from another thread while the backend start blocks
    release.set()
    t.join(5)
    assert not t.is_alive()
    assert inflight == [1]  # mid-start fallback fired from stop()
    assert isinstance(errors[0], lm.LauncherError)
    assert errors[0].code == "cancelled"
    assert stopped == [spawned]  # late-wired backend torn down by start()


def test_stop_during_backend_health_wait_reports_cancelled(monkeypatch):
    import threading

    _patch_for_start(monkeypatch, set())
    entered, killed = threading.Event(), threading.Event()

    def _start_killed_mid_health_wait(**_k):
        entered.set()
        killed.wait(5)
        raise RuntimeError("langgraph dev exited immediately with code -15.")

    monkeypatch.setattr(lgm, "start_langgraph_dev", _start_killed_mid_health_wait)
    monkeypatch.setattr(
        lgm, "stop_inflight_owned_server", lambda: (killed.set(), 4321)[1]
    )
    launcher = lm.WebUILauncher(object(), _cfg(), _FakeRunner())
    errors: list[BaseException] = []

    def _boot():
        try:
            launcher.start()
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=_boot)
    t.start()
    assert entered.wait(5)
    launcher.stop()
    t.join(5)
    assert not t.is_alive()
    assert errors[0].code == "cancelled"


def test_start_waits_for_another_start_before_the_backend_decision(monkeypatch):
    """Another start holds the server start lock: the decision and the start
    wait for it, so a backend that is bound but not answering yet is never
    taken for a stale one."""
    import threading

    from tests.fakes.langgraph_dev import LockHolder

    _patch_for_start(monkeypatch, set())
    holder = LockHolder()
    checks: list[bool] = []
    monkeypatch.setattr(
        lgm,
        "_is_port_occupied",
        lambda *_a, **_k: checks.append(holder.release.is_set()) or False,
    )
    threading.Timer(0.3, holder.done).start()
    result = lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    assert checks
    assert all(checks)
    assert result.backend_started is True


def test_the_backend_starts_under_the_start_lock(monkeypatch):
    _patch_for_start(monkeypatch, set())
    held: list[bool] = []
    monkeypatch.setattr(
        lgm,
        "start_langgraph_dev",
        lambda **_k: held.append(start_lock_is_held()) or _FakeProc(),
    )
    lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    assert held == [True]
    assert not start_lock_is_held()


@pytest.mark.parametrize("then_free", [False, True])
def test_stop_while_waiting_for_the_start_lock_starts_nothing(monkeypatch, then_free):
    """A stop() during the wait starts nothing, also when the lock is freed
    right after it."""
    import threading

    from filelock import FileLock

    from tests.fakes.langgraph_dev import LockHolder

    _patch_for_start(monkeypatch, set())
    started: list = []
    monkeypatch.setattr(
        lgm, "start_langgraph_dev", lambda **k: started.append(k) or _FakeProc()
    )
    holder = LockHolder()
    launcher = lm.WebUILauncher(object(), _cfg(), _FakeRunner())

    def _stop() -> None:
        launcher.stop()
        if then_free:
            holder.done()

    threading.Timer(0.3, _stop).start()
    try:
        with pytest.raises(lm.LauncherError) as ei:
            launcher.start()
        assert then_free or holder.thread.is_alive()
    finally:
        holder.done()
    assert ei.value.code == "cancelled"
    assert started == []
    lock = FileLock(str(lgm.RUNTIME.lock_file))
    lock.acquire(timeout=0.5)
    lock.release()


def test_start_lock_held_too_long_is_backend_start_failed(monkeypatch):
    from tests.fakes.langgraph_dev import LockHolder

    _patch_for_start(monkeypatch, set())
    monkeypatch.setattr(lgm, "_FILE_LOCK_TIMEOUT", 0.2)
    holder = LockHolder()
    try:
        with pytest.raises(lm.LauncherError) as ei:
            lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    finally:
        holder.done()
    assert ei.value.code == "backend_start_failed"
    assert ei.value.fields == {"lock_file": str(lgm.RUNTIME.lock_file)}


@pytest.mark.parametrize("step", ["mkdir", "acquire"])
def test_start_lock_os_error_is_backend_start_failed(monkeypatch, tmp_path, step):
    """An OSError from the start lock is backend_start_failed, as before the
    lock, so run_webui shows its panel instead of a traceback."""
    _patch_for_start(monkeypatch, set())
    started: list = []
    monkeypatch.setattr(
        lgm, "start_langgraph_dev", lambda **k: started.append(k) or _FakeProc()
    )
    break_start_lock(monkeypatch, tmp_path, step)
    with pytest.raises(lm.LauncherError) as ei:
        lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    assert ei.value.code == "backend_start_failed"
    assert ei.value.message.startswith("langgraph dev startup failed: ")
    assert isinstance(ei.value.__cause__, OSError)
    assert started == []


@pytest.mark.parametrize("held", [False, True])
def test_start_tells_the_caller_it_waits_for_another_start(monkeypatch, held):
    """on_lock_wait hears True while another start holds the lock and False
    once this start has it, before the backend decision; nothing when the
    lock is free."""
    from tests.fakes.langgraph_dev import LockHolder

    _patch_for_start(monkeypatch, set())
    events: list = []
    monkeypatch.setattr(
        lgm,
        "_is_port_occupied",
        lambda *_a, **_k: events.append("port check") or False,
    )
    holder = LockHolder() if held else None

    def _on_lock_wait(waiting: bool) -> None:
        events.append(waiting)
        if waiting and holder is not None:
            holder.release.set()

    try:
        launcher = lm.WebUILauncher(
            object(), _cfg(), _FakeRunner(), on_lock_wait=_on_lock_wait
        )
        launcher.start()
    finally:
        if holder is not None:
            holder.done()
    if held:
        assert events[:2] == [True, False]
        events = events[2:]
    assert events
    assert set(events) == {"port check"}


def test_stop_mid_backend_start_stops_only_owned_process(monkeypatch):
    """stop() during start_langgraph_dev's run (proc handle not wired yet) tears
    down only the process THIS launcher spawned — never the on-disk recorded
    server, which before our own Popen still names a different session's backend
    (e.g. one on another port). So a stop mid-boot can't kill an unrelated
    session's backend."""
    calls: list[int] = []
    # The mid-start fallback must use the owned-process stop, not the disk one.
    monkeypatch.setattr(
        lgm, "stop_inflight_owned_server", lambda: (calls.append(1), 4321)[1]
    )
    monkeypatch.setattr(
        lgm,
        "stop_recorded_server",
        lambda: calls.append("disk"),  # must NOT fire
    )
    launcher = lm.WebUILauncher(object(), _cfg(), _FakeRunner())
    # Simulate being inside _start_backend's blocking call: start initiated,
    # no proc handle assigned yet.
    launcher._backend_start_initiated = True
    launcher.stop()
    assert calls == [1]  # owned-process fallback fired, disk path did not
    assert launcher._backend_start_initiated is False
    launcher.stop()  # idempotent — does not fire again
    assert calls == [1]


def test_stop_after_reuse_leaves_backend_running(monkeypatch):
    _patch_for_evosci_occupant(monkeypatch, sidecar={"workspace": "/tmp/wsA", "pid": 1})
    calls: list = []
    monkeypatch.setattr(lgm, "stop_langgraph_dev", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(lgm, "stop_inflight_owned_server", lambda: calls.append(0))
    launcher = lm.WebUILauncher(object(), _cfg(), _FakeRunner())
    assert launcher.start().backend_started is False
    launcher.stop()
    assert calls == []


def _patch_for_evosci_occupant(monkeypatch, sidecar, occupied=(6174,)):
    """An EvoSci langgraph dev occupies ``occupied`` with ``sidecar``."""
    occ = set(occupied)
    monkeypatch.setattr(lgm, "_is_port_occupied", lambda port, *a, **k: port in occ)
    monkeypatch.setattr(lgm, "is_langgraph_dev_running", lambda **k: True)
    monkeypatch.setattr(lgm, "_read_workspace_sidecar", lambda: sidecar)
    monkeypatch.setattr(lgm, "_server_config_fingerprint", lambda _c: "fp")
    # The sidecar genuinely describes the server on this port; these tests
    # exercise the workspace checks past the PID-serves-port guard.
    monkeypatch.setattr(lgm, "_pid_serves_port", lambda *a, **k: True)
    monkeypatch.setattr(lgm, "start_langgraph_dev", lambda **k: _FakeProc())


def test_start_refuses_server_started_for_another_project(monkeypatch):
    """The WebUI shows the project of the client that started the backend;
    no second backend starts."""
    _patch_for_evosci_occupant(monkeypatch, sidecar={"workspace": "/tmp/wsB"})
    started: list = []
    monkeypatch.setattr(lgm, "start_langgraph_dev", lambda **k: started.append(k))
    launcher = lm.WebUILauncher(object(), _cfg(workspace_dir="/tmp/wsA"), _FakeRunner())
    with pytest.raises(lm.LauncherError) as ei:
        launcher.start()
    assert ei.value.code == "workspace_mismatch"
    assert ei.value.fields == {"workspace": "/tmp/wsB"}
    assert started == []


def test_start_reuses_server_started_from_same_project(monkeypatch):
    from EvoScientist.langgraph_dev import registration

    _patch_for_evosci_occupant(monkeypatch, sidecar={"workspace": "/tmp/wsA"})
    registered: list = []
    monkeypatch.setattr(
        registration,
        "register_folder",
        lambda workspace, *, url: registered.append(workspace.root),
    )
    launcher = lm.WebUILauncher(object(), _cfg(workspace_dir="/tmp/wsA"), _FakeRunner())

    assert not launcher.start().backend_started
    assert registered == [Path("/tmp/wsA").resolve()]


def test_start_warns_when_webui_port_occupied(monkeypatch):
    """An occupied WebUI port is a warning, not an error, and the port is kept."""
    _patch_for_start(monkeypatch, {4716})  # webui port taken, backend free
    launcher = lm.WebUILauncher(object(), _cfg(), _FakeRunner())
    result = launcher.start()
    assert launcher._cfg.webui_port == 4716
    assert any("4716" in w and "already in use" in w for w in result.warnings)


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


def test_frontend_env_drops_a_config_folder_set_after_startup(monkeypatch):
    """A workspace ``.env`` merged into os.environ must not give the WebUI
    another config folder than this process uses."""
    from EvoScientist import paths

    monkeypatch.setattr(paths, "_XDG_CONFIG_HOME_SET", False)
    monkeypatch.setenv("XDG_CONFIG_HOME", "/from/a/dotenv")
    env = lm.WebUILauncher(object(), _cfg(), _FakeRunner())._build_frontend_env()
    assert "XDG_CONFIG_HOME" not in env


def test_frontend_env_has_this_process_data_folder(monkeypatch, tmp_path):
    """The WebUI reads the same local files as this process."""
    from EvoScientist import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv("EVOSCIENTIST_DATA_DIR", "rel-data")
    env = lm.WebUILauncher(object(), _cfg(), _FakeRunner())._build_frontend_env()
    assert env["EVOSCIENTIST_DATA_DIR"] == str(tmp_path / "data")


# --------------------------------------------------------------------------- #
# Runner preflight
# --------------------------------------------------------------------------- #
def _installed_env(monkeypatch, tmp_path, *, node_error=None, webui_error=None):
    """Stub Node / WebUI provisioning for :class:`InstalledWebUIRunner`."""
    from EvoScientist.setup import node as setup_node
    from EvoScientist.setup import webui as setup_webui

    node = _bundled_runner_files(tmp_path)
    calls: dict = {"marked": [], "cleanups": 0, "progress": []}

    def ensure_node(**kw):
        calls["node_progress"] = kw.get("progress")
        if node_error is not None:
            raise node_error
        return setup_node.NodeInfo("private", "24.21.0", node)

    def ensure_webui(node_exe, **kw):
        calls["webui_node"] = node_exe
        if webui_error is not None:
            raise webui_error
        return setup_webui.WebUIInfo("0.3.1", tmp_path)

    def mark_in_use(info):
        calls["marked"].append(info)
        return tmp_path / "marker"

    def cleanup():
        calls["cleanups"] += 1
        return []

    monkeypatch.setattr(setup_node, "ensure_node", ensure_node)
    monkeypatch.setattr(
        setup_node, "activate_runtime", lambda: calls.setdefault("activated", True)
    )
    monkeypatch.setattr(setup_webui, "ensure_webui", ensure_webui)
    monkeypatch.setattr(setup_webui, "mark_in_use", mark_in_use)
    monkeypatch.setattr(setup_webui, "cleanup_old_versions", cleanup)
    monkeypatch.setattr(
        setup_webui, "release_in_use", lambda m: calls.setdefault("released", m)
    )
    return node, calls


def test_installed_runner_provides_node_and_webui_then_runs_them(monkeypatch, tmp_path):
    node, calls = _installed_env(monkeypatch, tmp_path)
    captured = {}

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        return _DoneProc()

    monkeypatch.setattr(lm.subprocess, "Popen", fake_popen)
    progress: list = []
    runner = lm.InstalledWebUIRunner(progress=lambda f, m: progress.append((f, m)))
    runner.preflight(_cfg())
    assert calls["webui_node"] == node
    assert callable(calls["node_progress"])  # an on-demand Node install reports
    assert calls["activated"]  # the backend and the agent's shell find it too
    assert progress[-1] == (1.0, "WebUI 0.3.1")  # the caller's spinner moves on
    assert [i.version for i in calls["marked"]] == ["0.3.1"]
    assert calls["cleanups"] == 1
    proc = runner.start(_cfg(), {"PORT": "4716"})
    assert captured["argv"] == [str(node), str(tmp_path / "dist" / "server.js")]
    runner.stop(proc)
    assert calls["released"] == tmp_path / "marker"


@pytest.mark.parametrize(
    ("which", "code"),
    [("node", "node_missing"), ("webui", "download_failed")],
)
def test_installed_runner_preflight_reports_setup_failures(
    monkeypatch, tmp_path, which, code
):
    from EvoScientist.setup.protocol import StageError

    error = StageError("download_failed", "offline")
    _installed_env(
        monkeypatch,
        tmp_path,
        node_error=error if which == "node" else None,
        webui_error=error if which == "webui" else None,
    )
    with pytest.raises(lm.LauncherError) as ei:
        lm.InstalledWebUIRunner().preflight(_cfg())
    assert ei.value.code == code
    assert "offline" in ei.value.message
    assert "EvoSci setup" in ei.value.detail


def test_failed_install_starts_no_backend(monkeypatch, tmp_path):
    """preflight runs before the backend, so a failed install leaves nothing."""
    from EvoScientist.setup.protocol import StageError

    _installed_env(
        monkeypatch, tmp_path, webui_error=StageError("download_failed", "offline")
    )
    started = []
    monkeypatch.setattr(lm, "_resolve_backend", lambda *a: started.append(a))
    launcher = lm.WebUILauncher(object(), _cfg(), lm.InstalledWebUIRunner())
    with pytest.raises(lm.LauncherError):
        launcher.start()
    assert started == []


def test_installed_runner_update_check_runs_in_a_daemon_thread(monkeypatch, tmp_path):
    from EvoScientist.setup import webui as setup_webui

    node, _calls = _installed_env(monkeypatch, tmp_path)
    seen = []
    monkeypatch.setattr(setup_webui, "stage_update", lambda exe: seen.append(exe))
    runner = lm.InstalledWebUIRunner()
    assert runner.start_update_check() is None  # nothing to check before preflight
    runner.preflight(_cfg())
    thread = runner.start_update_check()
    thread.join(5)
    assert thread.daemon
    assert seen == [node]


@pytest.mark.parametrize("private", [True, False])
def test_bundled_runner_cleans_env_only_for_private_node(
    monkeypatch, tmp_path, private
):
    from EvoScientist.setup import node as setup_node

    node = _bundled_runner_files(tmp_path)
    captured = {}

    def fake_popen(argv, env, **_kw):
        captured["env"] = env
        return _DoneProc()

    monkeypatch.setattr(setup_node, "is_private", lambda _exe: private)
    monkeypatch.setattr(lm.subprocess, "Popen", fake_popen)
    env = {"PATH": "p", "npm_config_prefix": "x", "NODE_OPTIONS": "--x"}
    lm.BundledWebUIRunner(app_dir=tmp_path, node_exe=node).start(_cfg(), env)
    if private:
        assert captured["env"] == {
            "PATH": "p",
            "NODE_ENV": "production",
            "NODE_USE_SYSTEM_CA": "1",
        }
    else:
        assert captured["env"] == {**env, "NODE_ENV": "production"}


@pytest.mark.parametrize("private", [True, False])
def test_bundled_runner_gives_the_private_node_the_proxy_switch(
    monkeypatch, tmp_path, private
):
    from EvoScientist.setup import node as setup_node

    node = _bundled_runner_files(tmp_path)
    captured = {}

    def fake_popen(argv, env, **_kw):
        captured["env"] = env
        return _DoneProc()

    monkeypatch.setattr(setup_node, "is_private", lambda _exe: private)
    monkeypatch.setattr(lm.subprocess, "Popen", fake_popen)
    env = {"PATH": "p", "HTTPS_PROXY": "http://proxy.example:3128"}
    lm.BundledWebUIRunner(app_dir=tmp_path, node_exe=node).start(_cfg(), env)
    assert ("NODE_USE_ENV_PROXY" in captured["env"]) is private
    assert ("NODE_USE_SYSTEM_CA" in captured["env"]) is private
    if private:
        assert captured["env"]["NO_PROXY"] == "localhost,127.0.0.1,::1,[::1]"


def test_bundled_runner_keeps_the_users_proxy_switch(monkeypatch, tmp_path):
    from EvoScientist.setup import node as setup_node

    node = _bundled_runner_files(tmp_path)
    captured = {}

    def fake_popen(argv, env, **_kw):
        captured["env"] = env
        return _DoneProc()

    monkeypatch.setattr(setup_node, "is_private", lambda _exe: True)
    monkeypatch.setattr(lm.subprocess, "Popen", fake_popen)
    env = {
        "PATH": "p",
        "HTTPS_PROXY": "http://proxy.example:3128",
        "NODE_USE_ENV_PROXY": "0",
    }
    lm.BundledWebUIRunner(app_dir=tmp_path, node_exe=node).start(_cfg(), env)
    assert captured["env"]["NODE_USE_ENV_PROXY"] == "0"


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
    assert "EvoSci setup" in ei.value.detail


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


def test_poll_ready_treats_served_4xx_as_ready(monkeypatch):
    # The default opener RAISES HTTPError for any non-2xx; a served 404/401 still
    # means the Next server is up and answering -> ready, not a 90s timeout.
    from urllib.error import HTTPError

    class _Opener:
        def open(self, *_a, **_k):
            raise HTTPError("http://127.0.0.1:4716", 404, "Not Found", {}, None)

    monkeypatch.setattr(lm.urllib.request, "build_opener", lambda *_h: _Opener())
    lm._poll_ready("http://127.0.0.1:4716", timeout=1)  # returns; no raise


def test_poll_ready_keeps_waiting_on_5xx(monkeypatch):
    # A 5xx is a real server error, not readiness -> keep polling, then time out.
    from urllib.error import HTTPError

    class _Opener:
        def open(self, *_a, **_k):
            raise HTTPError("http://127.0.0.1:4716", 503, "Unavailable", {}, None)

    monkeypatch.setattr(lm.urllib.request, "build_opener", lambda *_h: _Opener())
    with pytest.raises(lm.LauncherError) as ei:
        lm._poll_ready("http://127.0.0.1:4716", timeout=0.05, interval=0.01)
    assert ei.value.code == "not_ready"


def test_poll_ready_fails_fast_when_webui_proc_exits(monkeypatch):
    # node crashed after the backend came up -> fail with webui_start_failed at
    # once, not a full not_ready timeout.
    class _Opener:
        def open(self, *_a, **_k):
            raise ConnectionRefusedError("not answering yet")

    class _DeadProc:
        def poll(self):
            return 1  # already exited

    monkeypatch.setattr(lm.urllib.request, "build_opener", lambda *_h: _Opener())
    with pytest.raises(lm.LauncherError) as ei:
        lm._poll_ready(
            "http://127.0.0.1:4716", timeout=5, interval=0.01, webui_proc=_DeadProc()
        )
    assert ei.value.code == "webui_start_failed"


def test_wait_ready_times_out_when_backend_never_up(monkeypatch):
    monkeypatch.setattr(lgm, "is_langgraph_dev_running", lambda **_k: False)
    launcher = lm.WebUILauncher(object(), _cfg(), _FakeRunner())
    with pytest.raises(lm.LauncherError) as ei:
        launcher.wait_ready(timeout=0.05)
    assert ei.value.code == "not_ready"


# --------------------------------------------------------------------------- #
# build_launcher_config resolution
# --------------------------------------------------------------------------- #
def test_wait_ready_returns_cancelled_when_stopped(monkeypatch):
    import threading

    monkeypatch.setattr(lgm, "is_langgraph_dev_running", lambda **k: False)
    launcher = lm.WebUILauncher(object(), _cfg(), _FakeRunner())
    errors: list[BaseException] = []

    def _wait():
        try:
            launcher.wait_ready(timeout=30)
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=_wait)
    t.start()
    launcher.stop()
    t.join(5)
    assert not t.is_alive()
    assert errors[0].code == "cancelled"


def test_build_launcher_config_resolves_ports_and_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(lm.os, "makedirs", lambda *a, **k: None)
    config = SimpleNamespace(
        default_workdir=str(tmp_path),
        langgraph_dev_port=6000,
        langgraph_dev_host="127.0.0.1",
        webui_port=4000,
        webui_host="0.0.0.0",
    )
    cfg = lm.build_launcher_config(config, workspace_dir=None)
    assert cfg.workspace_dir == str(Path(tmp_path).resolve())
    assert cfg.backend_port == 6000
    assert cfg.webui_port == 4000
    assert cfg.webui_host == "0.0.0.0"
    assert cfg.open_browser is True


def test_build_launcher_config_blank_host_falls_back_to_loopback(monkeypatch):
    monkeypatch.setattr(lm.os, "makedirs", lambda *a, **k: None)
    config = SimpleNamespace(default_workdir="/tmp/x", webui_host="   ")
    cfg = lm.build_launcher_config(config, workspace_dir=None)
    assert cfg.webui_host == "127.0.0.1"


def test_backend_failure_releases_the_in_use_marker(monkeypatch, tmp_path):
    _node, calls = _installed_env(monkeypatch, tmp_path)

    def broken_backend(*_a, **_k):
        raise lm.LauncherError("backend_start_failed", "boom")

    monkeypatch.setattr(lm, "_resolve_backend", broken_backend)
    launcher = lm.WebUILauncher(object(), _cfg(), lm.InstalledWebUIRunner())
    with pytest.raises(lm.LauncherError):
        launcher.start()
    assert calls["released"] == tmp_path / "marker"


def test_failed_bundled_preflight_releases_the_in_use_marker(monkeypatch, tmp_path):
    _node, calls = _installed_env(monkeypatch, tmp_path)
    (tmp_path / "dist" / "server.js").unlink()
    with pytest.raises(lm.LauncherError):
        lm.InstalledWebUIRunner().preflight(_cfg())
    assert calls["released"] == tmp_path / "marker"


@pytest.mark.parametrize("opened", [True, False])
def test_wait_ready_reports_whether_the_browser_opened(monkeypatch, opened):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(lgm, "is_langgraph_dev_running", lambda **_k: True)
    monkeypatch.setattr(lm, "_poll_ready", lambda *a, **k: None)
    monkeypatch.setattr(lm.webbrowser, "open", lambda _url: opened)
    launcher = lm.WebUILauncher(object(), _cfg(open_browser=True), _FakeRunner())
    assert launcher.wait_ready(timeout=1).browser_opened is opened


def test_wait_ready_opens_no_console_browser_without_a_display(monkeypatch):
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "BROWSER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(lm.sys, "platform", "linux")
    monkeypatch.setattr(lgm, "is_langgraph_dev_running", lambda **_k: True)
    monkeypatch.setattr(lm, "_poll_ready", lambda *a, **k: None)
    opened = []
    monkeypatch.setattr(lm.webbrowser, "open", opened.append)
    launcher = lm.WebUILauncher(object(), _cfg(open_browser=True), _FakeRunner())
    assert launcher.wait_ready(timeout=1).browser_opened is False
    assert opened == []


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS case-insensitive volume")
def test_resolve_backend_same_folder_in_other_case_reuses(monkeypatch, tmp_path):
    """A stored sidecar and typed config can name the same APFS folder."""
    (tmp_path / "Project").mkdir()
    if not (tmp_path / "project").exists():
        pytest.skip("case-sensitive volume")
    _patch_backend_probes(
        monkeypatch,
        occupied=True,
        running=True,
        sidecar={
            "workspace": str(tmp_path / "Project"),
            "config_fingerprint": "fp-now",
        },
    )
    decision = lm._resolve_backend(
        _cfg(workspace_dir=str(tmp_path / "project")), object()
    )
    assert decision.action == "reuse"
