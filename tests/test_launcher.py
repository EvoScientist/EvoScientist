"""Tests for the shell-agnostic launcher core (``EvoScientist.deploy.launcher``).

These pin the pieces that moved out of ``run_webui`` so they can be reused by
other front-ends: the backend's start or reuse and its error-code taxonomy,
the secret-scrubbing env, the front-end runners' preflight and readiness
polling.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from EvoScientist.deploy import launcher as lm
from EvoScientist.langgraph_dev import manager as lgm
from tests.fakes.langgraph_dev import Server, break_start_lock, start_lock_is_held

# Before any test patches it.
_FINGERPRINT = lgm._server_config_fingerprint


@pytest.fixture(autouse=True)
def _isolated_runtime(runtime_paths):
    """The server's files in the test's server home."""


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
# server_error / reuse_warnings — what a launch reports of ensure_server
# --------------------------------------------------------------------------- #
def test_server_error_for_foreign_occupant_names_its_port(server):
    server.occupied.add(6174)
    with pytest.raises(lm.LauncherError) as ei:
        lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    assert "lsof -i :6174" in ei.value.detail
    assert ei.value.detail not in ei.value.message


@pytest.mark.parametrize("reason", ["foreign", "other_data_folder", "older_server"])
def test_server_error_names_port_conflict(reason):
    error = lm.server_error(lgm.ServerResult(problem=reason, error="taken", port=1))
    assert error.code == "port_conflict"
    assert error.fields == {"reason": reason}
    assert error.message == "taken"
    assert error.detail


@pytest.mark.parametrize(
    "result", [lgm._unrecorded_lock, lgm._lock_unreadable], ids=lambda f: f.__name__
)
def test_server_error_for_server_lock_names_it(result):
    error = lm.server_error(result())
    assert error.code == "backend_start_failed"
    assert error.fields == {"lock_file": str(lgm.RUNTIME.server_lock)}


def test_server_error_for_start_lock_held_too_long_names_it():
    """As the start reports another process holding the start lock."""
    error = lm.server_error(
        lgm.ServerResult(problem="busy", error="held", lock_file=lgm.RUNTIME.lock_file)
    )
    assert error.code == "backend_start_failed"
    assert error.fields == {"lock_file": str(lgm.RUNTIME.lock_file)}


_RECORD = lgm.ServerRecord(pid=4242, port=6174, host="127.0.0.1")


@pytest.mark.parametrize(
    "result",
    [
        lambda: lgm._stopping(_RECORD),
        lambda: lgm._unresponsive(_RECORD, "did not answer"),
        lambda: lgm._unreachable(_RECORD, "foreign"),
    ],
    ids=["stopping", "unresponsive", "unreachable"],
)
def test_server_error_for_found_server_names_its_log(result):
    """Nothing was started: the recorded server's log, no exit code."""
    error = lm.server_error(result())
    assert error.code == "backend_start_failed"
    assert error.fields == {"log_file": str(lgm.RUNTIME.log_file)}


@pytest.mark.parametrize(
    ("problem", "exit_code"), [("exited", 3), ("start_failed", None)]
)
def test_server_error_for_failed_start_names_log(problem, exit_code):
    error = lm.server_error(
        lgm.ServerResult(
            problem=problem,
            error="boom",
            exit_code=exit_code,
            log_file=lgm.RUNTIME.log_file,
        )
    )
    assert error.code == "backend_start_failed"
    assert error.message.startswith("langgraph dev startup failed: ")
    assert error.fields["log_file"] == str(lgm.RUNTIME.log_file)
    assert error.fields.get("exit_code") == exit_code


def test_server_error_for_cancel_is_cancelled():
    assert lm.server_error(lgm.ServerResult(problem="cancelled")).code == "cancelled"


def test_launcher_error_fields_default_to_empty():
    assert lm.LauncherError("port_conflict", "m").fields == {}


def _reused(**kw) -> lgm.ServerResult:
    record = lgm.ServerRecord(pid=1, port=kw.pop("port", 6174), host="127.0.0.1")
    return lgm.ServerResult(available=True, port=record.port, record=record, **kw)


def test_reuse_warnings_none_for_matching_server():
    assert lm.reuse_warnings(_reused(), 6174) == []


def test_reuse_warnings_none_for_started_server():
    record = lgm.ServerRecord(pid=1, port=7000, host="127.0.0.1")
    result = lgm.ServerResult(
        available=True, proc=_FakeProc(), port=7000, drift=True, record=record
    )
    assert lm.reuse_warnings(result, 6174) == []


def test_reuse_warnings_name_drift_since_launch():
    warnings = lm.reuse_warnings(_reused(drift=True), 6174)
    assert any("EvoSci server stop" in w for w in warnings)


def test_reuse_warnings_name_servers_shell():
    warnings = lm.reuse_warnings(_reused(shell_drift="keeps /conda/bin/python"), 6174)
    assert warnings == ["keeps /conda/bin/python"]


def test_reuse_warnings_name_port_server_runs_on():
    warnings = lm.reuse_warnings(_reused(port=7000), 6174)
    assert any("7000" in w and "6174" in w for w in warnings)


def test_reuse_warnings_name_older_releases_server():
    record = lgm.ServerRecord(pid=1, port=6174, host="127.0.0.1", older=True)
    result = lgm.ServerResult(available=True, port=6174, record=record)
    assert lgm.OLDER_SERVER_MESSAGE in lm.reuse_warnings(result, 6174)


# --------------------------------------------------------------------------- #
# WebUILauncher.start — conflicts, teardown, concurrent stop
# --------------------------------------------------------------------------- #
class _FakeProc:
    def poll(self):
        return None


class _FakeRunner:
    handles_browser_open = False

    def __init__(self):
        self.env: dict | None = None

    def preflight(self, cfg):
        pass

    def start(self, cfg, env):
        self.env = env
        return _FakeProc()

    def stop(self, proc):
        pass


@pytest.fixture
def server(monkeypatch):
    """The backends ``ensure_server`` finds, and the ones it starts."""
    from EvoScientist.langgraph_dev import registration

    fake = Server(monkeypatch)
    monkeypatch.setattr(
        registration,
        "register_folder",
        lambda workspace, *, url: registration.ServerExposure(),
    )
    return fake


def test_start_raises_on_foreign_occupant(server):
    """A foreign process on the backend port is an explicit error."""
    server.occupied.add(6174)
    launcher = lm.WebUILauncher(object(), _cfg(), _FakeRunner())
    with pytest.raises(lm.LauncherError) as ei:
        launcher.start()
    assert ei.value.code == "port_conflict"
    assert ei.value.fields == {"reason": "foreign"}


def test_start_raises_on_another_data_folders_backend(server):
    server.answering.add(6174)
    server.whose[6174] = "other"
    with pytest.raises(lm.LauncherError) as ei:
        lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    assert ei.value.fields == {"reason": "other_data_folder"}
    assert server.spawns == []


def test_start_passes_backend_allowed_origins(server):
    config = SimpleNamespace(backend_allowed_origins="https://smith.langchain.com")
    lm.WebUILauncher(config, _cfg(), _FakeRunner()).start()
    assert server.spawns[0]["allowed_origins"] == "https://smith.langchain.com"


def test_start_starts_backend_that_outlives_launch(server):
    result = lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    assert result.backend_started is True
    assert server.spawns[0]["lasting"] is True


def test_start_registers_workspace_before_front_end(server, monkeypatch):
    from EvoScientist.langgraph_dev import registration

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


def test_start_registration_failure_tears_backend_down(server, monkeypatch):
    from EvoScientist.langgraph_dev import registration

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
    assert stopped == [server.procs[0]]


def test_start_failure_after_backend_start_tears_backend_down(server, monkeypatch):
    """A front-end failure after the backend was spawned must not leave that
    backend running: start()'s error path tears down what it started."""
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
    assert stopped == [server.procs[0]]
    assert launcher._backend_proc is None


def test_start_reports_backend_that_exits(server):
    server.exit_code = 1
    with pytest.raises(lm.LauncherError) as ei:
        lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    assert ei.value.code == "backend_start_failed"
    assert ei.value.fields["exit_code"] == 1


def test_stop_cancels_in_flight_start_and_its_child_is_stopped(server):
    """A stop() from another thread while ensure_server waits for the
    started server: the call is cancelled, it stops its own child, and
    start() reports ``cancelled``."""
    import threading

    server.records = False  # the child never records: the start waits
    launcher = lm.WebUILauncher(object(), _cfg(), _FakeRunner())
    spawned = threading.Event()
    server.on_pause = lambda n: spawned.set() if server.spawns else None
    errors: list[BaseException] = []

    def _boot():
        try:
            launcher.start()
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=_boot)
    t.start()
    assert spawned.wait(5)
    launcher.stop()
    t.join(5)
    assert not t.is_alive()
    assert isinstance(errors[0], lm.LauncherError)
    assert errors[0].code == "cancelled"
    assert server.stopped == [server.pid]
    assert lgm._PROCESS is None


def test_stop_right_after_finished_start_stops_backend(monkeypatch):
    """ensure_server returned a started backend just as stop() fired: start()
    sees the stop and tears down the backend it was handed."""
    spawned = _FakeProc()
    launcher = lm.WebUILauncher(object(), _cfg(), _FakeRunner())

    def _ensure(*_a, **_k):
        launcher._stopped = True
        return lgm.ServerResult(available=True, proc=spawned, port=6174)

    monkeypatch.setattr(lgm, "ensure_server", _ensure)
    stopped: list = []
    monkeypatch.setattr(
        lgm, "stop_langgraph_dev", lambda proc=None: stopped.append(proc)
    )

    with pytest.raises(lm.LauncherError) as ei:
        launcher.start()

    assert ei.value.code == "cancelled"
    assert stopped == [spawned]


def test_start_waits_for_another_start_then_starts(server):
    """Another start holds the start lock and then gives up: this launch
    waits for it, then starts the backend itself."""
    import threading

    from tests.fakes.langgraph_dev import LockHolder

    holder = LockHolder()
    threading.Timer(0.05, holder.done).start()
    waits: list = []
    result = lm.WebUILauncher(
        object(), _cfg(), _FakeRunner(), on_lock_wait=waits.append
    ).start()
    assert result.backend_started is True
    assert waits == [True, False]


def test_backend_starts_under_start_lock(server, monkeypatch):
    held: list[bool] = []
    spawn = server._spawn
    monkeypatch.setattr(
        lgm,
        "_spawn_server",
        lambda **k: held.append(start_lock_is_held()) or spawn(**k),
    )
    lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    assert held == [True]
    assert not start_lock_is_held()


@pytest.mark.parametrize("then_free", [False, True])
def test_stop_while_waiting_for_start_lock_starts_nothing(
    server, monkeypatch, then_free
):
    """A stop() during the wait starts nothing, also when the lock is freed
    right after it."""
    import threading

    from filelock import FileLock

    from tests.fakes.langgraph_dev import LockHolder

    monkeypatch.setattr(lgm, "_START_BUDGET", 5.0)
    holder = LockHolder()
    launcher = lm.WebUILauncher(object(), _cfg(), _FakeRunner())

    def _stop() -> None:
        launcher.stop()
        if then_free:
            holder.done()

    threading.Timer(0.1, _stop).start()
    try:
        with pytest.raises(lm.LauncherError) as ei:
            launcher.start()
        assert then_free or holder.thread.is_alive()
    finally:
        holder.done()
    assert ei.value.code == "cancelled"
    assert server.spawns == []
    lock = FileLock(str(lgm.RUNTIME.lock_file))
    lock.acquire(timeout=0.5)
    lock.release()


def test_start_lock_held_too_long_is_backend_start_failed(server):
    from tests.fakes.langgraph_dev import LockHolder

    holder = LockHolder()
    try:
        with pytest.raises(lm.LauncherError) as ei:
            lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    finally:
        holder.done()
    assert ei.value.code == "backend_start_failed"
    assert ei.value.fields == {"lock_file": str(lgm.RUNTIME.lock_file)}


@pytest.mark.parametrize("step", ["mkdir", "acquire"])
def test_start_lock_os_error_is_backend_start_failed(
    server, monkeypatch, tmp_path, step
):
    """An OSError from the start lock is backend_start_failed, so run_webui
    shows its panel instead of a traceback."""
    break_start_lock(monkeypatch, tmp_path, step)
    with pytest.raises(lm.LauncherError) as ei:
        lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    assert ei.value.code == "backend_start_failed"
    assert ei.value.message.startswith("langgraph dev startup failed: ")
    # Only a lock that exists is named; the server home that cannot be
    # created is in the message.
    if step == "acquire":
        assert ei.value.fields == {"lock_file": str(lgm.RUNTIME.lock_file)}
    else:
        assert ei.value.fields == {}
        assert str(lgm.paths.SERVER_HOME) in ei.value.message
    assert server.spawns == []


def test_spawn_failing_before_launch_names_no_file(server, monkeypatch):
    """The langgraph CLI is missing: nothing ran, so no log was written."""

    def _missing(**_kw):
        raise FileNotFoundError("langgraph")

    monkeypatch.setattr(lgm, "_spawn_server", _missing)
    with pytest.raises(lm.LauncherError) as ei:
        lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    assert ei.value.code == "backend_start_failed"
    assert ei.value.fields == {}


def test_start_tells_caller_nothing_when_nobody_else_starts(server):
    waits: list = []
    lm.WebUILauncher(object(), _cfg(), _FakeRunner(), on_lock_wait=waits.append).start()
    assert waits == []


def test_stop_after_reuse_leaves_backend_running(server, monkeypatch):
    server.run()
    calls: list = []
    monkeypatch.setattr(lgm, "stop_langgraph_dev", lambda *a, **k: calls.append(a))
    launcher = lm.WebUILauncher(object(), _cfg(), _FakeRunner())
    assert launcher.start().backend_started is False
    launcher.stop()
    assert calls == []


def test_start_reuses_running_server_and_registers_project(server, monkeypatch):
    """The backend serves every registered project: the launch registers its
    own and starts no second backend."""
    from EvoScientist.langgraph_dev import registration

    server.run()
    registered: list = []
    monkeypatch.setattr(
        registration,
        "register_folder",
        lambda workspace, *, url: registered.append(workspace.root),
    )
    launcher = lm.WebUILauncher(object(), _cfg(workspace_dir="/tmp/wsA"), _FakeRunner())

    assert not launcher.start().backend_started
    assert registered == [Path("/tmp/wsA").resolve()]
    assert server.spawns == []


def test_start_reuses_this_data_folders_backend_on_another_port(server, monkeypatch):
    """The front end and the registration follow the backend to its port."""
    from EvoScientist.langgraph_dev import registration

    server.run(port=7000)
    urls: list = []
    monkeypatch.setattr(
        registration,
        "register_folder",
        lambda workspace, *, url: urls.append(url) or registration.ServerExposure(),
    )
    runner = _FakeRunner()
    launcher = lm.WebUILauncher(object(), _cfg(), runner)

    result = launcher.start()

    assert result.backend_url == "http://127.0.0.1:7000"
    assert launcher.backend_port == 7000
    assert urls == ["http://127.0.0.1:7000"]
    assert runner.env["EVOSCIENTIST_LANGGRAPH_DEV_PORT"] == "7000"
    assert any("7000" in w for w in result.warnings)
    assert server.spawns == []


def test_start_warns_of_drift_since_reused_servers_launch(server, monkeypatch):
    server.run(config_fingerprint="fp-old")
    result = lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    assert any("EvoSci server stop" in w for w in result.warnings)


def test_start_no_drift_for_session_started_with_auto_flags(server, monkeypatch):
    """``EvoSci --auto-approve`` / ``--auto-mode`` override fields the server
    reads from config.yaml; a second launcher with the file's values must not
    report drift (and advise ``EvoSci server stop``)."""
    from EvoScientist.config.settings import EvoScientistConfig

    monkeypatch.setattr(lgm, "_server_config_fingerprint", _FINGERPRINT)
    recorded = _FINGERPRINT(
        EvoScientistConfig(auto_approve=True, auto_mode=True, enable_ask_user=False)
    )
    server.run(config_fingerprint=recorded)
    result = lm.WebUILauncher(EvoScientistConfig(), _cfg(), _FakeRunner()).start()
    assert not any("EvoSci server stop" in w for w in result.warnings)


@pytest.mark.parametrize(
    ("recorded", "own", "hinted"),
    [
        ("/conda/bin/python", None, False),
        (None, None, True),
        (None, "/usr/bin/python", False),
    ],
)
def test_start_hint_follows_reused_server(server, monkeypatch, recorded, own, hinted):
    from EvoScientist.setup import research_env

    server.run(agent_python=recorded)
    monkeypatch.setattr(research_env, "agent_python", lambda: own)
    result = lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    assert (research_env.MISSING_PYTHON_HINT in result.warnings) is hinted


def test_start_warns_of_reused_servers_python(server, monkeypatch):
    from EvoScientist.setup import research_env

    server.run(agent_python="/conda/bin/python")
    monkeypatch.setattr(research_env, "agent_python", lambda: "/env/bin/python")
    result = lm.WebUILauncher(object(), _cfg(), _FakeRunner()).start()
    assert any("/conda/bin/python" in w for w in result.warnings)


def test_start_warns_when_webui_port_occupied(server):
    """An occupied WebUI port is a warning, not an error, and the port is kept."""
    server.occupied.add(4716)  # webui port taken, backend free
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
    monkeypatch.setattr(lgm, "ensure_server", lambda *a, **k: started.append(a))
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

    monkeypatch.setattr(
        lgm,
        "ensure_server",
        lambda *_a, **_k: lgm.ServerResult(problem="start_failed", error="boom"),
    )
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
