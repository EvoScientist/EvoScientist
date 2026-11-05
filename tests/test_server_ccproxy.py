"""The server's ccproxy (``langgraph_dev.server_ccproxy``): started at import
when Anthropic uses OAuth, recorded before it serves, started at a run start
that needs Anthropic when it is not running, stopped at exit, and the server
exits with its own code when it cannot start it."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from EvoScientist import ccproxy_manager, paths
from EvoScientist.langgraph_dev import manager, server_ccproxy, server_record


class _Popen:
    """A ccproxy that runs until *code* is set."""

    def __init__(self, pid: int = 4242, code: int | None = None) -> None:
        self.pid = pid
        self.code = code

    def poll(self):
        return self.code


@pytest.fixture
def server(monkeypatch):
    """A server holding its lock and recorded as starting, with Anthropic's
    OAuth configured and every ccproxy step faked; ``steps`` is what ran."""
    monkeypatch.setattr(paths, "SERVER_HOME", paths.DATA_DIR / "server")
    paths.SERVER_HOME.mkdir(parents=True)
    record = {"pid": 20, server_record.PHASE_KEY: server_record.STARTING}
    server_record.record_path().write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(server_record, "_written", record)
    monkeypatch.setattr(server_ccproxy, "_proc", None)
    monkeypatch.setattr(server_ccproxy, "_job", None)
    monkeypatch.setattr(server_ccproxy, "_stopping", False)
    monkeypatch.setattr(server_ccproxy, "_port", None)
    monkeypatch.setattr(server_ccproxy, "_attempts", 0)
    monkeypatch.setattr(server_ccproxy, "_last_error", None)
    monkeypatch.setattr(server_ccproxy, "_failed", 0)
    monkeypatch.setattr(server_ccproxy, "_noted_orphan", False)

    config = SimpleNamespace(
        anthropic_auth_mode="oauth",
        provider="anthropic",
        auxiliary_provider="",
        model_fallbacks="",
        ccproxy_port=_free_port(),
    )
    monkeypatch.setattr(server_ccproxy, "get_effective_config", lambda: config)
    steps: list = []
    proc = _Popen()

    def _spawn(port, *, output):
        steps.append(("spawn", port, Path(output.name)))
        return proc

    def _wait(_proc, port):
        steps.append(("wait", _read().get(server_record.CCPROXY_PID_KEY)))

    monkeypatch.setattr(ccproxy_manager, "check_ccproxy_ready", lambda: None)
    monkeypatch.setattr(ccproxy_manager, "spawn_ccproxy", _spawn)
    monkeypatch.setattr(ccproxy_manager, "wait_for_ccproxy", _wait)
    monkeypatch.setattr(
        ccproxy_manager, "setup_ccproxy_env", lambda port: steps.append(("env", port))
    )
    monkeypatch.setattr(
        ccproxy_manager, "stop_ccproxy_tree", lambda p: steps.append(("stop", p))
    )
    monkeypatch.setattr(
        ccproxy_manager, "stop_recorded_ccproxy", lambda: steps.append(("older",))
    )
    monkeypatch.setattr(
        ccproxy_manager,
        "end_ccproxy",
        lambda pid, created, port: steps.append(("end", pid, created)) or True,
    )
    monkeypatch.setattr(manager, "_create_time", lambda pid: 555.0)
    monkeypatch.setattr(manager, "_is_port_occupied", lambda port, host: False)
    registered: list = []
    monkeypatch.setattr(server_ccproxy.atexit, "register", registered.append)
    return SimpleNamespace(config=config, steps=steps, proc=proc, registered=registered)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _read() -> dict:
    return json.loads(server_record.record_path().read_text(encoding="utf-8"))


def _exit_code(left=None) -> int:
    with pytest.raises(SystemExit) as raised:
        server_ccproxy.start(left)
    return raised.value.code


def test_start_without_oauth_starts_nothing(server):
    server.config.anthropic_auth_mode = "api_key"

    server_ccproxy.start(None)

    assert server.steps == [("older",)]
    assert server_record.CCPROXY_PID_KEY not in _read()


_LEFT = {
    server_record.CCPROXY_PID_KEY: 77,
    server_record.CCPROXY_CREATE_TIME_KEY: 123.5,
}


def test_start_without_oauth_ends_ccproxy_previous_server_left(server):
    """Its record is replaced: no later server would find it."""
    server.config.anthropic_auth_mode = "api_key"

    server_ccproxy.start(_LEFT)

    assert server.steps == [("end", 77, 123.5), ("older",)]


def test_start_ends_left_ccproxy_without_creation_time(server):
    """Its creation time could not be read when it was recorded."""
    server.config.anthropic_auth_mode = "api_key"

    server_ccproxy.start({**_LEFT, server_record.CCPROXY_CREATE_TIME_KEY: None})

    assert server.steps == [("end", 77, None), ("older",)]


def test_start_not_signed_in_ends_ccproxy_previous_server_left(server, monkeypatch):
    def _unready():
        raise RuntimeError("Run: ccproxy auth login claude_api")

    monkeypatch.setattr(ccproxy_manager, "check_ccproxy_ready", _unready)

    assert _exit_code(_LEFT) == server_ccproxy.CCPROXY_FAILED_EXIT
    assert server.steps == [("end", 77, 123.5), ("older",)]


def test_start_with_oauth_starts_ccproxy_without_anthropic_model(server):
    """/model can pick an Anthropic model for a run whatever is configured."""
    server.config.provider = "openrouter"

    server_ccproxy.start(_LEFT)

    assert server.steps[:2] == [("end", 77, 123.5), ("older",)]
    assert ("env", server.config.ccproxy_port) in server.steps
    assert server_ccproxy.needed()


@pytest.mark.parametrize(
    ("auxiliary", "fallbacks"),
    [("anthropic", ""), ("", "gpt-5:openai,claude-sonnet-4-5:anthropic")],
)
def test_start_failure_with_anthropic_model_exits(
    server, monkeypatch, auxiliary, fallbacks
):
    server.config.provider = "openrouter"
    server.config.auxiliary_provider = auxiliary
    server.config.model_fallbacks = fallbacks

    def _unready():
        raise RuntimeError("Run: ccproxy auth login claude_api")

    monkeypatch.setattr(ccproxy_manager, "check_ccproxy_ready", _unready)

    assert _exit_code() == server_ccproxy.CCPROXY_FAILED_EXIT


def test_start_failure_without_anthropic_model_warns_and_serves(
    server, monkeypatch, caplog
):
    """A stale OAuth setting must not keep the server from starting."""
    server.config.provider = "openrouter"

    def _unready():
        raise RuntimeError("Run: ccproxy auth login claude_api")

    monkeypatch.setattr(ccproxy_manager, "check_ccproxy_ready", _unready)

    server_ccproxy.start(None)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("ccproxy auth login" in r.getMessage() for r in warnings)
    assert any("/model" in r.getMessage() for r in warnings)
    assert not [step for step in server.steps if step[0] == "env"]
    # The next run start that needs it tries to start it.
    assert server_ccproxy.needed()


def test_start_checks_port_before_sign_in(server, monkeypatch, caplog):
    """Checking the sign-in can take seconds; a taken port is told at once."""
    checked = []
    monkeypatch.setattr(
        ccproxy_manager, "check_ccproxy_ready", lambda: checked.append(1)
    )
    monkeypatch.setattr(manager, "_is_port_occupied", lambda port, host: True)

    assert _exit_code() == server_ccproxy.CCPROXY_FAILED_EXIT
    assert "EvoSci config set ccproxy_port" in caplog.text
    assert checked == []


def test_start_runs_ccproxy_with_its_own_log_and_sets_env(server):
    port = server.config.ccproxy_port

    server_ccproxy.start(None)

    assert ("spawn", port, manager.RUNTIME.ccproxy_log) in server.steps
    assert manager.RUNTIME.ccproxy_log.parent == paths.SERVER_HOME
    assert server.steps[-1] == ("env", port)


def test_start_rotates_large_ccproxy_log(server, monkeypatch):
    monkeypatch.setattr(manager, "_LOG_ROTATION_BYTES", 10)
    manager.RUNTIME.ccproxy_log.write_text("x" * 20)

    server_ccproxy.start(None)

    rotated = manager.RUNTIME.ccproxy_log.with_name("ccproxy.log.1")
    assert rotated.read_text() == "x" * 20
    assert manager.RUNTIME.ccproxy_log.stat().st_size == 0


def test_start_records_ccproxy_before_waiting_for_it(server):
    """A server killed while ccproxy starts leaves it recorded."""
    server_ccproxy.start(None)

    assert ("wait", server.proc.pid) in server.steps
    record = _read()
    assert record[server_record.CCPROXY_CREATE_TIME_KEY] == 555.0
    assert record[server_record.PHASE_KEY] == server_record.STARTING


def test_start_ends_ccproxy_previous_server_left(server):
    server_ccproxy.start(_LEFT)

    assert server.steps.index(("end", 77, 123.5)) < server.steps.index(
        ("spawn", server.config.ccproxy_port, manager.RUNTIME.ccproxy_log)
    )


def test_start_ends_older_release_lasting_ccproxy(server, monkeypatch):
    ended = []
    monkeypatch.setattr(
        ccproxy_manager, "stop_recorded_ccproxy", lambda: ended.append(1) or 99
    )

    server_ccproxy.start({"pid": 20})

    assert ended == [1]
    assert not [step for step in server.steps if step[0] == "end"]


@pytest.mark.parametrize(
    "message",
    [
        "ccproxy is required for OAuth mode but not found.",
        "ccproxy Anthropic OAuth not authenticated\nRun: ccproxy auth login claude_api",
    ],
)
def test_start_unready_ccproxy_exits_with_its_code(
    server, monkeypatch, caplog, message
):
    def _unready():
        raise RuntimeError(message)

    monkeypatch.setattr(ccproxy_manager, "check_ccproxy_ready", _unready)

    assert _exit_code() == server_ccproxy.CCPROXY_FAILED_EXIT
    assert message in caplog.text
    assert server.steps == [("older",)]


def test_start_on_taken_port_exits_naming_variable(server, caplog):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        server.config.ccproxy_port = taken.getsockname()[1]

        assert _exit_code() == server_ccproxy.CCPROXY_FAILED_EXIT

    assert "EVOSCIENTIST_CCPROXY_PORT" in caplog.text
    assert str(paths.DATA_DIR) in caplog.text
    assert server.steps == [("older",)]


def test_start_waits_for_port_after_ending_ccproxy_on_it(server, monkeypatch):
    """A ccproxy just ended may hold its port for a moment."""
    waited = []

    def _wait(port, host):
        waited.append(port)
        return True

    monkeypatch.setattr(manager, "_wait_for_port_bindable", _wait)

    server_ccproxy.start(_LEFT)

    assert waited == [server.config.ccproxy_port]


def test_start_without_ccproxy_ended_checks_port_once(server, monkeypatch):
    def _wait(port, host):
        raise AssertionError("waited for a port nothing was stopped on")

    monkeypatch.setattr(manager, "_wait_for_port_bindable", _wait)

    server_ccproxy.start(None)

    assert ("env", server.config.ccproxy_port) in server.steps


@pytest.mark.parametrize(("exited", "waits"), [(True, True), (False, False)])
def test_restart_waits_for_port_only_after_ccproxy_exited(
    serving, monkeypatch, exited, waits
):
    """A ccproxy that exited frees its port a moment later; a port taken
    since the server's start is not worth the wait."""
    waited = []

    def _wait(port, host):
        waited.append(port)
        return True

    monkeypatch.setattr(manager, "_wait_for_port_bindable", _wait)
    if exited:
        server_ccproxy._proc.code = 1
    else:
        monkeypatch.setattr(server_ccproxy, "_proc", None)

    server_ccproxy.ensure_serving()

    assert waited == ([serving.config.ccproxy_port] if waits else [])


def test_start_beside_wildcard_listener_exits(server, monkeypatch):
    """The loopback bind can pass beside another program's listener on the
    wildcard address (macOS, Windows)."""
    monkeypatch.setattr(manager, "_is_port_occupied", lambda port, host: True)

    assert _exit_code() == server_ccproxy.CCPROXY_FAILED_EXIT
    assert not [step for step in server.steps if step[0] == "spawn"]


def test_start_never_serving_ccproxy_exits_naming_its_log(server, monkeypatch, caplog):
    def _never(proc, port):
        raise RuntimeError("ccproxy did not become healthy within 180 seconds")

    monkeypatch.setattr(ccproxy_manager, "wait_for_ccproxy", _never)

    assert _exit_code() == server_ccproxy.CCPROXY_FAILED_EXIT
    assert "did not become healthy" in caplog.text
    assert str(manager.RUNTIME.ccproxy_log) in caplog.text
    # Stopped by the wait: the exit's stop has nothing to do.
    assert server_ccproxy._proc is None
    assert not [step for step in server.steps if step[0] == "env"]


def test_start_registers_stop_after_record_removal(server):
    """Exit handlers run in reverse order: ccproxy stops before the record
    goes."""
    server_ccproxy.start(None)

    assert server.registered == [server_ccproxy.stop]


def test_stop_stops_ccproxy_once(server):
    server_ccproxy.start(None)

    server_ccproxy.stop()
    server_ccproxy.stop()

    assert server.steps[-2:] == [("stop", server.proc), ("stop", None)]


def test_write_record_returns_record_it_replaced(server, monkeypatch):
    left = {"pid": 9, server_record.CCPROXY_PID_KEY: 77}
    server_record.record_path().write_text(json.dumps(left), encoding="utf-8")
    monkeypatch.setattr(server_record, "_build", lambda: {"pid": 10})

    assert server_record.write_record() == left
    assert _read() == {"pid": 10}


def test_mark_serving_keeps_recorded_ccproxy(server, monkeypatch):
    monkeypatch.setattr(
        "EvoScientist.setup.research_env.agent_python", lambda: "/usr/bin/python3"
    )
    monkeypatch.setattr("EvoScientist.agent_shell.sidecar_bash", lambda: None)
    server_record.record_ccproxy(4242, 555.0)

    server_record.mark_serving()

    record = _read()
    assert record[server_record.PHASE_KEY] == server_record.SERVING
    assert record[server_record.CCPROXY_PID_KEY] == 4242


def test_remove_record_removes_record_with_ccproxy(server):
    server_record.record_ccproxy(4242, 555.0)

    server_record.remove_record()

    assert not server_record.record_path().exists()


class _Process:
    def __init__(self, pid, created, cmdline):
        self.pid = pid
        self._created = created
        self._cmdline = cmdline

    def create_time(self):
        return self._created

    def cmdline(self):
        return self._cmdline


@pytest.mark.parametrize(
    ("created", "cmdline", "ended"),
    [
        (123.5, ["/venv/bin/ccproxy", "serve"], True),
        (123.9, ["/venv/bin/ccproxy", "serve"], True),
        (999.0, ["/venv/bin/ccproxy", "serve"], False),
        # A clock step moves the creation time; the port still names it.
        (999.0, ["/venv/bin/ccproxy", "serve", "--port", "8000"], True),
        (999.0, ["/venv/bin/ccproxy", "serve", "--port", "8001"], False),
        (123.5, ["python", "other.py"], False),
    ],
)
def test_end_ccproxy_ends_only_recorded_ccproxy(monkeypatch, created, cmdline, ended):
    """A reused PID, or another program on it, is left alone."""
    import psutil

    ends = []
    monkeypatch.setattr(psutil, "Process", lambda pid: _Process(pid, created, cmdline))
    monkeypatch.setattr(
        ccproxy_manager, "end_tree", lambda proc: ends.append(proc.pid) or True
    )

    assert ccproxy_manager.end_ccproxy(77, 123.5, 8000) is ended
    assert ends == ([77] if ended else [])


@pytest.mark.parametrize(
    ("cmdline", "ended"),
    [
        (["/venv/bin/ccproxy", "serve", "--port", "8000"], True),
        (["/venv/bin/ccproxy", "serve"], False),
    ],
)
def test_end_ccproxy_without_recorded_creation_time_goes_by_port(
    monkeypatch, cmdline, ended
):
    """Its creation time could not be read when it was recorded."""
    import psutil

    monkeypatch.setattr(psutil, "Process", lambda pid: _Process(pid, 123.5, cmdline))
    monkeypatch.setattr(ccproxy_manager, "end_tree", lambda proc: True)

    assert ccproxy_manager.end_ccproxy(77, None, 8000) is ended


def test_end_tree_ends_process_whose_children_it_cannot_list(monkeypatch):
    import psutil

    class _Denied(_Process):
        def children(self, recursive=False):
            raise psutil.AccessDenied(self.pid)

        def terminate(self):
            ended.append(self.pid)

    ended = []
    monkeypatch.setattr(psutil, "wait_procs", lambda procs, timeout: (procs, []))

    assert ccproxy_manager.end_tree(_Denied(77, 123.5, [])) is True
    assert ended == [77]


def test_spawn_ccproxy_output_goes_where_asked(monkeypatch):
    """The server's ccproxy writes to its log, a session's to nowhere."""
    calls = []
    monkeypatch.setattr(
        ccproxy_manager.subprocess, "Popen", lambda cmd, **kw: calls.append(kw)
    )

    log = object()
    ccproxy_manager.spawn_ccproxy(8000, output=log)
    ccproxy_manager.spawn_ccproxy(8000)

    assert calls[0]["stdout"] is log
    assert calls[0]["stderr"] is log
    assert calls[1]["stdout"] == ccproxy_manager.subprocess.DEVNULL


def test_end_ccproxy_leaves_process_it_cannot_inspect(monkeypatch):
    import psutil

    class _Denied(_Process):
        def cmdline(self):
            raise psutil.AccessDenied(self.pid)

    ends = []
    monkeypatch.setattr(psutil, "Process", lambda pid: _Denied(pid, 123.5, []))
    monkeypatch.setattr(ccproxy_manager, "end_tree", ends.append)

    assert ccproxy_manager.end_ccproxy(77, 123.5, 8000) is False
    assert ends == []


class _Child:
    """A Popen that runs until stopped."""

    pid = 4242

    def __init__(self, returncode=None):
        self.returncode = returncode
        self.waited = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waited = True
        self.returncode = -15
        return self.returncode


def test_stop_ccproxy_tree_ends_tree_then_reaps(monkeypatch):
    """On Windows ``ccproxy.exe`` is a launcher whose child holds the port."""
    import psutil

    steps = []
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(pid=pid))
    monkeypatch.setattr(
        ccproxy_manager, "end_tree", lambda proc: steps.append(("tree", proc.pid))
    )
    monkeypatch.setattr(
        ccproxy_manager, "stop_ccproxy", lambda proc: steps.append(("stop", proc))
    )
    child = _Child()

    ccproxy_manager.stop_ccproxy_tree(child)

    # One stop: the tree, then only the reap.
    assert steps == [("tree", 4242)]
    assert child.waited


def test_stop_ccproxy_tree_stops_process_it_cannot_inspect(monkeypatch):
    import psutil

    def _denied(pid):
        raise psutil.AccessDenied(pid)

    steps = []
    monkeypatch.setattr(psutil, "Process", _denied)
    monkeypatch.setattr(ccproxy_manager, "end_tree", steps.append)
    monkeypatch.setattr(
        ccproxy_manager, "stop_ccproxy", lambda proc: steps.append(("stop", proc))
    )
    child = _Child()

    ccproxy_manager.stop_ccproxy_tree(child)

    assert steps == [("stop", child)]


def test_stop_ccproxy_tree_leaves_reaped_process(monkeypatch):
    """Its PID may be another process's by now."""
    steps = []
    monkeypatch.setattr(ccproxy_manager, "end_tree", steps.append)
    monkeypatch.setattr(ccproxy_manager, "stop_ccproxy", steps.append)

    ccproxy_manager.stop_ccproxy_tree(_Child(returncode=1))
    ccproxy_manager.stop_ccproxy_tree(None)

    assert steps == []


def test_end_ccproxy_without_process_does_nothing(monkeypatch):
    import psutil

    def _gone(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", _gone)

    assert ccproxy_manager.end_ccproxy(77, 123.5, 8000) is False


@pytest.fixture
def serving(server, monkeypatch):
    """A server whose ccproxy started, with no Anthropic model configured;
    ``server.proc`` is the one a restart starts."""
    monkeypatch.setattr(server_ccproxy, "_port", server.config.ccproxy_port)
    monkeypatch.setattr(server_ccproxy, "_proc", _Popen(5000))
    monkeypatch.setattr(ccproxy_manager, "is_ccproxy_running", lambda port: False)
    server.proc.pid = 5001
    return server


def _spawned(server) -> list:
    return [step for step in server.steps if step[0] == "spawn"]


def _unready():
    raise RuntimeError("Run: ccproxy auth login claude_api")


def test_ensure_serving_leaves_running_ccproxy(serving):
    server_ccproxy.ensure_serving()

    assert _spawned(serving) == []


def test_ensure_serving_restarts_ccproxy_that_is_gone(serving):
    server_ccproxy._proc.code = 1

    server_ccproxy.ensure_serving()

    assert len(_spawned(serving)) == 1
    assert _read()[server_record.CCPROXY_PID_KEY] == 5001
    assert ("env", serving.config.ccproxy_port) in serving.steps


def _burst(count: int, hold) -> list:
    """*count* run starts at once; what each raised, None for none. *hold*
    keeps the restart going a moment, so the others queue behind it."""
    both = threading.Barrier(count)
    outcomes: list = []

    def _run():
        both.wait()
        try:
            server_ccproxy.ensure_serving()
        except RuntimeError as exc:
            outcomes.append(str(exc))
        else:
            outcomes.append(None)
        hold.set()

    runs = [threading.Thread(target=_run) for _ in range(count)]
    for run in runs:
        run.start()
    for run in runs:
        run.join(5)
    return outcomes


def test_ensure_serving_restarts_once_for_concurrent_runs(serving, monkeypatch):
    server_ccproxy._proc.code = 1
    hold = threading.Event()
    monkeypatch.setattr(
        ccproxy_manager, "wait_for_ccproxy", lambda p, port: hold.wait(0.2)
    )

    assert _burst(2, hold) == [None, None]
    assert len(_spawned(serving)) == 1


def test_ensure_serving_burst_shares_one_failed_attempt(serving, monkeypatch):
    """Runs that waited on a restart get its reason; it counts once."""
    server_ccproxy._proc.code = 1
    hold = threading.Event()
    checks = []

    def _slow_unready():
        checks.append(1)
        hold.wait(0.2)
        _unready()

    monkeypatch.setattr(ccproxy_manager, "check_ccproxy_ready", _slow_unready)

    outcomes = _burst(3, hold)

    assert len(checks) == 1
    assert server_ccproxy._failed == 1
    assert len(outcomes) == 3
    assert all("ccproxy auth login" in outcome for outcome in outcomes)


def test_ensure_serving_leaves_ccproxy_answering_without_its_process(
    serving, monkeypatch, caplog
):
    """On Windows a launcher can exit while its child serves on."""
    server_ccproxy._proc.code = 1
    monkeypatch.setattr(ccproxy_manager, "is_ccproxy_running", lambda port: True)

    server_ccproxy.ensure_serving()
    server_ccproxy.ensure_serving()

    assert _spawned(serving) == []
    assert caplog.text.count("not from the process the server started") == 1
    # Runs reach it even when the server started without its own.
    assert ("env", serving.config.ccproxy_port) in serving.steps


def test_ensure_serving_failure_fails_anthropic_run(serving, monkeypatch):
    server_ccproxy._proc.code = 1
    monkeypatch.setattr(ccproxy_manager, "check_ccproxy_ready", _unready)

    with pytest.raises(RuntimeError, match="ccproxy auth login"):
        server_ccproxy.ensure_serving()


@pytest.mark.parametrize(
    ("main", "auxiliary", "fallbacks", "fails"),
    [
        ("anthropic", "", "", True),
        ("openai", "anthropic", "", True),
        ("openai", "", "gpt-5:openai,claude-sonnet-4-5:anthropic", True),
        ("openai", "", "", False),
    ],
)
def test_ensure_serving_failure_fails_run_by_configured_models(
    serving, monkeypatch, caplog, main, auxiliary, fallbacks, fails
):
    """The server's start rule: without a configured Anthropic model the
    failure is only logged, as /model may still pick one."""
    # Read at each run start: a model saved while the server runs counts.
    serving.config.provider = main
    serving.config.auxiliary_provider = auxiliary
    serving.config.model_fallbacks = fallbacks
    server_ccproxy._proc.code = 1
    monkeypatch.setattr(ccproxy_manager, "check_ccproxy_ready", _unready)

    if fails:
        with pytest.raises(RuntimeError, match="ccproxy auth login"):
            server_ccproxy.ensure_serving()
    else:
        server_ccproxy.ensure_serving()
        assert "Could not start ccproxy" in caplog.text

    assert server_ccproxy._failed == 1


def test_ensure_serving_stops_trying_after_failures_in_row(serving, monkeypatch):
    server_ccproxy._proc.code = 1
    checks = []
    monkeypatch.setattr(
        ccproxy_manager, "check_ccproxy_ready", lambda: checks.append(1) or _unready()
    )

    for _ in range(5):
        with pytest.raises(RuntimeError):
            server_ccproxy.ensure_serving()

    assert len(checks) == server_ccproxy._FAILED_RESTART_LIMIT


def test_ensure_serving_after_giving_up_tells_anthropic_run(serving, monkeypatch):
    """Not a raw connection error: what to do."""
    server_ccproxy._proc.code = 1
    monkeypatch.setattr(server_ccproxy, "_failed", server_ccproxy._FAILED_RESTART_LIMIT)

    with pytest.raises(RuntimeError, match="after 3 tries"):
        server_ccproxy.ensure_serving()
    serving.config.provider = "openai"
    server_ccproxy.ensure_serving()

    assert _spawned(serving) == []


def test_ensure_serving_success_resets_failures(serving, monkeypatch):
    server_ccproxy._proc.code = 1
    monkeypatch.setattr(server_ccproxy, "_failed", 2)
    monkeypatch.setattr(server_ccproxy, "_noted_orphan", True)

    server_ccproxy.ensure_serving()

    assert server_ccproxy._failed == 0
    assert server_ccproxy._noted_orphan is False


def test_ensure_serving_while_server_stops_does_nothing(serving):
    server_ccproxy.stop()

    server_ccproxy.ensure_serving()

    assert _spawned(serving) == []


def test_ensure_serving_without_oauth_does_nothing(serving, monkeypatch):
    monkeypatch.setattr(server_ccproxy, "_port", None)
    server_ccproxy._proc.code = 1

    server_ccproxy.ensure_serving()

    assert _spawned(serving) == []


def test_ensure_serving_on_taken_port_says_to_restart_server(serving, monkeypatch):
    server_ccproxy._proc.code = 1
    monkeypatch.setattr(server_ccproxy, "_port_free", lambda port, ended: False)

    with pytest.raises(RuntimeError, match="EvoSci server stop") as raised:
        server_ccproxy.ensure_serving()
    assert "EVOSCIENTIST_CCPROXY_PORT" in str(raised.value)
    assert _spawned(serving) == []


def test_factory_refused_run_does_not_restart_ccproxy(monkeypatch):
    """The folder check comes first: its refusal reaches the run's client."""
    from EvoScientist.langgraph_dev import factories
    from EvoScientist.sessions import ThreadBindingError

    def _refused(*a):
        raise ThreadBindingError("bound to another project")

    calls = []
    monkeypatch.setattr(factories, "_graph_for", _refused)
    monkeypatch.setattr(server_ccproxy, "needed", lambda: True)
    monkeypatch.setattr(server_ccproxy, "ensure_serving", lambda *a: calls.append(a))

    with pytest.raises(ThreadBindingError):
        asyncio.run(
            factories.served_graph("writing-agent")(
                {}, SimpleNamespace(access_context="threads.create_run")
            )
        )

    assert calls == []


@pytest.mark.parametrize(
    ("access", "checked"),
    [("threads.create_run", True), ("threads.read", False)],
)
def test_factory_checks_ccproxy_only_at_run_start(monkeypatch, access, checked):
    from EvoScientist.langgraph_dev import factories

    calls = []
    monkeypatch.setattr(factories, "_graph_for", lambda *a: ("graph", None))
    monkeypatch.setattr(server_ccproxy, "needed", lambda: True)
    monkeypatch.setattr(server_ccproxy, "ensure_serving", lambda *a: calls.append(a))
    config = {"configurable": {}}

    graph = asyncio.run(
        factories.served_graph("writing-agent")(
            config, SimpleNamespace(access_context=access)
        )
    )

    assert graph == "graph"
    assert calls == ([()] if checked else [])


def test_factory_without_oauth_skips_ccproxy_check(monkeypatch):
    from EvoScientist.langgraph_dev import factories

    calls = []
    monkeypatch.setattr(factories, "_graph_for", lambda *a: ("graph", None))
    monkeypatch.setattr(server_ccproxy, "ensure_serving", lambda *a: calls.append(a))

    asyncio.run(
        factories.served_graph("writing-agent")(
            {}, SimpleNamespace(access_context="threads.create_run")
        )
    )

    assert calls == []
