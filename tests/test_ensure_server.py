"""How ``ensure_server`` finds this data folder's server, or starts one.

One test per state and outcome: a record that is serving (asked at the
caller's address), a record that is starting (waited for), the server lock
held without a record (a server stopping, waited for), and neither (the port
asked whose server answers, else a start under the start lock).
"""

from __future__ import annotations

import threading

import httpx
import pytest

from EvoScientist.langgraph_dev import manager, registration, server_record
from tests.fakes.langgraph_dev import LockHolder, Server, write_record


def _config(port: int = 6174):
    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    cfg.langgraph_dev_port = port
    return cfg


@pytest.fixture
def server(monkeypatch, runtime_paths):
    return Server(monkeypatch)


@pytest.fixture
def registered(monkeypatch):
    calls: list = []

    def register_folder(workspace, *, url):
        calls.append((url, workspace))
        return registration.ServerExposure()

    monkeypatch.setattr(registration, "register_folder", register_folder)
    return calls


# ---------------------------------------------------------------------------
# A serving record
# ---------------------------------------------------------------------------


def test_ensure_serving_record_answering_as_ours_is_reused_and_never_killed(server):
    server.run(lasting=True)

    result = manager.ensure_server(_config())

    assert result.available is True
    assert result.started is False
    assert result.record.pid == server.pid
    assert server.spawns == []
    assert server.stopped == []


def test_ensure_serving_record_counts_while_lock_is_held_and_process_runs(server):
    """The record parses, the server lock is held and its process runs as
    recorded: that is a valid record."""
    server.answering.add(6174)
    write_record(9999, created=1.0)

    result = manager.ensure_server(_config())

    assert result.available is True
    assert result.record.pid == 9999


def test_ensure_record_without_server_lock_is_ignored(server, monkeypatch):
    """A record left behind by a server that is gone: the lock is free."""
    import json

    from EvoScientist.langgraph_dev import server_home

    write_record(9999)

    def _held(wait=0):
        record = manager.RUNTIME.record
        held = record.exists() and json.loads(record.read_text())["pid"] != 9999
        return server_home.HELD if held else server_home.FREE

    monkeypatch.setattr(server_home, "lock_state", _held)

    result = manager.ensure_server(_config())

    assert result.started is True


def test_ensure_serving_record_on_another_port_is_refused(server, monkeypatch):
    server.run(port=16400)

    result = manager.ensure_server(_config(6174))

    assert result.served is False
    assert result.problem == "other_port"
    assert "16400" in result.error
    assert "6174" in result.error
    assert manager.SERVER_UNAVAILABLE_REASON == result.reason
    assert server.spawns == []


def test_ensure_serving_record_on_another_port_is_used_with_any_port(server):
    server.run(port=16400)

    result = manager.ensure_server(_config(6174), any_port=True)

    assert result.available is True
    assert (result.port, result.url) == (16400, "http://127.0.0.1:16400")


def test_ensure_serving_record_is_refused_without_reuse(server):
    server.run(port=16401)

    result = manager.ensure_server(_config(), reuse=False)

    assert result.problem == "running"
    assert str(server.pid) in result.error
    assert "16401" in result.error
    assert server.spawns == []


def test_ensure_serving_record_is_asked_at_its_own_address_only(server, monkeypatch):
    """Whatever host is configured, the recorded server is asked where it
    records itself, once per look, and the session uses that address. The
    first look, before any record was read, asks the configured address;
    the record it reads sends the next probe to its own at once."""
    from EvoScientist.langgraph_dev import sdk

    server.run(host="10.0.0.5")
    asked: list = []
    monkeypatch.setattr(
        manager, "ask", lambda url: asked.append(url) or server._ask(url)
    )
    cfg = _config()
    cfg.langgraph_dev_host = "192.168.1.5"
    waits: list = []

    result = manager.ensure_server(cfg, on_wait=waits.append)

    assert asked == ["http://192.168.1.5:6174", "http://10.0.0.5:6174"]
    assert True not in waits
    assert result.url == "http://10.0.0.5:6174"
    assert sdk.configured_langgraph_dev_url() == "http://10.0.0.5:6174"


def test_ensure_serving_record_answered_by_another_server_is_unreachable(
    server, monkeypatch
):
    server.run(host="10.0.0.5")
    monkeypatch.setattr(manager, "ask", lambda url: "other")

    result = manager.ensure_server(_config())

    assert result.problem == "unreachable"
    assert "http://10.0.0.5:6174" in result.error
    assert server.stopped == []


def test_ensure_served_url_falls_back_to_config_without_server(server):
    from EvoScientist.langgraph_dev import sdk

    server.occupied.add(6174)
    cfg = _config()

    result = manager.ensure_server(cfg)

    assert result.available is False
    assert sdk.configured_langgraph_dev_url(cfg) == sdk.langgraph_dev_url(cfg)


def test_ensure_serving_record_that_misses_one_probe_is_reused(server, monkeypatch):
    server.run()
    answers = iter(["silent", "own"])
    monkeypatch.setattr(manager, "ask", lambda url: next(answers))

    result = manager.ensure_server(_config())

    assert result.available is True
    assert server.spawns == []


def test_ensure_serving_record_that_misses_every_probe_is_unresponsive(
    server, monkeypatch
):
    server.run()
    asked: list = []
    monkeypatch.setattr(manager, "ask", lambda url: asked.append(url) or "silent")

    result = manager.ensure_server(_config())

    assert result.problem == "unresponsive"
    assert len(asked) == manager._SERVING_PROBES
    assert server.stopped == []
    assert server.spawns == []


def test_settle_wait_starts_over_once_record_was_seen(monkeypatch):
    """A server whose start fails removes its record before its lock goes:
    that look gets a settle wait of its own, not the one already spent."""
    from types import SimpleNamespace

    from EvoScientist.langgraph_dev import server_home

    now = [100.0]
    monkeypatch.setattr(
        manager,
        "time",
        SimpleNamespace(monotonic=lambda: now[0], sleep=lambda _s: None),
    )
    look = manager._Look(
        asked_url="http://127.0.0.1:6174",
        port=6174,
        host="127.0.0.1",
        any_port=False,
        reuse=True,
        fingerprint="fp",
    )
    clock = manager._Clock()
    held = manager._Seen(
        record=None, lock=server_home.HELD, url=look.asked_url, answer="refused"
    )
    starting = manager._Seen(
        record=manager.ServerRecord(
            pid=4242, port=6174, host="127.0.0.1", phase=server_record.STARTING
        ),
        lock=server_home.HELD,
        url=look.asked_url,
        answer="refused",
    )

    assert manager._decide(held, look, clock, manager._START_BUDGET) == "wait"
    now[0] += 10
    assert manager._decide(starting, look, clock, manager._START_BUDGET) == "wait"
    assert manager._decide(held, look, clock, manager._START_BUDGET) == "wait"


# ---------------------------------------------------------------------------
# A starting record
# ---------------------------------------------------------------------------


def test_ensure_starting_record_is_waited_for_until_it_answers(server):
    server.run(phase=server_record.STARTING)
    server.answering.clear()
    server.on_pause = lambda n: server.answering.add(6174) if n == 3 else None

    result = manager.ensure_server(_config())

    assert result.available is True
    assert server.pauses >= 3
    assert server.spawns == []


def test_ensure_starting_record_that_never_answers_is_unresponsive(server):
    server.run(phase=server_record.STARTING)
    server.answering.clear()
    server.silent.add(6174)

    result = manager.ensure_server(_config())

    assert result.problem == "unresponsive"
    assert server.stopped == []


def test_ensure_record_gone_while_waiting_leads_to_start(server):
    """A stop during the wait: the record goes away, and the call starts a
    server instead of timing out."""
    server.run(pid=7000, phase=server_record.STARTING)
    server.answering.clear()

    def stopped(n):
        if n == 2:
            manager.RUNTIME.record.unlink()

    server.on_pause = stopped

    result = manager.ensure_server(_config())

    assert result.started is True
    assert len(server.spawns) == 1


# ---------------------------------------------------------------------------
# The server lock held without a record
# ---------------------------------------------------------------------------


def test_ensure_lock_held_without_record_is_waited_for_until_record(server):
    server.locked = True
    waits: list = []

    def recorded(n):
        if n == 3:
            server.run()
            server.locked = False

    server.on_pause = recorded

    result = manager.ensure_server(_config(), on_wait=waits.append)

    assert result.available is True
    assert server.spawns == []
    assert waits == [True, False]


def test_ensure_lock_freed_without_record_leads_to_start(server):
    """The server that held it stopped."""
    server.locked = True
    server.on_pause = lambda n: setattr(server, "locked", n < 3)

    result = manager.ensure_server(_config())

    assert result.started is True


def test_ensure_lock_held_without_record_past_settle_is_unrecorded_lock(server):
    """A server that keeps the lock without a record, and does not answer as
    this data folder's: named, with how to stop it by hand; nothing is
    started."""
    server.locked = True

    result = manager.ensure_server(_config())

    assert result.problem == "unrecorded_lock"
    assert str(manager.RUNTIME.server_lock) in result.error
    assert "by hand" in result.error
    assert "try again" not in result.error.lower()
    assert server.spawns == []


def test_ensure_settle_waits_only_briefly_for_record(server, monkeypatch):
    server.locked = True
    monkeypatch.setattr(manager, "_SETTLE_WAIT", 0.0)

    result = manager.ensure_server(_config())

    assert result.problem == "unrecorded_lock"
    assert server.pauses == 0


def _unreadable_lock(monkeypatch):
    from EvoScientist.langgraph_dev import server_home

    looks: list = []

    def _unknown(wait=0):
        looks.append(1)
        return server_home.UNKNOWN

    monkeypatch.setattr(server_home, "lock_state", _unknown)
    return looks


def test_ensure_unreadable_lock_is_lock_unreadable_without_loop(server, monkeypatch):
    looks = _unreadable_lock(monkeypatch)
    server.run()  # its record is not trusted: the lock proves nothing

    server.answering.clear()
    result = manager.ensure_server(_config())

    assert result.problem == "lock_unreadable"
    assert str(manager.RUNTIME.server_lock) in result.error
    assert len(looks) <= 2
    assert server.spawns == []


def test_ensure_unreadable_lock_reuses_server_answering_as_own(server, monkeypatch):
    _unreadable_lock(monkeypatch)
    server.answering.add(6174)

    result = manager.ensure_server(_config())

    assert result.available is True
    assert server.spawns == []


# ---------------------------------------------------------------------------
# No record, the server lock free: the port
# ---------------------------------------------------------------------------


def test_ensure_free_port_answering_as_ours_is_reused(server):
    """A server of this data folder started by hand, outside the server
    home: it holds no lock and keeps no record."""
    server.answering.add(6174)

    result = manager.ensure_server(_config())

    assert result.available is True
    assert result.record is None
    assert server.spawns == []


@pytest.mark.parametrize(
    ("whose", "problem"),
    [("other", "other_data_folder"), ("older", "older_server")],
)
def test_ensure_free_port_answering_as_another_server_is_conflict(
    server, whose, problem
):
    server.answering.add(6174)
    server.whose[6174] = whose

    result = manager.ensure_server(_config())

    assert result.problem == problem
    assert server.spawns == []


def test_ensure_for_project_server_too_old_to_say_whose_it_is_raises(server, tmp_path):
    server.answering.add(6174)
    server.whose[6174] = "older"

    with pytest.raises(registration.OlderServerError) as raised:
        manager.ensure_server(_config(), workspace_dir=tmp_path)
    assert manager.SERVER_UNAVAILABLE is True
    # It left no record server stop could find it by.
    assert "port 6174" in str(raised.value)
    assert "EvoSci server stop" not in str(raised.value)


def test_ensure_port_another_program_holds_is_foreign_after_bind_wait(server):
    server.occupied.add(6174)

    result = manager.ensure_server(_config())

    assert result.problem == "foreign"
    assert result.port == 6174
    assert server.spawns == []


def test_ensure_port_closing_listener_frees_is_started_on(server, monkeypatch):
    """Something still holds the port but frees it within the bind wait."""
    server.silent.add(6174)
    monkeypatch.setattr(manager, "_wait_for_port_bindable", lambda port, host: True)

    result = manager.ensure_server(_config())

    assert result.started is True


def test_ensure_free_port_answering_as_ours_is_refused_without_reuse(server):
    server.answering.add(6174)

    result = manager.ensure_server(_config(), reuse=False)

    assert result.problem == "running"


def test_ask_ok_answer_is_judged_by_instance(monkeypatch, runtime_paths):
    monkeypatch.setattr(manager.httpx, "get", lambda url, **_kw: httpx.Response(200))
    monkeypatch.setattr(manager, "_whose_server", lambda url: "other")

    assert manager.ask("http://127.0.0.1:1") == "other"


def test_whose_server_tells_this_data_folders_from_another(monkeypatch, runtime_paths):
    from EvoScientist import local_secret

    monkeypatch.setattr(local_secret, "read_local_secret", lambda: "s3cret")
    own = local_secret.instance_of("s3cret")
    monkeypatch.setattr(registration, "server_instance", lambda url, **_k: own)
    assert manager._whose_server("http://127.0.0.1:6174") == "own"

    monkeypatch.setattr(
        registration, "server_instance", lambda url, **_k: "someone-else"
    )
    assert manager._whose_server("http://127.0.0.1:6174") == "other"

    monkeypatch.setattr(local_secret, "read_local_secret", lambda: None)
    assert manager._whose_server("http://127.0.0.1:6174") == "other"

    def _older(url, **_k):
        raise registration.OlderServerError()

    monkeypatch.setattr(registration, "server_instance", _older)
    assert manager._whose_server("http://127.0.0.1:6174") == "older"


@pytest.mark.parametrize("locked", [True, False])
def test_ensure_for_project_older_server_without_record_names_its_stop(
    server, monkeypatch, tmp_path, locked
):
    """An older server reused without a record that turns out too old for
    the registry: by the lock it holds, or by its port when it holds none;
    never EvoSci server stop, which cannot stop it."""
    server.locked = locked
    server.answering.add(6174)

    def _older(workspace, *, url):
        raise registration.OlderServerError()

    monkeypatch.setattr(registration, "register_folder", _older)

    with pytest.raises(registration.OlderServerError) as raised:
        manager.ensure_server(_config(), workspace_dir=tmp_path)
    message = str(raised.value)
    assert "EvoSci server stop" not in message
    if locked:
        assert str(manager.RUNTIME.server_lock) in message
    else:
        assert "port 6174" in message


# ---------------------------------------------------------------------------
# The start
# ---------------------------------------------------------------------------


def test_ensure_starts_server_and_owns_its_process(server):
    result = manager.ensure_server(_config())

    assert result.started is True
    assert result.proc is server.procs[0]
    assert result.record.pid == server.pid
    assert result.url == "http://127.0.0.1:6174"


def test_ensure_passes_start_what_it_was_asked(server):
    class _Job:
        started_with = None

        def started(self, proc):
            self.started_with = proc

    job = _Job()

    manager.ensure_server(
        _config(),
        port=16402,
        host="0.0.0.0",
        lasting=False,
        tunnel=True,
        job=lambda: job,
        any_port=True,
    )

    spawn = server.spawns[0]
    assert (spawn["port"], spawn["host"]) == (16402, "0.0.0.0")
    assert spawn["lasting"] is False
    assert spawn["tunnel"] is True
    assert spawn["config_fingerprint"] == "fp"
    assert job.started_with is server.procs[0]


def test_ensure_runs_before_start_right_before_start_only(server):
    calls: list = []

    def before():
        assert manager.RUNTIME.lock_file.exists()
        calls.append(len(server.spawns))

    manager.ensure_server(_config(), before_start=before)
    assert calls == [0]

    manager.ensure_server(_config(), before_start=before)
    assert calls == [0], "a reuse starts nothing"


def test_ensure_reports_exit_of_server_it_started(server, runtime_paths):
    runtime_paths.log_file.write_text("Traceback: address already in use\n")
    server.exit_code = 3

    result = manager.ensure_server(_config())

    assert result.problem == "exited"
    assert result.exit_code == 3
    assert "address already in use" in result.error
    assert result.served is False


def test_unavailable_reason_names_log_not_its_end(server, runtime_paths):
    """The notice is one line; the log's end stays in the error."""
    runtime_paths.log_file.write_text("Traceback\n  line\naddress in use\n")
    server.exit_code = 3

    manager.ensure_server(_config())

    reason = manager.SERVER_UNAVAILABLE_REASON
    assert "\n" not in reason
    assert "address in use" not in reason
    assert str(runtime_paths.log_file) in reason


def test_ensure_keeps_started_server_whose_serving_rewrite_failed(server, monkeypatch):
    """Its record stays ``starting``; it answers as ours, so it is started
    and not stopped."""
    spawn = server._spawn

    def _stays_starting(**kwargs):
        proc = spawn(**kwargs)
        write_record(proc.pid, port=kwargs["port"], phase=server_record.STARTING)
        return proc

    monkeypatch.setattr(manager, "_spawn_server", _stays_starting)

    result = manager.ensure_server(_config())

    assert result.started is True
    assert server.stopped == []


def test_ensure_stops_server_whose_first_write_failed(server):
    """A server that cannot write its record exits; one that does not is
    stopped once the start budget is spent."""
    server.records = False

    result = manager.ensure_server(_config())

    assert result.problem == "start_failed"
    assert server.stopped == [server.pid]
    assert manager._PROCESS is None


def test_ensure_stops_server_it_started_that_never_answers(server, monkeypatch):
    spawn = server._spawn

    def _silent(**kwargs):
        proc = spawn(**kwargs)
        server.answering.clear()
        server.silent.add(kwargs["port"])
        return proc

    monkeypatch.setattr(manager, "_spawn_server", _silent)

    result = manager.ensure_server(_config())

    assert result.problem == "start_failed"
    assert server.stopped == [server.pid]


def test_ensure_reports_start_that_cannot_run(server, monkeypatch):
    def _missing(**_kw):
        raise FileNotFoundError("langgraph CLI not found")

    monkeypatch.setattr(manager, "_spawn_server", _missing)

    result = manager.ensure_server(_config())

    assert result.problem == "start_failed"
    assert "langgraph CLI not found" in result.error


def test_ensure_reports_start_lock_it_cannot_use(server, monkeypatch, tmp_path):
    from tests.fakes.langgraph_dev import break_start_lock

    break_start_lock(monkeypatch, tmp_path, "acquire")

    result = manager.ensure_server(_config())

    assert result.problem == "start_failed"
    assert server.spawns == []


def test_ensure_waits_for_another_start_and_reuses_its_server(server):
    holder = LockHolder()
    waits: list = []

    def other_started():
        server.run(pid=8000)
        holder.release.set()

    timer = threading.Timer(0.05, other_started)
    timer.start()
    try:
        result = manager.ensure_server(_config(), on_wait=waits.append)
    finally:
        timer.join()
        holder.done()

    assert result.available is True
    assert result.record.pid == 8000
    assert server.spawns == []
    assert waits[0] is True
    assert waits[-1] is False


def test_ensure_gives_up_on_start_lock_held_too_long(server):
    holder = LockHolder()
    try:
        result = manager.ensure_server(_config())
    finally:
        holder.done()

    assert result.problem == "busy"
    assert server.spawns == []


def test_ensure_cancelled_while_waiting_starts_nothing(server):
    holder = LockHolder()
    try:
        result = manager.ensure_server(_config(), cancelled=lambda: True)
    finally:
        holder.done()

    assert result.problem == "cancelled"
    assert server.spawns == []


def test_ensure_cancelled_during_start_stops_its_server(server, monkeypatch):
    server.records = False
    cancel = {"now": False}
    server.on_pause = lambda n: cancel.update(now=True)

    result = manager.ensure_server(_config(), cancelled=lambda: cancel["now"])

    assert result.problem == "cancelled"
    assert server.stopped == [server.pid]


def test_ensure_reports_port_that_never_frees_before_start(server):
    server.unbindable.add(6174)

    result = manager.ensure_server(_config())

    assert result.problem == "foreign"
    assert result.port == 6174
    assert server.spawns == []


def test_ensure_runs_nothing_before_start_on_port_that_never_frees(server):
    server.unbindable.add(6174)
    calls: list = []

    result = manager.ensure_server(_config(), before_start=lambda: calls.append(1))

    assert result.problem == "foreign"
    assert calls == []


def test_ensure_installs_node_under_start_lock(server, monkeypatch):
    from tests.fakes.langgraph_dev import start_lock_is_held

    held: list = []
    monkeypatch.setattr(
        manager,
        "_install_node_for_mcp",
        lambda waits: held.append(start_lock_is_held()),
    )

    manager.ensure_server(_config())

    assert held == [True]


def test_node_install_stops_waiting_on_cancel(monkeypatch):
    import threading

    from EvoScientist.mcp import client as mcp_client

    release = threading.Event()
    monkeypatch.setattr(mcp_client, "load_mcp_config", lambda: {})
    monkeypatch.setattr(
        mcp_client, "_ensure_node_for_stdio", lambda _c: release.wait(5)
    )
    monkeypatch.setattr(manager, "_POLL_INTERVAL", 0.01)
    try:
        manager._install_node_for_mcp(
            manager._Waits(lambda: True, None, manager._START_BUDGET)
        )
    finally:
        release.set()


def test_ensure_releases_start_lock(server):
    from tests.fakes.langgraph_dev import start_lock_is_held

    manager.ensure_server(_config())

    assert not start_lock_is_held()


# ---------------------------------------------------------------------------
# The registration
# ---------------------------------------------------------------------------


def test_ensure_registers_project_with_server_it_found(server, registered, tmp_path):
    server.run()

    result = manager.ensure_server(_config(), workspace_dir=tmp_path)

    assert result.served is True
    assert [url for url, _ in registered] == ["http://127.0.0.1:6174"]


def test_ensure_without_registration_falls_back(server, monkeypatch, tmp_path):
    server.run()

    def _fail(workspace, *, url):
        raise registration.RegistrationError("did not register")

    monkeypatch.setattr(registration, "register_folder", _fail)

    result = manager.ensure_server(_config(), workspace_dir=tmp_path)

    assert result.served is False
    assert result.problem == "unregistered"
    assert manager.is_async_subagents_available() is False


def test_ensure_leaves_session_as_it_was_for_missing_project(
    server, monkeypatch, tmp_path
):
    server.run()
    monkeypatch.setattr(manager, "_ASYNC_SUBAGENTS_AVAILABLE", True)

    def _missing(workspace, *, url):
        raise registration.ProjectFolderMissingError(tmp_path)

    monkeypatch.setattr(registration, "register_folder", _missing)

    with pytest.raises(registration.ProjectFolderMissingError):
        manager.ensure_server(_config(), workspace_dir=tmp_path)
    assert manager.is_async_subagents_available() is True


def test_ensure_does_not_register_without_server(server, registered, tmp_path):
    server.occupied.add(6174)

    manager.ensure_server(_config(), workspace_dir=tmp_path)

    assert registered == []


def test_ensure_record_written_by_hand_counts(server):
    """Any process can be the recorded server, as long as it is live."""
    server.processes.add(9100)
    server.answering.add(6174)
    write_record(9100)

    result = manager.ensure_server(_config())

    assert result.record.pid == 9100


def test_ensure_older_releases_record_kept_across_probes(server, monkeypatch):
    from tests.fakes.langgraph_dev import write_older_record

    server.processes.add(4141)
    write_older_record(4141)
    answers = iter(["silent", "own"])
    monkeypatch.setattr(manager, "ask", lambda url: next(answers))

    result = manager.ensure_server(_config())

    assert result.available is True
    assert result.record.older is True


# ---------------------------------------------------------------------------
# The answers at the record's own address, interruptions
# ---------------------------------------------------------------------------


def test_ensure_serving_record_refused_at_its_own_address_is_probed_again(
    server, monkeypatch
):
    """Marked serving before it binds, or draining before it removes its
    record: a refusal at its own address is a missed probe."""
    server.run()
    answers = iter(["refused", "own"])
    monkeypatch.setattr(manager, "ask", lambda url: next(answers))

    result = manager.ensure_server(_config())

    assert result.available is True


def test_ensure_serving_record_refused_at_its_own_address_every_time_is_stopping(
    server, monkeypatch
):
    """Nothing listens while it holds the lock: waited out as a stop."""
    server.run()
    monkeypatch.setattr(manager, "ask", lambda url: "refused")

    result = manager.ensure_server(_config())

    assert result.problem == "stopping"
    assert result.record.url in result.error
    assert server.stopped == []


def test_ensure_serving_record_answering_as_older_release_is_older_server(
    server, monkeypatch, tmp_path
):
    server.run()
    monkeypatch.setattr(manager, "ask", lambda url: "older")

    assert manager.ensure_server(_config()).problem == "older_server"
    with pytest.raises(registration.OlderServerError) as raised:
        manager.ensure_server(_config(), workspace_dir=tmp_path)
    # server stop would end the recorded server, not the one answering.
    assert "EvoSci server stop" not in str(raised.value)


def test_unresponsive_older_releases_server_names_its_own_log():
    record = manager.ServerRecord(pid=1, port=6174, host="127.0.0.1", older=True)

    result = manager._unresponsive(record, "does not answer")

    assert result.log_file == manager.OLDER_RUNTIME_DIR / "langgraph_dev.log"
    assert str(result.log_file) in result.error


def test_ensure_interrupt_while_waiting_for_start_stops_child(server):
    server.records = False

    def _interrupt(n):
        if server.spawns:
            raise KeyboardInterrupt

    server.on_pause = _interrupt

    with pytest.raises(KeyboardInterrupt):
        manager.ensure_server(_config())

    assert server.stopped == [server.pid]
    assert manager._PROCESS is None


def test_ensure_interrupt_leaves_server_started_earlier_alone(server, monkeypatch):
    """A lasting server this process started in an earlier call is not this
    call's child."""
    from tests.fakes.langgraph_dev import FakeProc

    earlier = FakeProc(1234)
    monkeypatch.setattr(manager, "_PROCESS", earlier)
    server.locked = True

    def _interrupt(n):
        raise KeyboardInterrupt

    server.on_pause = _interrupt

    with pytest.raises(KeyboardInterrupt):
        manager.ensure_server(_config())

    assert server.stopped == []
    assert manager._PROCESS is earlier


def test_ensure_failing_before_start_surfaces_its_own_error(server):
    def _fails():
        raise PermissionError(13, "job object refused")

    with pytest.raises(PermissionError, match="job object refused"):
        manager.ensure_server(_config(), before_start=_fails)
    assert server.spawns == []


def test_ensure_start_that_finds_another_server_pauses_before_looking_again(
    server, monkeypatch
):
    """Another process changed what runs while this one waited for the
    start lock: the call looks again after a pause, not in a tight loop."""
    spawn = manager._spawn
    calls: list = []

    def _changed_once(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return None
        return spawn(*a, **k)

    monkeypatch.setattr(manager, "_spawn", _changed_once)

    result = manager.ensure_server(_config())

    assert result.started is True
    assert server.pauses >= 1


def test_ensure_draining_server_removing_its_record_leads_to_start(server):
    """A stopping server closed its listener, then removed its record."""
    server.run(pid=7000)
    server.answering.clear()

    def drained(n):
        if n == 3:
            manager.RUNTIME.record.unlink()

    server.on_pause = drained

    result = manager.ensure_server(_config())

    assert result.started is True
    assert server.stopped == []


def test_ensure_serving_record_not_bound_yet_is_reused_once_it_answers(server):
    """Marked serving before it binds: refused, then it answers."""
    server.run()
    server.answering.clear()
    server.on_pause = lambda n: server.answering.add(6174) if n == 2 else None

    result = manager.ensure_server(_config())

    assert result.available is True
    assert server.spawns == []


def test_ensure_silent_serving_record_is_unresponsive_after_three_probes(
    server, monkeypatch
):
    server.run()
    asked: list = []
    monkeypatch.setattr(manager, "ask", lambda url: asked.append(url) or "silent")

    result = manager.ensure_server(_config())

    assert result.problem == "unresponsive"
    assert len(asked) == 3


def test_ensure_starting_record_waited_for_at_its_own_address(server, monkeypatch):
    server.run(host="10.0.0.5", phase=server_record.STARTING)
    answers = iter(["refused", "refused", "own"])
    monkeypatch.setattr(manager, "ask", lambda url: next(answers))

    result = manager.ensure_server(_config())

    assert result.available is True
    assert result.url == "http://10.0.0.5:6174"


def test_ensure_starting_record_at_configured_address_reused_once_it_answers(server):
    server.run(phase=server_record.STARTING)

    result = manager.ensure_server(_config())

    assert result.available is True


def test_ensure_older_record_gone_after_missed_probe_leads_to_start(
    server, monkeypatch
):
    from tests.fakes.langgraph_dev import write_older_record

    server.processes.add(4141)
    write_older_record(4141)
    answers = iter(["silent"])

    def _ask(url):
        answer = next(answers, None)
        if answer is None:
            return server._ask(url)
        server.processes.end(4141)  # its server is gone
        return answer

    monkeypatch.setattr(manager, "ask", _ask)

    result = manager.ensure_server(_config())

    assert result.started is True


def test_ensure_start_budget_covers_record_and_first_answer(server, monkeypatch):
    """One budget from the spawn: a record that comes late leaves less time
    for the first answer, and the failure says how long it took."""
    import re
    from types import SimpleNamespace

    clock = {"now": 0.0}

    def _monotonic():
        clock["now"] += 1.0
        return clock["now"]

    monkeypatch.setattr(
        manager, "time", SimpleNamespace(monotonic=_monotonic, sleep=lambda _s: None)
    )
    monkeypatch.setattr(manager, "_START_BUDGET", 10.0)
    server.records = False

    def recorded_late(n):
        if n == 3 and server.spawns and not manager.RUNTIME.record.exists():
            write_record(server.pid)
            server.silent.add(6174)

    server.on_pause = recorded_late

    result = manager.ensure_server(_config())

    assert result.problem == "start_failed"
    elapsed = int(re.search(r"within (\d+) seconds", result.error).group(1))
    assert elapsed <= 12


# ---------------------------------------------------------------------------
# The started server's process, whatever the outcome
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("whose", "problem"),
    [("other", "unreachable"), ("older", "older_server"), ("foreign", "unreachable")],
)
def test_ensure_spawned_child_stopped_when_its_address_answers_otherwise(
    server, monkeypatch, whose, problem
):
    spawn = server._spawn

    def _answered_by_another(**kwargs):
        proc = spawn(**kwargs)
        server.whose[kwargs["port"]] = whose
        return proc

    monkeypatch.setattr(manager, "_spawn_server", _answered_by_another)

    result = manager.ensure_server(_config())

    assert result.problem == problem
    assert server.stopped == [server.pid]
    assert manager._PROCESS is None


def test_ensure_spawned_child_stopped_on_interrupt_after_its_record(
    server, monkeypatch
):
    spawn = server._spawn

    def _silent(**kwargs):
        proc = spawn(**kwargs)
        server.answering.clear()
        server.silent.add(kwargs["port"])
        return proc

    monkeypatch.setattr(manager, "_spawn_server", _silent)

    def _interrupt(n):
        if manager.RUNTIME.record.exists():
            raise KeyboardInterrupt

    server.on_pause = _interrupt

    with pytest.raises(KeyboardInterrupt):
        manager.ensure_server(_config())

    assert server.stopped == [server.pid]


def test_ensure_spawn_accepts_record_of_descendant(server, monkeypatch):
    """On Windows the server runs below the launcher the starter started."""
    spawn = server._spawn

    def _child_records(**kwargs):
        proc = spawn(**kwargs)
        server.run(5555, port=kwargs["port"])
        return proc

    monkeypatch.setattr(manager, "_spawn_server", _child_records)
    monkeypatch.setattr(manager, "_spawned_pids", lambda proc: {proc.pid, 5555})

    result = manager.ensure_server(_config())

    assert result.started is True
    assert result.record.pid == 5555


def test_ensure_spawn_never_takes_record_of_another_process(server, monkeypatch):
    """A record a killed server left, its PID alive again in another process,
    while the started one has not written its own."""
    server.records = False
    spawn = server._spawn

    def _stale(**kwargs):
        proc = spawn(**kwargs)
        server.run(7777, port=kwargs["port"])
        return proc

    monkeypatch.setattr(manager, "_spawn_server", _stale)

    result = manager.ensure_server(_config())

    assert result.problem == "start_failed"
    assert server.stopped == [server.pid]


# ---------------------------------------------------------------------------
# What a look sees
# ---------------------------------------------------------------------------


def test_ensure_record_switching_to_another_server_resets_probes(server, monkeypatch):
    server.run(pid=1001)
    asked: list = []
    monkeypatch.setattr(manager, "ask", lambda url: asked.append(url) or "silent")
    server.on_pause = lambda n: server.run(pid=1002) if n == 2 else None

    result = manager.ensure_server(_config())

    assert result.problem == "unresponsive"
    assert result.record.pid == 1002
    # Two misses of the first server, then three of the second.
    assert len(asked) == 5


def test_ensure_reuse_judges_record_read_after_probe(server, monkeypatch):
    """The answer comes from a server that has marked itself serving: the
    record read after it carries what the serving rewrite added."""
    server.run(phase=server_record.STARTING)

    def _serving_then_own(url):
        write_record(server.pid, agent_python="/env/bin/python")
        return "own"

    monkeypatch.setattr(manager, "ask", _serving_then_own)

    result = manager.ensure_server(_config())

    assert result.record.phase == server_record.SERVING
    assert result.record.fields["agent_python"] == "/env/bin/python"


def test_ensure_stale_record_with_dead_pid_is_ignored_while_lock_held(server):
    """A new server holds the lock and has not yet written its record over
    the one a killed server left."""
    write_record(9999)
    server.dead.add(9999)
    server.locked = True
    server.on_pause = lambda n: server.run() if n == 2 else None

    result = manager.ensure_server(_config())

    assert result.available is True
    assert result.record.pid == server.pid


def test_ensure_silent_probes_count_only_in_row(server, monkeypatch):
    """A refused answer between two silent ones starts the count again."""
    server.run()
    answers = iter(["silent", "silent", "refused", "silent", "silent", "silent"])
    asked: list = []

    def _ask(url):
        asked.append(url)
        return next(answers)

    monkeypatch.setattr(manager, "ask", _ask)

    result = manager.ensure_server(_config())

    assert result.problem == "unresponsive"
    assert len(asked) == 6


def test_ensure_reads_record_once_per_look(server, monkeypatch):
    server.run()
    reads: list = []
    read = manager.read_record

    def _read(**kwargs):
        reads.append(kwargs)
        return read(**kwargs)

    monkeypatch.setattr(manager, "read_record", _read)

    manager.ensure_server(_config())

    assert len(reads) == 1
    assert reads[0]["older"] is True
    assert reads[0]["lock"] == "held"


def test_ensure_lock_held_without_record_answering_own_is_older_release(server):
    """An older release's server keeps the lock and no record here: used,
    with the notice to restart it."""
    server.locked = True
    server.answering.add(6174)

    result = manager.ensure_server(_config())

    assert result.available is True
    assert result.older is True
    # It left no record server stop could stop it by.
    assert manager.OLDER_SERVER == manager.older_unrecorded_message()
    assert "EvoSci server stop" not in manager.OLDER_SERVER


def test_ensure_foreign_answer_without_record_is_foreign_at_once(server, monkeypatch):
    server.occupied.add(6174)
    waited: list = []
    monkeypatch.setattr(
        manager, "_wait_for_port_bindable", lambda *a, **k: waited.append(1)
    )

    result = manager.ensure_server(_config())

    assert result.problem == "foreign"
    assert waited == []
    assert server.spawns == []


def test_ensure_silent_port_without_record_waits_for_bind(server, monkeypatch):
    server.silent.add(6174)
    waited: list = []
    monkeypatch.setattr(
        manager, "_wait_for_port_bindable", lambda *a, **k: waited.append(1) or False
    )

    result = manager.ensure_server(_config())

    assert result.problem == "foreign"
    assert waited == [1]


def test_ensure_foreign_answer_at_record_address_is_unreachable(server, monkeypatch):
    server.run()
    monkeypatch.setattr(manager, "ask", lambda url: "foreign")

    result = manager.ensure_server(_config())

    assert result.problem == "unreachable"


@pytest.mark.parametrize(
    ("status", "answer"),
    [(400, "foreign"), (404, "foreign"), (500, "silent"), (503, "silent")],
)
def test_ask_ok_error_status_answer(monkeypatch, runtime_paths, status, answer):
    """A 4xx is not an EvoScientist server's answer; a 5xx is a busy or
    failing server's, asked again."""
    monkeypatch.setattr(manager.httpx, "get", lambda url, **_kw: httpx.Response(status))

    assert manager.ask("http://127.0.0.1:1") == answer


def test_ensure_spawned_child_exiting_is_exited(server, monkeypatch):
    spawn = server._spawn

    def _exits(**kwargs):
        proc = spawn(**kwargs)
        proc.exit(3)
        return proc

    monkeypatch.setattr(manager, "_spawn_server", _exits)

    result = manager.ensure_server(_config())

    assert result.problem == "exited"
    assert result.exit_code == 3


@pytest.mark.parametrize("missed", ["silent", "refused"])
def test_ensure_spawned_child_not_answering_within_budget_is_stopped(
    server, monkeypatch, missed
):
    spawn = server._spawn

    def _quiet(**kwargs):
        proc = spawn(**kwargs)
        server.answering.clear()
        (server.silent if missed == "silent" else server.unbindable).add(kwargs["port"])
        return proc

    monkeypatch.setattr(manager, "_spawn_server", _quiet)

    result = manager.ensure_server(_config())

    assert result.problem == "start_failed"
    assert server.stopped == [server.pid]


def test_ensure_spawned_child_answering_counts_without_its_record(server, monkeypatch):
    """Once the started server recorded itself, only its process and its
    answer count: a record that cannot be read meanwhile changes nothing."""
    from EvoScientist.langgraph_dev import server_home

    spawn = server._spawn

    def _quiet(**kwargs):
        proc = spawn(**kwargs)
        server.answering.clear()
        return proc

    def _record_lost_then_answers(n):
        if server.spawns:
            manager.RUNTIME.record.unlink(missing_ok=True)
            monkeypatch.setattr(
                server_home, "lock_state", lambda wait=0: server_home.UNKNOWN
            )
            server.answering.add(6174)

    monkeypatch.setattr(manager, "_spawn_server", _quiet)
    server.on_pause = _record_lost_then_answers

    result = manager.ensure_server(_config())

    assert result.started is True
    assert result.available is True
    # The record read before it answered.
    assert result.record.pid == server.pid
    assert server.stopped == []


_FAILURES = {
    "timeout": lambda: httpx.ReadTimeout("slow"),
    "lost": lambda: httpx.ReadError("connection reset"),
    "unsent": lambda: httpx.WriteError("broken pipe"),
    "refused": lambda: httpx.ConnectError("refused"),
    "not_http": lambda: httpx.RemoteProtocolError("not HTTP"),
    "undecodable": lambda: httpx.DecodingError("bad encoding"),
}


@pytest.mark.parametrize("request_path", ["/ok", "/api/instance"])
@pytest.mark.parametrize(
    ("failure", "answer"),
    [
        ("timeout", "silent"),
        ("lost", "silent"),
        ("unsent", "silent"),
        ("refused", "refused"),
        ("not_http", "silent"),
        ("undecodable", "silent"),
    ],
)
def test_ask_request_failure_answer(
    monkeypatch, runtime_paths, request_path, failure, answer
):
    """Either request failing the same way gives the same answer."""
    import httpx

    def _get(url, **_kw):
        if url.endswith(request_path):
            raise _FAILURES[failure]()
        return httpx.Response(200)

    monkeypatch.setattr(manager.httpx, "get", _get)

    assert manager.ask("http://127.0.0.1:1") == answer


@pytest.mark.parametrize(
    ("status", "answer"),
    [(404, "older"), (400, "silent"), (403, "silent"), (500, "silent")],
)
def test_ask_instance_error_status_answer(monkeypatch, runtime_paths, status, answer):
    """``/ok`` answered as a server's: an error on ``/api/instance`` other
    than 404 is asked again."""

    def _get(url, **_kw):
        return httpx.Response(200 if url.endswith("/ok") else status)

    monkeypatch.setattr(manager.httpx, "get", _get)

    assert manager.ask("http://127.0.0.1:1") == answer


def test_ensure_started_result_carries_serving_record(server, monkeypatch):
    """The record as the server rewrote it once serving, with what it added
    then, not the first one the start read."""
    spawn = server._spawn

    def _starting(**kwargs):
        proc = spawn(**kwargs)
        write_record(proc.pid, port=kwargs["port"], phase=server_record.STARTING)
        server.answering.clear()
        return proc

    def _serving(n):
        if server.spawns and not server.answering:
            write_record(server.pid, python="/usr/bin/python3")
            server.answering.add(6174)

    monkeypatch.setattr(manager, "_spawn_server", _starting)
    server.on_pause = _serving

    result = manager.ensure_server(_config())

    assert result.started is True
    assert result.record.phase == server_record.SERVING
    assert result.record.fields["python"] == "/usr/bin/python3"


def test_configured_langgraph_dev_url_prefers_served_url():
    from EvoScientist.langgraph_dev import sdk

    cfg = _config()
    assert sdk.configured_langgraph_dev_url(cfg) == "http://127.0.0.1:6174"

    sdk.set_served_url("http://10.0.0.5:6174")
    assert sdk.configured_langgraph_dev_url(cfg) == "http://10.0.0.5:6174"


def test_ask_malformed_url_raises(monkeypatch, runtime_paths):
    def _get(url, **_kw):
        raise httpx.InvalidURL("bad host")

    monkeypatch.setattr(manager.httpx, "get", _get)

    with pytest.raises(httpx.InvalidURL):
        manager.ask("http://bad host:1")


# ---------------------------------------------------------------------------
# A start that another process stops, or still runs
# ---------------------------------------------------------------------------


def test_start_in_progress_while_another_process_holds_start_lock(server):
    assert manager.start_in_progress() is False
    starting = LockHolder()
    try:
        assert manager.start_in_progress() is True
    finally:
        starting.done()


def test_ensure_server_reports_start_ended_by_sigterm_as_stopped(server):
    """`EvoSci server stop` from another terminal ends a start this way (a
    negative code: POSIX)."""
    server.exit_code = -15

    result = manager.ensure_server(_config())

    assert result.problem == "exited"
    assert result.exit_code == -15
    assert "stopped (SIGTERM)" in result.error
    assert "exited immediately" not in result.error


def test_ensure_server_reports_start_ended_by_other_signal_with_its_code(server):
    server.exit_code = -11

    result = manager.ensure_server(_config())

    assert "exited immediately with code -11" in result.error


def test_ensure_server_reports_start_that_could_not_start_ccproxy(server):
    """The server leaves the reason as it exits; a session gets what to do
    on one line."""
    from EvoScientist.langgraph_dev.server_ccproxy import CCPROXY_FAILED_EXIT

    server.exit_code = CCPROXY_FAILED_EXIT
    manager.RUNTIME.ccproxy_failure.parent.mkdir(parents=True, exist_ok=True)
    manager.RUNTIME.ccproxy_failure.write_text(
        "ccproxy is not signed in.\nRun: ccproxy auth login claude_api",
        encoding="utf-8",
    )

    result = manager.ensure_server(_config())

    assert result.problem == "ccproxy_failed"
    assert result.log_file == manager.RUNTIME.log_file
    assert "\n" not in result.reason
    assert "ccproxy auth login" in result.reason


def test_ensure_for_project_stops_when_server_cannot_start_ccproxy(server, tmp_path):
    """Anthropic is out of reach in the session too: no fallback helps."""
    from EvoScientist.langgraph_dev.server_ccproxy import CCPROXY_FAILED_EXIT

    server.exit_code = CCPROXY_FAILED_EXIT
    manager.RUNTIME.ccproxy_failure.parent.mkdir(parents=True, exist_ok=True)
    manager.RUNTIME.ccproxy_failure.write_text(
        "Run: ccproxy auth login claude_api", encoding="utf-8"
    )

    with pytest.raises(registration.ServerUnusableError, match="ccproxy auth login"):
        manager.ensure_server(_config(), workspace_dir=tmp_path)


def test_start_budget_covers_ccproxy_start_with_anthropic_oauth():
    from EvoScientist import ccproxy_manager

    config = _config()
    without = manager.start_budget(config)
    config.anthropic_auth_mode = "oauth"

    # Its port may need a moment to come free first.
    assert manager.start_budget(config) == (
        without + manager._BIND_WAIT + ccproxy_manager.START_BUDGET_SECONDS
    )


def test_forget_record_ends_ccproxy_it_names(runtime_paths, monkeypatch):
    from EvoScientist import ccproxy_manager

    ended = []
    monkeypatch.setattr(
        ccproxy_manager,
        "end_ccproxy",
        lambda pid, created, port: ended.append((pid, created)) or True,
    )
    write_record(
        9,
        **{
            server_record.CCPROXY_PID_KEY: 77,
            server_record.CCPROXY_CREATE_TIME_KEY: 12.5,
        },
    )

    # A run that started while the server stopped started this ccproxy.
    manager._forget_record(9)

    assert ended == [(77, 12.5)]
    assert not manager.RUNTIME.record.exists()


def _oauth_config(provider: str = "anthropic"):
    config = _config()
    config.provider = provider
    config.anthropic_auth_mode = "oauth"
    return config


@pytest.mark.parametrize("available", [True, False])
def test_ccproxy_missing_notice_when_nothing_answers(monkeypatch, available):
    from EvoScientist import ccproxy_manager

    monkeypatch.setattr(ccproxy_manager, "is_ccproxy_running", lambda port: False)
    monkeypatch.setattr(registration, "ensure_ccproxy", lambda url: None)

    notice = manager._ccproxy_missing(
        _oauth_config(), manager.ServerResult(available=available, url="http://s")
    )

    assert notice is not None
    # A server that runs can be restarted; without one, its problem comes first.
    assert ("EvoSci server stop" in notice) is available


def test_ccproxy_missing_no_notice_when_server_starts_ccproxy(monkeypatch):
    from EvoScientist import ccproxy_manager

    answers = iter([False, True])
    monkeypatch.setattr(
        ccproxy_manager, "is_ccproxy_running", lambda port: next(answers)
    )
    asked = []
    monkeypatch.setattr(
        registration, "ensure_ccproxy", lambda url: asked.append(url) or None
    )

    notice = manager._ccproxy_missing(
        _oauth_config(), manager.ServerResult(available=True, url="http://s")
    )

    assert notice is None
    assert asked == ["http://s"]


def test_ccproxy_missing_notice_says_why_server_could_not_start_it(monkeypatch):
    from EvoScientist import ccproxy_manager

    monkeypatch.setattr(ccproxy_manager, "is_ccproxy_running", lambda port: False)
    monkeypatch.setattr(
        registration, "ensure_ccproxy", lambda url: "Run: ccproxy auth login"
    )

    notice = manager._ccproxy_missing(
        _oauth_config(), manager.ServerResult(available=True, url="http://s")
    )

    assert "ccproxy auth login" in notice


def test_ccproxy_missing_no_notice_when_ccproxy_answers(monkeypatch):
    from EvoScientist import ccproxy_manager

    monkeypatch.setattr(ccproxy_manager, "is_ccproxy_running", lambda port: True)

    notice = manager._ccproxy_missing(
        _oauth_config(), manager.ServerResult(available=True)
    )

    assert notice is None


@pytest.mark.parametrize(
    ("config", "problem"),
    [
        (_oauth_config(provider="openai"), None),
        (_oauth_config(), "ccproxy_failed"),
        (_config(), None),
    ],
    ids=["no_anthropic_model", "server_said_why", "api_key"],
)
def test_ccproxy_missing_no_notice_when_not_needed_or_reported(
    monkeypatch, config, problem
):
    from EvoScientist import ccproxy_manager

    monkeypatch.setattr(ccproxy_manager, "is_ccproxy_running", lambda port: False)

    notice = manager._ccproxy_missing(config, manager.ServerResult(problem=problem))

    assert notice is None
