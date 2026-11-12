"""Tests for the server's record (``read_record``), the reuse of a recorded
server, and the stop of the recorded one.

The server records itself in the server home; clients read the record and
never write it, and a stop deletes the record of the server it ended. A
record counts while the server lock is held and the process it names is
still the one recorded; a stop checks that too. A record an older release
kept in the shared folder still counts, for this data folder only.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from EvoScientist import paths
from EvoScientist.langgraph_dev import manager, registration, server_record
from tests.fakes.langgraph_dev import (
    CREATED,
    LockHolder,
    Processes,
    Server,
    lock_follows_record,
    write_older_record,
    write_record,
)


@pytest.fixture(autouse=True)
def _server_lock(monkeypatch):
    """A written record holds the server lock, as the server does."""
    lock_follows_record(monkeypatch)


def _wait_for_child_argv(pid: int, needle: str, timeout: float = 5.0) -> bool:
    """Wait until a just-started child shows its own argv.

    ``Popen`` can return while the child is still inside ``exec``, before
    Linux has recorded the new argv, so a read right then comes back empty.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            if needle in psutil.Process(pid).cmdline():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# read_record
# ---------------------------------------------------------------------------


def test_runtime_files_follow_server_home(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "SERVER_HOME", tmp_path / "home")

    for path in (
        manager.RUNTIME.record,
        manager.RUNTIME.lock_file,
        manager.RUNTIME.log_file,
        manager.RUNTIME.webui_log,
    ):
        assert path.parent == tmp_path / "home"
    assert manager.RUNTIME.record.name == server_record.RECORD_FILE


def test_read_record_none_without_record(runtime_paths):
    assert manager.read_record() is None


def test_read_record_reads_record(monkeypatch, runtime_paths):
    write_record(4242, port=16300, host="0.0.0.0", lasting=True)

    record = manager.read_record()

    assert (record.pid, record.port, record.host) == (4242, 16300, "0.0.0.0")
    assert record.created == CREATED
    assert record.lasting is True
    assert record.older is False
    assert record.url == "http://127.0.0.1:16300"


def test_read_record_ignores_record_without_server_lock(monkeypatch, runtime_paths):
    """Its server is gone: the lock it held is free."""
    from EvoScientist.langgraph_dev import server_home

    write_record(4242)
    monkeypatch.setattr(server_home, "lock_state", lambda wait=0: server_home.FREE)

    assert manager.read_record() is None
    assert runtime_paths.record.exists(), "a client never deletes the record"


def test_read_record_counts_while_lock_is_held_and_process_runs(runtime_paths):
    """The lock and a live PID that runs as recorded make a record count."""
    write_record(4242, created=1.0)

    record = manager.read_record()

    assert record.pid == 4242
    assert record.created == 1.0


_REAL_RUNS_AS_RECORDED = manager._runs_as_recorded


def _record_of_this_process(created: float) -> manager.ServerRecord:
    return manager.ServerRecord(
        pid=os.getpid(), port=6174, host="127.0.0.1", created=created
    )


def test_runs_as_recorded_while_pid_is_that_process():
    created = psutil.Process().create_time()

    assert _REAL_RUNS_AS_RECORDED(_record_of_this_process(created)) is True


def test_read_record_ignores_record_whose_pid_went_to_another_process(
    monkeypatch, runtime_paths
):
    """A record a stopped server left is not taken for a process that got
    its PID later, also while a new server holds the lock."""
    monkeypatch.setattr(manager, "_runs_as_recorded", _REAL_RUNS_AS_RECORDED)
    monkeypatch.setattr(manager, "_serves_this_home", lambda pid: False)
    created = psutil.Process().create_time() - 60
    write_record(os.getpid(), created=created)

    assert _REAL_RUNS_AS_RECORDED(_record_of_this_process(created)) is False
    assert manager.read_record() is None


def test_runs_as_recorded_for_this_homes_server_after_clock_step(monkeypatch):
    """A clock step moves the creation time psutil reports for a live server."""
    monkeypatch.setattr(manager, "_serves_this_home", lambda pid: True)
    created = psutil.Process().create_time() - 60

    assert _REAL_RUNS_AS_RECORDED(_record_of_this_process(created)) is True


def test_runs_as_recorded_when_process_cannot_be_read(monkeypatch):
    """An elevated server on Windows refuses its creation time."""
    monkeypatch.setattr(manager, "_create_time", lambda pid: None)

    assert _REAL_RUNS_AS_RECORDED(_record_of_this_process(1.0)) is True


def test_serves_this_home_for_langgraph_dev_working_in_it(monkeypatch):
    monkeypatch.setattr(manager, "_langgraph_cmdline", lambda pid: ["langgraph"])
    monkeypatch.setattr(paths, "SERVER_HOME", Path.cwd())

    assert manager._serves_this_home(os.getpid()) is True


def test_serves_this_home_not_for_langgraph_dev_of_another_home(monkeypatch, tmp_path):
    """Another data folder's server that got the PID works elsewhere."""
    monkeypatch.setattr(manager, "_langgraph_cmdline", lambda pid: ["langgraph"])
    monkeypatch.setattr(paths, "SERVER_HOME", tmp_path)

    assert manager._serves_this_home(os.getpid()) is False


def test_serves_this_home_not_for_other_process(monkeypatch):
    monkeypatch.setattr(manager, "_langgraph_cmdline", lambda pid: None)
    monkeypatch.setattr(paths, "SERVER_HOME", Path.cwd())

    assert manager._serves_this_home(os.getpid()) is False


def test_read_record_distrusts_record_when_lock_cannot_be_opened(
    monkeypatch, runtime_paths
):
    from EvoScientist.langgraph_dev import server_home

    write_record(4242)
    monkeypatch.setattr(server_home, "lock_state", lambda wait=0: server_home.UNKNOWN)

    assert manager.read_record() is None


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        (server_record.STARTING, server_record.STARTING),
        (server_record.SERVING, server_record.SERVING),
        (None, server_record.SERVING),
    ],
)
def test_read_record_reads_phase(runtime_paths, phase, expected):
    """A record without a phase (an older release's) counts as serving."""
    write_record(4242, phase=phase)

    assert manager.read_record().phase == expected


def test_read_record_reads_older_place_only_when_asked(monkeypatch, runtime_paths):
    Processes(monkeypatch).add(4141)
    write_older_record(4141)

    assert manager.read_record() is None
    assert manager.read_record(older=True).older is True


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        "[]",
        "{}",
        '"just a string"',
        "null",
        '{"pid": null, "port": 6174, "create_time": 1.0}',
        '{"pid": "4242", "port": 6174, "create_time": 1.0}',
        '{"pid": true, "port": 6174, "create_time": 1.0}',
        '{"pid": 0, "port": 6174, "create_time": 1.0}',
        '{"pid": 4242, "create_time": 1.0}',
    ],
)
def test_read_record_ignores_malformed_record(payload, monkeypatch, runtime_paths):
    Processes(monkeypatch).add(4242, created=1.0)
    runtime_paths.record.write_text(payload)

    assert manager.read_record() is None


def test_read_record_names_folders_server_uses(monkeypatch, runtime_paths):
    Processes(monkeypatch).add(4242)
    write_record(4242, data_dir="/srv/data", config_dir="/srv/config")

    record = manager.read_record()

    assert (record.data_dir, record.config_dir) == ("/srv/data", "/srv/config")


def test_read_record_trusts_record_for_process_group(monkeypatch, runtime_paths):
    Processes(monkeypatch).add(4242)
    write_record(4242, **{server_record.BREAK_GROUP_KEY: True})

    assert manager.read_record().break_group is True


# ---------------------------------------------------------------------------
# The record an older release kept
# ---------------------------------------------------------------------------


def test_read_record_falls_back_to_older_releases_record(monkeypatch, runtime_paths):
    Processes(monkeypatch).add(4141, port=6174)
    write_older_record(4141, port=6174, lasting=True, host="0.0.0.0")

    record = manager.read_record(older=True)

    assert record.older is True
    assert (record.pid, record.port, record.host) == (4141, 6174, "0.0.0.0")
    assert record.lasting is True


def test_read_record_prefers_server_homes_record(monkeypatch, runtime_paths):
    processes = Processes(monkeypatch)
    processes.add(4242)
    processes.add(4141)
    write_record(4242)
    write_older_record(4141)

    assert manager.read_record(older=True).pid == 4242


def test_read_record_leaves_another_data_folders_older_record(
    monkeypatch, runtime_paths
):
    Processes(monkeypatch).add(4141)
    folder = write_older_record(4141, data_dir="/another/data")

    assert manager.read_record(older=True) is None
    assert (folder / server_record.RECORD_FILE).exists()
    assert (folder / "langgraph_dev.pid").exists()


def test_read_record_matches_data_folder_by_another_spelling(
    monkeypatch, runtime_paths, tmp_path
):
    """A symlink to the data folder names the same folder."""
    link = tmp_path / "data-link"
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    link.symlink_to(paths.DATA_DIR, target_is_directory=True)
    Processes(monkeypatch).add(4141)
    write_older_record(4141, data_dir=str(link))

    assert manager.read_record(older=True).pid == 4141


def test_read_record_ignores_older_record_without_data_folder(
    monkeypatch, runtime_paths
):
    """Releases before the data folder was recorded."""
    Processes(monkeypatch).add(4141)
    write_older_record(4141, data_dir=None)

    assert manager.read_record(older=True) is None


def test_read_record_deletes_older_record_whose_server_is_gone(
    monkeypatch, runtime_paths
):
    Processes(monkeypatch)
    folder = write_older_record(4141)

    assert manager.read_record(older=True) is None
    assert not (folder / server_record.RECORD_FILE).exists()
    assert not (folder / "langgraph_dev.pid").exists()


def test_read_record_keeps_older_record_whose_server_refuses_access(
    monkeypatch, runtime_paths
):
    """The server may still run: this read ignores the record, a later one
    with access finds it."""
    processes = Processes(monkeypatch)
    processes.add(4141)
    processes.denied.add(4141)
    folder = write_older_record(4141)

    assert manager.read_record(older=True) is None
    assert (folder / server_record.RECORD_FILE).exists()
    assert (folder / "langgraph_dev.pid").exists()


def test_read_record_deletes_older_record_whose_pid_went_elsewhere(
    monkeypatch, runtime_paths
):
    Processes(monkeypatch).add(4141, cmdline=["python", "editor.py"])
    folder = write_older_record(4141)

    assert manager.read_record(older=True) is None
    assert not (folder / server_record.RECORD_FILE).exists()


def test_read_record_deletes_older_record_whose_pid_serves_another_port(
    monkeypatch, runtime_paths
):
    """The PID went to another langgraph dev."""
    Processes(monkeypatch).add(4141, port=16500)
    folder = write_older_record(4141, port=6174)

    assert manager.read_record(older=True) is None
    assert not (folder / server_record.RECORD_FILE).exists()


def test_read_record_takes_older_records_port_from_command_line(
    monkeypatch, runtime_paths
):
    """Releases before the port was recorded."""
    Processes(monkeypatch).add(4141, port=16501)
    write_older_record(4141, port=None)

    assert manager.read_record(older=True).port == 16501


@pytest.mark.parametrize(("offset", "expected"), [(0.0, True), (-60.0, False)])
def test_read_record_trusts_older_record_for_process_group_while_pid_is_its(
    monkeypatch, runtime_paths, offset, expected
):
    Processes(monkeypatch).add(4141)
    write_older_record(4141, ctrl_break_create_time=CREATED + offset)

    assert manager.read_record(older=True).break_group is expected


def test_is_lasting_only_when_record_says_so(monkeypatch, runtime_paths):
    assert manager.is_lasting(None) is False
    assert manager.is_lasting(manager.ServerRecord(1, 6174, "h")) is False
    assert (
        manager.is_lasting(manager.ServerRecord(1, 6174, "h", lasting=False)) is False
    )
    assert manager.is_lasting(manager.ServerRecord(1, 6174, "h", lasting=True)) is True


# ---------------------------------------------------------------------------
# Reuse of the recorded server
# ---------------------------------------------------------------------------


def _config():
    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    return cfg


@pytest.fixture
def registered(monkeypatch):
    """The registrations the server answered, by URL and project."""
    calls: list = []

    def register_folder(workspace, *, url):
        calls.append((url, workspace))
        return registration.ServerExposure()

    monkeypatch.setattr(registration, "register_folder", register_folder)
    return calls


def test_ensure_serves_another_project_from_recorded_server(
    tmp_path, monkeypatch, runtime_paths, registered
):
    server = Server(monkeypatch)
    server.run(fingerprint="fp")

    result = manager.ensure_server(_config(), workspace_dir=tmp_path / "B")

    assert result.served is True
    assert result.started is False
    assert registered == [("http://127.0.0.1:6174", paths.Workspace(tmp_path / "B"))]
    assert manager.is_async_subagents_available()
    assert server.spawns == []


def test_ensure_with_any_port_reuses_recorded_server_at_its_host(
    tmp_path, monkeypatch, runtime_paths, registered
):
    Server(monkeypatch).run(host="10.0.0.5")

    result = manager.ensure_server(
        _config(), workspace_dir=tmp_path / "B", any_port=True
    )

    assert result.url == "http://10.0.0.5:6174"


def test_started_server_outlives_session(
    tmp_path, monkeypatch, runtime_paths, registered
):
    """Every session shares it until ``EvoSci server stop``."""
    import atexit

    server = Server(monkeypatch)
    exits = []
    monkeypatch.setattr(atexit, "register", lambda *a, **kw: exits.append(a))

    result = manager.ensure_server(_config(), workspace_dir=tmp_path / "A")

    assert result.started is True
    assert exits == []
    assert server.spawns[0]["lasting"] is True


def _reuse(tmp_path, monkeypatch, fingerprint, **fields):
    """Reuse a recorded server launched with *fingerprint*."""
    fingerprint_of = manager._server_config_fingerprint
    server = Server(monkeypatch)
    monkeypatch.setattr(manager, "_server_config_fingerprint", fingerprint_of)
    server.run(config_fingerprint=fingerprint, **fields)
    return manager.ensure_server(_config(), workspace_dir=tmp_path / "A")


def test_reuse_sets_drift_flag_on_fingerprint_mismatch(
    tmp_path, monkeypatch, runtime_paths, registered
):
    result = _reuse(tmp_path, monkeypatch, "stale-fingerprint")

    assert result.drift is True
    assert manager.CONFIG_DRIFT_SINCE_LAUNCH is True


def test_reuse_clears_drift_flag_on_matching_fingerprint(
    tmp_path, monkeypatch, runtime_paths, registered
):
    _reuse(tmp_path, monkeypatch, manager._server_config_fingerprint(_config()))

    assert manager.CONFIG_DRIFT_SINCE_LAUNCH is False


def test_reuse_sets_drift_flag_on_version_change(
    tmp_path, monkeypatch, runtime_paths, registered
):
    """A server left running across an upgrade counts as drift: the session
    reuses it and warns, and the server is not stopped."""
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.3.4")
    old_fp = manager._server_config_fingerprint(_config())
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.3.5")

    result = _reuse(tmp_path, monkeypatch, old_fp)

    assert result.served is True
    assert manager.CONFIG_DRIFT_SINCE_LAUNCH is True
    assert manager.is_async_subagents_available() is True


def test_reuse_without_fingerprint_sets_no_drift(
    tmp_path, monkeypatch, runtime_paths, registered
):
    """A server started by hand records no configuration."""
    Server(monkeypatch).run()

    manager.ensure_server(_config(), workspace_dir=tmp_path / "A")

    assert manager.CONFIG_DRIFT_SINCE_LAUNCH is False


def test_server_config_fingerprint_tracks_relevant_fields():
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    assert base == manager._server_config_fingerprint(cfg), "must be deterministic"
    cfg.model = "some-other-model"
    assert manager._server_config_fingerprint(cfg) != base


def test_server_config_fingerprint_ignores_port():
    """The port is checked on its own, so a server reused on another port
    than the caller's is no drift."""
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    cfg.langgraph_dev_port += 1
    assert manager._server_config_fingerprint(cfg) == base


def test_server_config_fingerprint_ignores_the_agent_python(monkeypatch):
    """The fingerprint covers launch settings only; a PATH difference is
    reported on its own (``AGENT_SHELL_DRIFT``)."""
    from EvoScientist.setup import research_env

    cfg = manager.EvoScientistConfig()
    monkeypatch.setattr(research_env, "agent_python", lambda: None)
    base = manager._server_config_fingerprint(cfg)
    monkeypatch.setattr(research_env, "agent_python", lambda: "/env/bin/python")
    assert manager._server_config_fingerprint(cfg) == base


def test_reuse_warns_when_the_server_shell_differs(
    tmp_path, monkeypatch, runtime_paths, registered
):
    """A server started before `EvoSci setup` recorded Git Bash keeps cmd.exe
    for its agents; this session runs bash."""
    from EvoScientist import agent_shell
    from EvoScientist.setup import research_env

    monkeypatch.setattr(research_env, "agent_python", lambda: "/env/bin/python")
    bash = r"C:\Git\bin\bash.exe"
    monkeypatch.setattr(agent_shell, "agent_bash", lambda: SimpleNamespace(bash=bash))

    _reuse(tmp_path, monkeypatch, "fp", agent_python="/env/bin/python", agent_bash=None)

    assert "cmd.exe" in manager.AGENT_SHELL_DRIFT
    assert bash in manager.AGENT_SHELL_DRIFT
    assert "python" not in manager.AGENT_SHELL_DRIFT


def _reuse_with_agent_python(tmp_path, monkeypatch, recorded):
    """Reuse a server whose record holds ``recorded`` (or nothing, when
    ``recorded`` is ``...``) from a session that resolves the research env."""
    from EvoScientist.setup import research_env

    monkeypatch.setattr(research_env, "agent_python", lambda: "/env/bin/python")
    fields = {} if recorded is ... else {"agent_python": recorded}
    _reuse(tmp_path, monkeypatch, "fp", **fields)


def test_reuse_warns_when_the_server_python_differs(
    tmp_path, monkeypatch, runtime_paths, registered
):
    """A server launched from a shell with conda active, reused from one
    without: its agents keep conda's python."""
    _reuse_with_agent_python(tmp_path, monkeypatch, "/conda/bin/python")
    assert "/conda/bin/python" in manager.AGENT_SHELL_DRIFT
    assert "/env/bin/python" in manager.AGENT_SHELL_DRIFT


def test_reuse_does_not_warn_when_the_server_python_matches(
    tmp_path, monkeypatch, runtime_paths, registered
):
    _reuse_with_agent_python(tmp_path, monkeypatch, "/env/bin/python")
    assert manager.AGENT_SHELL_DRIFT is None


def test_reuse_does_not_warn_without_a_recorded_python(
    tmp_path, monkeypatch, runtime_paths, registered
):
    """A server started by hand has no record of it."""
    _reuse_with_agent_python(tmp_path, monkeypatch, ...)
    assert manager.AGENT_SHELL_DRIFT is None


def test_reuse_of_older_releases_server_says_to_restart_it(
    tmp_path, monkeypatch, runtime_paths, registered
):
    server = Server(monkeypatch)
    server.processes.add(4141)
    server.answering.add(6174)
    write_older_record(4141, lasting=True)

    result = manager.ensure_server(_config(), workspace_dir=tmp_path / "A")

    assert result.served is True
    assert result.record.older is True
    assert manager.OLDER_SERVER == manager.OLDER_SERVER_MESSAGE
    assert server.spawns == []


# ---------------------------------------------------------------------------
# stop_server
# ---------------------------------------------------------------------------


def test_stop_server_none_without_record(runtime_paths):
    assert manager.stop_server() is None


def test_stop_server_never_signals_reused_pid(monkeypatch, runtime_paths):
    """The record's PID went to a process started later."""
    Processes(monkeypatch).add(4242, created=CREATED + 60)
    write_record(4242)
    ended: list = []
    monkeypatch.setattr(
        manager, "_end_server_tree", lambda pid, *a, **k: ended.append(pid)
    )

    assert manager.stop_server() is None
    assert ended == []


def _record_live(pid: int, **fields) -> None:
    """Record the live process ``pid`` as the server, as the server does."""
    write_record(pid, created=psutil.Process(pid).create_time(), **fields)


def test_stop_server_refuses_process_created_after_record(runtime_paths):
    """The record's PID went to another process: it is not killed."""
    victim = _sleeper("langgraph")
    try:
        assert _wait_for_child_argv(victim.pid, "langgraph")
        write_record(victim.pid, created=psutil.Process(victim.pid).create_time() - 60)
        assert manager.stop_server() is None
        assert victim.poll() is None, "a reused PID must not be killed"
    finally:
        victim.kill()
        victim.wait()


def test_stop_server_stops_recorded_server(runtime_paths):
    victim = _sleeper("langgraph")
    try:
        assert _wait_for_child_argv(victim.pid, "langgraph"), (
            "child never reported its own argv"
        )
        _record_live(victim.pid)
        assert manager.stop_server() == victim.pid
        assert victim.poll() is not None, "the recorded server must be stopped"
        assert not runtime_paths.record.exists(), "the stop removes the record"
    finally:
        if victim.poll() is None:
            victim.kill()
        victim.wait()


@pytest.mark.parametrize("lasting", [True, None])
def test_stop_server_keep_owned_stops_background_service(
    monkeypatch, runtime_paths, lasting
):
    Processes(monkeypatch).add(4242)
    write_record(4242, **({} if lasting is None else {"lasting": lasting}))
    monkeypatch.setattr(manager, "_end_server_tree", lambda *a, **k: None)

    assert manager.stop_server(keep_owned=True) == 4242


def test_stop_server_keep_owned_leaves_server_its_starter_owns(
    monkeypatch, runtime_paths
):
    Processes(monkeypatch).add(4242)
    write_record(4242, lasting=False)
    ended: list = []
    monkeypatch.setattr(
        manager, "_end_server_tree", lambda pid, *a, **k: ended.append(pid)
    )

    with pytest.raises(manager.ServerKeptError, match="left running"):
        manager.stop_server(keep_owned=True)
    assert ended == []
    assert manager.owned_by_its_starter(manager.read_record()) is True


def test_stop_server_reaps_server_this_process_started(monkeypatch, runtime_paths):
    proc = SimpleNamespace(pid=4242, returncode=None, poll=lambda: None)
    Processes(monkeypatch).add(4242)
    write_record(4242)
    monkeypatch.setattr(manager, "_PROCESS", proc)
    seen: list = []
    monkeypatch.setattr(
        manager,
        "_end_server_tree",
        lambda pid, popen=None, *, break_group=False: seen.append(popen),
    )

    assert manager.stop_server() == 4242
    assert seen[0] is proc
    assert manager._PROCESS is None


def test_stop_server_stops_older_releases_server_and_forgets_it(
    monkeypatch, runtime_paths
):
    Processes(monkeypatch).add(4141)
    folder = write_older_record(4141)
    ended: list = []
    monkeypatch.setattr(
        manager, "_end_server_tree", lambda pid, *a, **k: ended.append(pid)
    )

    assert manager.stop_server() == 4141
    assert ended == [4141]
    assert not (folder / server_record.RECORD_FILE).exists()
    assert not (folder / "langgraph_dev.pid").exists()


def test_stop_langgraph_dev_removes_its_servers_record(monkeypatch, runtime_paths):
    """The stop's signal ends the server before its own removal at exit."""
    write_record(4242)
    proc = SimpleNamespace(pid=4242, poll=lambda: 0)

    manager.stop_langgraph_dev(proc)

    assert not runtime_paths.record.exists()


def test_stop_langgraph_dev_keeps_record_while_server_holds_lock(runtime_paths):
    write_record(4242)
    holder = LockHolder(server=True)
    try:
        manager.stop_langgraph_dev(SimpleNamespace(pid=4242, poll=lambda: 0))
    finally:
        holder.done()

    assert runtime_paths.record.exists()


def test_stop_server_stops_this_homes_server_after_clock_step(
    monkeypatch, runtime_paths
):
    """What read_record takes after a clock step, the stop stops."""
    processes = Processes(monkeypatch)
    processes.add(4242, created=CREATED + 60)
    write_record(4242)
    monkeypatch.setattr(manager, "_serves_this_home", lambda pid: pid == 4242)
    ended: list = []
    monkeypatch.setattr(
        manager, "_end_server_tree", lambda pid, *_a, **_k: ended.append(pid)
    )

    assert manager.stop_server() == 4242
    assert ended == [4242]


def test_stop_server_keeps_record_that_names_another_server(monkeypatch, runtime_paths):
    """A server that took the lock and wrote its record by the time the
    stopped one is gone keeps that record."""
    processes = Processes(monkeypatch)
    processes.add(4242)
    write_record(4242)

    def _end(pid, popen=None, *, break_group=False):
        processes.end(pid)
        processes.add(5151)
        write_record(5151)

    monkeypatch.setattr(manager, "_end_server_tree", _end)

    assert manager.stop_server() == 4242
    assert manager.read_record().pid == 5151


# ---------------------------------------------------------------------------
# Windows: Ctrl+Break before the hard kill, so a cut run is run again
# ---------------------------------------------------------------------------


def _sleeper(*extra: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", *extra]
    )


@pytest.fixture
def stop_steps(monkeypatch):
    """Records, in order, the clean-stop requests and ``terminate()`` calls.
    The clean stop is faked by the test (``clean_stop``); terminate is real."""
    steps: list[tuple] = []
    real_terminate = psutil.Process.terminate

    def _terminate(self):
        steps.append(("terminate", self.pid))
        return real_terminate(self)

    monkeypatch.setattr(psutil.Process, "terminate", _terminate)
    return steps


def _clean_stop(monkeypatch, steps, *, sent: bool = True, then=None):
    def _request(pid):
        steps.append(("clean", pid))
        if then is not None:
            then()
        return sent

    monkeypatch.setattr(manager, "_request_clean_stop", _request)


def test_clean_stop_that_exits_in_time_is_never_terminated(monkeypatch, stop_steps):
    proc = _sleeper()
    try:
        # The server shuts down on the Ctrl+Break.
        _clean_stop(monkeypatch, stop_steps, then=proc.kill)
        manager._end_server_tree(proc.pid, proc, break_group=True)
        assert stop_steps == [("clean", proc.pid)]
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_clean_stop_that_does_not_exit_ends_hard_with_a_warning(
    monkeypatch, stop_steps, caplog
):
    proc = _sleeper()
    try:
        monkeypatch.setattr(manager, "CLEAN_STOP_TIMEOUT", 0.2)
        _clean_stop(monkeypatch, stop_steps)  # sent, but ignored
        with caplog.at_level("WARNING", logger=manager.__name__):
            manager._end_server_tree(proc.pid, proc, break_group=True)
        assert stop_steps == [("clean", proc.pid), ("terminate", proc.pid)]
        assert proc.poll() is not None
        assert "did not exit within 0.2 s of Ctrl+Break" in caplog.text
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_failed_clean_stop_ends_hard_at_once(monkeypatch, stop_steps):
    proc = _sleeper()
    try:
        monkeypatch.setattr(manager, "CLEAN_STOP_TIMEOUT", 30)
        _clean_stop(monkeypatch, stop_steps, sent=False)
        started = time.monotonic()
        manager._end_server_tree(proc.pid, proc, break_group=True)
        assert time.monotonic() - started < 10, "waited for a Ctrl+Break never sent"
        assert stop_steps == [("clean", proc.pid), ("terminate", proc.pid)]
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_recorded_server_clean_stop_waits_for_the_exit(monkeypatch, stop_steps):
    """The recorded-PID stop has no Popen handle: the wait goes through psutil."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    )
    try:
        _clean_stop(monkeypatch, stop_steps, then=proc.stdin.close)
        manager._end_server_tree(proc.pid, break_group=True)
        assert stop_steps == [("clean", proc.pid)]
    finally:
        proc.stdin.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def test_no_clean_stop_without_a_known_process_group(monkeypatch, stop_steps):
    proc = _sleeper()
    try:
        _clean_stop(monkeypatch, stop_steps)
        manager._end_server_tree(proc.pid, proc)
        assert stop_steps == [("terminate", proc.pid)]
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.mark.parametrize("wait_denied", [False, True])
def test_clean_stop_kills_children_left_after_the_server_exits(
    monkeypatch, stop_steps, wait_denied
):
    """The children are listed before the Ctrl+Break: after the server exits
    they can no longer be found from it. A denied wait (an elevated child on
    Windows) does not end the stop early: every listed child still gets a kill
    attempt."""
    wait_procs = psutil.wait_procs
    code = (
        "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "print(c.pid, flush=True)\n"
        "time.sleep(30)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    grandchild = None
    try:
        grandchild = psutil.Process(int(proc.stdout.readline()))
        if wait_denied:

            def _denied(*_a, **_k):
                raise psutil.AccessDenied(grandchild.pid)

            monkeypatch.setattr(psutil, "wait_procs", _denied)
        _clean_stop(monkeypatch, stop_steps, then=proc.kill)  # only the server
        manager._end_server_tree(proc.pid, proc, break_group=True)
        _, alive = wait_procs([grandchild], timeout=5)
        assert alive == []
    finally:
        for p in (proc, grandchild):
            try:
                if p is not None:
                    p.kill()
            except psutil.NoSuchProcess:
                pass
        proc.wait()
        proc.stdout.close()


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM can be ignored on POSIX only")
def test_server_that_ignores_terminate_is_killed(monkeypatch):
    """The escalation both stop paths had before sharing one helper."""
    code = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(30)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    try:
        assert proc.stdout.readline().strip() == b"ready"
        monkeypatch.setattr(manager, "_TERMINATE_TIMEOUT", 0.2)
        manager._end_server_tree(proc.pid, proc)
        assert proc.returncode == -9
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()


_SERVER_WITH_CHILD = (
    "import subprocess, sys, time\n"
    "child = (\n"
    "    'import signal, time\\n'\n"
    "    'signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n'\n"
    "    'print(\"ready\", flush=True)\\n'\n"
    "    'time.sleep(30)\\n'\n"
    ")\n"
    "c = subprocess.Popen([sys.executable, '-c', child], stdout=subprocess.PIPE)\n"
    "c.stdout.readline()\n"
    "print(c.pid, flush=True)\n"
    "time.sleep(30)\n"
)


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM can be ignored on POSIX only")
def test_hard_stop_kills_a_child_that_outlives_its_server(monkeypatch):
    """The server exits on SIGTERM, its child ignores it: after the grace the
    sweep kills the child."""
    monkeypatch.setattr(manager, "_SURVIVOR_GRACE", 0.2)
    proc = subprocess.Popen(
        [sys.executable, "-c", _SERVER_WITH_CHILD], stdout=subprocess.PIPE
    )
    child = None
    try:
        child = psutil.Process(int(proc.stdout.readline()))
        manager._end_server_tree(proc.pid, proc)
        assert proc.poll() is not None
        _, alive = psutil.wait_procs([child], timeout=5)
        assert alive == []
    finally:
        for p in (proc, child):
            try:
                if p is not None:
                    p.kill()
            except psutil.NoSuchProcess:
                pass
        proc.wait()
        proc.stdout.close()


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM can be ignored on POSIX only")
def test_hard_stop_kills_a_child_started_after_the_first_listing(monkeypatch):
    """A server that ignores SIGTERM and starts a child meanwhile: the kill
    lists the children again, so that child goes too."""
    code = (
        "import signal, subprocess, sys, time\n"
        "def late(*_):\n"
        "    c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "    print(c.pid, flush=True)\n"
        "signal.signal(signal.SIGTERM, late)\n"
        "print('ready', flush=True)\n"
        "while True:\n"
        "    time.sleep(0.05)\n"
    )
    monkeypatch.setattr(manager, "_TERMINATE_TIMEOUT", 1.0)
    monkeypatch.setattr(manager, "_SURVIVOR_GRACE", 0.2)
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    late = None
    try:
        assert proc.stdout.readline().strip() == b"ready"
        manager._end_server_tree(proc.pid, proc)
        assert proc.returncode == -9
        late = psutil.Process(int(proc.stdout.readline()))
        _, alive = psutil.wait_procs([late], timeout=5)
        assert alive == []
    except psutil.NoSuchProcess:
        pass  # already gone, as it should be
    finally:
        for p in (proc, late):
            try:
                if p is not None:
                    p.kill()
            except psutil.NoSuchProcess:
                pass
        proc.wait()
        proc.stdout.close()


@pytest.mark.skipif(os.name == "nt", reason="the POSIX stop")
def test_request_clean_stop_runs_no_helper_on_posix(monkeypatch):
    monkeypatch.setattr(
        manager.subprocess,
        "run",
        lambda *a, **k: pytest.fail("the Ctrl+Break helper ran on POSIX"),
    )
    assert manager._request_clean_stop(4242) is False


def test_request_clean_stop_runs_the_helper_without_a_window(monkeypatch):
    run = subprocess.run
    seen: dict = {}

    def _run(args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(manager.os, "name", "nt")
    monkeypatch.setattr(manager.subprocess, "run", _run)
    assert manager._request_clean_stop(4242) is True
    helper = os.path.join(os.path.dirname(manager.__file__), "console_break.py")
    assert seen["args"] == [sys.executable, "-I", helper, "4242"]
    assert seen["kwargs"]["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert seen["kwargs"]["timeout"] == manager._CLEAN_STOP_HELPER_TIMEOUT
    usage = run([*seen["args"][:-1], "0"], capture_output=True, text=True)
    assert "usage: console_break <pid>" in usage.stderr


@pytest.mark.parametrize(
    ("outcome", "logged"),
    [
        (
            subprocess.CompletedProcess(
                [], 3, "", "AttachConsole(4242) failed: error 6\n"
            ),
            "exited with 3 (AttachConsole(4242) failed: error 6)",
        ),
        (OSError("no python"), "(no python)"),
        (subprocess.TimeoutExpired("python", 5), "timed out"),
    ],
)
def test_request_clean_stop_warns_when_the_helper_fails(
    monkeypatch, caplog, outcome, logged
):
    def _run(*_a, **_k):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(manager.os, "name", "nt")
    monkeypatch.setattr(manager.subprocess, "run", _run)
    with caplog.at_level("WARNING", logger=manager.__name__):
        assert manager._request_clean_stop(4242) is False
    assert logged in caplog.text
    assert "ending it hard" in caplog.text


@pytest.mark.parametrize(
    ("spawned_here", "clean", "expected"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_stop_langgraph_dev_sends_ctrl_break_only_to_its_own_group(
    monkeypatch, runtime_paths, spawned_here, clean, expected
):
    monkeypatch.setattr(manager, "_PROCESS", None)
    proc = SimpleNamespace(pid=4242, poll=lambda: None)
    seen: list = []
    monkeypatch.setattr(
        manager,
        "_end_server_tree",
        lambda pid, popen=None, *, break_group=False: seen.append(break_group),
    )
    monkeypatch.setattr(
        manager, "_BREAK_GROUP_PIDS", {proc.pid} if spawned_here else set()
    )
    manager.stop_langgraph_dev(proc, clean=clean)
    assert seen == [expected]
    assert proc.pid not in manager._BREAK_GROUP_PIDS


@pytest.mark.parametrize("break_group", [True, False])
def test_stop_server_trusts_record_for_process_group(
    runtime_paths, monkeypatch, break_group
):
    victim = _sleeper("langgraph")
    try:
        assert _wait_for_child_argv(victim.pid, "langgraph")
        _record_live(victim.pid, **{server_record.BREAK_GROUP_KEY: break_group})
        seen: list = []

        def _end(pid, popen=None, *, break_group=False):
            seen.append((pid, break_group))
            victim.kill()
            victim.wait()

        monkeypatch.setattr(manager, "_end_server_tree", _end)
        assert manager.stop_server() == victim.pid
        assert seen == [(victim.pid, break_group)]
    finally:
        if victim.poll() is None:
            victim.kill()
            victim.wait()


def _fake_hold(monkeypatch, steps: list, refused=None):
    """``_hold_pid`` recording its hold and release, or raising *refused*
    (an exception class, given the PID) as a refused ``OpenProcess`` does."""

    @contextlib.contextmanager
    def _hold(pid):
        if refused is not None:
            raise refused(pid)
        steps.append(("hold", pid))
        try:
            yield
        finally:
            steps.append(("release", pid))

    monkeypatch.setattr(manager, "_hold_pid", _hold)


def test_stop_server_holds_pid_until_server_is_stopped(monkeypatch, runtime_paths):
    """A process given the server's PID during the stop gets neither the
    Ctrl+Break nor the kills: the check that it is still the recorded
    process runs while the PID is held."""
    victim = _sleeper("langgraph")
    try:
        assert _wait_for_child_argv(victim.pid, "langgraph")
        _record_live(victim.pid, **{server_record.BREAK_GROUP_KEY: True})
        steps: list = []
        _fake_hold(monkeypatch, steps)
        real_same = manager._same_process

        def _same(pid, created):
            steps.append(("check", pid))
            return real_same(pid, created)

        monkeypatch.setattr(manager, "_same_process", _same)

        def _end(pid, popen=None, *, break_group=False):
            steps.append(("end", pid, break_group))
            victim.kill()
            victim.wait()

        monkeypatch.setattr(manager, "_end_server_tree", _end)
        assert manager.stop_server() == victim.pid
        held = steps[steps.index(("hold", victim.pid)) :]
        assert held == [
            ("hold", victim.pid),
            ("check", victim.pid),
            ("end", victim.pid, True),
            ("release", victim.pid),
        ]
    finally:
        if victim.poll() is None:
            victim.kill()
            victim.wait()


def test_stop_server_releases_pid_when_stop_fails(monkeypatch, runtime_paths):
    victim = _sleeper("langgraph")
    try:
        assert _wait_for_child_argv(victim.pid, "langgraph")
        _record_live(victim.pid)
        steps: list = []
        _fake_hold(monkeypatch, steps)

        def _end(pid, popen=None, *, break_group=False):
            raise RuntimeError("stop failed")

        monkeypatch.setattr(manager, "_end_server_tree", _end)
        with pytest.raises(RuntimeError, match="stop failed"):
            manager.stop_server()
        assert steps == [("hold", victim.pid), ("release", victim.pid)]
    finally:
        if victim.poll() is None:
            victim.kill()
            victim.wait()


def test_stop_server_refused_signal_leaves_server_running(monkeypatch, runtime_paths):
    """Another user's or an elevated server: inspecting it works, signalling
    it is refused, so the stop fails and the record stays for a retry."""
    Processes(monkeypatch).add(4242)
    write_record(4242)

    def _refused(pid, popen=None, *, break_group=False):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(manager, "_end_server_tree", _refused)

    with pytest.raises(manager.ServerStopFailedError, match="4242"):
        manager.stop_server()
    assert runtime_paths.record.exists()


@pytest.fixture
def held_victim(monkeypatch, runtime_paths):
    """A recorded langgraph dev, and the list of pids the stop signals
    (none is ever signalled for real)."""
    victim = _sleeper("langgraph")
    try:
        assert _wait_for_child_argv(victim.pid, "langgraph")
        _record_live(victim.pid)
        ended: list = []
        monkeypatch.setattr(
            manager,
            "_end_server_tree",
            lambda pid, popen=None, **_kw: ended.append(pid),
        )
        yield victim, ended
    finally:
        victim.kill()
        victim.wait()


def test_stop_server_finds_nothing_for_unknown_pid(
    monkeypatch, runtime_paths, held_victim
):
    """``OpenProcess`` says no process has the PID: nothing to stop."""
    victim, ended = held_victim
    _fake_hold(monkeypatch, [], refused=psutil.NoSuchProcess)

    assert manager.stop_server() is None
    assert ended == []
    assert runtime_paths.record.exists()
    assert victim.poll() is None


def test_stop_server_refused_hold_fails_without_signal(
    monkeypatch, runtime_paths, held_victim
):
    """``OpenProcess`` refuses access: the stop fails, sends no signal, and
    the record stays."""
    victim, ended = held_victim
    _fake_hold(monkeypatch, [], refused=psutil.AccessDenied)

    with pytest.raises(manager.ServerStopFailedError):
        manager.stop_server()
    assert ended == []
    assert runtime_paths.record.exists()
    assert victim.poll() is None


def test_stop_server_keeps_server_it_could_not_open(monkeypatch, runtime_paths):
    """Any other refusal (on Windows, out of resources): the server may be
    running and is not held, so it gets no signal, and a retry finds it."""
    victim = _sleeper("langgraph")
    try:
        assert _wait_for_child_argv(victim.pid, "langgraph")
        _record_live(victim.pid)
        _fake_hold(monkeypatch, [], refused=OSError)
        ended: list = []
        monkeypatch.setattr(
            manager, "_end_server_tree", lambda pid, **_kw: ended.append(pid)
        )
        with pytest.raises(manager.ServerStopFailedError, match="try again"):
            manager.stop_server()
        assert ended == []
        assert runtime_paths.record.exists()
        assert victim.poll() is None
    finally:
        victim.kill()
        victim.wait()


class _FakeKernel32:
    def __init__(self, handle):
        self.handle = handle
        self.calls: list = []

    def OpenProcess(self, access, inherit, pid):
        self.calls.append(("open", access, inherit, pid))
        return self.handle

    def CloseHandle(self, handle):
        self.calls.append(("close", handle))
        return True


def _windows_kernel32(monkeypatch, handle, error=0) -> _FakeKernel32:
    kernel32 = _FakeKernel32(handle)
    monkeypatch.setattr(manager.os, "name", "nt")
    monkeypatch.setattr(manager, "_kernel32", lambda: (kernel32, lambda: error))
    return kernel32


def test_hold_pid_keeps_a_handle_open_for_the_block(monkeypatch):
    """Query-limited access only: Windows grants it for an elevated server too,
    and any open handle keeps the PID from being given to another process."""
    kernel32 = _windows_kernel32(monkeypatch, handle=1234)

    def _fail_while_held():
        with manager._hold_pid(4242):
            assert kernel32.calls == [("open", 0x1000, False, 4242)]
            raise RuntimeError

    with pytest.raises(RuntimeError):
        _fail_while_held()
    assert kernel32.calls[-1] == ("close", 1234)


@pytest.mark.parametrize(
    ("error", "raised"),
    [
        (87, psutil.NoSuchProcess),  # ERROR_INVALID_PARAMETER: no such PID
        # The two errors psutil counts as permission errors.
        (5, psutil.AccessDenied),  # ERROR_ACCESS_DENIED
        (1314, psutil.AccessDenied),  # ERROR_PRIVILEGE_NOT_HELD
        (8, OSError),  # ERROR_NOT_ENOUGH_MEMORY, or any other
    ],
)
def test_hold_pid_refused_runs_nothing(monkeypatch, error, raised):
    kernel32 = _windows_kernel32(monkeypatch, handle=None, error=error)
    # Windows only: an OSError with the system's message for the error.
    monkeypatch.setattr(
        ctypes, "WinError", lambda code: OSError(f"system text {code}"), raising=False
    )
    ran: list = []

    def _hold():
        with manager._hold_pid(4242):
            ran.append(True)

    with pytest.raises(raised) as refused:
        _hold()
    if raised is OSError:
        assert str(refused.value) == f"system text {error}"
    assert ran == []
    assert [call[0] for call in kernel32.calls] == ["open"]


def test_hold_pid_holds_nothing_off_windows(monkeypatch):
    monkeypatch.setattr(manager.os, "name", "posix")
    monkeypatch.setattr(manager, "_kernel32", lambda: pytest.fail("kernel32 loaded"))
    ran: list = []
    with manager._hold_pid(4242):
        ran.append(True)
    assert ran == [True]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process handles")
def test_hold_pid_on_windows():
    proc = _sleeper()
    try:
        with manager._hold_pid(proc.pid):
            assert proc.poll() is None
        # OpenProcess refuses the System Idle Process with the error it gives
        # for a PID no process has.
        with pytest.raises(psutil.NoSuchProcess):
            with manager._hold_pid(0):
                pass
    finally:
        proc.kill()
        proc.wait()


def test_server_config_fingerprint_tracks_dangerous_mode():
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    cfg.dangerous_mode = True
    assert manager._server_config_fingerprint(cfg) != base


def test_server_config_fingerprint_covers_new_fields_by_default():
    """The fingerprint iterates the dataclass minus an exclusion set, so
    server-affecting values din0s called out — reasoning effort, provider
    keys/base URLs — count without per-field registration."""
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    cfg.reasoning_effort = "definitely-different"
    after_effort = manager._server_config_fingerprint(cfg)
    assert after_effort != base
    cfg.anthropic_base_url = "https://proxy.example"
    assert manager._server_config_fingerprint(cfg) != after_effort


def test_server_config_fingerprint_ignores_cli_only_fields():
    """Channel/UI-side fields never reach the server — no spurious drift."""
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    cfg.show_thinking = not cfg.show_thinking
    cfg.telegram_bot_token = "tg-token"
    cfg.default_mode = "run"
    assert manager._server_config_fingerprint(cfg) == base


def test_server_config_fingerprint_ignores_shell_allow_list():
    """shell_allow_list is read only by CLI/channel approval resolvers —
    it never reaches the subprocess, so it must not cause drift."""
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    cfg.shell_allow_list = "git status,ls"
    assert manager._server_config_fingerprint(cfg) == base


def test_server_config_fingerprint_ignores_gateway_backend():
    """The gateway settings only decide, in the CLI, whether a server is
    needed; nothing in the server reads them, so they must not cause drift."""
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    cfg.gateway_backend_serve = "langgraph_server"
    cfg.gateway_backend_tui = "langgraph_server"
    assert manager._server_config_fingerprint(cfg) == base


def test_server_config_fingerprint_tracks_mcp_yaml_content(tmp_path, monkeypatch):
    from EvoScientist.config import settings as settings_mod

    monkeypatch.setattr(settings_mod, "get_config_dir", lambda: tmp_path)
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    (tmp_path / "mcp.yaml").write_text("server-a:\n  transport: stdio\n")
    changed = manager._server_config_fingerprint(cfg)
    assert changed != base
    (tmp_path / "mcp.yaml").write_text("server-b:\n  transport: http\n")
    assert manager._server_config_fingerprint(cfg) != changed


def test_server_config_fingerprint_tracks_subagent_yaml_content(tmp_path, monkeypatch):
    fake_dir = tmp_path / "subagents"
    fake_dir.mkdir()
    (fake_dir / "research.yaml").write_text("name: research\nprompt: v1\n")
    monkeypatch.setattr(manager, "_SUBAGENTS_DIR", fake_dir)
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)
    (fake_dir / "research.yaml").write_text("name: research\nprompt: v2\n")
    assert manager._server_config_fingerprint(cfg) != base


def test_server_config_fingerprint_reads_flagged_field_from_file():
    """A lasting server does not get this session's --dangerous: the session
    sees no drift for it, and a change in config.yaml still is one."""
    from EvoScientist.config import settings

    plain = manager.EvoScientistConfig()
    flagged = manager.EvoScientistConfig(dangerous_mode=True)
    assert manager._server_config_fingerprint(flagged) != (
        manager._server_config_fingerprint(plain)
    )

    settings.record_env_origin("EVOSCIENTIST_DANGEROUS_MODE", "cli")
    assert manager._server_config_fingerprint(flagged) == (
        manager._server_config_fingerprint(plain)
    )


@pytest.mark.parametrize("editable", [True, False])
def test_server_config_fingerprint_tracks_source_changes(
    tmp_path, monkeypatch, editable
):
    """A source checkout changes its code without a version bump: an edited
    file and a deleted one both count. Two installs of one release do not."""
    import os

    from EvoScientist import ccproxy_manager

    monkeypatch.setattr(ccproxy_manager, "_is_editable_install", lambda: editable)
    package = tmp_path / "pkg"
    package.mkdir()
    old, edited = package / "old.py", package / "edited.py"
    for module in (old, edited):
        module.write_text("x = 1\n", encoding="utf-8")
    os.utime(old, ns=(1_000_000_000, 1_000_000_000))
    monkeypatch.setattr(manager, "_PACKAGE_DIR", package)
    cfg = manager.EvoScientistConfig()
    base = manager._server_config_fingerprint(cfg)

    later = edited.stat().st_mtime_ns + 1_000_000_000
    os.utime(edited, ns=(later, later))
    after_edit = manager._server_config_fingerprint(cfg)
    old.unlink()

    seen = {base, after_edit, manager._server_config_fingerprint(cfg)}
    assert len(seen) == (3 if editable else 1)


def test_server_config_fingerprint_keeps_flagged_auth_mode():
    """A lasting server takes its starter's --auth-mode: the fingerprint
    hashes that one, so a later session with another one sees the drift."""
    from EvoScientist.config import settings

    plain = manager.EvoScientistConfig()
    flagged = manager.EvoScientistConfig(openai_auth_mode="oauth")
    settings.record_env_origin("EVOSCIENTIST_OPENAI_AUTH_MODE", "cli")
    assert manager._server_config_fingerprint(flagged) != (
        manager._server_config_fingerprint(plain)
    )


def test_read_record_ignores_dead_pid_while_lock_is_held(monkeypatch, runtime_paths):
    """A killed server's record, read while a new server holds the lock
    before it wrote its own."""
    dead = {4242}
    lock_follows_record(monkeypatch, {True}, dead)
    write_record(4242)

    assert manager.read_record() is None


def test_stop_server_stops_server_hung_in_its_shutdown(runtime_paths):
    """It keeps its record until it exits, so a stop finds it by pid even
    though it no longer listens."""
    victim = _sleeper("langgraph")
    try:
        assert _wait_for_child_argv(victim.pid, "langgraph")
        _record_live(victim.pid)
        assert manager.stop_server() == victim.pid
        assert victim.poll() is not None
    finally:
        if victim.poll() is None:
            victim.kill()
        victim.wait()
