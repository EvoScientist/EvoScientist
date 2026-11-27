"""The server's record of itself (``langgraph_dev.server_record``): what it
writes at import, and that it removes only its own record."""

from __future__ import annotations

import json
import os
import sys

import psutil
import pytest

from EvoScientist import paths
from EvoScientist.agent_shell import SIDECAR_KEY as BASH_KEY
from EvoScientist.langgraph_dev import server_record
from EvoScientist.setup.research_env import SIDECAR_KEY as PYTHON_KEY


class _Proc:
    """A process in a fake psutil tree, linked to its parent."""

    def __init__(self, pid, parent=None, created=100.0):
        self.pid = pid
        self._parent = parent
        self._created = created

    def ppid(self):
        return self._parent.pid if self._parent else 0

    def parent(self):
        return self._parent

    def create_time(self):
        return self._created


@pytest.fixture
def server_env(monkeypatch):
    """The server's environment as its starter sets it, in a fake tree:
    the starter (pid 10) started a launcher (20) whose child (30) is this
    process."""
    monkeypatch.setattr(paths, "SERVER_HOME", paths.DATA_DIR / "server")
    paths.SERVER_HOME.mkdir(parents=True)
    starter = _Proc(10)
    launcher = _Proc(20, starter, created=200.0)
    me = _Proc(30, launcher, created=300.0)
    monkeypatch.setattr(server_record.psutil, "Process", lambda *a: me)
    monkeypatch.setenv(server_record.SPAWNER_PID_VAR, "10")
    monkeypatch.setenv(server_record.LASTING_VAR, "1")
    monkeypatch.setenv(server_record.FINGERPRINT_VAR, "abc123")
    monkeypatch.setattr(
        sys, "argv", ["langgraph", "dev", "--host", "0.0.0.0", "--port", "16200"]
    )
    monkeypatch.setattr(
        "EvoScientist.setup.research_env.agent_python", lambda: "/usr/bin/python3"
    )
    monkeypatch.setattr("EvoScientist.agent_shell.sidecar_bash", lambda: None)
    monkeypatch.setattr(server_record, "_written", None)
    registered = []
    monkeypatch.setattr(server_record.atexit, "register", registered.append)
    return registered


def _read():
    return json.loads(server_record.record_path().read_text(encoding="utf-8"))


def test_write_record_names_process_its_starter_started(server_env):
    server_record.write_record()

    record = _read()
    assert record["pid"] == 20
    assert record[server_record.CREATE_TIME_KEY] == 200.0


def test_write_record_carries_what_starter_passed(server_env):
    server_record.write_record()

    record = _read()
    assert record["lasting"] is True
    assert record["config_fingerprint"] == "abc123"
    assert record["port"] == 16200
    assert record["host"] == "0.0.0.0"
    assert record["data_dir"] == str(paths.DATA_DIR)
    assert record["config_dir"] == str(paths.CONFIG_DIR)


def test_write_record_leaves_agents_shell_to_serving_rewrite(server_env):
    """Written first and fast: resolving the agents' python can take a while,
    and until the record is written a crashed server's one would count."""
    server_record.write_record()

    assert PYTHON_KEY not in _read()

    server_record.mark_serving()

    record = _read()
    assert record[PYTHON_KEY] == "/usr/bin/python3"
    assert record[BASH_KEY] is None


def test_write_record_removes_starters_variables(server_env):
    server_record.write_record()

    for name in (
        server_record.SPAWNER_PID_VAR,
        server_record.LASTING_VAR,
        server_record.FINGERPRINT_VAR,
    ):
        assert name not in os.environ


def test_write_record_records_server_its_starter_does_not_own(server_env, monkeypatch):
    monkeypatch.setenv(server_record.LASTING_VAR, "0")

    server_record.write_record()

    assert _read()["lasting"] is False


def test_write_record_names_owner_of_server_it_does_not_outlive(
    server_env, monkeypatch
):
    """``EvoSci deploy`` (pid 10 in the fake tree) owns the server it started."""
    monkeypatch.setenv(server_record.LASTING_VAR, "0")

    server_record.write_record()

    assert _read()[server_record.OWNER_PID_KEY] == 10


def test_write_record_names_no_owner_of_lasting_server(server_env):
    server_record.write_record()

    assert server_record.OWNER_PID_KEY not in _read()


def test_write_record_names_no_owner_it_cannot_find(server_env, monkeypatch):
    """No starter among its ancestors: the server counts as owned."""
    monkeypatch.setenv(server_record.LASTING_VAR, "0")
    monkeypatch.setenv(server_record.SPAWNER_PID_VAR, "999")

    server_record.write_record()

    assert server_record.OWNER_PID_KEY not in _read()


def test_write_record_without_starter_names_itself(server_env, monkeypatch):
    """A server started by hand: no starter, no ownership, no Ctrl+Break."""
    for name in (
        server_record.SPAWNER_PID_VAR,
        server_record.LASTING_VAR,
        server_record.FINGERPRINT_VAR,
    ):
        monkeypatch.delenv(name)

    server_record.write_record()

    record = _read()
    assert record["pid"] == 30
    assert "lasting" not in record
    assert "config_fingerprint" not in record
    assert record[server_record.BREAK_GROUP_KEY] is False


def test_write_record_names_itself_when_starter_is_not_ancestor(
    server_env, monkeypatch
):
    monkeypatch.setenv(server_record.SPAWNER_PID_VAR, "999")

    server_record.write_record()

    record = _read()
    assert record["pid"] == 30
    assert record[server_record.BREAK_GROUP_KEY] is False


@pytest.mark.parametrize(("os_name", "expected"), [("nt", True), ("posix", False)])
def test_write_record_says_server_leads_its_group_only_on_windows(
    server_env, monkeypatch, os_name, expected
):
    monkeypatch.setattr(server_record.os, "name", os_name)

    server_record.write_record()

    assert _read()[server_record.BREAK_GROUP_KEY] is expected


@pytest.mark.parametrize(
    "argv",
    [
        ["langgraph", "dev", "--host", "::1", "--port", "2030"],
        ["langgraph", "dev", "--host=::1", "--port=2030"],
    ],
)
def test_write_record_prefers_address_on_command_line(server_env, monkeypatch, argv):
    """A server started by hand: the variables are configuration, the
    command line is its bind, in either option form."""
    monkeypatch.setattr(sys, "argv", argv)

    server_record.write_record()

    record = _read()
    assert record["port"] == 2030
    assert record["host"] == "::1"


def test_write_record_without_address_records_langgraph_defaults(
    server_env, monkeypatch
):
    """A server started by hand without --port or --host binds langgraph
    dev's defaults, whatever the configuration variables say."""
    monkeypatch.setenv("EVOSCIENTIST_LANGGRAPH_DEV_PORT", "16200")
    monkeypatch.setenv("EVOSCIENTIST_LANGGRAPH_DEV_HOST", "0.0.0.0")
    monkeypatch.setattr(sys, "argv", ["langgraph", "dev"])

    server_record.write_record()

    record = _read()
    assert record["port"] == server_record.DEFAULT_PORT
    assert record["host"] == server_record.DEFAULT_HOST


def test_server_record_registers_removal_at_import_not_at_write(server_env):
    """Exit handlers run in reverse order: registered when the module is
    imported, before the server's others, the removal runs last."""
    from tests.fakes.langgraph_dev import reloaded

    server_record.write_record()
    assert server_env == []

    with reloaded(server_record) as module:
        assert server_env == [module.remove_record]


def test_write_record_retries_while_reader_holds_old_record(server_env, monkeypatch):
    """Windows refuses the replace while a client has the record open."""
    import os

    replace = os.replace
    refusals = [PermissionError(32, "in use"), PermissionError(32, "in use")]

    def _replace(src, dst):
        if refusals:
            raise refusals.pop()
        replace(src, dst)

    monkeypatch.setattr(server_record.os, "replace", _replace)
    monkeypatch.setattr(server_record, "_REPLACE_RETRY", 0.001)

    server_record.write_record()

    assert _read()["pid"] == 20


def test_write_record_gives_up_on_record_that_stays_in_use(
    server_env, monkeypatch, caplog
):
    """The server does not start: it would hold the lock unseen."""

    def _refuse(src, dst):
        raise PermissionError(32, "in use")

    monkeypatch.setattr(server_record.os, "replace", _refuse)
    monkeypatch.setattr(server_record, "_REPLACE_WAIT", 0.01)
    monkeypatch.setattr(server_record, "_REPLACE_RETRY", 0.001)

    with pytest.raises(PermissionError):
        server_record.write_record()

    assert "Could not write the server's record" in caplog.text
    assert list(paths.SERVER_HOME.iterdir()) == []
    assert server_env == []


def test_write_record_failure_stops_server(server_env, monkeypatch, caplog):
    monkeypatch.setattr(paths, "SERVER_HOME", paths.DATA_DIR / "missing" / "server")

    with pytest.raises(FileNotFoundError):
        server_record.write_record()

    assert "Could not write the server's record" in caplog.text
    assert server_env == []


def test_write_record_writes_server_as_starting(server_env):
    server_record.write_record()

    assert _read()[server_record.PHASE_KEY] == server_record.STARTING


def test_mark_serving_without_agents_shell_still_writes(server_env, monkeypatch):
    """Resolving the agents' python is best effort: the record holds None."""

    def _broken():
        raise RuntimeError("probe failed")

    monkeypatch.setattr("EvoScientist.setup.research_env.agent_python", _broken)
    server_record.write_record()

    server_record.mark_serving()

    assert _read()[server_record.PHASE_KEY] == server_record.SERVING
    assert _read()[PYTHON_KEY] is None


def test_mark_serving_rewrites_phase(server_env):
    server_record.write_record()

    server_record.mark_serving()

    assert _read()[server_record.PHASE_KEY] == server_record.SERVING
    server_record.remove_record()
    assert not server_record.record_path().exists()


def test_mark_serving_failure_is_logged_only(server_env, monkeypatch, caplog):
    server_record.write_record()

    def _refuse(src, dst):
        raise PermissionError(32, "in use")

    monkeypatch.setattr(server_record.os, "replace", _refuse)
    monkeypatch.setattr(server_record, "_REPLACE_WAIT", 0.01)
    monkeypatch.setattr(server_record, "_REPLACE_RETRY", 0.001)

    server_record.mark_serving()

    assert "Could not mark the server's record serving" in caplog.text
    assert _read()[server_record.PHASE_KEY] == server_record.STARTING


def test_remove_record_removes_its_own(server_env):
    server_record.write_record()

    server_record.remove_record()

    assert not server_record.record_path().exists()


def test_remove_record_retries_while_reader_holds_it(server_env, monkeypatch):
    server_record.write_record()
    refusals = [PermissionError(32, "in use")]
    unlink = type(server_record.record_path()).unlink

    def _unlink(self, *a, **k):
        if refusals:
            raise refusals.pop()
        return unlink(self, *a, **k)

    monkeypatch.setattr(type(server_record.record_path()), "unlink", _unlink)
    monkeypatch.setattr(server_record, "_REPLACE_RETRY", 0.001)

    server_record.remove_record()

    assert not server_record.record_path().exists()


def test_remove_record_gives_up_on_record_that_stays_in_use(
    server_env, monkeypatch, caplog
):
    server_record.write_record()

    def _refuse(self, *a, **k):
        raise PermissionError(32, "in use")

    monkeypatch.setattr(type(server_record.record_path()), "unlink", _refuse)
    monkeypatch.setattr(server_record, "_REPLACE_WAIT", 0.01)
    monkeypatch.setattr(server_record, "_REPLACE_RETRY", 0.001)

    server_record.remove_record()

    assert "Could not remove the server's record" in caplog.text


def test_remove_record_leaves_another_servers_record(server_env):
    server_record.write_record()
    other = {**_read(), "pid": 77}
    server_record.record_path().write_text(json.dumps(other), encoding="utf-8")

    server_record.remove_record()

    assert _read()["pid"] == 77


def test_remove_record_runs_once(server_env):
    """After the lifespan removed it, the exit handler leaves a record the
    next server wrote alone, even one that reads the same."""
    server_record.write_record()
    written = server_record.record_path().read_text(encoding="utf-8")
    server_record.remove_record()
    server_record.record_path().write_text(written, encoding="utf-8")

    server_record.remove_record()

    assert server_record.record_path().exists()


def test_remove_record_without_one_written_does_nothing(server_env):
    server_record.record_path().write_text("{}", encoding="utf-8")

    server_record.remove_record()

    assert server_record.record_path().exists()


def test_started_process_stops_at_process_it_cannot_read(server_env, monkeypatch):
    class _Gone(_Proc):
        def ppid(self):
            raise psutil.NoSuchProcess(self.pid)

    monkeypatch.setattr(server_record.psutil, "Process", lambda *a: _Gone(30))

    process, found = server_record._started_process()

    assert (process.pid, found) == (30, False)


def test_server_lifespan_marks_record_serving_and_keeps_it(monkeypatch):
    from starlette.testclient import TestClient

    from EvoScientist.langgraph_dev.http import app

    steps = []
    monkeypatch.setattr(server_record, "mark_serving", lambda: steps.append("serving"))
    monkeypatch.setattr(server_record, "remove_record", lambda: steps.append("removed"))
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)):
        assert steps == ["serving"]
    # Removed only as the process exits, with the lock.
    assert steps == ["serving"]
