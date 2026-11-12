"""The server outlives the sessions that use it: ``EvoSci server start``, the
offer after onboarding, and the setup stage after an upgrade."""

from __future__ import annotations

import contextlib
import json
import time

import pytest
import yaml
from typer.testing import CliRunner

from EvoScientist import paths
from EvoScientist.cli._app import app
from EvoScientist.config.settings import get_config_path
from EvoScientist.langgraph_dev import manager, registration, server_home, server_record
from EvoScientist.langgraph_dev.registration import OlderServerError
from EvoScientist.setup import server as server_stage


def _record(pid=4242, *, port=6174, host="127.0.0.1", lasting=True, **kw):
    """The record of a running server, as ``read_record`` returns it."""
    return manager.ServerRecord(pid=pid, port=port, host=host, lasting=lasting, **kw)


def _result(*, started: bool, record=None) -> manager.ServerResult:
    from tests.fakes.langgraph_dev import FakeProc

    return manager.ServerResult(
        available=True,
        served=True,
        proc=FakeProc() if started else None,
        url="http://127.0.0.1:6174",
        record=record,
    )


@pytest.fixture
def started(monkeypatch) -> list:
    """Fake the server start: this process starts the server unless one
    answers already; return the projects it was started for."""
    calls: list = []

    def ensure(config, *, workspace_dir, backend):
        calls.append((workspace_dir, backend))
        return _result(started=not manager.is_langgraph_dev_running())

    monkeypatch.setattr(manager, "ensure_server", ensure)
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    return calls


def _emit():
    events: list[dict] = []
    return events, events.append


def _saved(**values) -> None:
    get_config_path().parent.mkdir(parents=True, exist_ok=True)
    get_config_path().write_text(yaml.safe_dump(values))


# ---------------------------------------------------------------------------
# EvoSci server start
# ---------------------------------------------------------------------------


def test_server_start_starts_server_for_project(started, tmp_path):
    result = CliRunner().invoke(app, ["server", "start", "--workdir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert started == [(tmp_path, "langgraph_server")]
    assert "EvoSci server stop" in " ".join(result.output.split())


def test_server_start_running_server_serves_project_too(started, monkeypatch, tmp_path):
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: True)

    result = CliRunner().invoke(app, ["server", "start", "--workdir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "already running" in " ".join(result.output.split())


def test_server_start_failure_names_log_in_one_line(monkeypatch, tmp_path):
    """The log's end is in the log, not on the terminal."""
    failed = manager.ServerResult(
        problem="exited",
        exit_code=1,
        error="langgraph dev exited immediately with code 1.\nLog tail:\nboom",
    )
    monkeypatch.setattr(manager, "ensure_server", lambda *_a, **_k: failed)

    result = CliRunner().invoke(app, ["server", "start", "--workdir", str(tmp_path)])

    assert result.exit_code == 1
    assert "boom" not in result.output
    assert "exited immediately with code 1" in " ".join(result.output.split())


def test_server_start_refusal_exits(started, monkeypatch, tmp_path):
    def older(*_a, **_k):
        raise OlderServerError()

    monkeypatch.setattr(manager, "ensure_server", older)

    result = CliRunner().invoke(app, ["server", "start", "--workdir", str(tmp_path)])

    assert result.exit_code == 1
    assert "older EvoScientist" in " ".join(result.output.split())


# ---------------------------------------------------------------------------
# After onboarding
# ---------------------------------------------------------------------------


def _answer(monkeypatch, answer):
    import questionary

    asked = []

    def confirm(question, **_k):
        asked.append(question)
        return type("Q", (), {"ask": lambda self: answer})()

    monkeypatch.setattr(questionary, "confirm", confirm)
    return asked


@pytest.mark.parametrize("answer", [True, False])
def test_offer_server_start_names_project(monkeypatch, tmp_path, answer):
    from EvoScientist.cli import commands, server_cmd

    monkeypatch.setenv("EVOSCIENTIST_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    asked = _answer(monkeypatch, answer)
    starts = []
    monkeypatch.setattr(
        server_cmd,
        "start_background",
        lambda workdir: starts.append(workdir),
    )

    commands._offer_server_start()

    assert str(tmp_path) in asked[0]
    assert starts == ([None] if answer else [])


def test_offer_without_default_workdir_starts_nothing(monkeypatch, tmp_path):
    """The folder onboarding ran in is not the project the user works in."""
    from EvoScientist.cli import commands, server_cmd

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EVOSCIENTIST_WORKSPACE_DIR", raising=False)
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    asked = _answer(monkeypatch, True)
    starts = []
    monkeypatch.setattr(
        server_cmd,
        "start_background",
        lambda workdir: starts.append(workdir),
    )

    commands._offer_server_start()

    assert asked == []
    assert starts == []


def _running_service(monkeypatch, record, *, jobs=(), busy=False):
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: True)
    monkeypatch.setattr(manager, "read_record", lambda **_k: record)
    monkeypatch.setattr(
        "EvoScientist.deploy.busy.running_bg_process_names", lambda url: list(jobs)
    )
    monkeypatch.setattr("EvoScientist.deploy.busy.has_busy_thread", lambda url: busy)


def _record_steps(monkeypatch) -> list:
    from EvoScientist.cli import server_cmd

    steps: list = []
    monkeypatch.setattr(server_cmd, "server_stop", lambda: steps.append("stop"))
    monkeypatch.setattr(
        server_cmd, "start_background", lambda workdir: steps.append("start")
    )
    return steps


@pytest.mark.parametrize(
    "record",
    [None, _record(lasting=False), _record(lasting=True)],
    ids=["unrecorded", "owned", "lasting"],
)
def test_offer_leaves_running_server_alone(monkeypatch, record):
    """It says how to apply the new settings; it restarts nothing."""
    from EvoScientist.cli import commands

    _running_service(monkeypatch, record)
    asked = _answer(monkeypatch, True)
    steps = _record_steps(monkeypatch)

    with commands.console.capture() as captured:
        commands._offer_server_start()

    assert asked == []
    assert steps == []
    lasting = record is not None and record.lasting
    hint = "EvoSci server stop" if lasting else "not the background server"
    assert hint in " ".join(captured.get().split())


@pytest.mark.parametrize("default_workdir", [True, False])
def test_offer_restart_hint_follows_default_workdir(
    monkeypatch, tmp_path, default_workdir
):
    """With a ``default_workdir`` a plain server start restarts it; without
    one a session that needs it does."""
    from EvoScientist.cli import commands

    if default_workdir:
        _saved(default_workdir=str(tmp_path))
    _running_service(monkeypatch, _record())
    _answer(monkeypatch, True)
    steps = _record_steps(monkeypatch)

    with commands.console.capture() as captured:
        commands._offer_server_start()

    output = " ".join(captured.get().split())
    assert steps == []
    assert "EvoSci server stop" in output
    assert ("EvoSci server start" in output) is default_workdir
    assert ("EvoSci session" in output) is not default_workdir
    assert "--workdir" not in output


@pytest.mark.parametrize(
    ("changed", "older"),
    [(False, False), (True, False), (False, True)],
    ids=["unchanged", "changed", "older-release"],
)
def test_offer_restart_hint_only_on_drift(monkeypatch, changed, older):
    """The hint shows only when settings the server reads changed since its
    launch, or when an older release started the server."""
    from EvoScientist.cli import commands
    from EvoScientist.config import get_effective_config

    fingerprint = manager._server_config_fingerprint(get_effective_config())
    recorded = "stale" if changed else fingerprint
    _running_service(
        monkeypatch, _record(fields={"config_fingerprint": recorded}, older=older)
    )
    _answer(monkeypatch, True)
    steps = _record_steps(monkeypatch)

    with commands.console.capture() as captured:
        commands._offer_server_start()

    assert steps == []
    output = " ".join(captured.get().split())
    assert ("EvoSci server stop" in output) is (changed or older)


def test_offer_server_start_failure_is_reported(monkeypatch, tmp_path):
    """The configuration is saved; no traceback."""
    from EvoScientist.cli import commands, server_cmd

    monkeypatch.setenv("EVOSCIENTIST_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    _answer(monkeypatch, True)

    def unwritable(workdir):
        raise PermissionError("read-only")

    monkeypatch.setattr(server_cmd, "start_background", unwritable)

    with commands.console.capture() as captured:
        commands._offer_server_start()

    assert "next EvoSci session" in captured.get()


def test_onboard_without_terminal_offers_nothing(monkeypatch):
    from EvoScientist.cli import commands

    monkeypatch.setattr(commands, "_run_onboard_cli", lambda **_k: True)
    offered = []
    monkeypatch.setattr(commands, "_offer_server_start", lambda: offered.append(1))

    CliRunner().invoke(app, ["onboard"])

    assert offered == []


# ---------------------------------------------------------------------------
# The setup stage after an upgrade
# ---------------------------------------------------------------------------


def test_stage_announces_lasting_server_once_after_upgrade(started):
    _saved(model="m")
    _, emit = _emit()

    first = server_stage.run_stage(emit, "default")
    again = server_stage.run_stage(emit, "default")

    assert first.status == "done"
    assert first.message == server_stage.LASTING_MESSAGE
    assert again.status == "skipped"
    assert started == []


def test_stage_on_new_install_announces_nothing(started):
    result = server_stage.run_stage(_emit()[1], "default")

    assert result.status == "skipped"
    assert not get_config_path().exists()


def _updated(monkeypatch) -> None:
    """The update replaced the installed version since the record."""
    monkeypatch.setattr(server_stage, "_installed_version", lambda: "new")


def test_stage_restarts_server_update_stopped(started, monkeypatch):
    """The server restores every registered project: the restart registers
    none."""
    server_stage.remember_for_restart()
    _updated(monkeypatch)
    events, emit = _emit()

    result = server_stage.run_stage(emit, "default")

    assert started == [(None, "langgraph_server")]
    assert result.detail["restarted"] is True
    assert "Started" in result.message
    assert not (paths.SERVER_HOME / "restart-after-update.json").exists()
    assert [e["status"] for e in events] == ["running"]


def test_stage_restart_finding_running_server_claims_no_start(started, monkeypatch):
    """Another server took the port since the update: this process started
    nothing, and says where to look."""
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: True)
    server_stage.remember_for_restart()
    _updated(monkeypatch)

    result = server_stage.run_stage(_emit()[1], "default")

    assert started == [(None, "langgraph_server")]
    assert result.detail["restarted"] is False
    assert "Started" not in result.message
    assert "EvoSci server status" in result.message


def test_stage_restart_ignores_project_of_older_record(started, monkeypatch, tmp_path):
    """An older updater recorded the project the server was started from;
    the restart still registers none, so a project removed since stays
    removed."""
    project = tmp_path / "proj"
    paths.SERVER_HOME.mkdir(parents=True, exist_ok=True)
    (paths.SERVER_HOME / "restart-after-update.json").write_text(
        json.dumps(
            {
                "workspace": str(project),
                "stopped_at": time.time(),
                "from_version": "old",
            }
        )
    )
    _updated(monkeypatch)

    result = server_stage.run_stage(_emit()[1], "default")

    assert started == [(None, "langgraph_server")]
    assert result.detail["restarted"] is True
    assert not project.exists()


def test_stage_ignores_record_of_failed_update(started):
    """An update that failed after it stopped the server left it."""
    paths.SERVER_HOME.mkdir(parents=True, exist_ok=True)
    (paths.SERVER_HOME / "restart-after-update.json").write_text(
        json.dumps({"stopped_at": time.time() - 7200})
    )

    result = server_stage.run_stage(_emit()[1], "default")

    assert started == []
    assert result.detail["restarted"] is False
    assert not (paths.SERVER_HOME / "restart-after-update.json").exists()


def test_stage_restart_failure_is_reported_not_raised(started, monkeypatch):
    def failed(*_a, **_k):
        raise RuntimeError("ccproxy did not start")

    monkeypatch.setattr(manager, "ensure_server", failed)
    server_stage.remember_for_restart()
    _updated(monkeypatch)

    result = server_stage.run_stage(_emit()[1], "default")

    assert result.detail["restarted"] is False
    assert "Could not start" in result.message


def test_stage_restart_ccproxy_failure_says_why(started, monkeypatch):
    """The next session's start would fail the same way."""
    failed = manager.ServerResult(
        problem="ccproxy_failed",
        error="The server could not start ccproxy: Run: ccproxy auth login",
    )
    monkeypatch.setattr(manager, "ensure_server", lambda *_a, **_k: failed)
    server_stage.remember_for_restart()
    _updated(monkeypatch)

    result = server_stage.run_stage(_emit()[1], "default")

    assert result.detail["restarted"] is False
    assert "ccproxy auth login" in result.message


def test_stage_ignores_record_when_update_did_not_install(started):
    """The update failed after it stopped the server: same version."""
    server_stage.remember_for_restart()

    result = server_stage.run_stage(_emit()[1], "default")

    assert started == []
    assert result.detail["restarted"] is False


def test_stage_unreadable_restart_record_is_dropped(started):
    paths.SERVER_HOME.mkdir(parents=True, exist_ok=True)
    (paths.SERVER_HOME / "restart-after-update.json").write_text(json.dumps([]))

    server_stage.run_stage(_emit()[1], "default")

    assert started == []
    assert not (paths.SERVER_HOME / "restart-after-update.json").exists()


def test_service_stop_is_server_stop_with_notice(monkeypatch):
    stopped = []
    monkeypatch.setattr(
        manager, "stop_server", lambda **_k: stopped.append("server") or 42
    )

    result = CliRunner().invoke(app, ["service", "stop"])

    assert result.exit_code == 0, result.output
    assert stopped == ["server"]
    assert "deprecated" in result.output


def _recorded_server(monkeypatch, record, stopped: list) -> None:
    """*record* names the running server; ending it is recorded in
    *stopped*."""
    monkeypatch.setattr(manager, "read_record", lambda **_k: record)
    monkeypatch.setattr(manager, "_langgraph_cmdline", lambda pid: ["langgraph"])
    monkeypatch.setattr(manager, "_hold_pid", lambda pid: contextlib.nullcontext())
    monkeypatch.setattr(
        manager, "_end_server_tree", lambda *a, **k: stopped.append("server")
    )


def test_server_stop_leaves_deploy_owned_server(monkeypatch):
    """EvoSci deploy or the desktop app stops its own server."""
    from EvoScientist.cli import server_cmd

    stopped = []
    _recorded_server(monkeypatch, _record(lasting=False), stopped)

    result = CliRunner().invoke(app, ["server", "stop"])

    assert result.exit_code == 0, result.output
    assert stopped == []
    output = " ".join(result.output.split())
    assert "not the background server" in output
    # Its owner may be gone.
    assert server_cmd.manual_stop_command(6174) in output


def test_server_stop_stops_server_nobody_else_owns(monkeypatch):
    """A record that does not say who owns its server (a server started by
    hand, an older release) is the stop's to end."""
    stopped = []
    _recorded_server(monkeypatch, _record(lasting=None), stopped)
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)

    result = CliRunner().invoke(app, ["server", "stop"])

    assert result.exit_code == 0, result.output
    assert stopped == ["server"]


def test_server_stop_refused_access_exits_without_claiming_stop(monkeypatch):
    """Another user's or an elevated server: the stop says why it is left
    running and exits 1, and never reports it stopped."""
    monkeypatch.setattr(
        manager, "read_record_and_lock", lambda: (_record(), server_home.FREE)
    )

    def refused(**_k):
        raise manager._stop_failed(4242, manager._refused_cause())

    monkeypatch.setattr(manager, "stop_server", refused)

    result = CliRunner().invoke(app, ["server", "stop", "--yes"])

    assert result.exit_code == 1
    output = " ".join(result.output.split())
    assert "4242" in output
    assert manager._refused_cause().split(";")[0] in output
    assert "Stopped" not in output
    assert "no ownership record" not in output


def test_server_stop_looks_for_gone_server_at_its_recorded_address(monkeypatch):
    """A recorded server that is gone is looked for at its own address, not
    the configured one."""
    asked = []
    monkeypatch.setattr(
        manager,
        "read_record_and_lock",
        lambda: (_record(port=7000), server_home.FREE),
    )
    monkeypatch.setattr(manager, "stop_server", lambda **_k: None)
    monkeypatch.setattr(
        manager,
        "is_langgraph_dev_running",
        lambda *, port, host: asked.append(port) or False,
    )

    result = CliRunner().invoke(app, ["server", "stop", "--yes"])

    assert result.exit_code == 0, result.output
    assert asked == [7000]


def test_server_start_that_lost_race_says_it_reused(monkeypatch, tmp_path):
    """Another start took the port first: this one only registered its project."""
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    monkeypatch.setattr(
        manager, "ensure_server", lambda *_a, **_k: _result(started=False)
    )

    result = CliRunner().invoke(app, ["server", "start", "--workdir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "already running" in output
    assert "started from" not in output


def test_server_status_names_configured_port_of_server_on_another(
    monkeypatch,
):
    """langgraph_dev_port changed while the server ran."""
    _status_server(monkeypatch, projects=[], record=_record(port=7000))

    output = _status_output()

    assert "on port 7000" in output
    assert "configured port is 6174" in output


def test_server_status_lists_projects_of_server_on_another_port(monkeypatch):
    """The projects are the recorded server's, whatever answers on the
    configured port."""
    urls = _status_server(
        monkeypatch,
        projects=[{"workspace_dir": "/p/a", "missing": False}],
        record=_record(port=7000, host="192.0.2.10"),
    )

    output = _status_output()

    assert [url.rsplit("//", 1)[-1] for url in urls] == ["192.0.2.10:7000"]
    assert "on port 7000" in output
    assert "/p/a" in output


def test_server_status_shows_recorded_address_and_probes_only_it(monkeypatch):
    """Whatever host is configured, status names the address the recorded
    server listens on, and asks only there."""
    urls = _status_server(monkeypatch, projects=[], record=_record(host="192.168.1.5"))
    monkeypatch.setattr(manager, "configured_address", lambda _c: (6174, "10.0.0.5"))
    probed: list = []
    monkeypatch.setattr(
        manager,
        "is_langgraph_dev_running",
        lambda **kw: probed.append((kw.get("port"), kw.get("host"))) or True,
    )

    output = _status_output()

    assert "http://192.168.1.5:6174" in output
    assert probed == [(6174, "192.168.1.5")]
    assert urls == ["http://192.168.1.5:6174"]
    assert "listens on" not in output


@pytest.mark.parametrize(
    ("host", "shown"),
    [("192.168.1.5", True), ("0.0.0.0", True), ("10.0.0.5", False)],
)
def test_server_status_names_configured_host_other_than_recorded(
    monkeypatch, host, shown
):
    _status_server(monkeypatch, projects=[], record=_record(host=host))
    monkeypatch.setattr(manager, "configured_address", lambda _c: (6174, "10.0.0.5"))

    output = _status_output()

    assert ("configured host: 10.0.0.5" in output) is shown


@pytest.mark.parametrize(
    ("host", "address"),
    [("0.0.0.0", "http://127.0.0.1:6174"), ("::", "http://[::1]:6174")],
)
def test_server_status_names_wildcard_bind_next_to_address(monkeypatch, host, address):
    _status_server(monkeypatch, projects=[], record=_record(host=host))

    output = _status_output()

    assert address in output
    assert f"listens on {host}" in output


def test_server_status_says_to_restart_older_releases_server(monkeypatch):
    _status_server(monkeypatch, projects=[], record=_record(older=True))

    output = _status_output()

    assert "older EvoScientist release" in output


def test_server_status_names_lock_held_without_record(monkeypatch):
    from EvoScientist.langgraph_dev import server_home

    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    monkeypatch.setattr(manager, "read_record", lambda **_k: None)
    monkeypatch.setattr(server_home, "lock_state", lambda wait=0: server_home.HELD)

    output = _status_output()

    assert "without a record" in output
    assert "server.lock" in "".join(output.split())


def test_server_status_shows_starting_record_as_starting(monkeypatch):

    _status_server(
        monkeypatch, projects=[], record=_record(phase=server_record.STARTING)
    )
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)

    output = _status_output()

    assert "starting" in output
    assert "not responding" not in output


@pytest.mark.parametrize("group", ["server", "service"])
def test_stop_names_lock_held_without_record(monkeypatch, group):
    from EvoScientist.langgraph_dev import server_home

    stopped: list = []
    monkeypatch.setattr(manager, "read_record", lambda **_k: None)
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    monkeypatch.setattr(server_home, "lock_state", lambda wait=0: server_home.HELD)
    monkeypatch.setattr(manager, "stop_server", lambda **_k: stopped.append(1))

    result = CliRunner().invoke(app, [group, "stop", "--yes"])

    assert result.exit_code == 1
    assert stopped == []
    output = " ".join(result.output.split())
    assert "without a record" in output
    assert "try again" not in output


def _status_server(monkeypatch, *, projects=None, error=None, record=None):
    """A running server, recorded (on the configured port unless *record*
    says otherwise); *projects* is what it lists, *error* what the read
    raises instead."""
    from EvoScientist.langgraph_dev import registration

    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: True)
    record = record or _record()
    monkeypatch.setattr(manager, "read_record", lambda **_k: record)
    urls: list[str] = []

    def served_projects(url):
        urls.append(url)
        if error is not None:
            raise error
        return projects

    monkeypatch.setattr(registration, "served_projects", served_projects)
    return urls


def _status_output() -> str:
    result = CliRunner().invoke(app, ["server", "status"])
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def test_server_status_lists_projects_of_own_server(monkeypatch):
    kept, gone = "/p/kept", "/p/gone"
    _status_server(
        monkeypatch,
        projects=[
            {"workspace_dir": kept, "name": "kept", "missing": False},
            {"workspace_dir": gone, "name": "gone", "missing": True},
        ],
    )

    output = _status_output()

    assert "on port 6174" in output
    assert "4242" in output
    assert f"{kept} (folder missing)" not in output
    assert kept in output
    assert f"{gone} (folder missing)" in output


def test_server_status_without_projects_says_none(monkeypatch):
    _status_server(monkeypatch, projects=[])

    output = _status_output()

    assert "projects: none" in output


@pytest.mark.parametrize(
    ("error", "said"),
    [
        (lambda: registration.OtherServerError("another data folder"), "not listed"),
        (lambda: OlderServerError(), "could not be read"),
        (lambda: registration.RegistrationError("timed out"), "could not be read"),
        (
            lambda: registration.RegistrationError("Cannot read the local secret"),
            "could not be read",
        ),
    ],
    ids=["another_data_folder", "older_server", "read_failure", "unreadable_secret"],
)
def test_server_status_says_why_projects_are_not_listed(monkeypatch, error, said):
    _status_server(monkeypatch, error=error())

    output = _status_output()

    assert said in output
    assert "projects: none" not in output


def test_server_status_not_running_lists_no_projects(monkeypatch):
    urls = _status_server(monkeypatch, projects=[])
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    monkeypatch.setattr(manager, "read_record", lambda **_k: None)

    output = _status_output()

    assert "No background langgraph dev server is running" in output
    assert urls == []


def _oauth(monkeypatch, oauth: bool) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        "EvoScientist.config.get_effective_config",
        lambda: SimpleNamespace(
            anthropic_auth_mode="oauth" if oauth else "api_key",
            ccproxy_port=8000,
            langgraph_dev_port=6174,
            langgraph_dev_host="127.0.0.1",
        ),
    )


@pytest.mark.parametrize("running", [True, False])
def test_server_status_shows_server_ccproxy_with_oauth(monkeypatch, running):
    import EvoScientist.ccproxy_manager as ccproxy_mod

    _status_server(
        monkeypatch,
        projects=[],
        record=_record(fields={server_record.CCPROXY_PID_KEY: 77}),
    )
    _oauth(monkeypatch, True)
    monkeypatch.setattr(ccproxy_mod, "is_ccproxy_running", lambda port: running)

    output = _status_output()

    assert "ccproxy: port 8000" in output
    assert ("pid 77" in output) is running
    assert ("not running" in output) is not running


def test_server_status_shows_no_ccproxy_for_older_releases_server(monkeypatch):
    """It starts no ccproxy of its own."""
    _status_server(monkeypatch, projects=[], record=_record(older=True))
    _oauth(monkeypatch, True)

    assert "ccproxy" not in _status_output()


def test_server_status_shows_no_ccproxy_without_oauth(monkeypatch):
    _status_server(monkeypatch, projects=[])
    _oauth(monkeypatch, False)

    assert "ccproxy" not in _status_output()


def test_service_start_is_server_start_with_notice(started, tmp_path):
    result = CliRunner().invoke(app, ["service", "start", "--workdir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert started == [(tmp_path, "langgraph_server")]
    output = " ".join(result.output.split())
    assert "deprecated" in output
    assert "EvoSci server stop" in output


def test_server_start_oauth_points_at_server_ccproxy(started, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import EvoScientist.ccproxy_manager as ccproxy_mod

    monkeypatch.setattr(
        "EvoScientist.config.get_effective_config",
        lambda: SimpleNamespace(
            anthropic_auth_mode="oauth",
            openai_auth_mode="api_key",
            ccproxy_port=8000,
            default_workdir="",
            langgraph_dev_port=6174,
            langgraph_dev_host="127.0.0.1",
        ),
    )
    pointed: list[int] = []
    monkeypatch.setattr(ccproxy_mod, "setup_ccproxy_env", pointed.append)

    result = CliRunner().invoke(app, ["server", "start", "--workdir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert pointed == [8000]


def test_stop_recorded_ccproxy_keeps_record_when_denied(monkeypatch, tmp_path):
    """It may still be ours; a later stop can try again."""
    import psutil

    import EvoScientist.ccproxy_manager as ccproxy_mod

    monkeypatch.setattr(paths, "SERVER_HOME", tmp_path)
    (tmp_path / "ccproxy.pid").write_text("4242")

    class _Denied:
        def __init__(self, _pid):
            raise psutil.AccessDenied(4242)

    monkeypatch.setattr(psutil, "Process", _Denied)

    assert ccproxy_mod.stop_recorded_ccproxy() is None
    assert (tmp_path / "ccproxy.pid").exists()


def test_update_wait_spares_older_ccproxy(monkeypatch, tmp_path):
    """The update stops it itself, after the wait."""
    import os

    import EvoScientist.ccproxy_manager as ccproxy_mod
    from EvoScientist import updater

    monkeypatch.setattr(ccproxy_mod, "recorded_ccproxy_running", os.getpid)
    monkeypatch.setattr(manager, "read_record", lambda **_k: None)

    assert os.getpid() in updater._server_tree()


@pytest.fixture
def busy_server(monkeypatch):
    """A running server with a background job; returns the stops made."""
    from EvoScientist.deploy import busy

    stopped: list = []
    monkeypatch.setattr(manager, "read_record", lambda **_k: _record())
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: True)
    monkeypatch.setattr(manager, "_is_port_occupied", lambda port, host="": True)
    monkeypatch.setattr(busy, "backend_has_active_runs", lambda url: True)
    monkeypatch.setattr(busy, "running_bg_process_names", lambda url: ["train.py"])
    monkeypatch.setattr(busy, "has_busy_thread", lambda url: False)
    monkeypatch.setattr(
        manager,
        "stop_server",
        lambda **_k: stopped.append("server") or 42,
    )
    return stopped


def test_stop_without_record_asks_nothing(busy_server, monkeypatch):
    """A busy server on the port that no record names is not one the stop
    ends: no question, nor --yes in a script."""
    monkeypatch.setattr(manager, "read_record", lambda **_k: None)

    result = CliRunner().invoke(app, ["server", "stop"])

    assert result.exit_code == 0, result.output
    assert "--yes" not in result.output


def test_server_stop_asks_nothing_of_owned_busy_server(busy_server, monkeypatch):
    """EvoSci deploy or the desktop app owns it: the stop leaves it running,
    so its work needs no question, nor --yes in a script."""

    def stop(*, keep_owned=False, record=None):
        raise manager.ServerKeptError("left running")

    monkeypatch.setattr(manager, "owned_by_its_starter", lambda _record: True)
    monkeypatch.setattr(manager, "stop_server", stop)

    result = CliRunner().invoke(app, ["server", "stop"])

    assert result.exit_code == 0, result.output
    assert "--yes" not in result.output
    assert "left running" in result.output


def test_server_stop_says_to_retry_a_server_it_could_not_open(busy_server, monkeypatch):
    """The stop says to try again, not that it has no record, nor how to
    kill it by hand, and exits 1 for a script."""

    def stop(*, keep_owned=False, record=None):
        raise manager.ServerStopFailedError(
            "Could not stop langgraph dev (pid 42): out of resources. It keeps "
            "running; try again."
        )

    monkeypatch.setattr(manager, "stop_server", stop)

    result = CliRunner().invoke(app, ["server", "stop", "--yes"])

    assert result.exit_code == 1
    output = " ".join(result.output.split())
    assert "try again" in output
    assert "ownership record" not in output
    assert "kill" not in output
    assert "Stop-Process" not in output


def test_manual_stop_command_ends_only_listener(monkeypatch):
    """A session connected to the port is not the server it means."""
    from EvoScientist.cli import server_cmd

    monkeypatch.setattr(server_cmd.sys, "platform", "linux")

    assert "-sTCP:LISTEN" in server_cmd.manual_stop_command(6174)


def test_server_stop_asks_about_busy_server_slow_to_answer(busy_server, monkeypatch):
    """A busy server may miss /ok's second; its work is asked all the same."""
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)

    result = CliRunner().invoke(app, ["server", "stop"])

    assert result.exit_code == 1
    assert busy_server == []
    assert "--yes" in " ".join(result.output.split())


def test_stop_asks_recorded_server_on_another_port(busy_server, monkeypatch):
    """langgraph_dev_port changed while the server ran: the stop ends the
    recorded server, so that is the one asked whether it is busy."""
    from EvoScientist.deploy import busy

    asked = []
    monkeypatch.setattr(manager, "read_record", lambda **_k: _record(port=7000))
    monkeypatch.setattr(
        manager, "is_langgraph_dev_running", lambda *, port, host: port == 7000
    )
    monkeypatch.setattr(
        busy, "backend_has_active_runs", lambda url: asked.append(url) or True
    )

    result = CliRunner().invoke(app, ["server", "stop"])

    assert result.exit_code == 1
    assert busy_server == []
    assert asked
    assert all(":7000" in url for url in asked)


@pytest.mark.parametrize("command", [["server", "stop"], ["service", "stop"]])
def test_stop_without_terminal_refuses_busy_server(busy_server, command):
    """It serves every project; a script must say --yes to end its work."""
    result = CliRunner().invoke(app, command)

    assert result.exit_code == 1
    assert busy_server == []
    output = " ".join(result.output.split())
    assert "train.py" in output
    assert "--yes" in output


@pytest.mark.parametrize(
    ("jobs", "running", "said", "unsaid"),
    [
        (
            ["train.py"],
            False,
            "background jobs that a stop ends (train.py)",
            "runs again",
        ),
        ([], True, "a run in progress that a stop cuts and that runs again", "jobs"),
        (["train.py"], True, "ends (train.py); a run in progress", "work in progress"),
        ([], False, "work in progress", "runs again"),
    ],
    ids=["jobs", "run", "both", "neither found"],
)
def test_stop_question_says_what_a_stop_does(
    busy_server, monkeypatch, jobs, running, said, unsaid
):
    """Jobs end and runs run again at the next start; work whose kind the
    lookups missed promises neither."""
    from EvoScientist.deploy import busy

    monkeypatch.setattr(busy, "running_bg_process_names", lambda url: jobs)
    monkeypatch.setattr(busy, "has_busy_thread", lambda url: running)

    result = CliRunner().invoke(app, ["server", "stop"])

    assert result.exit_code == 1
    output = " ".join(result.output.split())
    assert said in output
    assert unsaid not in output


@pytest.mark.parametrize("group", ["server", "service"])
def test_stop_with_yes_stops_busy_server(busy_server, monkeypatch, group):
    import EvoScientist.ccproxy_manager as ccproxy_mod

    monkeypatch.setattr(ccproxy_mod, "stop_recorded_ccproxy", lambda: None)

    result = CliRunner().invoke(app, [group, "stop", "--yes"])

    assert result.exit_code == 0, result.output
    assert busy_server == ["server"]


@pytest.mark.parametrize(("answer", "stops"), [("y", ["server"]), ("n", [])])
def test_stop_asks_before_ending_busy_server(busy_server, monkeypatch, answer, stops):
    from EvoScientist.cli import server_cmd

    monkeypatch.setattr(server_cmd, "_interactive", lambda: True)

    result = CliRunner().invoke(app, ["server", "stop"], input=f"{answer}\n")

    assert result.exit_code == 0, result.output
    assert busy_server == stops


def test_stop_idle_server_asks_nothing(busy_server, monkeypatch):
    from EvoScientist.deploy import busy

    monkeypatch.setattr(busy, "backend_has_active_runs", lambda url: False)

    result = CliRunner().invoke(app, ["server", "stop"])

    assert result.exit_code == 0, result.output
    assert busy_server == ["server"]


def test_server_stop_stops_nothing_when_another_server_replaced_asked_one(
    busy_server, monkeypatch
):
    """The question may take a while: a server that took the place of the
    one asked about is not stopped without asking."""
    from EvoScientist.cli import server_cmd

    records = iter([_record(4242), _record(5151)])
    monkeypatch.setattr(manager, "read_record", lambda **_k: next(records))
    monkeypatch.setattr(server_cmd, "_interactive", lambda: True)

    result = CliRunner().invoke(app, ["server", "stop"], input="y\n")

    assert result.exit_code == 1
    assert busy_server == []
    assert "5151" in " ".join(result.output.split())


def test_server_stop_stops_nothing_that_appeared_after_prompt(busy_server, monkeypatch):
    """No server when the stop looked; one runs once it would stop: it was
    never asked about, so it is not stopped."""
    records = iter([None, _record(6161)])
    monkeypatch.setattr(manager, "read_record", lambda **_k: next(records))

    result = CliRunner().invoke(app, ["server", "stop"])

    assert result.exit_code == 1
    assert busy_server == []
    output = " ".join(result.output.split())
    assert "6161" in output
    assert "starting now" in output
    assert "not the one asked about" not in output


def test_server_status_names_unreadable_lock(monkeypatch):
    from EvoScientist.langgraph_dev import server_home

    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    monkeypatch.setattr(manager, "read_record", lambda **_k: None)
    monkeypatch.setattr(server_home, "lock_state", lambda wait=0: server_home.UNKNOWN)

    output = _status_output()

    assert "cannot be opened" in output
    assert "server.lock" in "".join(output.split())


@pytest.mark.parametrize("group", ["server", "service"])
def test_stop_names_unreadable_lock(monkeypatch, group):
    from EvoScientist.langgraph_dev import server_home

    monkeypatch.setattr(manager, "read_record", lambda **_k: None)
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    monkeypatch.setattr(server_home, "lock_state", lambda wait=0: server_home.UNKNOWN)

    result = CliRunner().invoke(app, [group, "stop", "--yes"])

    assert result.exit_code == 1
    output = " ".join(result.output.split())
    assert "cannot be opened" in output


def test_server_stop_while_server_starts_stops_nothing(monkeypatch):
    """A server being started has no record yet: the stop says so instead of
    finding none."""
    stopped = []
    monkeypatch.setattr(manager, "stop_server", lambda **_k: stopped.append(1))
    monkeypatch.setattr(
        manager, "read_record_and_lock", lambda: (None, server_home.FREE)
    )
    monkeypatch.setattr(manager, "start_in_progress", lambda: True)

    result = CliRunner().invoke(app, ["server", "stop", "--yes"])

    assert result.exit_code == 1
    assert "starting the server" in result.output
    assert stopped == []


def test_server_stop_after_starts_record_just_appeared_stops_nothing(monkeypatch):
    """The start released its lock between the stop's two reads."""
    stopped = []
    monkeypatch.setattr(manager, "stop_server", lambda **_k: stopped.append(1))
    monkeypatch.setattr(
        manager, "read_record_and_lock", lambda: (None, server_home.FREE)
    )
    monkeypatch.setattr(manager, "start_in_progress", lambda: False)
    monkeypatch.setattr(manager, "read_record", lambda **_k: _record())

    result = CliRunner().invoke(app, ["server", "stop", "--yes"])

    assert result.exit_code == 1
    assert "starting the server" in result.output
    assert stopped == []
