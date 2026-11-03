"""The server outlives the sessions that use it: ``EvoSci server start``, the
offer after onboarding, and the setup stage after an upgrade."""

from __future__ import annotations

import json
import time

import pytest
import yaml
from typer.testing import CliRunner

from EvoScientist import paths
from EvoScientist.cli._app import app
from EvoScientist.config.settings import get_config_path
from EvoScientist.langgraph_dev import manager, registration
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
    import EvoScientist.ccproxy_manager as ccproxy_mod

    calls: list = []

    def ensure(config, *, workspace_dir, backend):
        calls.append((workspace_dir, backend))
        return _result(started=not manager.is_langgraph_dev_running())

    monkeypatch.setattr(manager, "ensure_server", ensure)
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    monkeypatch.setattr(
        ccproxy_mod, "maybe_start_ccproxy", lambda config, *, lasting: None
    )
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
    import EvoScientist.ccproxy_manager as ccproxy_mod

    failed = manager.ServerResult(
        problem="exited",
        exit_code=1,
        error="langgraph dev exited immediately with code 1.\nLog tail:\nboom",
    )
    monkeypatch.setattr(manager, "ensure_server", lambda *_a, **_k: failed)
    monkeypatch.setattr(
        ccproxy_mod, "maybe_start_ccproxy", lambda config, *, lasting: None
    )

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
def test_offer_service_start_names_project(monkeypatch, tmp_path, answer):
    from EvoScientist.cli import commands, server_cmd

    monkeypatch.setenv("EVOSCIENTIST_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    asked = _answer(monkeypatch, answer)
    starts = []
    monkeypatch.setattr(
        server_cmd,
        "start_background",
        lambda workdir, *, with_ccproxy: starts.append(with_ccproxy),
    )

    commands._offer_server_start()

    assert str(tmp_path) in asked[0]
    assert starts == ([True] if answer else [])


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
        lambda workdir, *, with_ccproxy: starts.append(with_ccproxy),
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
    from EvoScientist.cli import server_cmd, service_cmd

    steps: list = []
    monkeypatch.setattr(service_cmd, "service_stop", lambda: steps.append("stop"))
    monkeypatch.setattr(
        server_cmd,
        "start_background",
        lambda workdir, *, with_ccproxy: steps.append("start"),
    )
    return steps


@pytest.mark.parametrize(
    "record",
    [None, _record(lasting=False), _record(lasting=True)],
    ids=["unrecorded", "owned", "lasting"],
)
def test_offer_leaves_running_service_alone(monkeypatch, record):
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
    hint = "EvoSci service stop" if lasting else "not the background service"
    assert hint in " ".join(captured.get().split())


@pytest.mark.parametrize("default_workdir", [True, False])
def test_offer_restart_hint_follows_default_workdir(
    monkeypatch, tmp_path, default_workdir
):
    """With a ``default_workdir`` a plain service start restarts it; without
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
    assert "EvoSci service stop" in output
    assert ("EvoSci service start" in output) is default_workdir
    assert ("EvoSci session" in output) is not default_workdir
    assert "--workdir" not in output


def test_offer_service_start_failure_is_reported(monkeypatch, tmp_path):
    """The configuration is saved; no traceback."""
    from EvoScientist.cli import commands, server_cmd

    monkeypatch.setenv("EVOSCIENTIST_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    _answer(monkeypatch, True)

    def unwritable(workdir, *, with_ccproxy):
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


def test_server_stop_leaves_ccproxy(monkeypatch):
    """Open sessions on the local backend route through it."""
    import EvoScientist.ccproxy_manager as ccproxy_mod

    monkeypatch.setattr(manager, "stop_server", lambda **_k: 42)
    stopped = []
    monkeypatch.setattr(ccproxy_mod, "stop_recorded_ccproxy", lambda: stopped.append(1))

    result = CliRunner().invoke(app, ["server", "stop"])

    assert result.exit_code == 0, result.output
    assert stopped == []


def test_service_stop_stops_server_and_ccproxy(monkeypatch):
    import EvoScientist.ccproxy_manager as ccproxy_mod

    stopped = []
    monkeypatch.setattr(
        manager,
        "stop_server",
        lambda **_k: stopped.append("server") or 42,
    )
    monkeypatch.setattr(
        ccproxy_mod, "stop_recorded_ccproxy", lambda: stopped.append("ccproxy") or 7
    )

    result = CliRunner().invoke(app, ["service", "stop"])

    assert result.exit_code == 0, result.output
    assert stopped == ["server", "ccproxy"]


def _recorded_server(monkeypatch, record, stopped: list) -> None:
    """*record* names the running server; ending it is recorded in
    *stopped*."""
    monkeypatch.setattr(manager, "read_record", lambda **_k: record)
    monkeypatch.setattr(manager, "_langgraph_cmdline", lambda pid: ["langgraph"])
    monkeypatch.setattr(
        manager, "_end_server_tree", lambda *a, **k: stopped.append("server")
    )


@pytest.mark.parametrize("ccproxy_pid", [7, None])
def test_service_stop_leaves_deploy_owned_server(monkeypatch, ccproxy_pid):
    """EvoSci deploy or the desktop app stops its own server; the ccproxy that
    server may route through stays too."""
    import EvoScientist.ccproxy_manager as ccproxy_mod

    stopped = []
    _recorded_server(monkeypatch, _record(lasting=False), stopped)
    monkeypatch.setattr(ccproxy_mod, "recorded_ccproxy_running", lambda: ccproxy_pid)
    monkeypatch.setattr(
        ccproxy_mod, "stop_recorded_ccproxy", lambda: stopped.append("ccproxy") or 7
    )

    result = CliRunner().invoke(app, ["service", "stop"])

    assert result.exit_code == 0, result.output
    assert stopped == []
    output = " ".join(result.output.split())
    assert "not the background service" in output
    assert ("EvoSci ccproxy stop" in output) is (ccproxy_pid is not None)


def test_service_stop_leaves_ccproxy_for_unrecorded_server(monkeypatch):
    """A server with no ownership record is not touched; it may route through
    the service's ccproxy."""
    import EvoScientist.ccproxy_manager as ccproxy_mod
    from EvoScientist.deploy import busy

    stopped = []
    monkeypatch.setattr(manager, "stop_server", lambda **_k: None)
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: True)
    monkeypatch.setattr(busy, "backend_has_active_runs", lambda url: False)
    monkeypatch.setattr(ccproxy_mod, "recorded_ccproxy_running", lambda: 7)
    monkeypatch.setattr(
        ccproxy_mod, "stop_recorded_ccproxy", lambda: stopped.append("ccproxy") or 7
    )

    result = CliRunner().invoke(app, ["service", "stop"])

    assert result.exit_code == 0, result.output
    assert stopped == []
    assert "EvoSci ccproxy stop" in " ".join(result.output.split())


@pytest.mark.parametrize(
    ("record", "expected"),
    [(None, ["ccproxy"]), (_record(lasting=None), ["server", "ccproxy"])],
    ids=["server-gone", "unsaid-owner"],
)
def test_service_stop_stops_server_nobody_else_owns(monkeypatch, record, expected):
    """A record that does not say who owns its server (a server started by
    hand, an older release) is the service's to stop; without a server, the
    service stops its ccproxy."""
    import EvoScientist.ccproxy_manager as ccproxy_mod

    stopped = []
    _recorded_server(monkeypatch, record, stopped)
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    monkeypatch.setattr(
        ccproxy_mod, "stop_recorded_ccproxy", lambda: stopped.append("ccproxy") or 7
    )

    result = CliRunner().invoke(app, ["service", "stop"])

    assert result.exit_code == 0, result.output
    assert stopped == expected


def test_server_stop_stops_deploy_owned_server(monkeypatch):
    """EvoSci server stop is the explicit way to stop any recorded server."""
    stopped = []
    _recorded_server(monkeypatch, _record(lasting=False), stopped)

    result = CliRunner().invoke(app, ["server", "stop"])

    assert result.exit_code == 0, result.output
    assert stopped == ["server"]


def test_server_start_that_lost_race_says_it_reused(monkeypatch, tmp_path):
    """Another start took the port first: this one only registered its project."""
    import EvoScientist.ccproxy_manager as ccproxy_mod

    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    monkeypatch.setattr(
        manager, "ensure_server", lambda *_a, **_k: _result(started=False)
    )
    monkeypatch.setattr(
        ccproxy_mod, "maybe_start_ccproxy", lambda config, *, lasting: None
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


@pytest.mark.parametrize(("answer", "flagged"), [("refused", True), ("own", False)])
def test_server_status_names_server_configured_address_does_not_reach(
    monkeypatch, answer, flagged
):
    """langgraph_dev_host changed while the server ran: the configured
    address is asked whose server answers."""
    _status_server(monkeypatch, projects=[], record=_record(host="192.168.1.5"))
    monkeypatch.setattr(manager, "ask", lambda url: answer)

    output = _status_output()

    assert ("does not reach" in output) is flagged


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
    assert "server.lock" in output


def test_server_status_shows_starting_record_as_starting(monkeypatch):
    from EvoScientist.langgraph_dev import server_record

    _status_server(
        monkeypatch, projects=[], record=_record(phase=server_record.STARTING)
    )
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)

    output = _status_output()

    assert "starting" in output
    assert "not responding" not in output


@pytest.mark.parametrize("group", ["server", "service"])
def test_stop_names_lock_held_without_record(monkeypatch, group):
    import EvoScientist.ccproxy_manager as ccproxy_mod
    from EvoScientist.langgraph_dev import server_home

    stopped: list = []
    monkeypatch.setattr(manager, "read_record", lambda **_k: None)
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: False)
    monkeypatch.setattr(server_home, "lock_state", lambda wait=0: server_home.HELD)
    monkeypatch.setattr(
        ccproxy_mod, "stop_recorded_ccproxy", lambda: stopped.append("ccproxy")
    )

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


def test_service_start_starts_ccproxy_with_server(started, tmp_path, monkeypatch):
    import EvoScientist.ccproxy_manager as ccproxy_mod

    lasting = []
    monkeypatch.setattr(
        ccproxy_mod,
        "maybe_start_ccproxy",
        lambda config, **kw: lasting.append(kw.get("lasting")),
    )

    result = CliRunner().invoke(app, ["service", "start", "--workdir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert lasting == [True]
    assert started == [(tmp_path, "langgraph_server")]
    assert "EvoSci service stop" in " ".join(result.output.split())


def test_server_start_oauth_warns_without_ccproxy(started, tmp_path, monkeypatch):
    """Only the server: ccproxy is its own command."""
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
    monkeypatch.setattr(ccproxy_mod, "setup_ccproxy_env", lambda port: None)
    monkeypatch.setattr(ccproxy_mod, "is_ccproxy_running", lambda port: False)

    result = CliRunner().invoke(app, ["server", "start", "--workdir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "ccproxy is not running" in " ".join(result.output.split())


def test_ccproxy_start_without_oauth_does_nothing(monkeypatch):
    import EvoScientist.ccproxy_manager as ccproxy_mod

    started = []
    monkeypatch.setattr(
        ccproxy_mod, "maybe_start_ccproxy", lambda *a, **k: started.append(1)
    )

    result = CliRunner().invoke(app, ["ccproxy", "start"])

    assert result.exit_code == 0, result.output
    assert started == []
    assert "not needed" in " ".join(result.output.split())


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


def test_update_wait_spares_service_ccproxy(monkeypatch, tmp_path):
    """The update stops it itself, after the wait."""
    import os

    import EvoScientist.ccproxy_manager as ccproxy_mod
    from EvoScientist import updater

    monkeypatch.setattr(ccproxy_mod, "recorded_ccproxy_running", os.getpid)
    monkeypatch.setattr(manager, "read_record", lambda **_k: None)

    assert os.getpid() in updater._server_tree()


def test_lost_ccproxy_start_race_is_not_recorded(monkeypatch, tmp_path):
    """The one that lost the port exits; the winner records itself."""
    from types import SimpleNamespace

    import EvoScientist.ccproxy_manager as ccproxy_mod

    monkeypatch.setattr(paths, "SERVER_HOME", tmp_path)

    class _Lost:
        pid = 5

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(ccproxy_mod, "is_ccproxy_available", lambda: True)
    monkeypatch.setattr(ccproxy_mod, "check_ccproxy_auth", lambda _p: (True, ""))
    monkeypatch.setattr(
        ccproxy_mod, "ensure_ccproxy", lambda port, *, detached: _Lost()
    )
    monkeypatch.setattr(ccproxy_mod, "setup_ccproxy_env", lambda port: None)
    config = SimpleNamespace(anthropic_auth_mode="oauth", ccproxy_port=8000)

    ccproxy_mod.maybe_start_ccproxy(config, lasting=True)

    assert not (tmp_path / "ccproxy.pid").exists()


def test_interrupted_ccproxy_start_leaves_record(monkeypatch, tmp_path):
    """A Ctrl+C while the start checks the new ccproxy still leaves it
    recorded, so EvoSci ccproxy stop can stop it."""
    from types import SimpleNamespace

    import EvoScientist.ccproxy_manager as ccproxy_mod

    monkeypatch.setattr(paths, "SERVER_HOME", tmp_path)

    class _Started:
        pid = 5

        def wait(self, timeout=None):
            raise KeyboardInterrupt

    monkeypatch.setattr(ccproxy_mod, "is_ccproxy_available", lambda: True)
    monkeypatch.setattr(ccproxy_mod, "check_ccproxy_auth", lambda _p: (True, ""))
    monkeypatch.setattr(
        ccproxy_mod, "ensure_ccproxy", lambda port, *, detached: _Started()
    )
    monkeypatch.setattr(ccproxy_mod, "setup_ccproxy_env", lambda port: None)
    config = SimpleNamespace(anthropic_auth_mode="oauth", ccproxy_port=8000)

    with pytest.raises(KeyboardInterrupt):
        ccproxy_mod.maybe_start_ccproxy(config, lasting=True)

    assert ccproxy_mod.recorded_ccproxy_pid() == 5


def test_ccproxy_start_failure_keeps_install_hint(monkeypatch):
    """The hint's ``[oauth]`` extra is text, not markup."""
    import EvoScientist.ccproxy_manager as ccproxy_mod

    monkeypatch.setattr(ccproxy_mod, "uses_oauth", lambda _config: True)
    monkeypatch.setattr(ccproxy_mod, "is_ccproxy_running", lambda _port: False)

    def missing(config, *, lasting):
        raise RuntimeError("Install it with: pip install 'evoscientist[oauth]'")

    monkeypatch.setattr(ccproxy_mod, "maybe_start_ccproxy", missing)

    result = CliRunner().invoke(app, ["ccproxy", "start"])

    assert result.exit_code == 1
    assert "evoscientist[oauth]" in result.output


@pytest.fixture
def busy_server(monkeypatch):
    """A running server with a background job; returns the stops made."""
    from EvoScientist.deploy import busy

    stopped: list = []
    monkeypatch.setattr(manager, "read_record", lambda **_k: _record())
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: True)
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


def test_service_stop_asks_nothing_of_owned_busy_server(busy_server, monkeypatch):
    """EvoSci deploy or the desktop app owns it: service stop leaves it
    running, so its work needs no question, nor --yes in a script."""

    def stop(*, keep_owned=False, record=None):
        raise manager.ServerKeptError("left running")

    monkeypatch.setattr(manager, "owned_by_its_starter", lambda _record: True)
    monkeypatch.setattr(manager, "stop_server", stop)

    result = CliRunner().invoke(app, ["service", "stop"])

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


def test_server_stop_asks_about_owned_busy_server(busy_server, monkeypatch):
    """EvoSci server stop ends a server EvoSci deploy or the desktop app owns
    too, so its work is asked about; without a terminal it is refused."""
    monkeypatch.setattr(manager, "owned_by_its_starter", lambda _record: True)

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


@pytest.mark.parametrize(
    ("answer", "said", "unsaid"),
    [
        ("silent", "does not answer", "does not reach"),
        ("older", "older EvoScientist release", "does not reach"),
        ("other", "does not reach", "does not answer"),
    ],
)
def test_server_status_says_how_configured_address_answers(
    monkeypatch, answer, said, unsaid
):
    _status_server(monkeypatch, projects=[], record=_record(host="192.168.1.5"))
    monkeypatch.setattr(manager, "ask", lambda url: answer)

    output = _status_output()

    assert said in output
    assert unsaid not in output


def test_server_status_refused_at_records_own_address_is_no_mismatch(
    monkeypatch,
):
    _status_server(monkeypatch, projects=[], record=_record())
    monkeypatch.setattr(manager, "ask", lambda url: "refused")

    assert "does not reach" not in _status_output()


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
    assert "server.lock" in output


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
