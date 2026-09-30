"""Tests for the ``EvoSci setup`` event protocol, stage runner and command."""

from __future__ import annotations

import io
import json

import pytest

import EvoScientist.setup as setup_pkg
from EvoScientist.setup import Stage, StageResult, manifest, run_stages
from EvoScientist.setup.protocol import (
    PROTOCOL,
    ConsoleEmitter,
    JsonEmitter,
    StageError,
    make_event,
)

_ALLOWED_FIELDS = {
    "protocol",
    "stage",
    "status",
    "progress",
    "message",
    "detail",
    "code",
}


def _collect():
    events: list[dict] = []
    return events, events.append


def test_manifest_shape():
    assert manifest() == {
        "protocol": 1,
        "stages": [
            {"id": "node", "title": "Node.js"},
            {"id": "research-env", "title": "Python research environment"},
        ],
    }


def test_make_event_leaves_out_unset_fields_and_clamps_progress():
    assert make_event("node", "running") == {
        "protocol": PROTOCOL,
        "stage": "node",
        "status": "running",
    }
    assert make_event("node", "running", progress=1.7)["progress"] == 1.0


def test_stage_error_rejects_unknown_code():
    with pytest.raises(ValueError, match="unknown setup error code"):
        StageError("oops", "x")


def test_run_stages_done_carries_detail():
    stage = Stage(
        "node",
        "Node.js",
        None,
        lambda emit, mirror: StageResult("ok", {"source": "system"}),
    )
    events, emit = _collect()
    assert run_stages([stage], emit, "default") == 0
    assert events == [
        {
            "protocol": 1,
            "stage": "node",
            "status": "done",
            "message": "ok",
            "detail": {"source": "system"},
        }
    ]


def test_run_stages_skipped_result_is_the_only_terminal_event():
    def run(emit, mirror):
        emit(make_event("env", "running", message="Checking"))
        return StageResult("Using x", {"reason": "system_python"}, status="skipped")

    later = Stage("later", "Later", None, lambda emit, mirror: StageResult("ok", {}))
    events, emit = _collect()
    assert run_stages([Stage("env", "Env", None, run), later], emit, "default") == 0
    assert [(e["stage"], e["status"]) for e in events] == [
        ("env", "running"),
        ("env", "skipped"),
        ("later", "done"),
    ]
    assert events[1]["detail"] == {"reason": "system_python"}


def test_run_stages_error_carries_code_and_stops():
    def fail(emit, mirror):
        raise StageError("checksum_mismatch", "bad sum")

    later = Stage("later", "Later", None, lambda emit, mirror: StageResult("ok", {}))
    events, emit = _collect()
    assert (
        run_stages([Stage("node", "Node.js", None, fail), later], emit, "default") == 1
    )
    assert events == [
        {
            "protocol": 1,
            "stage": "node",
            "status": "error",
            "message": "bad sum",
            "code": "checksum_mismatch",
        }
    ]


def test_run_stages_skips_other_platforms():
    stage = Stage(
        "git", "Git", frozenset({"no-such-platform"}), lambda e, m: StageResult("", {})
    )
    events, emit = _collect()
    assert run_stages([stage], emit, "default") == 0
    assert events[0]["status"] == "skipped"


def test_run_stages_reports_an_unexpected_exception_as_an_error_event():
    def crash(emit, mirror):
        raise RuntimeError("boom")

    events, emit = _collect()
    assert run_stages([Stage("node", "Node.js", None, crash)], emit, "default") == 1
    assert [(e["status"], e["code"]) for e in events] == [("error", "install_failed")]
    assert "boom" in events[0]["message"]


def test_manifest_leaves_out_stages_for_other_platforms(monkeypatch):
    other = Stage(
        "git", "Git", frozenset({"no-such-platform"}), lambda e, m: StageResult("", {})
    )
    monkeypatch.setattr(setup_pkg, "STAGES", (*setup_pkg.STAGES, other))
    assert [s["id"] for s in manifest()["stages"]] == ["node", "research-env"]


def test_json_emitter_writes_one_line_per_event(tmp_path):
    out = (tmp_path / "out.txt").open("w+", encoding="utf-8")
    emit = JsonEmitter(out)
    emit(make_event("node", "running", progress=0.4, message="Downloading Node 24.x"))
    emit(make_event("node", "done", detail={"source": "system", "version": "22.11.0"}))
    out.seek(0)
    lines = [json.loads(line) for line in out.read().splitlines()]
    assert [e["status"] for e in lines] == ["running", "done"]
    assert all(set(e) <= _ALLOWED_FIELDS for e in lines)


def test_console_emitter_throttles_progress():
    printed: list[str] = []

    class _Console:
        def print(self, text):
            printed.append(text)

    emit = ConsoleEmitter(_Console())
    for pct in (0.0, 0.02, 0.05, 0.12, 0.13, 1.0):
        emit(make_event("node", "running", progress=pct, message="dl"))
    assert len(printed) == 3  # 0%, 12%, 100%


def test_json_emitter_survives_a_non_utf8_pipe():
    """A piped stdout on Windows uses the ANSI code page (e.g. cp1252)."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    JsonEmitter(stream)(
        make_event("node", "done", message=r"C:\Users\Jan Kowalski ąę", detail={})
    )
    line = raw.getvalue().decode("ascii")
    assert json.loads(line)["message"] == r"C:\Users\Jan Kowalski ąę"


def test_console_emitter_prints_markup_characters_literally():
    """Paths and exception text reach the console as written: `[/x]` must not
    raise MarkupError and a Windows `\\[` must keep its backslash."""
    from rich.console import Console

    out = io.StringIO()
    emit = ConsoleEmitter(Console(file=out, width=300, color_system=None))
    message = r"Could not move Node into C:\Users\[work]\tools: a[/b]"
    emit(make_event("node", "error", message=message, code="install_failed"))
    assert message in out.getvalue()


def test_console_emitter_shows_every_step_message():
    printed: list[str] = []

    class _Console:
        def print(self, text):
            printed.append(text)

    emit = ConsoleEmitter(_Console())
    for pct, message in (
        (0.85, "Downloading"),
        (0.86, "Verifying checksum"),
        (0.9, "Unpacking"),
        (0.96, "Checking the installed Node"),
    ):
        emit(make_event("node", "running", progress=pct, message=message))
    assert [p.split(": ")[1].split(" (")[0] for p in printed] == [
        "Downloading",
        "Verifying checksum",
        "Unpacking",
        "Checking the installed Node",
    ]


def test_main_activates_the_private_node(tmp_path, monkeypatch):
    """cli.main() must put the private Node on PATH before the app runs."""
    import EvoScientist.cli as cli_pkg
    from EvoScientist.cli import commands
    from EvoScientist.setup import node as setup_node

    order: list[str] = []
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(
        setup_node, "activate_runtime", lambda: order.append("activate")
    )
    monkeypatch.setattr(commands, "_configure_logging", lambda: None)
    monkeypatch.setattr(cli_pkg, "app", lambda: order.append("app"))
    cli_pkg.main()
    assert order == ["activate", "app"]


# --------------------------------------------------------------------------- #
# Command
# --------------------------------------------------------------------------- #
@pytest.fixture
def cli(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from EvoScientist.cli._app import app
    from EvoScientist.stream.console import console

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    # --json redirects the shared console for the rest of the process; undo it.
    monkeypatch.setattr(console, "_file", None)
    runner = CliRunner()
    return lambda *argv: runner.invoke(app, ["setup", *argv])


def _fake_stage(monkeypatch, run):
    monkeypatch.setattr(setup_pkg, "STAGES", (Stage("node", "Node.js", None, run),))


def test_cli_manifest(cli):
    result = cli("--manifest")
    assert result.exit_code == 0
    assert json.loads(result.stdout) == manifest()


def test_cli_json_stdout_holds_only_json_lines(cli, monkeypatch):
    from EvoScientist.stream.console import console

    def run(emit, mirror):
        console.print("human noise that must not reach stdout")
        emit(make_event("node", "running", progress=0.5, message="half"))
        return StageResult("ok", {"source": "private", "version": "24.21.0"})

    _fake_stage(monkeypatch, run)
    result = cli("--stage", "node", "--json")
    assert result.exit_code == 0
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert [e["status"] for e in events] == ["running", "done"]
    assert "human noise" in result.stderr


def test_cli_error_exits_non_zero(cli, monkeypatch):
    def run(emit, mirror):
        raise StageError("download_failed", "offline")

    _fake_stage(monkeypatch, run)
    result = cli("--json")
    assert result.exit_code == 1
    assert json.loads(result.stdout.splitlines()[-1])["code"] == "download_failed"


def test_cli_unknown_stage_exits_non_zero(cli):
    result = cli("--stage", "nope", "--json")
    assert result.exit_code == 2
    assert result.stdout == ""


def test_cli_cn_saves_mirror_and_passes_it(cli, monkeypatch):
    from EvoScientist.config import get_config_value

    seen: list[str] = []
    _fake_stage(
        monkeypatch,
        lambda emit, mirror: (seen.append(mirror), StageResult("ok", {}))[1],
    )
    assert cli("--cn").exit_code == 0
    assert seen == ["cn"]
    assert get_config_value("mirror") == "cn"
    # Later runs without --cn keep the saved mirror.
    assert cli().exit_code == 0
    assert seen == ["cn", "cn"]


def test_cli_cn_with_an_unwritable_config_still_runs_with_the_mirror(
    cli, monkeypatch, caplog
):
    import EvoScientist.config as config_pkg

    def read_only(key, value):
        raise PermissionError(13, "Read-only file system")

    monkeypatch.setattr(config_pkg, "set_config_value", read_only)
    seen: list[str] = []
    _fake_stage(monkeypatch, lambda emit, mirror: (seen.append(mirror), ("ok", {}))[1])
    with caplog.at_level("WARNING"):
        result = cli("--cn", "--json")
    assert result.exit_code == 0
    assert seen == ["cn"]
    assert [json.loads(line)["status"] for line in result.stdout.splitlines()] == [
        "done"
    ]
    assert "Could not save mirror" in caplog.text


def test_cli_json_keeps_config_warnings_off_stdout(cli, monkeypatch):
    import EvoScientist.config as config_pkg
    from EvoScientist.stream.console import console

    real_load = config_pkg.load_config

    def load_with_warning():
        console.print("config warning that must not reach stdout")
        return real_load()

    monkeypatch.setattr(config_pkg, "load_config", load_with_warning)
    _fake_stage(monkeypatch, lambda emit, mirror: StageResult("ok", {}))
    result = cli("--json")
    assert [json.loads(line)["status"] for line in result.stdout.splitlines()] == [
        "done"
    ]


def test_cli_full_run_leaves_out_stages_for_other_platforms(cli, monkeypatch):
    stages = (
        Stage(
            "git",
            "Git",
            frozenset({"no-such-platform"}),
            lambda e, m: StageResult("", {}),
        ),
        Stage("node", "Node.js", None, lambda e, m: StageResult("ok", {})),
    )
    monkeypatch.setattr(setup_pkg, "STAGES", stages)
    result = cli("--json")
    assert [json.loads(line)["stage"] for line in result.stdout.splitlines()] == [
        "node"
    ]
    skipped = cli("--stage", "git", "--json")
    assert skipped.exit_code == 0
    assert json.loads(skipped.stdout)["status"] == "skipped"
