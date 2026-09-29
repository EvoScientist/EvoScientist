"""Tests for the ``EvoSci setup`` event protocol, stage runner and command."""

from __future__ import annotations

import json

import pytest

import EvoScientist.setup as setup_pkg
from EvoScientist.setup import Stage, manifest, run_stages
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
    assert manifest() == {"protocol": 1, "stages": [{"id": "node", "title": "Node.js"}]}


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
        "node", "Node.js", None, lambda emit, mirror: ("ok", {"source": "system"})
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


def test_run_stages_error_carries_code_and_stops():
    def fail(emit, mirror):
        raise StageError("checksum_mismatch", "bad sum")

    later = Stage("later", "Later", None, lambda emit, mirror: ("ok", {}))
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
    stage = Stage("git", "Git", frozenset({"no-such-platform"}), lambda e, m: ("", {}))
    events, emit = _collect()
    assert run_stages([stage], emit, "default") == 0
    assert events[0]["status"] == "skipped"


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
        return "ok", {"source": "private", "version": "24.21.0"}

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
    _fake_stage(monkeypatch, lambda emit, mirror: (seen.append(mirror), ("ok", {}))[1])
    assert cli("--cn").exit_code == 0
    assert seen == ["cn"]
    assert get_config_value("mirror") == "cn"
    # Later runs without --cn keep the saved mirror.
    assert cli().exit_code == 0
    assert seen == ["cn", "cn"]
