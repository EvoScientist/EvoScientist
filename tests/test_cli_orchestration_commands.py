from pathlib import Path

from typer.testing import CliRunner

from EvoScientist.cli._app import app

runner = CliRunner()


class _TestConfig:
    provider = "anthropic"
    model = "claude-sonnet-4-5"


def _bootstrap_run(monkeypatch, tmp_path):
    import json

    config_dir = tmp_path / "evoscientist"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.yaml").write_text("provider: anthropic\nmodel: claude-sonnet-4-5\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    result = runner.invoke(
        app,
        ["-p", "test prompt", "--json", "--output", str(tmp_path / "artifacts")],
    )
    assert result.exit_code == 0
    return json.loads(result.stdout)


def _install_fake_stream(monkeypatch):
    from EvoScientist import sessions
    from EvoScientist.cli import commands
    from EvoScientist.stream import events

    class _DummyCheckpointer:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_event_stream(agent, message, thread_id, metadata=None, media=None):
        yield {"type": "text", "content": "hello"}
        yield {"type": "done"}

    monkeypatch.setattr(commands, "_load_agent", lambda **kwargs: object())
    monkeypatch.setattr(sessions, "get_checkpointer", lambda: _DummyCheckpointer())
    monkeypatch.setattr(events, "stream_agent_events", fake_event_stream)


def test_doctor_command_returns_json(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("provider: anthropic\nmodel: claude-sonnet-4-5\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code != 0
    assert '"ok": false' in result.stdout.lower()


def test_validate_command_returns_json_for_missing_config(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    result = runner.invoke(app, ["validate", "--json"])

    assert result.exit_code != 0
    assert '"ok": false' in result.stdout.lower()


def test_validate_command_returns_json_for_valid_config(monkeypatch, tmp_path):
    config_dir = tmp_path / "evoscientist"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text("provider: anthropic\nmodel: claude-sonnet-4-5\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    result = runner.invoke(app, ["validate", "--json"])

    assert result.exit_code == 0
    assert '"ok": true' in result.stdout.lower()


def test_run_prompt_json_orchestration_returns_structured_payload(monkeypatch, tmp_path):
    from EvoScientist.cli import commands

    _install_fake_stream(monkeypatch)

    payload = commands._run_prompt_json_orchestration(
        prompt="test prompt",
        workspace_dir=str(tmp_path),
        config=_TestConfig(),
        output=str(tmp_path / "artifacts"),
    )

    assert payload["run_id"].startswith("es-")
    assert payload["status"] == "completed"
    assert payload["workspace_dir"] == str(tmp_path)
    assert payload["artifact_dir"].startswith(str(tmp_path / "artifacts"))



def test_run_command_does_not_emit_ambiguous_ok_field(monkeypatch, tmp_path):
    _install_fake_stream(monkeypatch)
    payload = _bootstrap_run(monkeypatch, tmp_path)

    assert "ok" not in payload




def test_run_command_updates_run_json_to_final_status(monkeypatch, tmp_path):
    import json

    _install_fake_stream(monkeypatch)
    payload = _bootstrap_run(monkeypatch, tmp_path)
    artifact_dir = Path(payload["artifact_dir"])

    run_payload = json.loads((artifact_dir / "run.json").read_text())

    assert run_payload["status"] == "completed"


def test_status_command_returns_json_snapshot_for_run(monkeypatch, tmp_path):
    _install_fake_stream(monkeypatch)
    payload = _bootstrap_run(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        ["status", payload["run_id"], "--json", "--output", str(tmp_path / "artifacts")],
    )

    assert result.exit_code == 0
    assert '"run_id":' in result.stdout
    assert '"status": "completed"' in result.stdout


import pytest


@pytest.mark.parametrize("command", ["status", "artifacts", "resume"])
def test_orchestration_commands_emit_json_error_payload_for_missing_run(tmp_path, command):
    import json

    result = runner.invoke(
        app,
        [command, "missing-run", "--json", "--output", str(tmp_path / "artifacts")],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error"] == "run not found"
    assert payload["run_id"] == "missing-run"
    assert payload["run_dir"].endswith("artifacts/missing-run")



def test_artifacts_command_returns_expected_paths(monkeypatch, tmp_path):
    _install_fake_stream(monkeypatch)
    payload = _bootstrap_run(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        ["artifacts", payload["run_id"], "--json", "--output", str(tmp_path / "artifacts")],
    )

    assert result.exit_code == 0
    assert '"artifact_dir":' in result.stdout
    assert '"run_json":' in result.stdout
    assert '"status_json":' in result.stdout
    assert '"events_jsonl":' in result.stdout


def test_report_command_summarizes_run(monkeypatch, tmp_path):
    _install_fake_stream(monkeypatch)
    payload = _bootstrap_run(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        ["report", payload["run_id"], "--output", str(tmp_path / "artifacts")],
    )

    assert result.exit_code == 0
    assert payload["run_id"] in result.stdout
    assert "artifact" in result.stdout.lower()



def test_run_prompt_json_orchestration_marks_failed_when_startup_raises_before_first_event(
    monkeypatch, tmp_path
):
    import json

    from EvoScientist import sessions
    from EvoScientist.cli import commands

    class _DummyCheckpointer:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(sessions, "get_checkpointer", lambda: _DummyCheckpointer())
    monkeypatch.setattr(commands, "_load_agent", lambda **kwargs: object())

    def _boom(*args, **kwargs):
        raise RuntimeError("startup boom")

    monkeypatch.setattr(commands, "build_metadata", _boom)

    payload = commands._run_prompt_json_orchestration(
        prompt="test prompt",
        workspace_dir=str(tmp_path),
        config=_TestConfig(),
        output=str(tmp_path / "artifacts"),
    )

    assert payload["status"] == "failed"

    artifact_root = tmp_path / "artifacts"
    run_dirs = [path for path in artifact_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1

    status_payload = json.loads((run_dirs[0] / "status.json").read_text())
    run_payload = json.loads((run_dirs[0] / "run.json").read_text())
    failure_summary = json.loads(
        (run_dirs[0] / "diagnostics" / "failure_summary.json").read_text()
    )
    assert status_payload["status"] == "failed"
    assert status_payload["last_error"] == "startup boom"
    assert run_payload["status"] == "failed"
    assert run_payload["last_error"] == "startup boom"
    assert failure_summary["status"] == "failed"
    assert failure_summary["last_error"] == "startup boom"



def test_run_command_marks_failed_when_startup_raises_before_first_event(monkeypatch, tmp_path):
    import json

    from EvoScientist import sessions
    from EvoScientist.cli import commands

    class _DummyCheckpointer:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    config_dir = tmp_path / "evoscientist"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.yaml").write_text("provider: anthropic\nmodel: claude-sonnet-4-5\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    monkeypatch.setattr(sessions, "get_checkpointer", lambda: _DummyCheckpointer())
    monkeypatch.setattr(commands, "_load_agent", lambda **kwargs: object())

    def _boom(*args, **kwargs):
        raise RuntimeError("startup boom")

    monkeypatch.setattr(commands, "build_metadata", _boom)

    result = runner.invoke(
        app,
        ["-p", "test prompt", "--json", "--output", str(tmp_path / "artifacts")],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"

    artifact_root = tmp_path / "artifacts"
    run_dirs = [path for path in artifact_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1

    status_payload = json.loads((run_dirs[0] / "status.json").read_text())
    run_payload = json.loads((run_dirs[0] / "run.json").read_text())
    failure_summary = json.loads(
        (run_dirs[0] / "diagnostics" / "failure_summary.json").read_text()
    )
    assert status_payload["status"] == "failed"
    assert status_payload["last_error"] == "startup boom"
    assert run_payload["status"] == "failed"
    assert run_payload["last_error"] == "startup boom"
    assert failure_summary["status"] == "failed"
    assert failure_summary["last_error"] == "startup boom"
