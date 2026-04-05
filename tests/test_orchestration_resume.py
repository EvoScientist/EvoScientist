import json

from EvoScientist.orchestration.resume import build_resume_payload


def test_resume_command_restarts_from_stored_run_metadata(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "es-20260401-abc123"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "es-20260401-abc123",
                "thread_id": "deadbeef",
                "workspace_dir": "/tmp/work",
                "artifact_dir": str(artifact_dir),
                "status": "created",
                "current_stage": None,
                "completed_stages": [],
                "last_error": None,
                "metadata": {"prompt": "test prompt", "model": "claude-sonnet-4-5"},
                "created_at": "2026-04-01T00:00:00Z",
                "updated_at": "2026-04-01T00:00:00Z",
            }
        )
        + "\n"
    )

    payload = build_resume_payload(artifact_dir)

    assert payload["run_id"] == "es-20260401-abc123"
    assert payload["thread_id"] == "deadbeef"
    assert payload["workspace_dir"] == "/tmp/work"
    assert payload["artifact_dir"] == str(artifact_dir)
    assert payload["prompt"] == "test prompt"
    assert payload["model"] == "claude-sonnet-4-5"
    assert payload["resume_semantics"] == "restart_from_saved_run_context"
