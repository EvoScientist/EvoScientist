from EvoScientist.orchestration.models import (
    ArtifactIndex,
    DiagnosticReport,
    RunRecord,
    RunStatus,
    StatusSnapshot,
)


def test_run_record_defaults_are_stable():
    record = RunRecord(
        run_id="es-20260401-abc123",
        thread_id="deadbeef",
        workspace_dir="/tmp/work",
        artifact_dir="/tmp/artifacts/run1",
    )
    assert record.status == RunStatus.CREATED
    assert record.current_stage is None
    assert record.last_error is None
    assert record.completed_stages == []
    assert record.metadata == {}


def test_status_snapshot_defaults_are_stable():
    snapshot = StatusSnapshot(
        run_id="es-20260401-abc123",
        status=RunStatus.RUNNING,
        thread_id="deadbeef",
        workspace_dir="/tmp/work",
        artifact_dir="/tmp/artifacts/run1",
        updated_at="2026-04-01T00:00:00Z",
    )
    assert snapshot.current_stage is None
    assert snapshot.completed_stages == []
    assert snapshot.last_error is None
    assert snapshot.suggested_next_action is None


def test_supporting_models_capture_expected_fields():
    artifacts = ArtifactIndex(
        artifact_dir="/tmp/artifacts/run1",
        run_json="/tmp/artifacts/run1/run.json",
        status_json="/tmp/artifacts/run1/status.json",
        events_jsonl="/tmp/artifacts/run1/events.jsonl",
        outputs_dir="/tmp/artifacts/run1/outputs",
        deliverables_dir="/tmp/artifacts/run1/deliverables",
        diagnostics_dir="/tmp/artifacts/run1/diagnostics",
    )
    report = DiagnosticReport(
        ok=False,
        summary="Missing API key",
        checks=[{"name": "api_key", "ok": False}],
    )

    assert artifacts.outputs_dir.endswith("/outputs")
    assert artifacts.deliverables_dir.endswith("/deliverables")
    assert report.ok is False
    assert report.checks[0]["name"] == "api_key"
