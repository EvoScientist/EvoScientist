from EvoScientist.orchestration.models import RunRecord, RunStatus
from EvoScientist.orchestration.status import build_status_snapshot


def test_build_status_snapshot_includes_core_fields():
    record = RunRecord(
        run_id="es-20260401-abc123",
        thread_id="deadbeef",
        workspace_dir="/tmp/work",
        artifact_dir="/tmp/artifacts/run1",
        status=RunStatus.RUNNING,
        current_stage="planning",
        completed_stages=["bootstrap"],
    )

    snapshot = build_status_snapshot(record)

    assert snapshot["run_id"] == "es-20260401-abc123"
    assert snapshot["thread_id"] == "deadbeef"
    assert snapshot["workspace_dir"] == "/tmp/work"
    assert snapshot["artifact_dir"] == "/tmp/artifacts/run1"
    assert snapshot["status"] == "running"
    assert snapshot["current_stage"] == "planning"
    assert snapshot["completed_stages"] == ["bootstrap"]
    assert snapshot["last_error"] is None
    assert snapshot["suggested_next_action"] is None
    assert "updated_at" in snapshot


def test_build_status_snapshot_recommends_next_action_for_failed_run():
    record = RunRecord(
        run_id="es-20260401-def456",
        thread_id="feedbeef",
        workspace_dir="/tmp/work",
        artifact_dir="/tmp/artifacts/run2",
        status=RunStatus.FAILED,
        last_error="provider auth missing",
    )

    snapshot = build_status_snapshot(record)

    assert snapshot["status"] == "failed"
    assert snapshot["last_error"] == "provider auth missing"
    assert snapshot["suggested_next_action"] is not None
