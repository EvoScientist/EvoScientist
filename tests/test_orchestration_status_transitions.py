from EvoScientist.orchestration.models import RunRecord, RunStatus
from EvoScientist.orchestration.status import apply_event_to_run_record


def test_apply_event_sets_running_status_for_started_work():
    record = RunRecord(
        run_id="es-20260401-abc123",
        thread_id="deadbeef",
        workspace_dir="/tmp/work",
        artifact_dir="/tmp/artifacts/run1",
    )

    updated = apply_event_to_run_record(record, {"type": "text", "content": "hello"})

    assert updated.status == RunStatus.RUNNING


def test_apply_event_sets_blocked_status_for_interrupt():
    record = RunRecord(
        run_id="es-20260401-abc123",
        thread_id="deadbeef",
        workspace_dir="/tmp/work",
        artifact_dir="/tmp/artifacts/run1",
    )

    updated = apply_event_to_run_record(record, {"type": "interrupt"})

    assert updated.status == RunStatus.BLOCKED


def test_apply_event_sets_failed_status_and_last_error():
    record = RunRecord(
        run_id="es-20260401-abc123",
        thread_id="deadbeef",
        workspace_dir="/tmp/work",
        artifact_dir="/tmp/artifacts/run1",
    )

    updated = apply_event_to_run_record(record, {"type": "error", "message": "boom"})

    assert updated.status == RunStatus.FAILED
    assert updated.last_error == "boom"


def test_apply_event_sets_completed_status_for_done():
    record = RunRecord(
        run_id="es-20260401-abc123",
        thread_id="deadbeef",
        workspace_dir="/tmp/work",
        artifact_dir="/tmp/artifacts/run1",
    )

    updated = apply_event_to_run_record(record, {"type": "done"})

    assert updated.status == RunStatus.COMPLETED
