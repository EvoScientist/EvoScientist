import json

import pytest

from EvoScientist.orchestration.models import RunRecord, RunStatus, StatusSnapshot
from EvoScientist.orchestration.run_store import (
    append_run_event,
    apply_event_and_update_status,
    consume_event_stream,
    create_run_artifact_tree,
    load_status_snapshot,
    replay_events_into_status,
    write_run_record,
    write_status_snapshot,
)


def test_create_run_artifact_tree_rejects_existing_run_dir(tmp_path):
    create_run_artifact_tree(tmp_path, "es-20260401-abc123")

    with pytest.raises(FileExistsError):
        create_run_artifact_tree(tmp_path, "es-20260401-abc123")


def test_write_run_record_persists_json(tmp_path):
    run_dir = create_run_artifact_tree(tmp_path, "es-20260401-abc123")
    record = RunRecord(
        run_id="es-20260401-abc123",
        thread_id="deadbeef",
        workspace_dir="/tmp/work",
        artifact_dir=str(run_dir),
        status=RunStatus.CREATED,
    )

    write_run_record(run_dir, record)

    payload = json.loads((run_dir / "run.json").read_text())
    assert payload["run_id"] == "es-20260401-abc123"
    assert payload["thread_id"] == "deadbeef"
    assert payload["status"] == "created"


def test_write_and_load_status_snapshot_round_trip(tmp_path):
    run_dir = create_run_artifact_tree(tmp_path, "es-20260401-abc123")
    snapshot = StatusSnapshot(
        run_id="es-20260401-abc123",
        status=RunStatus.RUNNING,
        thread_id="deadbeef",
        workspace_dir="/tmp/work",
        artifact_dir=str(run_dir),
        current_stage="planning",
        updated_at="2026-04-01T00:00:00Z",
    )

    write_status_snapshot(run_dir, snapshot)
    loaded = load_status_snapshot(run_dir)

    assert loaded is not None
    assert loaded.run_id == "es-20260401-abc123"
    assert loaded.status == RunStatus.RUNNING
    assert loaded.current_stage == "planning"


def test_append_run_event_appends_json_lines(tmp_path):
    run_dir = create_run_artifact_tree(tmp_path, "es-20260401-abc123")

    append_run_event(run_dir, {"type": "text", "content": "hello"})
    append_run_event(run_dir, {"type": "done", "content": "ok"})

    lines = (run_dir / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["type"] == "text"
    assert second["type"] == "done"


def test_append_run_event_preserves_timestamp_field(tmp_path):
    run_dir = create_run_artifact_tree(tmp_path, "es-20260401-abc123")
    append_run_event(
        run_dir,
        {
            "type": "created",
            "run_id": "es-20260401-abc123",
            "timestamp": "2026-04-01T00:00:00Z",
        },
    )

    payload = json.loads((run_dir / "events.jsonl").read_text().strip())
    assert payload["type"] == "created"
    assert payload["timestamp"] == "2026-04-01T00:00:00Z"




def test_apply_event_and_update_status_writes_failure_summary(tmp_path):
    run_dir = create_run_artifact_tree(tmp_path, "es-20260401-abc123")
    record = RunRecord(
        run_id="es-20260401-abc123",
        thread_id="deadbeef",
        workspace_dir="/tmp/work",
        artifact_dir=str(run_dir),
        status=RunStatus.CREATED,
    )
    write_run_record(run_dir, record)
    write_status_snapshot(
        run_dir,
        StatusSnapshot(
            run_id=record.run_id,
            status=record.status,
            thread_id=record.thread_id,
            workspace_dir=record.workspace_dir,
            artifact_dir=record.artifact_dir,
            updated_at=record.updated_at,
        ),
    )

    apply_event_and_update_status(run_dir, record, {"type": "error", "message": "boom"})

    summary_path = run_dir / "diagnostics" / "failure_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["run_id"] == "es-20260401-abc123"
    assert payload["status"] == "failed"
    assert payload["last_error"] == "boom"


def test_replay_events_into_status_reaches_completed_state(tmp_path):
    run_dir = create_run_artifact_tree(tmp_path, "es-20260401-abc123")
    record = RunRecord(
        run_id="es-20260401-abc123",
        thread_id="deadbeef",
        workspace_dir="/tmp/work",
        artifact_dir=str(run_dir),
        status=RunStatus.CREATED,
    )
    write_run_record(run_dir, record)
    write_status_snapshot(
        run_dir,
        StatusSnapshot(
            run_id=record.run_id,
            status=record.status,
            thread_id=record.thread_id,
            workspace_dir=record.workspace_dir,
            artifact_dir=record.artifact_dir,
            updated_at=record.updated_at,
        ),
    )

    final_record = replay_events_into_status(
        run_dir,
        record,
        [{"type": "text", "content": "hello"}, {"type": "done"}],
    )
    updated = load_status_snapshot(run_dir)

    assert final_record.status == RunStatus.COMPLETED
    assert updated is not None
    assert updated.status == RunStatus.COMPLETED




@pytest.mark.anyio
async def test_consume_event_stream_marks_failed_when_stream_raises(tmp_path):
    run_dir = create_run_artifact_tree(tmp_path, "es-20260401-abc123")
    record = RunRecord(
        run_id="es-20260401-abc123",
        thread_id="deadbeef",
        workspace_dir="/tmp/work",
        artifact_dir=str(run_dir),
        status=RunStatus.CREATED,
    )
    write_run_record(run_dir, record)
    write_status_snapshot(
        run_dir,
        StatusSnapshot(
            run_id=record.run_id,
            status=record.status,
            thread_id=record.thread_id,
            workspace_dir=record.workspace_dir,
            artifact_dir=record.artifact_dir,
            updated_at=record.updated_at,
        ),
    )

    async def event_source():
        yield {"type": "text", "content": "hello"}
        raise RuntimeError("stream boom")

    final_record = await consume_event_stream(run_dir, record, event_source())
    updated = load_status_snapshot(run_dir)
    events = (run_dir / "events.jsonl").read_text().strip().splitlines()

    assert final_record.status == RunStatus.FAILED
    assert final_record.last_error == "stream boom"
    assert updated is not None
    assert updated.status == RunStatus.FAILED
    assert len(events) == 2
    assert json.loads(events[-1])["type"] == "error"
