from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import ArtifactIndex, RunRecord, RunStatus, StatusSnapshot
from .status import apply_event_to_run_record, build_status_snapshot


def create_run_artifact_tree(base_dir: str | Path, run_id: str) -> Path:
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "outputs").mkdir(exist_ok=True)
    (run_dir / "deliverables").mkdir(exist_ok=True)
    (run_dir / "diagnostics").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "run.json").write_text("{}\n")
    (run_dir / "status.json").write_text("{}\n")
    (run_dir / "events.jsonl").write_text("")
    return run_dir


def artifact_index_for(run_dir: str | Path) -> ArtifactIndex:
    run_dir = Path(run_dir)
    return ArtifactIndex(
        artifact_dir=str(run_dir),
        run_json=str(run_dir / "run.json"),
        status_json=str(run_dir / "status.json"),
        events_jsonl=str(run_dir / "events.jsonl"),
        outputs_dir=str(run_dir / "outputs"),
        deliverables_dir=str(run_dir / "deliverables"),
        diagnostics_dir=str(run_dir / "diagnostics"),
    )


def resolve_run_dir(base_dir: str | Path, run_id: str) -> Path:
    return Path(base_dir) / run_id


def write_run_record(run_dir: str | Path, record: RunRecord) -> None:
    payload = asdict(record)
    payload["status"] = record.status.value
    Path(run_dir, "run.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_status_snapshot(run_dir: str | Path, snapshot: StatusSnapshot) -> None:
    payload = asdict(snapshot)
    payload["status"] = snapshot.status.value
    Path(run_dir, "status.json").write_text(json.dumps(payload, indent=2) + "\n")


def load_status_snapshot(run_dir: str | Path) -> StatusSnapshot | None:
    status_file = Path(run_dir) / "status.json"
    if not status_file.exists():
        return None
    raw = status_file.read_text().strip()
    if not raw or raw == "{}":
        return None
    payload = json.loads(raw)
    status = RunStatus(payload.get("status", "created"))
    return StatusSnapshot(
        run_id=payload["run_id"],
        status=status,
        thread_id=payload["thread_id"],
        workspace_dir=payload["workspace_dir"],
        artifact_dir=payload["artifact_dir"],
        updated_at=payload["updated_at"],
        current_stage=payload.get("current_stage"),
        completed_stages=payload.get("completed_stages", []),
        last_error=payload.get("last_error"),
        suggested_next_action=payload.get("suggested_next_action"),
    )


def append_run_event(run_dir: str | Path, event: dict[str, Any]) -> None:
    event_file = Path(run_dir) / "events.jsonl"
    with event_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def write_failure_summary(run_dir: str | Path, snapshot: StatusSnapshot) -> None:
    if snapshot.status != RunStatus.FAILED:
        return
    payload = {
        "run_id": snapshot.run_id,
        "thread_id": snapshot.thread_id,
        "status": snapshot.status.value,
        "last_error": snapshot.last_error,
        "updated_at": snapshot.updated_at,
        "workspace_dir": snapshot.workspace_dir,
        "artifact_dir": snapshot.artifact_dir,
        "suggested_next_action": snapshot.suggested_next_action,
    }
    summary_file = Path(run_dir) / "diagnostics" / "failure_summary.json"
    summary_file.write_text(json.dumps(payload, indent=2) + "\n")


def apply_event_and_update_status(
    run_dir: str | Path, record: RunRecord, event: dict[str, Any]
) -> RunRecord:
    updated = apply_event_to_run_record(record, event)
    snapshot_payload = build_status_snapshot(updated)
    snapshot = StatusSnapshot(
        run_id=snapshot_payload["run_id"],
        status=RunStatus(snapshot_payload["status"]),
        thread_id=snapshot_payload["thread_id"],
        workspace_dir=snapshot_payload["workspace_dir"],
        artifact_dir=snapshot_payload["artifact_dir"],
        updated_at=snapshot_payload["updated_at"],
        current_stage=snapshot_payload["current_stage"],
        completed_stages=snapshot_payload["completed_stages"],
        last_error=snapshot_payload["last_error"],
        suggested_next_action=snapshot_payload["suggested_next_action"],
    )
    append_run_event(run_dir, event)
    write_status_snapshot(run_dir, snapshot)
    write_failure_summary(run_dir, snapshot)
    return updated


def replay_events_into_status(
    run_dir: str | Path, record: RunRecord, events: list[dict[str, Any]]
) -> RunRecord:
    current = record
    for event in events:
        current = apply_event_and_update_status(run_dir, current, event)
    return current


async def consume_event_stream(
    run_dir: str | Path, record: RunRecord, event_stream
) -> RunRecord:
    current = record
    try:
        async for event in event_stream:
            current = apply_event_and_update_status(run_dir, current, event)
    except Exception as exc:
        current = apply_event_and_update_status(
            run_dir,
            current,
            {"type": "error", "message": str(exc)},
        )
    return current
