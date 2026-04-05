from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import ArtifactIndex, RunRecord, RunStatus, StatusSnapshot
from .status import apply_event_to_run_record, build_status_snapshot


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = path.read_text().strip()
    if not raw or raw == "{}":
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        return None
    return payload


def _parse_updated_at(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).astimezone(UTC)
    except ValueError:
        return None


def _extract_run_summary(
    run_payload: dict[str, Any] | None, status_payload: dict[str, Any] | None
) -> str:
    prompt = None
    if run_payload:
        prompt = run_payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            metadata = run_payload.get("metadata")
            if isinstance(metadata, dict):
                prompt = metadata.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()

    if status_payload:
        current_stage = status_payload.get("current_stage")
        if isinstance(current_stage, str) and current_stage.strip():
            return current_stage.strip()

        last_error = status_payload.get("last_error")
        if isinstance(last_error, str) and last_error.strip():
            return last_error.strip()

        status = status_payload.get("status")
        if isinstance(status, str) and status.strip():
            return status.strip()

    return "unknown"


def _first_string(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _resolve_updated_at(
    status_payload: dict[str, Any] | None, run_payload: dict[str, Any] | None
) -> tuple[str | None, datetime | None]:
    candidates: list[object] = []
    if status_payload is not None:
        candidates.append(status_payload.get("updated_at"))
    if run_payload is not None:
        candidates.append(run_payload.get("updated_at"))

    fallback_value = _first_string(*candidates)
    for value in candidates:
        parsed = _parse_updated_at(value)
        if parsed is not None:
            return str(value), parsed
    return fallback_value, None


def _build_recent_run_entry(run_dir: Path) -> tuple[float, dict[str, object]] | None:
    status_path = run_dir / "status.json"
    run_path = run_dir / "run.json"

    try:
        status_payload = _read_json_mapping(status_path)
        run_payload = _read_json_mapping(run_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None

    if status_payload is None and run_payload is None:
        return None

    run_id = None
    if status_payload is not None:
        run_id = status_payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        if run_payload is not None:
            run_id = run_payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return None

    updated_at, sort_dt = _resolve_updated_at(status_payload, run_payload)
    if sort_dt is not None:
        sort_key = sort_dt.timestamp()
    else:
        mtime_candidates = [
            path.stat().st_mtime
            for path in (status_path, run_path)
            if path.exists()
        ]
        if not mtime_candidates:
            mtime_candidates = [run_dir.stat().st_mtime]
        sort_key = max(mtime_candidates)

    row_source = status_payload or run_payload or {}
    row: dict[str, object] = {
        "run_id": run_id,
        "status": row_source.get("status") or "unknown",
        "updated_at": updated_at,
        "current_stage": row_source.get("current_stage"),
        "summary": _extract_run_summary(run_payload, status_payload),
        "last_error": row_source.get("last_error"),
    }
    return sort_key, row


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


def resolve_latest_run_id(artifact_root: Path) -> str | None:
    rows = list_recent_runs(artifact_root, limit=1)
    if not rows:
        return None
    run_id = rows[0].get("run_id")
    return run_id if isinstance(run_id, str) else None


def list_recent_runs(artifact_root: Path, limit: int = 10) -> list[dict[str, object]]:
    artifact_root = Path(artifact_root)
    if limit <= 0 or not artifact_root.exists():
        return []

    rows: list[tuple[float, dict[str, object]]] = []
    for child in artifact_root.iterdir():
        if not child.is_dir():
            continue
        entry = _build_recent_run_entry(child)
        if entry is not None:
            rows.append(entry)

    rows.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in rows[:limit]]


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
