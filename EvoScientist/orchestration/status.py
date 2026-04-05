from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from .models import RunRecord, RunStatus


def apply_event_to_run_record(record: RunRecord, event: dict[str, object]) -> RunRecord:
    event_type = str(event.get("type", ""))
    updated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    if event_type in {"text", "thinking", "tool_call", "tool_result", "subagent_start", "subagent_text"}:
        return replace(record, status=RunStatus.RUNNING, updated_at=updated_at)
    if event_type in {"interrupt", "ask_user"}:
        return replace(record, status=RunStatus.BLOCKED, updated_at=updated_at)
    if event_type == "error":
        return replace(
            record,
            status=RunStatus.FAILED,
            last_error=str(event.get("message", "unknown error")),
            updated_at=updated_at,
        )
    if event_type == "done":
        return replace(record, status=RunStatus.COMPLETED, updated_at=updated_at)
    return replace(record, updated_at=updated_at)


def build_status_snapshot(record: RunRecord) -> dict[str, object]:
    suggested_next_action = None
    if record.status == RunStatus.FAILED:
        suggested_next_action = "run doctor, inspect diagnostics, then retry or resume"
    elif record.status == RunStatus.BLOCKED:
        suggested_next_action = "provide required approval or input and continue"

    return {
        "run_id": record.run_id,
        "thread_id": record.thread_id,
        "workspace_dir": record.workspace_dir,
        "artifact_dir": record.artifact_dir,
        "status": record.status.value,
        "current_stage": record.current_stage,
        "completed_stages": list(record.completed_stages),
        "last_error": record.last_error,
        "suggested_next_action": suggested_next_action,
        "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
