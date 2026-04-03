from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import StrEnum
from typing import Any


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    BLOCKED = "blocked"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class RunRecord:
    run_id: str
    thread_id: str
    workspace_dir: str
    artifact_dir: str
    status: RunStatus = RunStatus.CREATED
    current_stage: str | None = None
    completed_stages: list[str] = field(default_factory=list)
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )


@dataclass
class StatusSnapshot:
    run_id: str
    status: RunStatus
    thread_id: str
    workspace_dir: str
    artifact_dir: str
    updated_at: str
    current_stage: str | None = None
    completed_stages: list[str] = field(default_factory=list)
    last_error: str | None = None
    suggested_next_action: str | None = None


@dataclass
class DiagnosticReport:
    ok: bool
    summary: str
    checks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ArtifactIndex:
    artifact_dir: str
    run_json: str
    status_json: str
    events_jsonl: str
    outputs_dir: str
    deliverables_dir: str
    diagnostics_dir: str
