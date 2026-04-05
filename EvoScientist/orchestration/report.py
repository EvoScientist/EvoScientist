from __future__ import annotations

import json
from pathlib import Path


def build_run_report(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    status_file = run_dir / "status.json"
    if not status_file.exists():
        return f"Run report unavailable: missing status.json in {run_dir}"

    try:
        payload = json.loads(status_file.read_text())
    except json.JSONDecodeError:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}
    run_id = payload.get("run_id", run_dir.name)
    status = payload.get("status", "unknown")
    workspace_dir = payload.get("workspace_dir", "")
    current_stage = payload.get("current_stage") or "n/a"
    last_error = payload.get("last_error") or "none"

    return (
        f"Run {run_id}\n"
        f"Status: {status}\n"
        f"Workspace: {workspace_dir}\n"
        f"Artifact dir: {run_dir}\n"
        f"Current stage: {current_stage}\n"
        f"Last error: {last_error}"
    )
