from __future__ import annotations

import json
from pathlib import Path


def build_resume_payload(run_dir: str | Path) -> dict[str, object]:
    run_dir = Path(run_dir)
    payload = json.loads((run_dir / "run.json").read_text())
    metadata = payload.get("metadata", {})
    return {
        "run_id": payload["run_id"],
        "thread_id": payload["thread_id"],
        "workspace_dir": payload["workspace_dir"],
        "artifact_dir": payload["artifact_dir"],
        "prompt": metadata.get("prompt", ""),
        "model": metadata.get("model", ""),
        "resume_semantics": "restart_from_saved_run_context",
    }
