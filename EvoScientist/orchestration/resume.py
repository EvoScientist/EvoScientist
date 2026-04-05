from __future__ import annotations

import json
from pathlib import Path

_REQUIRED_KEYS = ("run_id", "thread_id", "workspace_dir", "artifact_dir")


def build_resume_payload(run_dir: str | Path) -> dict[str, object]:
    run_dir = Path(run_dir)
    try:
        payload = json.loads((run_dir / "run.json").read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"Run resume unavailable: missing run.json in {run_dir}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Run resume unavailable: malformed run.json in {run_dir}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Run resume unavailable: invalid run payload in {run_dir}")

    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(
            f"Run resume unavailable: missing required fields {', '.join(missing)} in {run_dir}"
        )

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "run_id": payload["run_id"],
        "thread_id": payload["thread_id"],
        "workspace_dir": payload["workspace_dir"],
        "artifact_dir": payload["artifact_dir"],
        "prompt": metadata.get("prompt", ""),
        "model": metadata.get("model", ""),
        "resume_semantics": "restart_from_saved_run_context",
    }
