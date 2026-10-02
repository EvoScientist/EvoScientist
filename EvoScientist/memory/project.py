"""Project identity helpers for file-backed memory."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .. import paths as _paths


def _short_hash(text: str, *, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def resolve_project_id(workspace: str | Path | None = None) -> str:
    """Return the stable id used for this workspace's project memory."""
    root = Path(workspace or _paths.WORKSPACE_ROOT).expanduser().resolve()
    git_root = _run_git(["rev-parse", "--show-toplevel"], root)
    if git_root:
        git_root_path = Path(git_root).expanduser().resolve()
        remote = _run_git(["remote", "get-url", "origin"], git_root_path)
        source = f"git-remote:{remote}" if remote else f"git-root:{git_root_path}"
        return f"P-{_short_hash(source)}"
    return f"P-{_short_hash(f'path:{root}')}"
