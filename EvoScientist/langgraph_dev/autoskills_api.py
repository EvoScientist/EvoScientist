"""AutoSkills drafts for the WebUI's review page.

Each function returns ``(status, payload)`` for a thin handler in
:mod:`~EvoScientist.langgraph_dev.http`. They read files synchronously, so
the handlers call them in a thread. Only drafts of served projects that
:func:`~EvoScientist.memory.autoskills.proposals.find_skill_proposal` reaches
are listed or read, so the page shows exactly what an approval would use.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from EvoScientist import paths
from EvoScientist.langgraph_dev import registry
from EvoScientist.memory.autoskills.proposals import (
    InstallTarget,
    SkillProposal,
    copied_files,
    find_skill_proposal,
    list_skill_proposals,
    proposal_install_target,
)

logger = logging.getLogger(__name__)

MAX_INLINE_CHARS = 200_000
MAX_FILE_BYTES = 1_048_576
_STATUSES = ("pending", "approved", "rejected")
_SCOPES = (None, "workspace", "global")
_APPROVERS = frozenset({"user", "auto"})


def _project_key(workspace_dir: object) -> str | None:
    if not isinstance(workspace_dir, str) or not workspace_dir.strip():
        return None
    try:
        return paths.Workspace(workspace_dir.strip()).key
    except paths.PATH_ERRORS:
        return None


def _inside(path: Path, root: Path) -> bool:
    """Whether ``path`` resolves inside the already-resolved ``root``."""
    try:
        return path.resolve().is_relative_to(root)
    except (OSError, RuntimeError):
        return False


def _served_keys(keys: list[str]) -> set[str]:
    served = set()
    for key in keys:
        try:
            if paths.Workspace(key).root.is_dir():
                served.add(key)
        except paths.PATH_ERRORS:
            continue
    return served


def _listed(proposal: SkillProposal, served: set[str]) -> str | None:
    """The project key a draft is listed under, or None when it is left out."""
    key = _project_key(proposal.workspace_dir)
    if key not in served or proposal.status not in _STATUSES:
        return None
    if proposal.scope not in _SCOPES:
        return None
    found = find_skill_proposal(paths.MEMORIES_DIR, proposal.proposal_id)
    if found is None or found.path != proposal.path:
        return None
    return key


def _inline(path: Path, folder: Path) -> tuple[str, bool]:
    """The start of a draft's text file, and whether it was cut."""
    try:
        if not path.is_file() or not _inside(path, folder):
            return "", False
        with path.open("rb") as handle:
            data = handle.read(MAX_INLINE_CHARS * 4 + 1)
            more = bool(handle.read(1))
    except OSError:
        return "", False
    text = data.decode("utf-8", errors="replace")
    truncated = more or len(text) > MAX_INLINE_CHARS
    return text[:MAX_INLINE_CHARS], truncated


def _install_target(proposal: SkillProposal, key: str) -> InstallTarget | str:
    """The pending draft's install target, or the error computing it raised."""
    try:
        return proposal_install_target(
            paths.MEMORIES_DIR,
            proposal.proposal_id,
            skills_dir=paths.Workspace(key).skills_dir,
        )
    except (OSError, ValueError) as exc:
        return str(exc) or type(exc).__name__


def _files(folder: Path, base: Path | None) -> list[dict[str, Any]]:
    """Every file an approval copies; metadata never follows a link outside."""
    entries = []
    for relative in copied_files(folder):
        path = folder / relative
        try:
            size = (path.stat() if _inside(path, folder) else path.lstat()).st_size
        except OSError:
            continue
        candidate = base / relative if base is not None else None
        entries.append(
            {
                "path": relative.as_posix(),
                "size": size,
                "installed": candidate is not None
                and _inside(candidate, base)
                and candidate.is_file(),
            }
        )
    return entries


def _draft(proposal: SkillProposal, key: str) -> dict[str, Any]:
    folder = proposal.path.resolve()
    skill_md, skill_md_truncated = _inline(folder / "SKILL.md", folder)
    rationale_md, rationale_md_truncated = _inline(folder / "RATIONALE.md", folder)
    target: InstallTarget | None = None
    install_error = None
    if proposal.status == "pending":
        computed = _install_target(proposal, key)
        if isinstance(computed, str):
            install_error = computed
        else:
            target, install_error = computed, computed.error
    base = target.base_path.resolve() if target and target.base_path else None
    present = None
    if proposal.status == "approved":
        present = (
            bool(proposal.approved_skill_path)
            and Path(str(proposal.approved_skill_path)).is_dir()
        )
    return {
        "proposal_id": proposal.proposal_id,
        "skill_name": proposal.skill_name,
        "description": proposal.description,
        "operation": proposal.operation,
        "target_skill_name": proposal.target_skill_name,
        "scope": proposal.effective_scope,
        "submitted_scope": proposal.submitted_scope,
        "status": proposal.status,
        "workspace_dir": key,
        "created_at": proposal.created_at,
        "updated_at": proposal.updated_at,
        "approved_at": proposal.approved_at,
        "approved_by": (
            proposal.approved_by if proposal.approved_by in _APPROVERS else None
        ),
        "rejected_at": proposal.rejected_at,
        "approved_skill_path": proposal.approved_skill_path,
        "approved_skill_present": present,
        "upstream_source": proposal.upstream_source,
        "workspace_shadow": proposal.workspace_shadow,
        "source_observation_ids": list(proposal.source_observation_ids),
        "skill_md": skill_md,
        "skill_md_truncated": skill_md_truncated,
        "rationale_md": rationale_md,
        "rationale_md_truncated": rationale_md_truncated,
        "files": _files(folder, base),
        "install_target": str(target.path) if target and target.path else None,
        "install_base": str(base) if base is not None else None,
        "install_error": install_error,
    }


def list_drafts(status: str | None = None) -> tuple[int, dict[str, Any]]:
    """Every served project's drafts; counts ignore the ``status`` filter."""
    if status is not None and status not in _STATUSES:
        return 400, {"error": "invalid_status"}
    keys = registry.REGISTRY.registered()
    served = _served_keys(keys)
    counts = dict.fromkeys(_STATUSES, 0)
    by_project = {key: dict.fromkeys(_STATUSES, 0) for key in keys}
    drafts = []
    for proposal in list_skill_proposals(paths.MEMORIES_DIR):
        key = _listed(proposal, served)
        if key is None:
            continue
        try:
            draft = _draft(proposal, key)
            json.dumps(draft, ensure_ascii=False).encode("utf-8")
        except Exception:
            # One draft that cannot be read or sent must not fail the page.
            logger.warning(
                "Could not list draft %s.", proposal.proposal_id, exc_info=True
            )
            continue
        counts[proposal.status] += 1
        by_project[key][proposal.status] += 1
        if status is None or proposal.status == status:
            drafts.append(draft)
    return 200, {"proposals": drafts, "counts": counts, "by_project": by_project}


def _relative_path(value: str) -> str | None:
    """A forward-slash path inside a folder, or None for anything else."""
    if not value or "\\" in value or "\x00" in value:
        return None
    posix = PurePosixPath(value)
    if posix.is_absolute() or PureWindowsPath(value).drive or ".." in posix.parts:
        return None
    return posix.as_posix()


def _read_text(path: Path) -> tuple[int, str | dict[str, Any]]:
    """A regular file's text, read once and never past the size limit."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return 404, {"error": "not_found"}
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            return 404, {"error": "not_found"}
        data = handle.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        return 413, {"error": "too_large"}
    if b"\x00" in data:
        return 415, {"error": "binary_file"}
    try:
        return 200, data.decode("utf-8")
    except UnicodeDecodeError:
        return 415, {"error": "binary_file"}


def read_draft_file(
    proposal_id: str, rel_path: str, *, installed: bool
) -> tuple[int, str | dict[str, Any]]:
    """A file an approval copies, from the draft or (``installed``) from the
    installed skill the approval starts from; a ``str`` body is its text."""
    relative = _relative_path(rel_path)
    if relative is None:
        return 400, {"error": "invalid_path"}
    proposal = find_skill_proposal(paths.MEMORIES_DIR, proposal_id)
    served = _served_keys(registry.REGISTRY.registered())
    key = _listed(proposal, served) if proposal is not None else None
    if proposal is None or key is None:
        return 404, {"error": "not_found"}
    folder = proposal.path.resolve()
    if relative not in {path.as_posix() for path in copied_files(folder)}:
        return 404, {"error": "not_found"}
    if installed:
        target = _install_target(proposal, key)
        if isinstance(target, str) or target.base_path is None:
            return 404, {"error": "not_found"}
        folder = target.base_path.resolve()
    path = folder / relative
    if not path.is_file():
        return 404, {"error": "not_found"}
    if not _inside(path, folder):
        return 400, {"error": "invalid_path"}
    return _read_text(path)
