"""AutoSkills drafts for the WebUI's review page.

Each function returns ``(status, payload)`` for a thin handler in
:mod:`~EvoScientist.langgraph_dev.http`. They read and write files
synchronously, so the handlers call them in a thread; ``run_now`` is a
coroutine that moves its own file work off the loop. Only drafts of served
projects that
:func:`~EvoScientist.memory.autoskills.proposals.find_skill_proposal` reaches
are listed or read, so the page shows exactly what an approval would use.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import stat
import threading
from collections.abc import AsyncIterator
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from EvoScientist import paths
from EvoScientist.langgraph_dev import folder_check, registry
from EvoScientist.memory.autoskills.proposals import (
    _OPEN_FLAGS,
    _SKILL_SCOPES,
    InstallTarget,
    SkillProposal,
    _failure_text,
    _resolves_within_root,
    approve_skill_proposal,
    copied_files,
    find_skill_proposal,
    list_skill_proposals,
    proposal_install_target,
    reject_skill_proposal,
    revert_skill_proposal,
)

logger = logging.getLogger(__name__)

MAX_INLINE_CHARS = 200_000
MAX_FILE_BYTES = 1_048_576
_STATUSES = ("pending", "approved", "rejected")
# A tuple, so an unhashable scope from a request body is simply not in it.
_SCOPES = (None, *sorted(_SKILL_SCOPES))
_APPROVERS = frozenset({"user", "auto"})
_APPROVED_KEYS = (
    "approved",
    "proposal_id",
    "skill_name",
    "operation",
    "path",
    "scope",
    "workspace_shadow",
    "upstream_source",
)
_REVERT_STATUS = {
    "not_found": 404,
    "not_decided": 409,
    "revert_unavailable": 409,
    "skill_changed": 409,
    "revert_failed": 409,
}
# One decision at a time, so two tabs approving one draft install it once.
_DECISIONS = threading.Lock()


def _project_key(workspace_dir: object) -> str | None:
    if not isinstance(workspace_dir, str) or not workspace_dir.strip():
        return None
    try:
        return paths.Workspace(workspace_dir.strip()).key
    except paths.PATH_ERRORS:
        return None


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


def _read_head(path: Path, size: int) -> bytes | None:
    """Up to ``size`` bytes of a regular file, or None; a FIFO never blocks it."""
    try:
        fd = os.open(path, _OPEN_FLAGS)
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return None
            return handle.read(size)
    except OSError:
        return None


def _inline(path: Path, folder: Path) -> tuple[str, bool]:
    """The start of a draft's text file, and whether it was cut."""
    # A character takes at most four bytes, so one byte more shows a longer text.
    limit = MAX_INLINE_CHARS * 4 + 1
    data = _read_head(path, limit) if _resolves_within_root(path, folder) else None
    if data is None:
        return "", False
    text = data.decode("utf-8", errors="replace")
    return text[:MAX_INLINE_CHARS], len(text) > MAX_INLINE_CHARS


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
    try:
        relatives = copied_files(folder)
    except OSError:
        # A pending draft's dry run reports the unreadable folder as install_error.
        return []
    entries = []
    for relative in relatives:
        path = folder / relative
        try:
            size = (
                path.stat() if _resolves_within_root(path, folder) else path.lstat()
            ).st_size
        except OSError:
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "size": size,
                "installed": base is not None and _installed(base, relative),
            }
        )
    return entries


def _installed(base: Path, relative: Path) -> bool:
    """Whether the installed skill has this file, inside its folder."""
    candidate = base / relative
    try:
        return _resolves_within_root(candidate, base) and candidate.is_file()
    except OSError:
        return False


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
        # os.path.isdir never raises: a folder it cannot stat must not drop the draft.
        present = bool(proposal.approved_skill_path) and os.path.isdir(
            str(proposal.approved_skill_path)
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


def list_drafts_body(status: str | None = None) -> tuple[int, bytes]:
    """:func:`list_drafts` encoded as the response body, off the event loop."""
    code, payload = list_drafts(status)
    body = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return code, body.encode("utf-8")


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
    data = _read_head(path, MAX_FILE_BYTES + 1)
    if data is None:
        return 404, {"error": "not_found"}
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
    try:
        names = {path.as_posix() for path in copied_files(folder)}
    except OSError:
        return 404, {"error": "not_found"}
    if relative not in names:
        return 404, {"error": "not_found"}
    if installed:
        target = _install_target(proposal, key)
        if isinstance(target, str) or target.base_path is None:
            return 404, {"error": "not_found"}
        folder = target.base_path.resolve()
    path = folder / relative
    try:
        if not path.is_file():
            return 404, {"error": "not_found"}
    except OSError:
        return 404, {"error": "not_found"}
    if not _resolves_within_root(path, folder):
        return 400, {"error": "invalid_path"}
    return _read_text(path)


def _refused(action: str, result: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Map a refusal of approve or reject to the routes' error codes."""
    error = str(result.get("error") or "")
    if error.startswith("Only pending proposals"):
        return 409, {"error": "not_pending", "status": result.get("status")}
    if error.startswith("Proposal changed since it was read"):
        return 409, {"error": "draft_changed", "updated_at": result.get("updated_at")}
    if error.startswith("scope must be"):
        return 400, {"error": "invalid_scope"}
    return 409, {"error": f"{action}_failed", "detail": _failure_text(result)}


def _apply(
    proposal: SkillProposal,
    workspace: paths.Workspace,
    action: str,
    scope: str | None,
    expected_updated_at: str | None,
) -> tuple[int, dict[str, Any]]:
    if action == "approve":
        result = approve_skill_proposal(
            paths.MEMORIES_DIR,
            proposal.proposal_id,
            skills_dir=workspace.skills_dir,
            workspace_dir=workspace.root,
            scope=scope,
            approved_by="user",
            exact=True,
            expected_updated_at=expected_updated_at,
        )
        if result.get("approved"):
            return 200, {key: result.get(key) for key in _APPROVED_KEYS}
        return _refused(action, result)
    if action == "reject":
        result = reject_skill_proposal(
            paths.MEMORIES_DIR,
            proposal.proposal_id,
            workspace_dir=workspace.root,
            exact=True,
        )
        if result.get("rejected"):
            return 200, {
                "rejected": True,
                "proposal_id": result.get("proposal_id"),
                "skill_name": result.get("skill_name"),
            }
        return _refused(action, result)
    result = revert_skill_proposal(paths.MEMORIES_DIR, proposal.proposal_id)
    if result.get("reverted"):
        return 200, {
            "reverted": True,
            "proposal_id": result.get("proposal_id"),
            "status": "pending",
        }
    code = str(result.get("error"))
    payload: dict[str, Any] = {"error": code}
    if code == "revert_failed":
        payload["detail"] = str(result.get("detail") or "")
    return _REVERT_STATUS.get(code, 409), payload


def _served(workspace: paths.Workspace) -> bool:
    """As the list does: a project folder the server cannot stat is not served."""
    try:
        return folder_check.is_served(workspace, registry.REGISTRY)
    except OSError:
        return False


def _decide_locked(
    proposal_id: str, action: str, scope: str | None, expected_updated_at: str | None
) -> tuple[int, dict[str, Any]]:
    proposal = find_skill_proposal(paths.MEMORIES_DIR, proposal_id)
    if proposal is None:
        return 404, {"error": "not_found"}
    key = _project_key(proposal.workspace_dir)
    if key is None:
        return 409, {"error": "no_project"}
    workspace = paths.Workspace(key)
    if not _served(workspace):
        return 409, {"error": "project_not_served"}
    return _apply(proposal, workspace, action, scope, expected_updated_at)


def _sendable(payload: dict[str, Any]) -> dict[str, Any]:
    """A path in a refusal may hold a lone surrogate, which JSON cannot carry."""
    return {
        key: value.encode("utf-8", "backslashreplace").decode("utf-8")
        if isinstance(value, str)
        else value
        for key, value in payload.items()
    }


def decide(
    proposal_id: str,
    action: Literal["approve", "reject", "revert"],
    *,
    scope: object = None,
    expected_updated_at: object = None,
) -> tuple[int, dict[str, Any]]:
    """Approve, reject or revert one draft of a served project; an approval
    with ``expected_updated_at`` refuses a draft resubmitted since the page read it."""
    if action == "approve" and scope is not None and scope not in _SCOPES:
        return 400, {"error": "invalid_scope"}
    if action == "approve" and not isinstance(expected_updated_at, str | None):
        return 400, {"error": "invalid_body"}
    with _DECISIONS:
        try:
            status, payload = _decide_locked(
                proposal_id,
                action,
                scope if isinstance(scope, str) else None,
                expected_updated_at if action == "approve" else None,
            )
        except Exception as exc:
            # One failed decision answers the client instead of failing the request.
            logger.warning("Could not %s draft %s.", action, proposal_id, exc_info=True)
            status = 409
            payload = {
                "error": f"{action}_failed",
                "detail": str(exc) or type(exc).__name__,
            }
    return status, _sendable(payload)


# One start at a time per project and event loop, so two Run now requests cannot
# both start a run; an entry is dropped when unused, as a waited-on lock holds its loop.
_RUN_LOCKS: dict[tuple[asyncio.AbstractEventLoop, str], tuple[asyncio.Lock, int]] = {}


@contextlib.asynccontextmanager
async def _run_lock(key: str) -> AsyncIterator[None]:
    entry = (asyncio.get_running_loop(), key)
    lock, users = _RUN_LOCKS.get(entry, (asyncio.Lock(), 0))
    _RUN_LOCKS[entry] = (lock, users + 1)
    try:
        async with lock:
            yield
    finally:
        lock, users = _RUN_LOCKS[entry]
        if users == 1:
            del _RUN_LOCKS[entry]
        else:
            _RUN_LOCKS[entry] = (lock, users - 1)


async def run_now(workspace_dir: object) -> tuple[int, dict[str, Any]]:
    """Start one manual AutoSkills run for a served project, unless one runs."""
    from EvoScientist.config import get_effective_config
    from EvoScientist.memory.autoskills import schedule

    workspace = await asyncio.to_thread(
        folder_check.read_workspace_or_none, workspace_dir
    )
    if workspace is None:
        return 400, {"error": "invalid_path"}
    try:
        if not await asyncio.to_thread(_served, workspace):
            return 404, {"error": "not_served"}
        async with _run_lock(workspace.key):
            # Read under the lock: a start that waited sees a setting changed meanwhile.
            config = await asyncio.to_thread(get_effective_config)
            if not config.memory_skill_synthesis_enabled:
                return 409, {"error": "autoskills_off"}
            running = await schedule.afind_running_autoskill(
                config, workspace_dir=workspace.key
            )
            if running is not None:
                return 409, {"error": "already_running", **running}
            started = await schedule.arun_autoskill_now(
                config, workspace_dir=workspace.key
            )
    except Exception as exc:
        logger.warning(
            "Could not start AutoSkills for %s.", workspace.key, exc_info=True
        )
        detail = str(exc) or type(exc).__name__
        return 502, _sendable({"error": "run_failed", "detail": detail})
    return 200, {
        "thread_id": str(started["thread_id"]),
        "run_id": str(started["run_id"]),
    }
