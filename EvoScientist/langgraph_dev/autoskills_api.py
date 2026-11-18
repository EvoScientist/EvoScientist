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
import threading
from collections.abc import AsyncIterator
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from EvoScientist import paths
from EvoScientist.langgraph_dev import folder_check, registry
from EvoScientist.memory.autoskills.proposals import (
    _SKILL_SCOPES,
    InstallTarget,
    SkillProposal,
    _failure_text,
    _read_regular,
    _resolves_within_root,
    approve_skill_proposal,
    copied_files,
    find_skill_proposal,
    installed_skills_read_once,
    list_skill_proposals,
    proposal_install_target,
    record_rendered_proposal,
    reject_skill_proposal,
    rendered_files,
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
    "updated_at",
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
    # The proposal came from the listing's own read of its manifest, so the
    # row is already the draft an approval would use. Confirming it with a
    # second lookup reads the folder again, and a folder removed between the
    # two reads fails the page. A single draft is still confirmed by
    # ``find_skill_proposal`` where one file is served.
    if proposal.proposal_id != proposal.path.name:
        return None
    return key


def _inline(path: Path, folder: Path) -> tuple[str, bool]:
    """The start of a draft's text file, and whether it was cut."""
    # A character takes at most four bytes, so one byte more shows a longer text.
    limit = MAX_INLINE_CHARS * 4 + 1
    data = _read_regular(path, limit) if _resolves_within_root(path, folder) else None
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
    """The draft's files an approval copies, without a fork's base skill;
    metadata never follows a link outside."""
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


def _shown_draft_record(
    proposal: SkillProposal, draft: dict[str, Any]
) -> tuple[Path, dict[str, Any], list[tuple[str, str]]] | None:
    """The review record of a draft whose content this response returns.

    The list inlines ``SKILL.md`` and ``RATIONALE.md``. Every other copied
    file is appended to ``skill_md``, which is the body the page shows, so
    the record is of text present in the payload. The file rows stay
    metadata. A draft that cannot be read in full, or whose text was cut,
    is left unrecorded.

    Nothing is recorded here. The record is built from the text the
    response carries and written by the caller once that response has been
    delivered, so a request that never delivers the payload leaves the
    draft unarmed.
    """
    assert proposal.manifest is not None
    try:
        rendered = rendered_files(proposal.path)
    except (OSError, ValueError):
        return None
    by_path = dict(rendered)
    skill = by_path.get("SKILL.md")
    if skill is None or draft.get("skill_md") != skill[:MAX_INLINE_CHARS]:
        # The inlined text is not the file an approval would copy.
        return None
    rationale = by_path.get("RATIONALE.md")
    if (
        rationale is not None
        and draft.get("rationale_md") != rationale[:MAX_INLINE_CHARS]
    ):
        return None
    if draft.get("skill_md_truncated") or draft.get("rationale_md_truncated"):
        # A cut text is not the whole draft, so it is not a review of it.
        return None
    # The two markdown texts are inlined above. Every other copied file is
    # returned in ``contents``, path to text, so the payload holds the whole
    # draft and the review record is of that text. A file row stays metadata.
    draft["contents"] = {
        relative: content
        for relative, content in rendered
        if relative not in {"SKILL.md", "RATIONALE.md"}
    }
    return (proposal.path, proposal.manifest, rendered)


def record_shown_drafts(
    records: list[tuple[Path, dict[str, Any], list[tuple[str, str]]]],
) -> None:
    """Record the reviews of the drafts whose text a response carried.

    Called once the response has been delivered: a request that never
    delivered the payload leaves its drafts unarmed.
    """
    for path, manifest, rendered in records:
        record_rendered_proposal(path, manifest, rendered)


def _draft(proposal: SkillProposal, key: str) -> dict[str, Any]:
    folder = proposal.path.resolve()
    # A decided draft's texts are read through the file route: the list is
    # polled and decided drafts only grow in number.
    skill_md: tuple[str | None, bool | None] = (None, None)
    rationale_md = skill_md
    target: InstallTarget | None = None
    install_error = None
    if proposal.status == "pending":
        skill_md = _inline(folder / "SKILL.md", folder)
        rationale_md = _inline(folder / "RATIONALE.md", folder)
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
        "skill_md": skill_md[0],
        "skill_md_truncated": skill_md[1],
        "rationale_md": rationale_md[0],
        "rationale_md_truncated": rationale_md[1],
        "contents": {} if proposal.status == "pending" else None,
        "files": _files(folder, base),
        "install_target": str(target.path) if target and target.path else None,
        "install_base": str(base) if base is not None else None,
        "install_error": install_error,
    }


def list_drafts(
    status: str | None = None,
    records: list[tuple[Path, dict[str, Any], list[tuple[str, str]]]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Every served project's drafts; counts ignore the ``status`` filter."""
    if status is not None and status not in _STATUSES:
        return 400, {"error": "invalid_status"}
    keys = registry.REGISTRY.registered()
    served = _served_keys(keys)
    counts = dict.fromkeys(_STATUSES, 0)
    by_project = {key: dict.fromkeys(_STATUSES, 0) for key in keys}
    drafts = []
    with installed_skills_read_once():
        for proposal in list_skill_proposals(paths.MEMORIES_DIR):
            key = _listed(proposal, served)
            if key is None:
                continue
            counts[proposal.status] += 1
            by_project[key][proposal.status] += 1
            if status is not None and proposal.status != status:
                continue
            record = None
            try:
                draft = _draft(proposal, key)
                if proposal.status == "pending" and proposal.manifest is not None:
                    # The row is in the response, so its inlined texts are
                    # what the page shows. Build that review record only. A
                    # filter that dropped the row never reaches here, and a
                    # draft that cannot be sent does not either: the payload
                    # is encoded below, before anything is recorded.
                    record = _shown_draft_record(proposal, draft)
                json.dumps(draft, ensure_ascii=False).encode("utf-8")
            except Exception:
                # One draft that cannot be read or sent must not fail the page.
                logger.warning(
                    "Could not list draft %s.", proposal.proposal_id, exc_info=True
                )
                continue
            if record is not None:
                if records is None:
                    # An in-process caller receives the payload as the return
                    # value, so the text is delivered with it.
                    record_shown_drafts([record])
                else:
                    # A response on its way out: the caller records once it
                    # has been delivered.
                    records.append(record)
            drafts.append(draft)
    return 200, {"proposals": drafts, "counts": counts, "by_project": by_project}


def list_drafts_body(
    status: str | None = None,
) -> tuple[int, bytes, list[tuple[Path, dict[str, Any], list[tuple[str, str]]]]]:
    """:func:`list_drafts` encoded as the response body, off the event loop.

    The review records come back with it, unrecorded: the caller writes them
    once the body has been delivered.
    """
    records: list[tuple[Path, dict[str, Any], list[tuple[str, str]]]] = []
    code, payload = list_drafts(status, records)
    body = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return code, body.encode("utf-8"), records


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
    data = _read_regular(path, MAX_FILE_BYTES + 1)
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
    """A file an approval copies (the draft's, or with ``installed`` the installed
    skill's) or the draft's own rationale; a ``str`` body is its text."""
    relative = _relative_path(rel_path)
    if relative is None:
        return 400, {"error": "invalid_path"}
    # Looking the draft up does not record a review. The installed copy is
    # not the pending draft, and one file is not the whole draft either.
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
    if relative not in names and (installed or relative != "RATIONALE.md"):
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
    status, body = _read_text(path)
    if (
        status == 200
        and not installed
        and isinstance(body, str)
        and proposal.status == "pending"
        and proposal.manifest is not None
    ):
        # This response returns one file. Record a review only when that
        # file is the whole draft an approval would copy: a single-file
        # draft's body is then on screen, and a larger draft stays unrecorded
        # until the list returns every copied file.
        try:
            rendered = rendered_files(proposal.path)
        except (OSError, ValueError):
            rendered = []
        if rendered == [(relative, body)]:
            record_rendered_proposal(proposal.path, proposal.manifest, rendered)
    return status, body


def _refused(action: str, result: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Map a refusal of approve or reject to the routes' error codes."""
    error = str(result.get("error") or "")
    if error.startswith("Only pending proposals"):
        return 409, {"error": "not_pending", "status": result.get("status")}
    if error.startswith("Proposal changed since it was read"):
        return 409, {"error": "draft_changed", "updated_at": result.get("updated_at")}
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
            expected_updated_at=expected_updated_at,
        )
        if result.get("rejected"):
            return 200, {
                "rejected": True,
                "proposal_id": result.get("proposal_id"),
                "skill_name": result.get("skill_name"),
                "updated_at": result.get("updated_at"),
            }
        return _refused(action, result)
    result = revert_skill_proposal(
        paths.MEMORIES_DIR,
        proposal.proposal_id,
        expected_updated_at=expected_updated_at,
    )
    if result.get("reverted"):
        return 200, {
            "reverted": True,
            "proposal_id": result.get("proposal_id"),
            "status": result.get("status"),
            "updated_at": result.get("updated_at"),
        }
    code = str(result.get("error"))
    payload: dict[str, Any] = {"error": code}
    if code == "revert_failed":
        payload["detail"] = str(result.get("detail") or "")
    if code == "draft_changed":
        payload["updated_at"] = result.get("updated_at")
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
    """Approve, reject or revert one draft of a served project; with
    ``expected_updated_at``, a draft resubmitted or decided since is refused."""
    if action == "approve" and scope is not None and scope not in _SCOPES:
        return 400, {"error": "invalid_scope"}
    if not isinstance(expected_updated_at, str | None):
        return 400, {"error": "invalid_body"}
    with _DECISIONS:
        try:
            status, payload = _decide_locked(
                proposal_id,
                action,
                scope if isinstance(scope, str) else None,
                expected_updated_at,
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
    """Start one manual AutoSkills run for a served project, unless a manual one
    is in flight; a run the cron fired is not counted, so the two can overlap."""
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
