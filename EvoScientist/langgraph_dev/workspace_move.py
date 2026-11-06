"""Moving a project to another folder: what is keyed on its old folder.

Everything a project owns lives outside its folder and names it by its
``Workspace.key``: its threads (the folders stamped on their checkpoints in
``sessions.db``, and the runtime's own copy of each thread), its scheduled
tasks and workspace assistants (IDs computed from the key), its memory
project (when its id comes from the path), its AutoSkills proposals and the
old store import's marks. Each function here re-points one of them from
:attr:`Move.old` to :attr:`Move.new`, touching only what still names the old
folder, so running it again after a crash finishes the job and is otherwise a
no-op. Moving no files: the user moved the folder.

The runtime's rows are edited in place, as the old store import edits
threads: the runtime's own update cannot change a task's assistant, and a
thread or task keeps its place in the lists and its next run time.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import aiosqlite

from ..paths import SessionDirs, Workspace
from ..sessions import _table_exists, get_db_path
from .assistants import served_graph_ids, workspace_assistant_id
from .server_crons import is_system_cron, metadata_of

logger = logging.getLogger(__name__)

_FOLDER_KEYS = ("workspace_dir", "run_dir")
# What :func:`~EvoScientist.memory.project.resolve_project_id` gives.
_PROJECT_ID = re.compile(r"P-[0-9a-f]{16}")


@dataclass(frozen=True)
class Move:
    """A project's move from the folder *old* to the folder *new*."""

    old: Workspace
    new: Workspace

    @cached_property
    def assistants(self) -> dict[str, str]:
        """Each of the old folder's workspace assistants to the new folder's
        for the same graph."""
        return {
            workspace_assistant_id(graph_id, self.old): workspace_assistant_id(
                graph_id, self.new
            )
            for graph_id in served_graph_ids()
        }

    def folders(
        self, workspace_dir: object, run_dir: object = None
    ) -> dict[str, str] | None:
        """The folders to store instead of the stored *workspace_dir* and
        *run_dir* when they name the old folder (a run folder keeps its
        place under the new one), else None. Resolves paths: call it off the
        event loop on the server."""
        if not isinstance(workspace_dir, str):
            return None
        dirs = SessionDirs.from_stored(
            workspace_dir, run_dir if isinstance(run_dir, str) else None
        )
        if dirs is None or dirs.workspace.key != self.old.key:
            return None
        moved_run = None
        if dirs.run_dir is not None:
            moved_run = self.new.root / dirs.run_dir.relative_to(self.old.root)
        return SessionDirs(self.new, moved_run).metadata()


def _set_folders(target: dict[str, Any], folders: dict[str, str]) -> None:
    target["workspace_dir"] = folders["workspace_dir"]
    if "run_dir" in folders:
        target["run_dir"] = folders["run_dir"]
    else:
        target.pop("run_dir", None)


def _moved_pairs(
    move: Move, pairs: set[tuple[object, object]]
) -> dict[tuple[object, object], dict[str, str]]:
    """The stored (workspace_dir, run_dir) pairs among *pairs* that name the
    old folder, with what to store instead. Blocking."""
    moved = {}
    for pair in pairs:
        folders = move.folders(*pair)
        if folders is not None:
            moved[pair] = folders
    return moved


async def move_checkpoints(move: Move) -> int:
    """Re-point every checkpoint row of the old folder's threads in
    ``sessions.db``, every row of a thread and not only its newest (readers
    take the newest stamped row, or the highest ``assistant_id`` over all of
    them): their folders, and the workspace assistant they name. One
    transaction; returns the number of threads changed."""
    db_path = str(await asyncio.to_thread(get_db_path))
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return 0
        async with conn.execute(
            "SELECT DISTINCT json_extract(metadata, '$.workspace_dir'), "
            "json_extract(metadata, '$.run_dir') FROM checkpoints "
            "WHERE json_valid(metadata) "
            "AND json_extract(metadata, '$.workspace_dir') IS NOT NULL"
        ) as cur:
            pairs = {(row[0], row[1]) for row in await cur.fetchall()}
        moved = await asyncio.to_thread(_moved_pairs, move, pairs)
        await conn.execute("BEGIN IMMEDIATE")
        try:
            changed = await _rewrite_checkpoints(conn, move, moved)
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise
    return changed


async def _rewrite_checkpoints(
    conn: aiosqlite.Connection,
    move: Move,
    moved: dict[tuple[object, object], dict[str, str]],
) -> int:
    await conn.execute(
        "CREATE TEMP TABLE _move_dirs (old_dir TEXT NOT NULL, old_run TEXT NOT "
        "NULL, new_dir TEXT NOT NULL, new_run TEXT, PRIMARY KEY (old_dir, old_run))"
    )
    await conn.execute(
        "CREATE TEMP TABLE _move_assistants (old TEXT PRIMARY KEY, new TEXT NOT NULL)"
    )
    try:
        await conn.executemany(
            "INSERT INTO _move_dirs VALUES (?, ?, ?, ?)",
            [
                (old_dir, old_run or "", f["workspace_dir"], f.get("run_dir"))
                for (old_dir, old_run), f in moved.items()
            ],
        )
        await conn.executemany(
            "INSERT INTO _move_assistants VALUES (?, ?)",
            list(move.assistants.items()),
        )
        # ``metadata`` is stored as a BLOB of JSON text; keep it one. A
        # correlated subquery rather than ``UPDATE ... FROM``, which needs
        # SQLite 3.33.
        match = (
            "FROM _move_dirs AS m "
            "WHERE m.old_dir = json_extract(checkpoints.metadata, '$.workspace_dir') "
            "AND m.old_run = "
            "COALESCE(json_extract(checkpoints.metadata, '$.run_dir'), '')"
        )
        assistant_match = (
            "json_extract(checkpoints.metadata, '$.assistant_id') "
            "IN (SELECT old FROM _move_assistants)"
        )
        async with conn.execute(
            "SELECT COUNT(DISTINCT thread_id) FROM checkpoints WHERE json_valid(metadata) "
            f"AND (EXISTS (SELECT 1 {match}) OR {assistant_match})"
        ) as cur:
            (changed,) = await cur.fetchone()
        await conn.execute(
            f"""
            UPDATE checkpoints SET metadata = CAST(
                (SELECT CASE WHEN m.new_run IS NULL
                    THEN json_remove(json_set(CAST(checkpoints.metadata AS TEXT),
                                              '$.workspace_dir', m.new_dir),
                                     '$.run_dir')
                    ELSE json_set(CAST(checkpoints.metadata AS TEXT),
                                  '$.workspace_dir', m.new_dir,
                                  '$.run_dir', m.new_run)
                    END {match})
                AS BLOB)
            WHERE json_valid(metadata) AND EXISTS (SELECT 1 {match})
            """
        )
        await conn.execute(
            f"""
            UPDATE checkpoints SET metadata = CAST(
                json_set(CAST(metadata AS TEXT), '$.assistant_id',
                    (SELECT a.new FROM _move_assistants AS a
                     WHERE a.old = json_extract(checkpoints.metadata,
                                                '$.assistant_id')))
                AS BLOB)
            WHERE json_valid(metadata) AND {assistant_match}
            """
        )
        return changed
    finally:
        await conn.execute("DROP TABLE _move_dirs")
        await conn.execute("DROP TABLE _move_assistants")


def _configurable(row: dict[str, Any]) -> dict[str, Any] | None:
    config = row.get("config")
    configurable = config.get("configurable") if isinstance(config, dict) else None
    return configurable if isinstance(configurable, dict) else None


def _stored_pairs(rows: list[Any], *, payload: bool = False) -> set[tuple]:
    """The (workspace_dir, run_dir) pairs stored in *rows*' metadata and run
    config (of the payload, for crons)."""
    pairs: set[tuple] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        found = [metadata_of(row), _configurable(row)]
        source = row.get("payload") if payload else None
        if isinstance(source, dict):
            context = source.get("context")
            found = [
                metadata_of(row),
                metadata_of(source),
                _configurable(source),
                context if isinstance(context, dict) else None,
            ]
        for fields in found:
            if fields:
                pairs.add((fields.get("workspace_dir"), fields.get("run_dir")))
    return pairs


def _move_fields(
    fields: dict[str, Any] | None, moved: dict[tuple, dict[str, str]]
) -> bool:
    if not fields:
        return False
    folders = moved.get((fields.get("workspace_dir"), fields.get("run_dir")))
    if folders is None:
        return False
    _set_folders(fields, folders)
    return True


async def move_threads(move: Move) -> int:
    """Re-point the runtime's own copy of the old folder's threads, in
    place, so they keep their place in the lists: their folders in metadata
    and config (a run takes the thread's ``workspace_dir`` over its
    assistant's, and a read builds its graph from it) and the assistant in
    metadata (the WebUI's thread search filters on it; the restore sets it
    only when absent). Found by those fields, not by the checkpoints, so a
    thread whose checkpoints moved before a crash is found again. Returns
    the number of threads changed."""
    from langgraph_runtime.database import connect

    async with connect() as conn:
        pairs = _stored_pairs(list(conn.store.get("threads") or []))
    moved = await asyncio.to_thread(_moved_pairs, move, pairs)
    changed = 0
    # No await from here on: the edits are atomic on the event loop.
    async with connect() as conn:
        for thread in conn.store.get("threads") or []:
            if not isinstance(thread, dict):
                continue
            metadata = metadata_of(thread)
            edited = _move_fields(metadata, moved)
            edited |= _move_fields(_configurable(thread), moved)
            new_assistant = move.assistants.get(str(metadata.get("assistant_id")))
            if new_assistant is not None:
                metadata["assistant_id"] = new_assistant
                edited = True
            changed += edited
    return changed


async def move_crons(move: Move) -> tuple[int, int]:
    """Re-point the old folder's scheduled tasks to the new folder's
    assistants, in place: the assistant (on the cron and in its payload),
    the run config and the tag. Each keeps its ID, its enabled state and its
    next run. Its AutoSkills and proactive crons are deleted instead: their
    reconciles create the new folder's. Returns (tasks moved, system crons
    deleted)."""
    from langgraph_runtime.database import connect

    async with connect() as conn:
        pairs = _stored_pairs(list(conn.store.get("crons") or []), payload=True)
    moved = await asyncio.to_thread(_moved_pairs, move, pairs)
    system: list[Any] = []
    tasks = 0
    # No await until the deletes: the edits are atomic on the event loop.
    async with connect() as conn:
        for cron in conn.store.get("crons") or []:
            if not isinstance(cron, dict):
                continue
            new_assistant = move.assistants.get(str(cron.get("assistant_id")))
            if new_assistant is None:
                continue
            if is_system_cron(cron):
                system.append(cron.get("cron_id"))
                continue
            cron["assistant_id"] = uuid.UUID(new_assistant)
            _move_fields(metadata_of(cron), moved)
            payload = cron.get("payload")
            if isinstance(payload, dict):
                payload["assistant_id"] = new_assistant
                _move_fields(metadata_of(payload), moved)
                _move_fields(_configurable(payload), moved)
                context = payload.get("context")
                if isinstance(context, dict):
                    _move_fields(context, moved)
            tasks += 1
    if system:
        from .server_crons import delete_crons

        await delete_crons(system)
    return tasks, len(system)


async def drop_import_marks(move: Move) -> None:
    """Forget that the old store in the old and the new folder was read: the
    new folder's next setup reads the moved file once more, as its own, so
    its threads get their old metadata and what it holds for projects not
    set up yet waits again. Imported tasks stay marked: none comes back."""
    from langgraph_runtime.database import connect

    from .legacy_import import OLD_STORE, _marks

    async with connect() as conn:
        marks = _marks(conn.store)
        for workspace in (move.old, move.new):
            marks["files"].pop(str(workspace.root / OLD_STORE), None)
        if move.old.key in marks["waiting"]:
            marks["waiting"].remove(move.old.key)


@dataclass
class MemoryMove:
    """What :func:`move_memory` did."""

    moved: list[str]
    copied: list[str]


def _merge_folder(source: Path, target: Path, *, keep: bool) -> None:
    """Give *target* every file of *source* it lacks; remove *source*
    unless *keep*. Renamed whole when *target* does not exist yet."""
    if not source.is_dir():
        return
    if not keep and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        return
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        destination = target / path.relative_to(source)
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Published whole: a copy cut short must not pass for a copied file.
        partial = destination.with_name(f".{destination.name}.moving")
        shutil.copy2(path, partial)
        os.replace(partial, destination)
    if not keep:
        shutil.rmtree(source)


def _merge_profile(source: Path, target: Path, old_key: str, template: str) -> None:
    """Give *target* the old project's profile *source*: in place of a
    missing or untouched one, else after it."""
    if not source.is_file():
        return
    text = source.read_text(encoding="utf-8")
    if text.strip() == template.strip():
        return
    current = target.read_text(encoding="utf-8") if target.is_file() else None
    if current is None or current.strip() == template.strip():
        merged = text
    elif text.strip() in current:
        return
    else:
        merged = (
            f"{current.rstrip()}\n\n## Before the move from {old_key}\n\n"
            f"{text.strip()}\n"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # Replaced whole: a write cut short must not lose the new profile.
    partial = target.with_name(f".{target.name}.moving")
    partial.write_text(merged, encoding="utf-8")
    os.replace(partial, target)


def move_memory(
    move: Move,
    memory_dir: Path,
    *,
    old_ids: set[str],
    new_id: str,
    shared: set[str],
) -> MemoryMove:
    """Merge the memory projects *old_ids* into *new_id*, under the
    observation store's writer lock: the observations (a file name is a
    hash of its content, so a name the new project has is the same
    observation) and the profile (see :func:`_merge_profile`). A project in
    *shared* is still another registered folder's, so it is copied and kept;
    any other is removed once merged. Blocking."""
    from ..memory.observations.store import OBSERVATION_DIR, _record_guard
    from ..middleware.memory import PROFILE_TEMPLATES

    profile_path = "/profile/projects/{project_id}/PROJECT_PROFILE.md"
    template = PROFILE_TEMPLATES[profile_path]
    observations = memory_dir / OBSERVATION_DIR.lstrip("/") / "projects"
    profiles = memory_dir / "profile" / "projects"
    result = MemoryMove(moved=[], copied=[])
    with _record_guard(memory_dir):
        for old_id in sorted(old_ids - {new_id}):
            if not any(
                folder.joinpath(old_id).is_dir() for folder in (observations, profiles)
            ):
                continue
            keep = old_id in shared
            _merge_folder(observations / old_id, observations / new_id, keep=keep)
            if profiles.joinpath(new_id).exists():
                _merge_profile(
                    profiles / old_id / "PROJECT_PROFILE.md",
                    profiles / new_id / "PROJECT_PROFILE.md",
                    move.old.key,
                    template,
                )
            _merge_folder(profiles / old_id, profiles / new_id, keep=keep)
            (result.copied if keep else result.moved).append(old_id)
    return result


class MoveRefusedError(RuntimeError):
    """A move that cannot be made; :attr:`code` names why."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass
class MoveReport:
    """What a move did: the new folder, the threads (``sessions.db``) and
    scheduled tasks it re-pointed, the memory projects and proposals it
    moved, and what could not be done but does not stop the move."""

    workspace_dir: str
    threads: int
    tasks: int
    memory: list[str]
    proposals: int
    warnings: list[str]
    # It finished a move cut short: what moved before is not counted.
    resumed: bool = False


def same_key(first: str, second: str) -> bool:
    """Whether two folder keys name the same folder: on macOS and Windows,
    whose folders ignore case, keys that differ only in case do."""
    if sys.platform in ("darwin", "win32"):
        return first.casefold() == second.casefold()
    return first == second


def same_folder(first: str, second: str) -> bool:
    """Whether two folder keys name the same folder: by the folder itself
    when both exist (a case-sensitive macOS volume can hold ``Proj`` and
    ``proj``), else as :func:`same_key`. Blocking."""
    if first == second:
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return same_key(first, second)


def match_registered(old: Workspace, registered: list[str]) -> str:
    """The registered key *old* names: its own, else, on macOS and Windows,
    the one that differs only in case (a folder that is gone keeps the case
    it was typed in)."""
    if old.key in registered:
        return old.key
    found = [key for key in registered if same_key(key, old.key)]
    if len(found) == 1:
        return found[0]
    if found:
        raise MoveRefusedError(
            "ambiguous_old",
            f"{old.root} matches more than one registered project: " + ", ".join(found),
        )
    raise MoveRefusedError("not_registered", f"{old.root} is not a registered project.")


async def _busy(move: Move) -> bool:
    """Whether a run of the old folder is pending or running: on one of its
    threads, or on one of its assistants (a queued run has no thread yet)."""
    from langgraph_runtime.database import connect

    old_assistants = set(move.assistants)
    async with connect() as conn:
        threads = {
            thread.get("thread_id")
            for thread in conn.store.get("threads") or []
            if isinstance(thread, dict)
            and (
                metadata_of(thread).get("workspace_dir") == move.old.key
                or str(metadata_of(thread).get("assistant_id")) in old_assistants
            )
        }
        return any(
            isinstance(run, dict)
            and run.get("status") in ("pending", "running")
            and (
                run.get("thread_id") in threads
                or str(run.get("assistant_id")) in old_assistants
            )
            for run in conn.store.get("runs") or []
        )


def _check_folders(move: Move, *, resuming: bool) -> None:
    """Blocking."""
    if not move.new.root.is_dir():
        if resuming:
            raise MoveRefusedError(
                "new_missing",
                f"The move of {move.old.root} to {move.new.root} is not done, and "
                f"{move.new.root} no longer exists: put the folder back there and "
                "run the move again.",
            )
        raise MoveRefusedError("new_missing", f"{move.new.root} does not exist.")
    if resuming:
        return
    if move.new.key == move.old.key:
        raise MoveRefusedError("same_folder", "The two folders are the same.")
    # A rename that changes only case, on a folder that ignores case.
    if move.old.root.exists() and not same_folder(
        str(move.old.root), str(move.new.root)
    ):
        raise MoveRefusedError(
            "old_exists",
            f"{move.old.root} still exists, so this is a copy, not a move: add "
            f"{move.new.root} as a new project.",
        )


async def move_project(old: Workspace, new: Workspace) -> MoveReport:
    """Move the registered project *old* to the folder *new*, or finish its
    move when one was cut short; the caller holds
    :func:`~EvoScientist.langgraph_dev.workspace_setup.registration_lock`.

    Refused (:class:`MoveRefusedError`) when *new* does not exist, *old* is
    not a registered project, its folder still exists (a copy), a run of it
    is in progress, or another move is not done yet.
    """
    from .registry import REGISTRY

    pending = await asyncio.to_thread(REGISTRY.pending_move)
    if pending is None:
        registered = await asyncio.to_thread(REGISTRY.registered)
        try:
            old_key = match_registered(old, registered)
        except MoveRefusedError as exc:
            if exc.code != "not_registered":
                raise
            done = await asyncio.to_thread(REGISTRY.done_moves)
            for done_old, done_new in done:
                if same_key(done_old, old.key) and await asyncio.to_thread(
                    same_folder, done_new, new.key
                ):
                    # Done already: as the server started, or after a client
                    # gave up waiting. A default_workdir the config lock kept
                    # it from rewriting is tried again.
                    done_move = Move(Workspace(done_old), Workspace(done_new))
                    await asyncio.to_thread(_check_folders, done_move, resuming=True)
                    warning = await asyncio.to_thread(_move_default_workdir, done_move)
                    return MoveReport(
                        new.key, 0, 0, [], 0, [warning] if warning else [], resumed=True
                    )
            raise
        move = Move(Workspace(old_key), new)
        await asyncio.to_thread(_check_folders, move, resuming=False)
    else:
        try:
            same_old = match_registered(old, [pending[0]]) == pending[0]
        except MoveRefusedError:
            same_old = False
        if not same_old or not await asyncio.to_thread(
            same_folder, new.key, pending[1]
        ):
            raise MoveRefusedError(
                "move_pending",
                f"The move of {pending[0]} to {pending[1]} is not done yet: run "
                "it again to finish it.",
            )
        move = Move(Workspace(pending[0]), Workspace(pending[1]))
        await asyncio.to_thread(_check_folders, move, resuming=True)
    if await _busy(move):
        raise MoveRefusedError(
            "busy", f"A run of {move.old.root} is in progress; wait for it to end."
        )
    if pending is None:
        await asyncio.to_thread(REGISTRY.record_move, move.old.key, move.new.key)
    report = await _run(move)
    report.resumed = pending is not None
    return report


async def resume_pending_move() -> None:
    """At the server's start: finish a move that was cut short. Logged on
    failure; the next ``move`` call, or the next start, tries again."""
    from .registry import REGISTRY

    pending = await asyncio.to_thread(REGISTRY.pending_move)
    if pending is None:
        return
    move = Move(Workspace(pending[0]), Workspace(pending[1]))
    try:
        await asyncio.to_thread(_check_folders, move, resuming=True)
        report = await _run(move)
    except Exception:
        logger.warning(
            "Could not finish moving %s to %s; run the move again.",
            pending[0],
            pending[1],
            exc_info=True,
        )
    else:
        logger.info("Finished moving %s to %s.", pending[0], pending[1])
        for warning in report.warnings:
            logger.warning("%s", warning)


def _move_memory(move: Move) -> tuple[MemoryMove, dict[str, str]]:
    """Move the old folder's memory projects; with the map of old ids to
    the new one, for the proposals. Blocking."""
    from .. import paths
    from ..memory.autoskills.proposals import proposal_project_ids
    from ..memory.project import path_project_ids, resolve_project_id
    from .registry import REGISTRY

    memory_dir = paths.MEMORIES_DIR
    path_ids = path_project_ids(move.old.root)
    # A proposal's id names a folder: only the form ids take is read.
    proposal_ids = {
        project_id
        for project_id in proposal_project_ids(memory_dir, move.old.key)
        if _PROJECT_ID.fullmatch(project_id)
    }
    candidates = path_ids | proposal_ids
    new_id = resolve_project_id(move.new.root)
    # Every candidate, its folder moved already or not: a move resumed after
    # its memory step still maps the proposals.
    project_ids = {old_id: new_id for old_id in candidates if old_id != new_id}
    old_ids = {
        project_id
        for project_id in candidates
        if any(
            (memory_dir / kind / "projects" / project_id).is_dir()
            for kind in ("observations", "profile")
        )
    }
    if not old_ids:
        return MemoryMove(moved=[], copied=[]), project_ids
    # An id only a proposal names may be a repository's remote id, which its
    # other clones share: it is copied, never removed.
    shared = (proposal_ids - path_ids) | {
        project_id
        for key in REGISTRY.registered()
        if key not in (move.old.key, move.new.key)
        and Path(key).is_dir()
        and (project_id := resolve_project_id(key)) in old_ids
    }
    moved = move_memory(move, memory_dir, old_ids=old_ids, new_id=new_id, shared=shared)
    return moved, project_ids


async def _delete_assistants(workspace: Workspace) -> None:
    """Delete *workspace*'s workspace assistants; one already gone is
    skipped."""
    from langgraph_runtime.database import connect
    from langgraph_runtime.ops import Assistants
    from starlette.exceptions import HTTPException

    from .assistants import workspace_assistant_ids

    async with connect() as conn:
        for assistant_id in workspace_assistant_ids(workspace):
            try:
                async for _ in await Assistants.delete(conn, uuid.UUID(assistant_id)):
                    pass
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise


def _move_default_workdir(move: Move) -> str | None:
    """Point ``default_workdir`` in the configuration file at the new
    folder when it names the old one, so the next launch registers the new
    folder; why it could not, or None. Blocking."""
    from ..config import store
    from ..config.settings import set_config_values

    try:
        value = store.read_config_mapping().get("default_workdir")
        if not isinstance(value, str) or not value.strip():
            return None
        if Workspace(value.strip()).key != move.old.key:
            return None
        set_config_values({"default_workdir": str(move.new.root)})
    except Exception as exc:
        return f"default_workdir still names {move.old.root}: {exc}"
    return None


async def _run(move: Move) -> MoveReport:
    """Every step of a move, in order, each re-pointing only what still
    names the old folder:

    1. the old store import's marks, before the new folder's setup reads
       the moved store;
    2. the threads, before anything runs on the new folder's assistants (a
       run into a thread still bound to the old folder is refused);
    3. the new folder's setup: its assistants, registration, threads and
       AutoSkills cron;
    4. the scheduled tasks, now that the new assistants exist;
    5. the memory, before the proposals, which name its old ids;
    6. the old folder's assistants, after every task left them (deleting
       one deletes its tasks), then its registry row and ``default_workdir``.
    """
    from .. import paths, sessions
    from ..memory.autoskills.proposals import move_proposals
    from . import workspace_setup
    from .registry import REGISTRY

    await drop_import_marks(move)
    threads = await move_checkpoints(move)
    await move_threads(move)
    await workspace_setup.register_workspace(SessionDirs(move.new))
    tasks, _system = await move_crons(move)
    memory, project_ids = await asyncio.to_thread(_move_memory, move)
    proposals = await asyncio.to_thread(
        move_proposals, paths.MEMORIES_DIR, move.old.key, move.new.key, project_ids
    )
    await _delete_assistants(move.old)
    await asyncio.to_thread(REGISTRY.forget, move.old.key)
    workspace_setup.SET_UP.discard(move.old.key)
    sessions.RESTORED_WORKSPACES.discard(move.old.key)
    warning = await asyncio.to_thread(_move_default_workdir, move)
    await asyncio.to_thread(REGISTRY.finish_move, move.old.key)
    return MoveReport(
        workspace_dir=move.new.key,
        threads=threads,
        tasks=tasks,
        memory=memory.moved + memory.copied,
        proposals=proposals,
        warnings=[warning] if warning else [],
    )
