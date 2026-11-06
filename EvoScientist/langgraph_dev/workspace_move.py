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
import shutil
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
    transaction; returns the number of rows changed."""
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
            "SELECT COUNT(*) FROM checkpoints WHERE json_valid(metadata) "
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
        shutil.copy2(path, destination)
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
    target.write_text(merged, encoding="utf-8")


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
