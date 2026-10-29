"""The import of the projects' old runtime stores.

Before one server served every project, each project's server kept its
scheduled tasks and its threads' metadata (renames, pins, the proactive arm
flag, ...) in ``<project>/.langgraph_api/.langgraph_ops.pckl``. A server
started in one folder also served the sessions it was pointed at elsewhere
(``--workdir``), so a file can hold other projects' tasks too. The server
reads a project's file at the project's setup, once its threads are
restored:

- every cron but the AutoSkills and proactive ones (their reconciles recreate
  those) goes, under its old ID, on the workspace assistant for its graph of
  the project it belongs to: the one its thread is bound to, else the one it
  is tagged with, else the file's project (untagged, tagged with one of its
  run folders, or with a folder that is gone: the old server's rule). A
  cron of a project not set up yet waits: every setup reads again the files
  that hold such crons, and imports those that can go somewhere now;
- each thread the restore brought back bound to the file's project gets the
  old metadata keys it lacks.

A cron is marked once it is imported, or reported as one that cannot be
(but one without an ID, reported at each first read of its file): it is
never imported again, so a task the user deleted does not come back
from a moved or copied folder, or from a file the previous release saved
again. The file is the source of what is left: a waiting cron the file no
longer holds is not imported. A file is not read again for its own project
until it changes (its size or modification time). The file itself is only
read, never changed; one that cannot be read for now (an I/O error) is read
at the next setup.

The marks live in the runtime's store, next to what the import put there
(:data:`IMPORTED_KEY`), so the two are written together: a store that is
lost, or never written (``langgraph_dev_file_persistence: false``), loses the
marks with the imported crons, and the next start imports again. They hold
only strings and numbers, so the runtime's store never depends on this
module to load. ``tests/test_legacy_import.py`` checks that the runtime's
startup loader keeps the key.
"""

from __future__ import annotations

import asyncio
import functools
import io
import logging
import os
import pickle
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..cron.schedule import belongs_to, cron_request
from ..paths import PATH_ERRORS, SessionDirs, Workspace, normalize_path
from . import registry
from .assistants import served_graph_ids
from .folder_check import is_served
from .server_crons import is_system_cron, metadata_of, request_payload

logger = logging.getLogger(__name__)

OLD_STORE = Path(".langgraph_api") / ".langgraph_ops.pckl"
# The runtime store's key for the import's marks (see :func:`_marks`).
IMPORTED_KEY = "evoscientist_legacy_import"

# The payload fields of a cron a client can set (the HTTP route's request
# body), carried over as they were; the assistant, config, context, metadata
# and schedule are set here.
_PAYLOAD_FIELDS = (
    "interrupt_before",
    "interrupt_after",
    "webhook",
    "stream_mode",
    "stream_subgraphs",
    "stream_resumable",
    "checkpoint_during",
)
# Where a thread works and on what: the restore sets these from the
# checkpoints, which win.
_BINDING_KEYS = frozenset({"graph_id", "workspace_dir", "run_dir", "assistant_id"})


class UnreadableStoreError(RuntimeError):
    """The old store exists but cannot be loaded."""


@dataclass
class ImportReport:
    """What one read of a file imported."""

    crons: int = 0
    threads: int = 0
    skipped: list[str] = field(default_factory=list)
    # Project key -> how many of its crons wait for its setup.
    waiting: dict[str, int] = field(default_factory=dict)


class _Gone:
    """What the old store holds of a class that cannot be loaded any more
    (in a run's or a thread's values, which the import does not read): it
    takes whatever the pickle gives it, a container's items too."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __setstate__(self, _state: Any) -> None:
        pass

    def __setitem__(self, _key: Any, _value: Any) -> None:
        pass

    def append(self, _item: Any) -> None:
        pass

    def extend(self, _items: Any) -> None:
        pass


class _GoneFinder(pickle.Pickler):
    """Pickles a value as the runtime's store would, noting whether it
    meets anything of a class gone since."""

    found = False

    def persistent_id(self, obj: Any) -> str | None:
        if obj is _Gone or isinstance(obj, _Gone):
            self.found = True
            return "gone"
        return None


def _holds_gone(value: Any) -> bool:
    """Whether *value* holds something of a class gone since, at any depth,
    or cannot be pickled: it cannot go into the runtime's store, which would
    then need this module to load, or fail to save."""
    # The protocol the runtime saves its store with.
    finder = _GoneFinder(io.BytesIO(), protocol=2)
    try:
        finder.dump(value)
    except Exception:
        return True
    return finder.found


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except Exception:
            # Gone since, or its module fails to import here.
            return _Gone


def read_old_store(
    root: Path, known: list[int] | None = None
) -> tuple[list[int], bytes] | None:
    """The signature of *root*'s old store (its size and modification time,
    which change when it is saved again) and its bytes; None without one, or
    when its signature is *known* (it is not read then). Blocking."""
    try:
        with (root / OLD_STORE).open("rb") as file:
            stat = os.fstat(file.fileno())
            signature = [stat.st_size, stat.st_mtime_ns]
            if signature == known:
                return None
            return signature, file.read()
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return None


def load_old_store(path: Path, data: bytes) -> dict[str, list[Any]]:
    """The crons, threads and assistants of the old store *data*, read from
    *path*.

    Raises :class:`UnreadableStoreError` for data that does not load, or
    loads into something else than what the old server saved.
    """
    try:
        store = _Unpickler(io.BytesIO(data)).load()
    except Exception as exc:
        raise UnreadableStoreError(f"{path}: {exc!r}") from exc
    if not isinstance(store, dict):
        raise UnreadableStoreError(f"{path}: it holds no store")
    old: dict[str, list[Any]] = {}
    for kind in ("crons", "threads", "assistants"):
        rows = store.get(kind) or []
        if not isinstance(rows, list):
            raise UnreadableStoreError(f"{path}: its {kind} are not a list")
        old[kind] = rows
    return old


def _marks(store: Any) -> dict[str, Any]:
    """The import's marks in the runtime *store*, created if need be:

    - ``crons``: the IDs of the crons imported, or reported as ones that
      cannot be;
    - ``files``: old store path -> its signature when it was read for its
      project;
    - ``waiting``: the keys of the projects whose files hold crons waiting
      for another project's setup.
    """
    marks = store.get(IMPORTED_KEY)
    if not isinstance(marks, dict):
        marks = store[IMPORTED_KEY] = {}
    marks.setdefault("crons", [])
    marks.setdefault("files", {})
    marks.setdefault("waiting", [])
    return marks


def _set_waiting(marks: dict[str, Any], key: str, waiting: bool) -> None:
    if waiting and key not in marks["waiting"]:
        marks["waiting"].append(key)
    elif not waiting and key in marks["waiting"]:
        marks["waiting"].remove(key)


async def import_once(workspace: Workspace) -> ImportReport | None:
    """At *workspace*'s setup: import what its old store has left to import,
    then what the files holding waiting crons can give now; the report of
    *workspace*'s file, or None when it was not read. Nothing runs before
    the project's threads are restored. Raises when its file's import
    failed (what it put before is marked; the project's next setup tries
    again)."""
    from .. import sessions

    # Thread-bound crons and thread metadata need the restored threads.
    if workspace.key not in sessions.RESTORED_WORKSPACES:
        return None
    # Its own file is read again only when this setup did not read it.
    skip: str | None = workspace.key
    try:
        report = await _import_file(workspace)
        if report is None:
            skip = None
    finally:
        await _import_waiting(workspace, skip=skip)
    return report


async def _import_file(workspace: Workspace) -> ImportReport | None:
    """Import what *workspace*'s old store has left to import; the report,
    or None when it was not read (none, or unchanged since it was)."""
    from langgraph_runtime.database import connect

    path = workspace.root / OLD_STORE
    async with connect() as conn:
        known = _marks(conn.store)["files"].get(str(path))
    read = await asyncio.to_thread(read_old_store, workspace.root, known)
    if read is None:
        return None
    signature, data = read
    # A pickle: loading one runs what it holds, as the per-project server
    # that loaded it at its start before this release did.
    logger.warning(
        "Importing %s, the store of the server that ran in this folder "
        "before this release (a pickle, loaded as that server loaded it).",
        path,
    )
    try:
        old = await asyncio.to_thread(load_old_store, path, data)
    except UnreadableStoreError as exc:
        report = ImportReport(skipped=[str(exc)])
    else:
        report = await _import(workspace, old, workspace, first_read=True)
    async with connect() as conn:
        marks = _marks(conn.store)
        marks["files"][str(path)] = signature
        _set_waiting(marks, workspace.key, bool(report.waiting))
    _log(workspace, report)
    for owner, count in report.waiting.items():
        logger.info(
            "%d scheduled task(s) in the old store of %s wait for %s: they "
            "are imported once EvoScientist is used there.",
            count,
            workspace.root,
            owner,
        )
    return report


async def _import_waiting(current: Workspace, *, skip: str | None) -> None:
    """At *current*'s setup, read again the files that hold waiting crons
    (but *skip*'s) and import those that can go somewhere now."""
    from langgraph_runtime.database import connect

    async with connect() as conn:
        holders = [key for key in _marks(conn.store)["waiting"] if key != skip]
    for key in holders:
        try:
            await _import_held(key, current)
        except Exception:
            logger.warning(
                "Could not import the waiting scheduled tasks of the old store "
                "of %s; the next setup tries again.",
                key,
                exc_info=True,
            )


async def _import_held(key: str, current: Workspace) -> None:
    from langgraph_runtime.database import connect

    holder = await asyncio.to_thread(Workspace, key)
    path = holder.root / OLD_STORE
    read = await asyncio.to_thread(read_old_store, holder.root)
    report = None
    if read is None:
        logger.warning(
            "%s is gone: the scheduled tasks it held for other projects are "
            "not imported.",
            path,
        )
    else:
        try:
            old = await asyncio.to_thread(load_old_store, path, read[1])
        except UnreadableStoreError as exc:
            logger.warning(
                "The scheduled tasks %s held for other projects are not imported: %s",
                path,
                exc,
            )
        else:
            report = await _import(holder, old, current, first_read=False)
    async with connect() as conn:
        _set_waiting(_marks(conn.store), key, bool(report and report.waiting))
    if report is not None:
        _log(holder, report)


def _log(holder: Workspace, report: ImportReport) -> None:
    if report.crons or report.threads:
        logger.info(
            "Imported %d scheduled task(s) and the metadata of %d thread(s) "
            "from the old store of %s.",
            report.crons,
            report.threads,
            holder.root,
        )
    for reason in report.skipped:
        logger.warning("Not imported from the old store of %s: %s", holder.root, reason)


def _graph_of(assistants: list[Any]) -> dict[str, Any]:
    """Assistant ID -> graph, from the old store's assistants and, for a
    snapshot without them, the fixed IDs of the graphs' own assistants."""
    from langgraph_api.graph import NAMESPACE_GRAPH

    graphs = {
        str(uuid.uuid5(NAMESPACE_GRAPH, graph_id)): graph_id
        for graph_id in served_graph_ids()
    }
    for assistant in assistants:
        if isinstance(assistant, dict) and assistant.get("graph_id"):
            graphs[str(assistant.get("assistant_id"))] = assistant["graph_id"]
    return graphs


def _cron_id(cron: Any) -> str | None:
    """*cron*'s ID, or None when it has none the runtime's routes can read
    (a UUID)."""
    cron_id = cron.get("cron_id") if isinstance(cron, dict) else None
    if isinstance(cron_id, uuid.UUID):
        return str(cron_id)
    if not isinstance(cron_id, str):
        return None
    try:
        return str(uuid.UUID(cron_id))
    except ValueError:
        return None


def _skip(cron: Any, exc: Exception) -> str:
    name = metadata_of(cron).get("name")
    if not isinstance(name, str) or not name:
        name = _cron_id(cron) or "without a name or ID"
    detail = getattr(exc, "detail", None) or exc
    return f"scheduled task {name}: {detail}"


def _threads(conn: Any) -> dict[str, dict[str, Any]]:
    return {str(t.get("thread_id")): t for t in conn.store["threads"]}


async def _import(
    holder: Workspace,
    old: dict[str, list[Any]],
    current: Workspace,
    *,
    first_read: bool,
) -> ImportReport:
    """Import *holder*'s old store *old* at *current*'s setup: each cron
    whose project can take it now (*current*, or one set up with its
    threads restored), and at the *first_read* of the file for its project,
    its threads' metadata."""
    from langgraph_runtime.database import connect

    from .. import sessions
    from .workspace_setup import SET_UP

    def ready(owner: Workspace) -> bool:
        # Its assistants are created, and its threads restored, at its setup.
        return owner.key == current.key or (
            owner.key in SET_UP and owner.key in sessions.RESTORED_WORKSPACES
        )

    report = ImportReport()
    graph_of = _graph_of(old["assistants"])
    async with connect() as conn:
        done = set(_marks(conn.store)["crons"])
        threads = _threads(conn)
    bound = {
        thread_id: metadata_of(thread).get("workspace_dir")
        for thread_id, thread in threads.items()
    }
    crons = []
    unreadable = []
    for cron in old["crons"]:
        cron_id = _cron_id(cron)
        # The AutoSkills and proactive crons are left to their reconciles.
        if is_system_cron(cron) or cron_id in done:
            continue
        if cron_id is None:
            # Without one, each read would import it again.
            if first_read:
                report.skipped.append(_skip(cron, _NotImported("it has no ID")))
        elif _unreadable(cron):
            report.skipped.append(
                _skip(cron, _NotImported("it holds data this version cannot read"))
            )
            unreadable.append(cron_id)
        else:
            crons.append(cron)
    old_threads = old["threads"] if first_read else []
    owners, merged = await asyncio.to_thread(
        _resolve, holder, crons, bound, old_threads
    )
    async with connect() as conn:
        marks = _marks(conn.store)
        marks["crons"].extend(unreadable)
        for cron, owner in zip(crons, owners, strict=True):
            cron_id = _cron_id(cron)
            if owner is None:
                label = f"the project of thread {cron.get('thread_id')}"
                report.waiting[label] = report.waiting.get(label, 0) + 1
                continue
            if isinstance(owner, Workspace) and not ready(owner):
                report.waiting[owner.key] = report.waiting.get(owner.key, 0) + 1
                continue
            try:
                if not isinstance(owner, Workspace):
                    raise _NotImported(owner)
                graph_id = graph_of.get(str(cron.get("assistant_id")))
                await _import_cron(conn, owner, cron, graph_id, threads)
            except Exception as exc:
                # One cron that cannot be imported (malformed, a thread the
                # restore did not bring back, a bad schedule) is reported
                # once; the others go on.
                report.skipped.append(_skip(cron, exc))
            else:
                report.crons += 1
            marks["crons"].append(cron_id)
        for old_thread in old_threads:
            if not isinstance(old_thread, dict):
                report.skipped.append("a thread entry that is not one")
                continue
            try:
                if _merge_thread_metadata(merged, old_thread, threads):
                    report.threads += 1
            except Exception as exc:
                thread_id = (
                    old_thread.get("thread_id")
                    if isinstance(old_thread, dict)
                    else None
                )
                report.skipped.append(f"thread {thread_id}: {exc}")
    return report


def _resolve(
    holder: Workspace,
    crons: list[Any],
    bound: dict[str, Any],
    old_threads: list[Any],
) -> tuple[list[Workspace | str | None], set[str]]:
    """The owners of *crons* (see :func:`_owners`), and the projects whose
    restored threads take the metadata *old_threads* give them: *holder*, and
    the others *holder*'s server served that have no old store of their own
    (one that has reads its own, maybe newer). Blocking."""
    owners = _owners(holder, crons, bound)
    projects = {
        bound.get(str(old_thread.get("thread_id")))
        for old_thread in old_threads
        if isinstance(old_thread, dict)
    }
    merged = {holder.key}
    for project in projects - {holder.key, None}:
        # Only a folder that is there tells it has no file of its own.
        if (
            isinstance(project, str)
            and os.path.isdir(project)
            and not os.path.isfile(Path(project) / OLD_STORE)
        ):
            merged.add(project)
    return owners, merged


def _owners(
    holder: Workspace, crons: list[Any], bound: dict[str, Any]
) -> list[Workspace | str | None]:
    """For each of *crons* of *holder*'s file, the project it belongs to,
    why it belongs to none, or None when that is not known yet; *bound* maps
    the restored threads' IDs to their projects. Blocking."""
    owners: list[Workspace | str | None] = []
    for cron in crons:
        thread_id = cron.get("thread_id") if isinstance(cron, dict) else None
        thread_project = bound.get(str(thread_id)) if thread_id is not None else None
        tag = metadata_of(cron).get("workspace_dir")
        if thread_id is not None and not thread_project and not tag:
            # Untagged, on a thread of a project not restored yet (one the
            # old server served from here).
            owners.append(None)
            continue
        try:
            if isinstance(thread_project, str) and thread_project:
                owner = Workspace(thread_project)
                if owner.key != holder.key and not os.path.isdir(owner.root):
                    owners.append(f"its thread's project {owner.key} is gone")
                    continue
            elif not tag:
                owner = holder
            else:
                owner = _tagged_project(holder, tag)
        except PATH_ERRORS:
            named = (
                f"its thread's project {thread_project!r}"
                if isinstance(thread_project, str) and thread_project
                else f"its project {tag!r}"
            )
            owners.append(f"{named} is not a folder on this machine")
            continue
        owners.append(owner)
    return owners


def _tagged_project(holder: Workspace, tag: Any) -> Workspace:
    """The project of a cron of *holder*'s file tagged *tag* (relative to
    the folder the old server ran in). Blocking."""
    path = normalize_path(holder.root / tag)
    # A generated run folder stored as the workspace names its project.
    tagged = SessionDirs.from_legacy(path).workspace
    if tagged.key != holder.key and is_served(tagged, registry.REGISTRY):
        return tagged
    # The old server ran in *holder* and served its runs.
    if tagged.key == holder.key or holder.is_run_folder(path):
        return holder
    # A named run folder of a registered project; a project may itself
    # live in a folder called ``runs``, so only of a registered one.
    runs = tagged.root.parent
    if (
        tagged.root == path
        and runs.name == "runs"
        and is_served(Workspace(runs.parent), registry.REGISTRY)
    ):
        return Workspace(runs.parent)
    # The old server's rule: a tag whose folder is gone (or cannot be
    # looked at) is the file's.
    if not os.path.isdir(tagged.root):
        return holder
    return tagged


class _NotImported(Exception):
    """Why a cron is not imported."""


async def _import_cron(
    conn: Any,
    workspace: Workspace,
    cron: Any,
    graph_id: Any,
    threads: dict[str, dict[str, Any]],
) -> None:
    """Put *cron* on *workspace*'s assistant for *graph_id*, or raise why it
    is not."""
    from langgraph_api.validation import RESERVED_CONFIGURABLE_KEYS
    from langgraph_runtime.ops import Crons

    if graph_id not in served_graph_ids():
        raise _NotImported(f"its graph {graph_id!r} is not served")
    thread_id = cron.get("thread_id")
    if thread_id is not None:
        thread = threads.get(str(thread_id))
        if thread is None:
            raise _NotImported(f"its thread {thread_id} was not restored")
    old_payload = dict(cron.get("payload") or {})
    old_config = dict(old_payload.get("config") or {})
    # A run takes either; the project's folders go in ``configurable``.
    configurable = {
        **(old_payload.get("context") or {}),
        **(old_config.pop("configurable", None) or {}),
    }
    # Keys the runtime keeps for itself, the old run user among them: a
    # client may not set them.
    for key in ("workspace_dir", "run_dir", "cron_id", *RESERVED_CONFIGURABLE_KEYS):
        configurable.pop(key, None)
    assistant_id, fields = cron_request(
        workspace=workspace,
        graph_id=graph_id,
        schedule=cron["schedule"],
        input=old_payload.get("input"),
        metadata={
            key: value
            for key, value in metadata_of(cron).items()
            if key not in _excluded_metadata_keys()
        },
        timezone=cron.get("timezone"),
        multitask_strategy=old_payload.get("multitask_strategy"),
        durability=old_payload.get("durability"),
    )
    # The old run config is kept; the project's folders replace the old ones.
    fields["config"] = {
        **old_config,
        "configurable": {**configurable, **fields["config"]["configurable"]},
    }
    payload = request_payload(assistant_id, fields)
    for key in _PAYLOAD_FIELDS:
        if key in old_payload and key not in payload:
            payload[key] = old_payload[key]
    put = [
        row
        async for row in await Crons.put(
            conn,
            payload=payload,
            schedule=cron["schedule"],
            cron_id=cron.get("cron_id"),
            thread_id=thread_id,
            on_run_completed=cron.get("on_run_completed"),
            end_time=cron.get("end_time"),
            metadata=fields["metadata"],
            enabled=bool(cron.get("enabled", True)),
            timezone=cron.get("timezone"),
        )
    ]
    if not put:
        raise _NotImported("the runtime did not create it")
    # A cron of that ID that is already there is returned as it is.
    if not belongs_to(put[0], workspace):
        raise _NotImported("a scheduled task of that ID belongs to another project")


def _unreadable(cron: dict[str, Any]) -> bool:
    """Whether *cron* holds something of a class gone since, but in the
    runtime's own keys of its run config, which the import drops (the old
    run user, say, whose class a later version may move)."""
    from langgraph_api.validation import RESERVED_CONFIGURABLE_KEYS

    return _holds_gone(_without_reserved(cron, RESERVED_CONFIGURABLE_KEYS))


def _without_reserved(cron: dict[str, Any], reserved: tuple[str, ...]) -> Any:
    """*cron* without the *reserved* keys of its run config."""
    payload = cron.get("payload")
    config = payload.get("config") if isinstance(payload, dict) else None
    configurable = config.get("configurable") if isinstance(config, dict) else None
    if not isinstance(configurable, dict):
        return cron
    kept = {key: value for key, value in configurable.items() if key not in reserved}
    return {**cron, "payload": {**payload, "config": {**config, "configurable": kept}}}


@functools.cache
def _excluded_metadata_keys() -> frozenset[str]:
    """The metadata keys an import leaves out: the bindings, which the
    import or the restore sets, and the runtime's own keys, which are not a
    client's to set either."""
    from langgraph_api.validation import RESERVED_METADATA_KEYS

    return _BINDING_KEYS.union(RESERVED_METADATA_KEYS)


def _merge_thread_metadata(
    projects: set[str],
    old_thread: dict[str, Any],
    threads: dict[str, dict[str, Any]],
) -> bool:
    """Give *old_thread*, when the restore brought it back bound to one of
    *projects*, the old metadata keys it lacks; whether it got any. In
    place, as the restore edits the registry, so a thread keeps its place in
    the lists (an update would move it to the top). Merging again changes
    nothing."""
    thread = threads.get(str(old_thread.get("thread_id")))
    if thread is None:
        return False
    # Edited in place.
    metadata = metadata_of(thread)
    # A copied folder's old store also names the original's threads, maybe
    # older.
    if metadata.get("workspace_dir") not in projects:
        return False
    old_metadata = old_thread.get("metadata")
    if old_metadata is _Gone or isinstance(old_metadata, _Gone):
        raise _NotImported("its metadata holds data this version cannot read")
    excluded = _excluded_metadata_keys()
    extra = {
        key: value
        for key, value in metadata_of(old_thread).items()
        if key not in excluded and key not in metadata and not _holds_gone((key, value))
    }
    metadata.update(extra)
    return bool(extra)
