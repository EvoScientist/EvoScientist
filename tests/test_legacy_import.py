"""The import of a project's old runtime store: its scheduled tasks and its
threads' metadata, once."""

from __future__ import annotations

import pickle
import shutil
import sys
import uuid
from datetime import UTC, datetime

import pytest

from EvoScientist import sessions
from EvoScientist.langgraph_dev import legacy_import, registry, workspace_setup
from EvoScientist.langgraph_dev.assistants import (
    ensure_workspace_assistants,
    workspace_assistant_id,
)
from EvoScientist.langgraph_dev.server_crons import is_system_cron
from EvoScientist.paths import Workspace

# The system assistants of the old store, as langgraph-api names them.
SCHEDULER = "3976a1c4-465e-5da3-a466-abd08ac6515a"
AUTOSKILLS = "6404d745-46b2-51e4-a928-b89b155f4d6a"


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    project = Workspace(tmp_path / "project")
    project.root.mkdir()
    registry.REGISTRY.set_registered(project.key, True)
    # The restore brought the project's threads back.
    sessions.RESTORED_WORKSPACES.add(project.key)
    return project


def _marked(project: Workspace) -> bool:
    """Whether *project*'s old store has nothing left to import."""
    from langgraph_runtime_inmem.database import GLOBAL_STORE

    marks = GLOBAL_STORE.get(legacy_import.IMPORTED_KEY) or {}
    return str(project.root / legacy_import.OLD_STORE) in marks.get("files", {})


def _project(path) -> Workspace:
    """A project set up in this server process."""
    project = Workspace(path)
    project.root.mkdir()
    registry.REGISTRY.set_registered(project.key, True)
    sessions.RESTORED_WORKSPACES.add(project.key)
    workspace_setup.SET_UP.add(project.key)
    return project


def _assistants(runtime_store) -> dict[str, str]:
    """Each scheduled task's assistant; a setup also adds AutoSkills crons."""
    return {
        str(c["cron_id"]): str(c["assistant_id"])
        for c in runtime_store["crons"]
        if not is_system_cron(c)
    }


def _cron(cron_id, *, assistant=SCHEDULER, run_kind="scheduled_task", **extra):
    return {
        "cron_id": uuid.UUID(cron_id),
        "assistant_id": uuid.UUID(assistant),
        "thread_id": None,
        "schedule": "0 9 * * *",
        "timezone": "UTC",
        "enabled": True,
        "end_time": None,
        "on_run_completed": "keep",
        "metadata": {"run_kind": run_kind, "name": f"task {cron_id[:4]}"},
        "payload": {
            "assistant_id": assistant,
            "input": {"messages": [{"role": "user", "content": "hi"}]},
            "config": {"configurable": {"workspace_dir": "/old", "cron_id": cron_id}},
            "metadata": {"run_kind": run_kind},
            "schedule": "0 9 * * *",
        },
        **extra,
    }


def _write_old_store(root, *, crons=(), threads=()) -> None:
    from langgraph.checkpoint.memory import PersistentDict

    path = root / legacy_import.OLD_STORE
    path.parent.mkdir(parents=True, exist_ok=True)
    store = PersistentDict(dict, filename=str(path))
    store["crons"] = list(crons)
    store["threads"] = list(threads)
    store["assistants"] = [
        {"assistant_id": uuid.UUID(SCHEDULER), "graph_id": "scheduler"},
        {"assistant_id": uuid.UUID(AUTOSKILLS), "graph_id": "evomemory-autoskills"},
    ]
    store.sync()


def _restored_thread(thread_id, workspace, **metadata):
    now = datetime.now(UTC)
    return {
        "thread_id": uuid.UUID(thread_id),
        "created_at": now,
        "updated_at": now,
        "metadata": {
            "graph_id": "EvoScientist",
            "workspace_dir": workspace.key,
            **metadata,
        },
        "status": "idle",
        "config": {},
        "values": None,
    }


# ---------------------------------------------------------------------------
# Reading the old store
# ---------------------------------------------------------------------------


def test_read_old_store_without_one_is_none(tmp_path):
    assert legacy_import.read_old_store(tmp_path) is None


def test_read_old_store_known_signature_is_not_read(tmp_path):
    _write_old_store(tmp_path)
    signature, _data = legacy_import.read_old_store(tmp_path)

    assert legacy_import.read_old_store(tmp_path, signature) is None


def test_load_old_store_reads_saved_store(langgraph_api_env, tmp_path):
    _write_old_store(tmp_path, crons=[_cron("11111111-1111-1111-1111-111111111111")])
    path = tmp_path / legacy_import.OLD_STORE

    _signature, data = legacy_import.read_old_store(tmp_path)

    old = legacy_import.load_old_store(path, data)

    assert len(old["crons"]) == 1
    assert len(old["assistants"]) == 2


def test_load_old_store_empty_store_has_nothing(tmp_path):
    old = legacy_import.load_old_store(tmp_path, pickle.dumps({}))

    assert old == {"crons": [], "threads": [], "assistants": []}


@pytest.mark.parametrize(
    "data",
    [b"\x80\x05", pickle.dumps([]), pickle.dumps({"crons": "not a list"})],
    ids=["truncated", "not-a-dict", "not-a-list"],
)
def test_load_old_store_unexpected_data_is_unreadable(tmp_path, data):
    with pytest.raises(legacy_import.UnreadableStoreError):
        legacy_import.load_old_store(tmp_path, data)


# ---------------------------------------------------------------------------
# The import
# ---------------------------------------------------------------------------


async def test_import_puts_user_cron_on_workspace_assistant(runtime_store, workspace):
    cron_id = "11111111-1111-1111-1111-111111111111"
    _write_old_store(workspace.root, crons=[_cron(cron_id)])
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    (cron,) = runtime_store["crons"]
    assert report.crons == 1
    assert str(cron["cron_id"]) == cron_id
    assert str(cron["assistant_id"]) == workspace_assistant_id("scheduler", workspace)
    assert cron["metadata"]["workspace_dir"] == workspace.key
    configurable = cron["payload"]["config"]["configurable"]
    assert configurable["workspace_dir"] == workspace.key
    assert configurable["cron_id"] == cron_id
    assert cron["on_run_completed"] == "keep"
    assert cron["timezone"] == "UTC"
    assert _marked(workspace)


async def test_import_leaves_system_crons_to_their_reconcile(runtime_store, workspace):
    _write_old_store(
        workspace.root,
        crons=[
            _cron(
                "22222222-2222-2222-2222-222222222222",
                assistant=AUTOSKILLS,
                run_kind="evomemory_autoskills",
            )
        ],
    )
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert runtime_store["crons"] == []
    assert report.skipped == []


async def test_import_runs_once(runtime_store, workspace):
    """The old store stays as it was: importing again would bring back the
    tasks the user deleted since."""
    _write_old_store(
        workspace.root, crons=[_cron("11111111-1111-1111-1111-111111111111")]
    )
    await ensure_workspace_assistants(workspace)
    await legacy_import.import_once(workspace)
    runtime_store["crons"].clear()

    assert await legacy_import.import_once(workspace) is None
    assert runtime_store["crons"] == []


async def test_import_reports_cron_bound_to_missing_thread(runtime_store, workspace):
    bound = _cron(
        "33333333-3333-3333-3333-333333333333",
        thread_id=uuid.UUID("44444444-4444-4444-4444-444444444444"),
    )
    bound["metadata"]["workspace_dir"] = workspace.key
    _write_old_store(
        workspace.root,
        crons=[bound, _cron("11111111-1111-1111-1111-111111111111")],
    )
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.crons == 1
    (reason,) = report.skipped
    assert "was not restored" in reason


async def test_import_untagged_cron_on_thread_not_restored_waits_for_it(
    runtime_store, workspace, tmp_path
):
    """The old server here served another project's sessions, made before
    tasks were tagged: the thread's project restores it later."""
    thread_id = "44444444-4444-4444-4444-444444444444"
    cron = _cron("33333333-3333-3333-3333-333333333333", thread_id=uuid.UUID(thread_id))
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.waiting == {f"the project of thread {thread_id}": 1}
    other = Workspace(tmp_path / "other")
    other.root.mkdir()
    runtime_store["threads"].append(_restored_thread(thread_id, other))
    sessions.RESTORED_WORKSPACES.add(other.key)

    await workspace_setup._set_up(other, restore=False)

    assert _assistants(runtime_store) == {
        "33333333-3333-3333-3333-333333333333": workspace_assistant_id(
            "scheduler", other
        )
    }


async def test_import_unreadable_store_is_reported_once(
    runtime_store, workspace, caplog
):
    path = workspace.root / legacy_import.OLD_STORE
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x80\x05")

    report = await legacy_import.import_once(workspace)

    assert report.skipped
    assert _marked(workspace)
    assert await legacy_import.import_once(workspace) is None
    assert path.read_bytes() == b"\x80\x05"
    assert caplog.text.count("Not imported from") == 1


async def test_import_merges_metadata_onto_restored_threads(
    runtime_store, workspace, tmp_path
):
    """Only the keys a restored thread lacks; the restore's binding wins,
    and a thread keeps its place in the lists."""
    mine = "55555555-5555-5555-5555-555555555555"
    theirs = "66666666-6666-6666-6666-666666666666"
    restored = _restored_thread(mine, workspace, model="m1")
    updated_at = restored["updated_at"]
    # A copy's file, saved again since, names the original's threads: the
    # original reads its own.
    original = Workspace(tmp_path / "original")
    original.root.mkdir()
    _write_old_store(original.root)
    runtime_store["threads"].extend([restored, _restored_thread(theirs, original)])
    _write_old_store(
        workspace.root,
        threads=[
            {
                "thread_id": uuid.UUID(mine),
                "metadata": {
                    "title": "Renamed",
                    "pinned": True,
                    "model": "old",
                    "workspace_dir": "/old",
                    "run_id": "old-run",
                },
            },
            {"thread_id": uuid.UUID(theirs), "metadata": {"title": "Not here"}},
        ],
    )
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.threads == 1
    assert restored["metadata"] == {
        "graph_id": "EvoScientist",
        "workspace_dir": workspace.key,
        "model": "m1",
        "title": "Renamed",
        "pinned": True,
    }
    assert restored["updated_at"] == updated_at
    assert "title" not in runtime_store["threads"][1]["metadata"]


async def test_project_setup_imports_old_store(runtime_store, workspace):
    _write_old_store(
        workspace.root, crons=[_cron("11111111-1111-1111-1111-111111111111")]
    )

    await workspace_setup._set_up(workspace, restore=False)

    # The setup also reconciles the AutoSkills cron.
    tasks = [
        str(c["cron_id"])
        for c in runtime_store["crons"]
        if c["metadata"].get("run_kind") == "scheduled_task"
    ]
    assert tasks == ["11111111-1111-1111-1111-111111111111"]
    assert _marked(workspace)


async def test_project_setup_retries_failed_import(
    runtime_store, monkeypatch, workspace, caplog
):
    """An import that fails as a whole is not marked."""
    _write_old_store(
        workspace.root, crons=[_cron("11111111-1111-1111-1111-111111111111")]
    )

    async def ensure(_workspace):
        return None

    async def fail(*_args, **_kwargs):
        raise RuntimeError("store busy")

    monkeypatch.setattr(
        workspace_setup.assistants, "ensure_workspace_assistants", ensure
    )
    monkeypatch.setattr(legacy_import, "_import", fail)

    await workspace_setup._set_up(workspace, restore=False)

    assert not _marked(workspace)
    assert "store busy" in caplog.text


async def test_import_malformed_cron_is_skipped_and_project_marked(
    runtime_store, workspace
):
    """Marked either way: an import that ran again would bring back the
    tasks the user deleted since."""
    broken = _cron("77777777-7777-7777-7777-777777777777")
    del broken["schedule"]
    _write_old_store(
        workspace.root,
        crons=[broken, _cron("11111111-1111-1111-1111-111111111111")],
    )
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.crons == 1
    assert len(report.skipped) == 1
    assert _marked(workspace)


async def test_import_keeps_old_run_config(runtime_store, workspace):
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["payload"]["config"] = {
        "recursion_limit": 200,
        "tags": ["nightly"],
        "configurable": {"model": "m2", "workspace_dir": "/old"},
    }
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)

    await legacy_import.import_once(workspace)

    (imported,) = runtime_store["crons"]
    config = imported["payload"]["config"]
    assert config["recursion_limit"] == 200
    assert config["tags"] == ["nightly"]
    assert config["configurable"]["model"] == "m2"
    assert config["configurable"]["workspace_dir"] == workspace.key


async def test_import_drops_reserved_run_config_keys(runtime_store, workspace):
    """A fired cron kept its run user in its config."""
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["payload"]["config"]["configurable"].update(
        {"langgraph_auth_user_id": "old-user", "__request_start_time_ms__": 1}
    )
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)

    await legacy_import.import_once(workspace)

    (imported,) = runtime_store["crons"]
    configurable = imported["payload"]["config"]["configurable"]
    assert "langgraph_auth_user_id" not in configurable
    assert "__request_start_time_ms__" not in configurable


async def test_import_unhashable_run_kind_is_user_cron(runtime_store, workspace):
    _write_old_store(
        workspace.root,
        crons=[_cron("11111111-1111-1111-1111-111111111111", run_kind=["odd"])],
    )
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report is not None
    assert _marked(workspace)


async def test_import_folds_context_into_run_config(runtime_store, workspace):
    """A run refuses both, and the project's folders go in ``configurable``."""
    cron = _cron("11111111-1111-1111-1111-111111111111")
    del cron["payload"]["config"]
    cron["payload"]["context"] = {"model": "m2"}
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)

    await legacy_import.import_once(workspace)

    (imported,) = runtime_store["crons"]
    assert not imported["payload"].get("context")
    configurable = imported["payload"]["config"]["configurable"]
    assert configurable["model"] == "m2"
    assert configurable["workspace_dir"] == workspace.key


async def test_import_cron_bound_to_project_thread(runtime_store, workspace):
    thread_id = "88888888-8888-8888-8888-888888888888"
    runtime_store["threads"].append(_restored_thread(thread_id, workspace))
    _write_old_store(
        workspace.root,
        crons=[
            _cron(
                "99999999-9999-9999-9999-999999999999",
                thread_id=uuid.UUID(thread_id),
            )
        ],
    )
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.crons == 1
    assert str(runtime_store["crons"][0]["thread_id"]) == thread_id


async def test_import_finds_graph_without_old_assistants(runtime_store, workspace):
    """A snapshot without assistants: the graphs' own assistants have fixed
    IDs."""
    from langgraph.checkpoint.memory import PersistentDict

    path = workspace.root / legacy_import.OLD_STORE
    path.parent.mkdir(parents=True)
    store = PersistentDict(dict, filename=str(path))
    store["crons"] = [_cron("11111111-1111-1111-1111-111111111111")]
    store["threads"] = []
    store["assistants"] = []
    store.sync()
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.crons == 1


async def test_import_malformed_entries_are_skipped(runtime_store, workspace):
    """Not even the reason may fail: the project must still be marked."""
    _write_old_store(workspace.root, threads=["not a thread"])
    path = workspace.root / legacy_import.OLD_STORE
    from langgraph.checkpoint.memory import PersistentDict

    store = PersistentDict(dict, filename=str(path))
    store.load()
    store["crons"] = ["not a cron", {"metadata": "not a dict"}]
    store.sync()
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert len(report.skipped) == 3
    assert _marked(workspace)


async def test_import_waits_for_restored_threads(runtime_store, workspace):
    """Thread-bound crons and thread metadata need them: a restore that
    failed at the server's start leaves the import to the project's
    registration, which restores first."""
    sessions.RESTORED_WORKSPACES.discard(workspace.key)
    _write_old_store(
        workspace.root, crons=[_cron("11111111-1111-1111-1111-111111111111")]
    )
    await ensure_workspace_assistants(workspace)

    assert await legacy_import.import_once(workspace) is None
    assert runtime_store["crons"] == []
    assert not _marked(workspace)


async def test_import_read_error_leaves_project_unmarked(
    runtime_store, workspace, monkeypatch
):
    """A store that cannot be read for now is read at the next setup."""
    _write_old_store(
        workspace.root, crons=[_cron("11111111-1111-1111-1111-111111111111")]
    )

    def denied(_root, _known=None):
        raise PermissionError("denied")

    monkeypatch.setattr(legacy_import, "read_old_store", denied)

    with pytest.raises(PermissionError):
        await legacy_import.import_once(workspace)
    assert not _marked(workspace)


async def test_import_mark_survives_runtime_restart(
    langgraph_api_env, tmp_path, monkeypatch
):
    """The mark is a key of its own in the runtime's store: the runtime's
    startup loader must keep it, or every start would import again and
    bring back the tasks the user deleted. This fails first when a
    langgraph-runtime-inmem update drops keys it does not know."""
    import langgraph_runtime_inmem.database as database

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".langgraph_api").mkdir()
    saved = database.GlobalStore(filename=database.OPS_FILENAME)
    marks = {"crons": ["11111111-1111-1111-1111-111111111111"], "files": {}}
    saved[legacy_import.IMPORTED_KEY] = marks
    saved.sync()
    fresh = database.GlobalStore(filename=database.OPS_FILENAME)
    monkeypatch.setattr(database, "GLOBAL_STORE", fresh)
    monkeypatch.setattr(database, "register_persistent_dict", lambda _d: None)

    async def no_stream():
        return None

    monkeypatch.setattr(database, "start_stream", no_stream)

    await database.start_pool()

    assert fresh[legacy_import.IMPORTED_KEY] == marks


async def test_import_moved_folder_brings_back_no_deleted_task(
    runtime_store, workspace, tmp_path
):
    """The old store moves with the folder."""
    _write_old_store(
        workspace.root, crons=[_cron("11111111-1111-1111-1111-111111111111")]
    )
    await ensure_workspace_assistants(workspace)
    await legacy_import.import_once(workspace)
    # The user deleted the task.
    runtime_store["crons"].clear()
    moved = tmp_path / "moved"
    shutil.move(workspace.root, moved)
    project = Workspace(moved)
    registry.REGISTRY.set_registered(project.key, True)
    sessions.RESTORED_WORKSPACES.add(project.key)
    await ensure_workspace_assistants(project)

    report = await legacy_import.import_once(project)

    assert report.crons == 0
    assert runtime_store["crons"] == []


async def test_import_store_saved_again_brings_back_no_deleted_task(
    runtime_store, workspace
):
    """The previous release, run here again, saves its store again: only
    its new tasks are imported."""
    old_task = _cron("11111111-1111-1111-1111-111111111111")
    _write_old_store(workspace.root, crons=[old_task])
    await ensure_workspace_assistants(workspace)
    await legacy_import.import_once(workspace)
    runtime_store["crons"].clear()
    _write_old_store(
        workspace.root,
        crons=[old_task, _cron("22222222-2222-2222-2222-222222222222")],
    )

    report = await legacy_import.import_once(workspace)

    assert report.crons == 1
    assert list(_assistants(runtime_store)) == ["22222222-2222-2222-2222-222222222222"]


async def test_import_cron_of_unregistered_project_waits_for_its_setup(
    runtime_store, workspace, tmp_path
):
    """A server started here also served the sessions pointed at another
    project: that project's setup imports its tasks."""
    other = Workspace(tmp_path / "other")
    other.root.mkdir()
    theirs = _cron("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    theirs["metadata"]["workspace_dir"] = other.key
    gone = _cron("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    gone["metadata"]["workspace_dir"] = str(tmp_path / "gone")
    _write_old_store(workspace.root, crons=[theirs, gone])
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.waiting == {other.key: 1}
    assert list(_assistants(runtime_store)) == ["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"]
    # Only the folder's key: the runtime's store loads without this module.
    assert runtime_store[legacy_import.IMPORTED_KEY]["waiting"] == [workspace.key]

    sessions.RESTORED_WORKSPACES.add(other.key)
    await workspace_setup._set_up(other, restore=False)

    assert _assistants(runtime_store)[
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    ] == workspace_assistant_id("scheduler", other)


async def test_import_cron_of_registered_project_goes_to_it(
    runtime_store, workspace, tmp_path
):
    other = _project(tmp_path / "other")
    await ensure_workspace_assistants(other)
    theirs = _cron("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    theirs["metadata"]["workspace_dir"] = other.key
    _write_old_store(workspace.root, crons=[theirs])
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.crons == 1
    assert _assistants(runtime_store) == {
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa": workspace_assistant_id(
            "scheduler", other
        )
    }
    assert _marked(workspace)


async def test_import_copy_set_up_first_leaves_original_tasks(
    runtime_store, workspace, tmp_path
):
    """A copied folder's old store names the original's tasks, tagged with
    it."""
    original = Workspace(tmp_path / "original")
    original.root.mkdir()
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["metadata"]["workspace_dir"] = original.key
    _write_old_store(original.root, crons=[cron])
    shutil.copytree(original.root / ".langgraph_api", workspace.root / ".langgraph_api")
    await ensure_workspace_assistants(workspace)
    await legacy_import.import_once(workspace)
    sessions.RESTORED_WORKSPACES.add(original.key)

    await workspace_setup._set_up(original, restore=False)

    assert _assistants(runtime_store) == {
        "11111111-1111-1111-1111-111111111111": workspace_assistant_id(
            "scheduler", original
        )
    }
    assert _marked(original)
    assert _marked(workspace)


async def test_import_cron_imported_from_other_folder_is_not_imported_again(
    runtime_store, workspace, tmp_path
):
    """An untagged task of a copied folder goes to the first copy set up."""
    original = _project(tmp_path / "original")
    cron = _cron("11111111-1111-1111-1111-111111111111")
    _write_old_store(original.root, crons=[cron])
    _write_old_store(workspace.root, crons=[cron])
    for project in (original, workspace):
        await ensure_workspace_assistants(project)
    await legacy_import.import_once(original)

    report = await legacy_import.import_once(workspace)

    assert report.crons == 0
    assert report.skipped == []
    assert _assistants(runtime_store) == {
        "11111111-1111-1111-1111-111111111111": workspace_assistant_id(
            "scheduler", original
        )
    }


async def test_import_cron_id_held_by_other_project_is_skipped(
    runtime_store, workspace, tmp_path
):
    original = _project(tmp_path / "original")
    cron = _cron("11111111-1111-1111-1111-111111111111")
    _write_old_store(original.root, crons=[cron])
    _write_old_store(workspace.root, crons=[cron])
    for project in (original, workspace):
        await ensure_workspace_assistants(project)
    await legacy_import.import_once(original)
    runtime_store[legacy_import.IMPORTED_KEY]["crons"].clear()

    report = await legacy_import.import_once(workspace)

    (reason,) = report.skipped
    assert "another project" in reason


async def test_import_cron_tagged_with_run_folder_goes_to_its_project(
    runtime_store, workspace
):
    """Run-mode sessions stored their run folder as the workspace."""
    run = workspace.root / "runs" / "20260101_120000"
    run.mkdir(parents=True)
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["metadata"]["workspace_dir"] = str(run)
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.crons == 1
    assert report.waiting == {}


async def test_import_cron_kept_for_project_survives_its_own_unreadable_store(
    runtime_store, workspace, tmp_path, monkeypatch
):
    """The crons other files kept for a project do not wait on its own."""
    other = Workspace(tmp_path / "other")
    other.root.mkdir()
    theirs = _cron("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    theirs["metadata"]["workspace_dir"] = other.key
    _write_old_store(workspace.root, crons=[theirs])
    await ensure_workspace_assistants(workspace)
    await legacy_import.import_once(workspace)
    _write_old_store(other.root)
    sessions.RESTORED_WORKSPACES.add(other.key)
    await ensure_workspace_assistants(other)
    read = legacy_import.read_old_store

    def denied(root, known=None):
        if root == other.root:
            raise PermissionError("denied")
        return read(root, known)

    monkeypatch.setattr(legacy_import, "read_old_store", denied)

    with pytest.raises(PermissionError):
        await legacy_import.import_once(other)
    assert "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa" in _assistants(runtime_store)


async def test_import_cron_without_id_is_not_imported(runtime_store, workspace):
    """Each read would import it again, under a new ID."""
    cron = _cron("11111111-1111-1111-1111-111111111111")
    del cron["cron_id"]
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert runtime_store["crons"] == []
    (reason,) = report.skipped
    assert "no ID" in reason


async def test_import_relative_tag_names_store_folder(runtime_store, workspace):
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["metadata"]["workspace_dir"] = "."
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.crons == 1


def test_load_old_store_reads_past_classes_gone_since(tmp_path):
    """Run and thread values hold classes a later version may lack; the
    crons do not need them."""
    data = pickle.dumps(
        {"crons": [{"cron_id": "1"}], "runs": [_Value()]}, protocol=0
    ).replace(b"tests.test_legacy_import", b"tests.gone_since")

    old = legacy_import.load_old_store(tmp_path, data)

    assert old["crons"] == [{"cron_id": "1"}]


class _Value:
    def __init__(self):
        self.content = "hi"


async def test_import_cron_bound_to_other_project_thread_waits_for_it(
    runtime_store, workspace, tmp_path
):
    """A copy set up first: the task goes with its thread, not to the copy."""
    original = Workspace(tmp_path / "original")
    original.root.mkdir()
    sessions.RESTORED_WORKSPACES.add(original.key)
    thread_id = "88888888-8888-8888-8888-888888888888"
    runtime_store["threads"].append(_restored_thread(thread_id, original))
    cron = _cron("99999999-9999-9999-9999-999999999999", thread_id=uuid.UUID(thread_id))
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.waiting == {original.key: 1}

    await workspace_setup._set_up(original, restore=False)

    assert _assistants(runtime_store) == {
        "99999999-9999-9999-9999-999999999999": workspace_assistant_id(
            "scheduler", original
        )
    }


async def test_import_cron_of_project_not_restored_waits(
    runtime_store, workspace, tmp_path
):
    """Set up while its threads could not be restored: its thread-bound
    tasks would fail."""
    other = Workspace(tmp_path / "other")
    other.root.mkdir()
    workspace_setup.SET_UP.add(other.key)
    theirs = _cron("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    theirs["metadata"]["workspace_dir"] = other.key
    _write_old_store(workspace.root, crons=[theirs])
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.waiting == {other.key: 1}


async def test_import_kept_cron_of_project_gone_since_goes_to_store_project(
    runtime_store, workspace, tmp_path
):
    """The old server's rule: a task whose project folder is gone is the
    file's project's."""
    other = Workspace(tmp_path / "other")
    other.root.mkdir()
    theirs = _cron("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    theirs["metadata"]["workspace_dir"] = other.key
    _write_old_store(workspace.root, crons=[theirs])
    await ensure_workspace_assistants(workspace)
    await legacy_import.import_once(workspace)
    other.root.rmdir()

    await legacy_import.import_once(workspace)

    assert _assistants(runtime_store) == {
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa": workspace_assistant_id(
            "scheduler", workspace
        )
    }
    assert runtime_store[legacy_import.IMPORTED_KEY]["waiting"] == []


def test_load_old_store_reads_past_container_classes_gone_since(tmp_path):
    """The runtime saves with protocol 2: a container's items come after."""
    data = pickle.dumps(
        {"crons": [{"cron_id": "1"}], "runs": [_Mapping(a=1), _Sequence([1])]},
        protocol=2,
    ).replace(b"tests.test_legacy_import", b"tests.gone_since")

    old = legacy_import.load_old_store(tmp_path, data)

    assert old["crons"] == [{"cron_id": "1"}]


class _Mapping(dict):
    pass


class _Sequence(list):
    pass


async def test_import_waiting_cron_deleted_from_store_since_is_not_imported(
    runtime_store, workspace, tmp_path
):
    """The previous release, run here again, deleted it: the file is the
    source of what is left."""
    other = Workspace(tmp_path / "other")
    other.root.mkdir()
    theirs = _cron("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    theirs["metadata"]["workspace_dir"] = other.key
    _write_old_store(workspace.root, crons=[theirs])
    await ensure_workspace_assistants(workspace)
    await legacy_import.import_once(workspace)
    _write_old_store(workspace.root)
    sessions.RESTORED_WORKSPACES.add(other.key)

    await workspace_setup._set_up(other, restore=False)

    assert _assistants(runtime_store) == {}
    assert runtime_store[legacy_import.IMPORTED_KEY]["waiting"] == []


async def test_import_cron_tagged_with_named_run_folder_goes_to_store_project(
    runtime_store, workspace
):
    """The old server ran in the project and served its runs."""
    run = workspace.root / "runs" / "my-run"
    run.mkdir(parents=True)
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["metadata"]["workspace_dir"] = str(run)
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.crons == 1
    assert report.waiting == {}


async def test_import_cron_tagged_with_named_run_of_registered_project_goes_to_it(
    runtime_store, workspace, tmp_path
):
    other = _project(tmp_path / "other")
    await ensure_workspace_assistants(other)
    run = other.root / "runs" / "my-run"
    run.mkdir(parents=True)
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["metadata"]["workspace_dir"] = str(run)
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)

    await legacy_import.import_once(workspace)

    assert _assistants(runtime_store) == {
        "11111111-1111-1111-1111-111111111111": workspace_assistant_id(
            "scheduler", other
        )
    }


async def test_import_cron_tagged_with_registered_project_in_runs_folder_goes_to_it(
    runtime_store, workspace, tmp_path
):
    """A project may live in a folder called ``runs`` of another project."""
    parent = _project(tmp_path / "parent")
    (parent.root / "runs").mkdir()
    nested = _project(parent.root / "runs" / "nested")
    await ensure_workspace_assistants(nested)
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["metadata"]["workspace_dir"] = nested.key
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)

    await legacy_import.import_once(workspace)

    assert _assistants(runtime_store) == {
        "11111111-1111-1111-1111-111111111111": workspace_assistant_id(
            "scheduler", nested
        )
    }


@pytest.mark.parametrize("own_store", [False, True])
async def test_import_merges_metadata_of_other_project_without_own_store(
    runtime_store, workspace, tmp_path, own_store
):
    """A server started here served the other project's sessions; a project
    with an old store of its own reads its own, maybe newer."""
    other = Workspace(tmp_path / "other")
    other.root.mkdir()
    if own_store:
        _write_old_store(other.root)
    thread_id = "55555555-5555-5555-5555-555555555555"
    restored = _restored_thread(thread_id, other)
    runtime_store["threads"].append(restored)
    _write_old_store(
        workspace.root,
        threads=[{"thread_id": uuid.UUID(thread_id), "metadata": {"title": "Theirs"}}],
    )
    await ensure_workspace_assistants(workspace)

    await legacy_import.import_once(workspace)

    assert ("title" in restored["metadata"]) is not own_store


async def test_import_leaves_out_data_of_classes_gone_since(runtime_store, workspace):
    """The runtime's store must load without this module."""
    thread_id = "55555555-5555-5555-5555-555555555555"
    restored = _restored_thread(thread_id, workspace)
    runtime_store["threads"].append(restored)
    gone = legacy_import._Gone()
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["payload"]["input"] = {"messages": [gone]}
    old = {
        "crons": [cron],
        "threads": [
            {
                "thread_id": uuid.UUID(thread_id),
                "metadata": {"title": "Kept", "odd": gone},
            }
        ],
        "assistants": [],
    }
    await ensure_workspace_assistants(workspace)

    report = await legacy_import._import(workspace, old, workspace, first_read=True)

    assert runtime_store["crons"] == []
    assert "cannot read" in report.skipped[0]
    assert restored["metadata"]["title"] == "Kept"
    assert "odd" not in restored["metadata"]


async def test_import_leaves_out_old_cron_bindings(runtime_store, workspace):
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["metadata"].update(
        {"run_dir": "/old/runs/x", "assistant_id": "old", "thread_id": "old"}
    )
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)

    await legacy_import.import_once(workspace)

    (imported,) = runtime_store["crons"]
    assert not {"run_dir", "assistant_id", "thread_id"} & set(imported["metadata"])
    assert imported["metadata"]["name"] == "task 1111"


async def test_import_generated_run_of_project_in_runs_folder_waits_for_it(
    runtime_store, workspace, tmp_path
):
    """The run names its own project, not the one whose ``runs`` folder
    holds that project."""
    parent = _project(tmp_path / "parent")
    nested = parent.root / "runs" / "nested"
    (nested / "runs" / "20260101_120000").mkdir(parents=True)
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["metadata"]["workspace_dir"] = str(nested / "runs" / "20260101_120000")
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.waiting == {Workspace(nested).key: 1}


def test_holds_gone_finds_placeholder_at_any_depth():
    assert legacy_import._holds_gone([_Message({"x": legacy_import._Gone()})])
    assert legacy_import._holds_gone({"kind": legacy_import._Gone})
    assert not legacy_import._holds_gone([_Message({"x": 1})])


class _Message:
    def __init__(self, extra):
        self.extra = extra


async def test_import_cron_of_registered_project_gone_since_goes_to_store_project(
    runtime_store, workspace, tmp_path
):
    """Its folder is gone: it can never be set up."""
    other = Workspace(tmp_path / "other")
    registry.REGISTRY.set_registered(other.key, True)
    theirs = _cron("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    theirs["metadata"]["workspace_dir"] = other.key
    _write_old_store(workspace.root, crons=[theirs])
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.crons == 1
    assert report.waiting == {}


async def test_import_cron_on_thread_of_project_gone_since_is_reported(
    runtime_store, workspace, tmp_path
):
    thread_id = "44444444-4444-4444-4444-444444444444"
    runtime_store["threads"].append(
        _restored_thread(thread_id, Workspace(tmp_path / "gone"))
    )
    cron = _cron("33333333-3333-3333-3333-333333333333", thread_id=uuid.UUID(thread_id))
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)

    report = await legacy_import.import_once(workspace)

    assert report.waiting == {}
    (reason,) = report.skipped
    assert "is gone" in reason


async def test_import_cron_tagged_with_deleted_named_run_goes_to_its_project(
    runtime_store, workspace, tmp_path
):
    """Run folders get cleaned up; the project stays."""
    other = _project(tmp_path / "other")
    await ensure_workspace_assistants(other)
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["metadata"]["workspace_dir"] = str(other.root / "runs" / "my-run")
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)

    await legacy_import.import_once(workspace)

    assert _assistants(runtime_store) == {
        "11111111-1111-1111-1111-111111111111": workspace_assistant_id(
            "scheduler", other
        )
    }


async def test_import_keeps_cron_whose_dropped_run_user_is_gone(
    runtime_store, workspace
):
    """A fired cron kept its run user, whose class a later version may move:
    the import drops it anyway."""
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["payload"]["config"]["configurable"]["langgraph_auth_user"] = (
        legacy_import._Gone()
    )
    old = {"crons": [cron], "threads": [], "assistants": []}
    await ensure_workspace_assistants(workspace)

    report = await legacy_import._import(workspace, old, workspace, first_read=True)

    assert report.crons == 1


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("thread_id",), legacy_import._Gone()),
        (("on_run_completed",), legacy_import._Gone()),
        (("enabled",), legacy_import._Gone()),
        (("metadata",), legacy_import._Gone()),
        (("payload",), legacy_import._Gone()),
        (("payload", "config"), legacy_import._Gone()),
        (("payload", "context"), legacy_import._Gone()),
        (("payload", "context"), legacy_import._Gone),
        (("payload", "config", "configurable"), legacy_import._Gone()),
    ],
)
async def test_import_reports_cron_with_field_gone(
    runtime_store, workspace, path, value
):
    cron = _cron("11111111-1111-1111-1111-111111111111")
    row = cron
    for key in path[:-1]:
        row = row[key]
    row[path[-1]] = value
    old = {"crons": [cron], "threads": [], "assistants": []}
    await ensure_workspace_assistants(workspace)

    report = await legacy_import._import(workspace, old, workspace, first_read=True)

    assert runtime_store["crons"] == []
    (reason,) = report.skipped
    assert "cannot read" in reason


async def test_import_reports_thread_metadata_gone(runtime_store, workspace):
    thread_id = "55555555-5555-5555-5555-555555555555"
    runtime_store["threads"].append(_restored_thread(thread_id, workspace))
    old = {
        "crons": [],
        "threads": [
            {"thread_id": uuid.UUID(thread_id), "metadata": legacy_import._Gone()}
        ],
        "assistants": [],
    }

    report = await legacy_import._import(workspace, old, workspace, first_read=True)

    (reason,) = report.skipped
    assert "cannot read" in reason


async def test_import_cron_with_unreadable_id_is_reported_without_mark(
    runtime_store, workspace
):
    """A mark of it would change at each read."""
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["cron_id"] = legacy_import._Gone()
    old = {"crons": [cron], "threads": [], "assistants": []}

    report = await legacy_import._import(workspace, old, workspace, first_read=True)

    (reason,) = report.skipped
    assert "no ID" in reason
    assert runtime_store[legacy_import.IMPORTED_KEY]["crons"] == []


async def test_import_report_names_unreadable_cron_by_id(runtime_store, workspace):
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["metadata"]["name"] = legacy_import._Gone()
    old = {"crons": [cron], "threads": [], "assistants": []}

    report = await legacy_import._import(workspace, old, workspace, first_read=True)

    (reason,) = report.skipped
    assert "11111111-1111-1111-1111-111111111111" in reason


@pytest.mark.parametrize("cron_id", ["", "not-a-uuid"])
async def test_import_cron_without_uuid_is_reported_as_without_id(
    runtime_store, workspace, cron_id
):
    """The runtime would give it a new ID, or one its routes cannot read."""
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["cron_id"] = cron_id
    old = {"crons": [cron], "threads": [], "assistants": []}

    report = await legacy_import._import(workspace, old, workspace, first_read=True)

    assert runtime_store["crons"] == []
    (reason,) = report.skipped
    assert "no ID" in reason


async def test_import_unreadable_system_cron_is_left_to_its_reconcile(
    runtime_store, workspace
):
    cron = _cron(
        "22222222-2222-2222-2222-222222222222",
        assistant=AUTOSKILLS,
        run_kind="evomemory_autoskills",
    )
    cron["payload"]["input"] = legacy_import._Gone()
    old = {"crons": [cron], "threads": [], "assistants": []}

    report = await legacy_import._import(workspace, old, workspace, first_read=True)

    assert report.skipped == []


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS case-insensitive volume")
async def test_import_cron_tagged_with_named_run_in_other_case_goes_to_its_project(
    runtime_store, workspace, tmp_path
):
    """A named run stored as Runs still belongs to its registered project."""
    other = _project(tmp_path / "other")
    await ensure_workspace_assistants(other)
    (other.root / "Runs" / "my-run").mkdir(parents=True)
    if not (other.root / "runs").exists():
        pytest.skip("case-sensitive volume")
    cron = _cron("11111111-1111-1111-1111-111111111111")
    cron["metadata"]["workspace_dir"] = str(other.root / "runs" / "my-run")
    _write_old_store(workspace.root, crons=[cron])
    await ensure_workspace_assistants(workspace)
    await legacy_import.import_once(workspace)
    assert _assistants(runtime_store) == {
        "11111111-1111-1111-1111-111111111111": workspace_assistant_id(
            "scheduler", other
        )
    }
