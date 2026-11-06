"""Moving a project to another folder: each store re-pointed from the old
folder to the new one, and nothing changed when it runs again."""

from __future__ import annotations

import json
import uuid

import aiosqlite
import pytest

from EvoScientist import sessions
from EvoScientist.langgraph_dev import legacy_import, workspace_move
from EvoScientist.langgraph_dev.assistants import chat_assistant_id
from EvoScientist.memory import project
from EvoScientist.memory.autoskills import proposals
from EvoScientist.memory.autoskills.schedule import AUTOSKILL_RUN_KIND
from EvoScientist.memory.project import path_project_ids, resolve_project_id
from EvoScientist.middleware.memory import PROFILE_TEMPLATES
from EvoScientist.paths import Workspace


@pytest.fixture
def move(tmp_path) -> workspace_move.Move:
    new = tmp_path / "new"
    new.mkdir()
    # The old folder is gone: the user moved it.
    return workspace_move.Move(Workspace(tmp_path / "old"), Workspace(new))


@pytest.fixture
def other(tmp_path) -> Workspace:
    folder = tmp_path / "other"
    folder.mkdir()
    return Workspace(folder)


# -- sessions.db -------------------------------------------------------------


async def _db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(str(sessions.get_db_path()))
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS checkpoints (thread_id TEXT NOT NULL, "
        "checkpoint_ns TEXT NOT NULL DEFAULT '', checkpoint_id TEXT NOT NULL, "
        "parent_checkpoint_id TEXT, type TEXT, checkpoint BLOB, "
        "metadata BLOB NOT NULL DEFAULT '{}', "
        "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id))"
    )
    return conn


async def _put_rows(rows: list[tuple[str, str, dict]]) -> None:
    conn = await _db()
    try:
        await conn.executemany(
            "INSERT INTO checkpoints (thread_id, checkpoint_id, checkpoint, metadata) "
            "VALUES (?, ?, x'00', ?)",
            [(t, c, json.dumps(m).encode()) for t, c, m in rows],
        )
        await conn.commit()
    finally:
        await conn.close()


async def _metadata(thread_id: str) -> list[dict]:
    conn = await _db()
    try:
        async with conn.execute(
            "SELECT metadata FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id",
            (thread_id,),
        ) as cur:
            return [json.loads(row[0]) for row in await cur.fetchall()]
    finally:
        await conn.close()


async def test_move_checkpoints_repoints_every_row_of_old_threads(move, other):
    old_assistant = chat_assistant_id(move.old)
    run = f"{move.old.key}/runs/20260101_120000"
    await _put_rows(
        [
            ("t1", "1", {"workspace_dir": move.old.key, "assistant_id": old_assistant}),
            ("t1", "2", {"workspace_dir": move.old.key, "run_dir": run}),
            ("t2", "1", {"workspace_dir": other.key, "assistant_id": "x"}),
        ]
    )

    assert await workspace_move.move_checkpoints(move) == 1

    first, second = await _metadata("t1")
    assert first == {
        "workspace_dir": move.new.key,
        "assistant_id": chat_assistant_id(move.new),
    }
    assert second == {
        "workspace_dir": move.new.key,
        "run_dir": f"{move.new.key}/runs/20260101_120000",
    }
    assert await _metadata("t2") == [{"workspace_dir": other.key, "assistant_id": "x"}]
    assert await workspace_move.move_checkpoints(move) == 0


async def test_move_checkpoints_matches_old_folder_spelled_otherwise(move):
    await _put_rows([("t1", "1", {"workspace_dir": f"{move.old.key}/"})])

    await workspace_move.move_checkpoints(move)

    assert await _metadata("t1") == [{"workspace_dir": move.new.key}]


# -- the runtime's threads and crons -----------------------------------------


async def test_move_threads_repoints_metadata_config_and_assistant(
    runtime_store, move, other
):
    moved = {
        "thread_id": uuid.uuid4(),
        "metadata": {
            "workspace_dir": move.old.key,
            "assistant_id": chat_assistant_id(move.old),
        },
        "config": {"configurable": {"workspace_dir": move.old.key}},
    }
    kept = {
        "thread_id": uuid.uuid4(),
        "metadata": {"workspace_dir": other.key},
        "config": {},
    }
    runtime_store["threads"].extend([moved, kept])

    assert await workspace_move.move_threads(move) == 1

    assert moved["metadata"] == {
        "workspace_dir": move.new.key,
        "assistant_id": chat_assistant_id(move.new),
    }
    assert moved["config"] == {"configurable": {"workspace_dir": move.new.key}}
    assert kept["metadata"] == {"workspace_dir": other.key}
    assert await workspace_move.move_threads(move) == 0


async def test_move_threads_finds_thread_by_assistant_alone(runtime_store, move):
    """Its checkpoints moved before a crash, and a restore re-set its folders."""
    thread = {
        "thread_id": uuid.uuid4(),
        "metadata": {
            "workspace_dir": move.new.key,
            "assistant_id": chat_assistant_id(move.old),
        },
        "config": {},
    }
    runtime_store["threads"].append(thread)

    await workspace_move.move_threads(move)

    assert thread["metadata"]["assistant_id"] == chat_assistant_id(move.new)


def _cron(assistant: str, workspace: Workspace, **metadata) -> dict:
    return {
        "cron_id": uuid.uuid4(),
        "assistant_id": uuid.UUID(assistant),
        "enabled": False,
        "metadata": {"workspace_dir": workspace.key, **metadata},
        "payload": {
            "assistant_id": assistant,
            "metadata": {"workspace_dir": workspace.key},
            "config": {"configurable": {"workspace_dir": workspace.key}},
        },
    }


async def test_move_crons_repoints_tasks_and_deletes_system_crons(
    runtime_store, move, other
):
    task = _cron(chat_assistant_id(move.old), move.old, run_kind="scheduled_task")
    system = _cron(chat_assistant_id(move.old), move.old, run_kind=AUTOSKILL_RUN_KIND)
    kept = _cron(chat_assistant_id(other), other, run_kind="scheduled_task")
    runtime_store["crons"].extend([task, system, kept])

    assert await workspace_move.move_crons(move) == (1, 1)

    new_assistant = chat_assistant_id(move.new)
    assert task["assistant_id"] == uuid.UUID(new_assistant)
    assert task["enabled"] is False
    assert task["metadata"]["workspace_dir"] == move.new.key
    assert task["payload"]["assistant_id"] == new_assistant
    assert task["payload"]["metadata"]["workspace_dir"] == move.new.key
    assert task["payload"]["config"]["configurable"]["workspace_dir"] == move.new.key
    assert [c["cron_id"] for c in runtime_store["crons"]] == [
        task["cron_id"],
        kept["cron_id"],
    ]
    assert kept["metadata"]["workspace_dir"] == other.key
    assert await workspace_move.move_crons(move) == (0, 0)


async def test_drop_import_marks_forgets_both_folders_files(runtime_store, move, other):
    marks = runtime_store[legacy_import.IMPORTED_KEY] = {
        "crons": ["c1"],
        "files": {
            str(move.old.root / legacy_import.OLD_STORE): [1, 2],
            str(move.new.root / legacy_import.OLD_STORE): [1, 2],
            str(other.root / legacy_import.OLD_STORE): [3, 4],
        },
        "waiting": [move.old.key, other.key],
    }

    await workspace_move.drop_import_marks(move)

    assert marks == {
        "crons": ["c1"],
        "files": {str(other.root / legacy_import.OLD_STORE): [3, 4]},
        "waiting": [other.key],
    }


# -- memory and proposals ----------------------------------------------------

_TEMPLATE = PROFILE_TEMPLATES["/profile/projects/{project_id}/PROJECT_PROFILE.md"]


def _memory(memory_dir, project_id, *, observations=(), profile=None) -> None:
    folder = memory_dir / "observations" / "projects" / project_id
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in observations:
        (folder / name).write_text(text)
    if profile is not None:
        path = memory_dir / "profile" / "projects" / project_id / "PROJECT_PROFILE.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(profile)


def _profile(memory_dir, project_id) -> str:
    path = memory_dir / "profile" / "projects" / project_id / "PROJECT_PROFILE.md"
    return path.read_text()


def test_move_memory_merges_into_new_project(tmp_path, move):
    memory = tmp_path / "memories"
    _memory(memory, "P-old", observations=[("O-1.md", "old"), ("O-2.md", "both")])
    _memory(memory, "P-old", profile="# Project profile\n\nUse uv.\n")
    _memory(memory, "P-new", observations=[("O-2.md", "both")], profile=_TEMPLATE)

    result = workspace_move.move_memory(
        move, memory, old_ids={"P-old"}, new_id="P-new", shared=set()
    )

    assert result.moved == ["P-old"]
    folder = memory / "observations" / "projects"
    assert sorted(p.name for p in (folder / "P-new").iterdir()) == ["O-1.md", "O-2.md"]
    assert not (folder / "P-old").exists()
    assert "Use uv." in _profile(memory, "P-new")
    assert not (memory / "profile" / "projects" / "P-old").exists()


def test_move_memory_keeps_both_edited_profiles(tmp_path, move):
    memory = tmp_path / "memories"
    _memory(memory, "P-old", profile="# Project profile\n\nUse uv.\n")
    _memory(memory, "P-new", profile="# Project profile\n\nRun make.\n")

    for _ in range(2):
        workspace_move.move_memory(
            move, memory, old_ids={"P-old"}, new_id="P-new", shared={"P-old"}
        )

    profile = _profile(memory, "P-new")
    assert "Run make." in profile
    assert profile.count("Use uv.") == 1


def test_move_memory_copies_project_another_folder_uses(tmp_path, move):
    memory = tmp_path / "memories"
    _memory(memory, "P-old", observations=[("O-1.md", "old")])

    result = workspace_move.move_memory(
        move, memory, old_ids={"P-old"}, new_id="P-new", shared={"P-old"}
    )

    assert result.copied == ["P-old"]
    folder = memory / "observations" / "projects"
    assert (folder / "P-old" / "O-1.md").exists()
    assert (folder / "P-new" / "O-1.md").exists()


@pytest.mark.parametrize("git_root", [False, True], ids=["folder", "repo_root"])
def test_path_project_ids_cover_folder_and_repo_root(tmp_path, monkeypatch, git_root):
    folder = tmp_path / "project"
    folder.mkdir()
    answers = {"rev-parse": str(folder) if git_root else None, "remote": None}
    monkeypatch.setattr(project, "_run_git", lambda args, cwd: answers[args[0]])

    assert resolve_project_id(folder) in path_project_ids(folder)


def test_move_memory_leaves_out_untouched_old_profile(tmp_path, move):
    memory = tmp_path / "memories"
    _memory(memory, "P-old", profile=_TEMPLATE)
    _memory(memory, "P-new", profile="# Project profile\n\nRun make.\n")

    workspace_move.move_memory(
        move, memory, old_ids={"P-old"}, new_id="P-new", shared=set()
    )

    assert _profile(memory, "P-new") == "# Project profile\n\nRun make.\n"


def _proposal(memory, name, workspace: Workspace, project_id: str) -> None:
    folder = memory / proposals.AUTOSKILL_PROPOSALS_DIR / name
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(
        json.dumps(
            {"workspace_dir": workspace.key, "project_id": project_id, "status": "x"}
        )
    )


def _manifest(memory, name) -> dict:
    path = memory / proposals.AUTOSKILL_PROPOSALS_DIR / name / "manifest.json"
    return json.loads(path.read_text())


def test_move_proposals_gives_every_proposal_to_new_folder(tmp_path, move, other):
    memory = tmp_path / "memories"
    _proposal(memory, "a", move.old, "P-old")
    _proposal(memory, "b", other, "P-other")

    assert proposals.proposal_project_ids(memory, move.old.key) == {"P-old"}
    assert (
        proposals.move_proposals(memory, move.old.key, move.new.key, {"P-old": "P-new"})
        == 1
    )

    assert _manifest(memory, "a")["workspace_dir"] == move.new.key
    assert _manifest(memory, "a")["project_id"] == "P-new"
    assert _manifest(memory, "b")["workspace_dir"] == other.key
    assert proposals.move_proposals(memory, move.old.key, move.new.key, {}) == 0


# -- the whole move ----------------------------------------------------------


@pytest.fixture
def moving(runtime_store, monkeypatch, tmp_path):
    """A registered project whose folder was moved, with its assistants, one
    thread and one scheduled task; the setup's background work skipped."""
    from EvoScientist.langgraph_dev import registry, workspace_setup
    from EvoScientist.langgraph_dev.assistants import ensure_workspace_assistants

    projects = registry.Registry()
    monkeypatch.setattr(registry, "REGISTRY", projects)
    monkeypatch.setattr(workspace_setup, "warm_up", lambda workspace: None)

    async def no_reconcile(_workspace):
        return None

    monkeypatch.setattr(workspace_setup, "_reconcile_autoskills", no_reconcile)
    monkeypatch.setattr(project, "_run_git", lambda args, cwd: None)
    old = Workspace(tmp_path / "old")
    old.root.mkdir()
    new = Workspace(tmp_path / "new")
    new.root.mkdir()
    projects.set_registered(old.key, True)
    return projects, old, new, ensure_workspace_assistants


async def _arrange(moving):
    projects, old, new, ensure = moving
    await ensure(old)
    old.root.rmdir()
    task = _cron(chat_assistant_id(old), old, run_kind="scheduled_task")
    import langgraph_runtime_inmem.database as db

    db.GLOBAL_STORE["crons"].append(task)
    await _put_rows(
        [
            (
                str(uuid.uuid4()),
                "1",
                {"workspace_dir": old.key, "assistant_id": chat_assistant_id(old)},
            )
        ]
    )
    return projects, old, new, task


async def test_move_project_moves_threads_tasks_and_registration(moving):
    projects, old, new, task = await _arrange(moving)

    report = await workspace_move.move_project(old, new)

    assert (report.workspace_dir, report.threads, report.tasks) == (new.key, 1, 1)
    assert task["assistant_id"] == uuid.UUID(chat_assistant_id(new))
    assert old.key not in projects.known()
    assert projects.is_registered(new.key)
    assert projects.pending_move() is None
    import langgraph_runtime_inmem.database as db

    stored = {str(a["assistant_id"]) for a in db.GLOBAL_STORE["assistants"]}
    assert chat_assistant_id(new) in stored
    assert chat_assistant_id(old) not in stored
    # Asked again (a client that gave up waiting): done already.
    again = await workspace_move.move_project(old, new)
    assert again.resumed is True


async def test_move_project_refuses_unregistered_old_into_registered_new(
    moving, tmp_path
):
    """Every client registers the folder it opens: a mistyped old folder
    must not pass for a finished move."""
    projects, _old, new, _task = await _arrange(moving)
    projects.set_registered(new.key, True)

    with pytest.raises(workspace_move.MoveRefusedError) as refused:
        await workspace_move.move_project(Workspace(tmp_path / "typo"), new)

    assert refused.value.code == "not_registered"


async def test_move_project_finishes_move_cut_short_at_start(moving):
    projects, old, new, task = await _arrange(moving)
    projects.record_move(old.key, new.key)

    await workspace_move.resume_pending_move()

    assert projects.pending_move() is None
    assert projects.is_registered(new.key)
    assert task["assistant_id"] == uuid.UUID(chat_assistant_id(new))


async def test_move_project_refuses_other_move_while_one_is_pending(moving, tmp_path):
    projects, old, new, _task = await _arrange(moving)
    projects.record_move(old.key, new.key)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(workspace_move.MoveRefusedError) as refused:
        await workspace_move.move_project(old, Workspace(elsewhere))

    assert refused.value.code == "move_pending"


@pytest.mark.parametrize(
    "case", ["old_exists", "new_missing", "not_registered", "busy"]
)
async def test_move_project_refuses_move_it_cannot_make(moving, tmp_path, case):
    projects, old, new, _task = await _arrange(moving)
    if case == "old_exists":
        old.root.mkdir()
    elif case == "new_missing":
        new.root.rmdir()
    elif case == "not_registered":
        old = Workspace(tmp_path / "unknown")
    else:
        import langgraph_runtime_inmem.database as db

        db.GLOBAL_STORE["runs"].append(
            {"assistant_id": chat_assistant_id(old), "status": "pending"}
        )

    with pytest.raises(workspace_move.MoveRefusedError) as refused:
        await workspace_move.move_project(old, new)

    assert refused.value.code == case
    assert projects.pending_move() is None


def test_move_default_workdir_follows_old_folder(monkeypatch, move, other):
    from EvoScientist.config import settings, store

    written = []
    monkeypatch.setattr(
        settings, "set_config_values", lambda changes: written.append(changes)
    )
    for value, expected in [(f"{move.old.key}/", 1), (other.key, 1)]:
        monkeypatch.setattr(
            store, "read_config_mapping", lambda value=value: {"default_workdir": value}
        )
        assert workspace_move._move_default_workdir(move) is None
        assert len(written) == expected

    assert written == [{"default_workdir": str(move.new.root)}]


# -- the route -----------------------------------------------------------------


@pytest.fixture
def server(monkeypatch, no_workspace_setup):
    from starlette.testclient import TestClient

    from EvoScientist import local_secret, paths
    from EvoScientist.langgraph_dev import registry
    from EvoScientist.langgraph_dev.http import app

    monkeypatch.setattr(registry, "REGISTRY", registry.Registry())
    monkeypatch.setattr(local_secret, "_value", None)
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as client:
        value = (paths.SERVER_HOME / local_secret.FILE_NAME).read_text()
        yield client, {local_secret.HEADER: value}


def test_move_route_needs_local_secret(server, move):
    client, _secret = server

    response = client.post(
        "/api/workspaces/move", json={"old": move.old.key, "new": move.new.key}
    )

    assert response.status_code == 403


def test_move_route_answers_refusal_with_its_code(server, move):
    client, secret = server

    response = client.post(
        "/api/workspaces/move",
        json={"old": move.old.key, "new": move.new.key},
        headers=secret,
    )

    assert response.status_code == 409
    assert response.json()["error"] == "not_registered"


def test_register_route_refuses_folder_being_moved_away(server, move, tmp_path):
    from EvoScientist.langgraph_dev import registry

    client, secret = server
    move.old.root.mkdir()
    registry.REGISTRY.record_move(move.old.key, move.new.key)

    response = client.post(
        "/api/workspaces", json={"workspace_dir": move.old.key}, headers=secret
    )

    assert response.status_code == 409
    assert response.json()["error"] == "move_pending"


@pytest.mark.parametrize(("platform", "same"), [("darwin", True), ("linux", False)])
def test_same_key_ignores_case_where_folders_do(monkeypatch, platform, same):
    monkeypatch.setattr(workspace_move.sys, "platform", platform)

    assert workspace_move.same_key("/u/Proj", "/u/proj") is same


async def test_move_project_says_when_it_finished_earlier_move(moving):
    projects, old, new, _task = await _arrange(moving)
    projects.record_move(old.key, new.key)

    report = await workspace_move.move_project(old, new)

    assert report.resumed is True


async def test_move_project_pending_without_new_folder_says_to_put_it_back(moving):
    projects, old, new, _task = await _arrange(moving)
    projects.record_move(old.key, new.key)
    new.root.rmdir()

    with pytest.raises(workspace_move.MoveRefusedError) as refused:
        await workspace_move.move_project(old, new)

    assert refused.value.code == "new_missing"
    assert "put the folder back" in str(refused.value)


def test_move_memory_keeps_proposal_only_ids_and_ignores_odd_ones(
    tmp_path, move, monkeypatch
):
    from EvoScientist import paths
    from EvoScientist.langgraph_dev import registry

    memory = tmp_path / "memories"
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory)
    monkeypatch.setattr(registry, "REGISTRY", registry.Registry())
    monkeypatch.setattr(project, "_run_git", lambda args, cwd: None)
    path_id = min(path_project_ids(move.old.root))
    remote_id = "P-" + "a" * 16
    _memory(memory, path_id, observations=[("O-1.md", "path")])
    _memory(memory, remote_id, observations=[("O-2.md", "remote")])
    _proposal(memory, "a", move.old, remote_id)
    _proposal(memory, "b", move.old, "../../outside")

    moved, project_ids = workspace_move._move_memory(move)

    assert moved.moved == [path_id]
    assert moved.copied == [remote_id]
    assert (memory / "observations" / "projects" / remote_id).is_dir()
    assert remote_id in project_ids
    assert path_id in project_ids
    assert "../../outside" not in project_ids


async def test_repeated_move_retries_default_workdir_it_could_not_update(
    moving, monkeypatch
):
    """The config file was locked as the move finished."""
    _projects, old, new, _task = await _arrange(moving)
    await workspace_move.move_project(old, new)
    retried = []
    monkeypatch.setattr(
        workspace_move, "_move_default_workdir", lambda move: retried.append(move)
    )

    again = await workspace_move.move_project(old, new)

    assert again.resumed is True
    assert [(m.old.key, m.new.key) for m in retried] == [(old.key, new.key)]


def test_same_folder_tells_apart_folders_differing_in_case(tmp_path, monkeypatch):
    """A case-sensitive macOS volume can hold both."""
    monkeypatch.setattr(workspace_move.sys, "platform", "darwin")
    upper, lower = tmp_path / "Proj", tmp_path / "proj"
    upper.mkdir()
    if lower.exists():
        pytest.skip("this file system ignores case")
    lower.mkdir()

    assert not workspace_move.same_folder(str(upper), str(lower))
    assert workspace_move.same_folder(str(upper), str(upper / "."))
    # A folder that is gone falls back to the keys.
    assert workspace_move.same_folder(str(tmp_path / "Gone"), str(tmp_path / "gone"))
