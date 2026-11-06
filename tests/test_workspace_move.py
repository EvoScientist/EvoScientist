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

    assert await workspace_move.move_checkpoints(move) == 2

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
