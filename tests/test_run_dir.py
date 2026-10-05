"""A ``--mode=run`` session: the workspace root plus the run folder it works in.

Stored ``workspace_dir`` is always the root; ``run_dir`` is added only for a
run-mode session. Daemon sessions keep the metadata, server and sandbox they
had before.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sqlite3
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from EvoScientist.gateway.server import _build_thread_metadata
from EvoScientist.langgraph_dev import manager
from EvoScientist.paths import SessionDirs, Workspace
from tests.fakes import FakeGraphGateway, FakeThreadStore

TS_RUN = "20260930_120000"


@pytest.fixture
def run_dirs(workspace) -> SessionDirs:
    return SessionDirs(workspace, workspace.runs_dir / TS_RUN)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def test_server_thread_metadata_carries_run_dir_only_when_set(run_dirs):
    daemon = _build_thread_metadata(
        graph_id="EvoScientist", **SessionDirs(run_dirs.workspace).metadata()
    )
    run = _build_thread_metadata(graph_id="EvoScientist", **run_dirs.metadata())
    assert "run_dir" not in daemon
    assert run["workspace_dir"] == run_dirs.workspace.key
    assert run["run_dir"] == run_dirs.run_dir.as_posix()


# ---------------------------------------------------------------------------
# Thread picker
# ---------------------------------------------------------------------------


def test_picker_groups_run_threads_under_their_run_folder(workspace):
    from EvoScientist.cli.widgets.thread_selector import _build_items

    run = workspace.runs_dir / TS_RUN
    threads = [
        {"thread_id": "first", **SessionDirs(workspace, run).metadata()},
        {"thread_id": "second", **SessionDirs(workspace, run).metadata()},
        {"thread_id": "daemon", **SessionDirs(workspace).metadata()},
    ]
    items = _build_items(threads)
    subheaders = [i["label"] for i in items if i["type"] == "subheader"]
    assert f"🔁 runs/{TS_RUN}" in subheaders
    run_rows = []
    for item in items:
        if item["type"] == "subheader":
            current = item["label"]
        elif item["type"] == "thread" and current.startswith("🔁"):
            run_rows.append(item["thread"]["thread_id"])
    assert sorted(run_rows) == ["first", "second"]


def test_picker_shortens_windows_style_home(monkeypatch):
    """Stored folders are POSIX on every platform; the home prefix must match."""
    import os

    from EvoScientist.cli.widgets.thread_selector import _normalize_path

    monkeypatch.setattr(
        os.path, "expanduser", lambda p: "C:\\Users\\x" if p == "~" else p
    )
    assert _normalize_path("C:/Users/x/proj") == "~/proj"


# ---------------------------------------------------------------------------
# /resume switches the whole session
# ---------------------------------------------------------------------------


class _ResumeUI:
    def __init__(self) -> None:
        self.resumed: list[tuple[str, SessionDirs | None]] = []

    def append_system(self, text: str, style: str = "dim") -> None:
        pass

    async def handle_session_resume(self, thread_id, dirs=None) -> None:
        self.resumed.append((thread_id, dirs))


async def _resume(ctx, metadata, thread_id="t2"):
    from EvoScientist.commands.implementation.session import ResumeCommand

    ctx.graph_gateway = FakeGraphGateway(
        thread_store=FakeThreadStore(resolved_thread_id=thread_id, metadata=metadata)
    )
    await ResumeCommand().execute(ctx, [thread_id])


def _ctx(workspace, ui, run_dir=None):
    from EvoScientist.commands.base import CommandContext

    return CommandContext(
        agent=None,
        thread_id="t1",
        ui=ui,
        dirs=SessionDirs(workspace, run_dir),
    )


async def test_resume_into_another_workspace_switches_workspace_and_run_dir(
    workspace, tmp_path
):
    other = Workspace(tmp_path / "other")
    run = other.runs_dir / TS_RUN
    ui = _ResumeUI()
    ctx = _ctx(workspace, ui)

    await _resume(ctx, SessionDirs(other, run).metadata())

    assert ctx.workspace == other
    assert ctx.run_dir == run
    assert ui.resumed == [("t2", SessionDirs(other, run))]


# ---------------------------------------------------------------------------
# Memory belongs to the workspace
# ---------------------------------------------------------------------------


def test_run_mode_memory_is_keyed_to_the_workspace_root(run_dirs, monkeypatch):
    import EvoScientist.EvoScientist as evo
    import EvoScientist.middleware as mw_mod
    import EvoScientist.middleware.model_fallback as fallback

    seen: list = []

    def _spy(memory_dir=None, *, workspace_dir, **_kwargs):
        seen.append(Path(workspace_dir))
        raise RuntimeError("captured")

    monkeypatch.setattr(mw_mod, "create_memory_middleware", _spy)
    monkeypatch.setattr(evo, "_ensure_config", lambda *_a: MagicMock())
    monkeypatch.setattr(evo, "_ensure_chat_model", lambda: object())
    monkeypatch.setattr(fallback, "seed_fallback_chain", lambda _cfg: None)
    monkeypatch.setattr(
        evo.MemoryControls, "from_config", classmethod(lambda cls, _cfg: MagicMock())
    )

    with pytest.raises(RuntimeError, match="captured"):
        evo._get_default_middleware(
            workspace=run_dirs.workspace, work_dir=run_dirs.work_dir
        )
    assert seen == [run_dirs.workspace.root]


# ---------------------------------------------------------------------------
# Server pinned to (workspace, run folder)
# ---------------------------------------------------------------------------


def test_server_starts_in_the_root_with_the_run_folder_in_its_env(
    run_dirs, monkeypatch, tmp_path, runtime_paths
):
    from tests.test_langgraph_dev_deploy_mode import _patch_start_prereqs, _PopenAbort

    monkeypatch.setenv("EVOSCIENTIST_RUN_DIR", "/inherited")
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(
            workspace_dir=run_dirs.workspace.root,
            run_dir=run_dirs.run_dir,
            port=16190,
        )
    assert captured["env"]["EVOSCIENTIST_WORKSPACE_DIR"] == str(run_dirs.workspace.root)
    assert captured["env"]["EVOSCIENTIST_RUN_DIR"] == str(run_dirs.run_dir)


def test_daemon_server_env_has_no_run_folder(
    workspace, monkeypatch, tmp_path, runtime_paths
):
    from tests.test_langgraph_dev_deploy_mode import _patch_start_prereqs, _PopenAbort

    monkeypatch.setenv("EVOSCIENTIST_RUN_DIR", "/inherited")
    captured = _patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    with pytest.raises(_PopenAbort):
        manager.start_langgraph_dev(workspace_dir=workspace.root, port=16191)
    assert "EVOSCIENTIST_RUN_DIR" not in captured["env"]


def _reuse_running_server(monkeypatch):
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: True)
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", None)
    monkeypatch.setattr(manager, "_PROCESS_RUN_DIR", None)
    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True
    return cfg


def test_reuse_refuses_another_run_folder_of_the_same_workspace(
    workspace, monkeypatch, runtime_paths
):
    manager._write_workspace_sidecar(
        workspace_dir=workspace.root, pid=1, run_dir=workspace.runs_dir / "a"
    )
    cfg = _reuse_running_server(monkeypatch)
    for run_dir in (workspace.runs_dir / "b", None):
        with pytest.raises(manager.WorkspaceMismatchError) as exc:
            manager.ensure_langgraph_dev(
                cfg, workspace_dir=workspace.root, run_dir=run_dir
            )
        # Same workspace: pointing at --workdir would name the user's own folder.
        assert "--workdir" not in str(exc.value)


def test_reuse_accepts_the_same_pair(run_dirs, monkeypatch, runtime_paths):
    manager._write_workspace_sidecar(
        workspace_dir=run_dirs.workspace.root, pid=1, run_dir=run_dirs.run_dir
    )
    cfg = _reuse_running_server(monkeypatch)
    manager.ensure_langgraph_dev(
        cfg, workspace_dir=run_dirs.workspace.root, run_dir=run_dirs.run_dir
    )


def test_owned_server_restarts_for_another_run_folder(
    workspace, monkeypatch, runtime_paths
):
    class _LiveProc:
        def poll(self):
            return None

    stopped: list = []
    started: list = []
    monkeypatch.setattr(manager, "_PROCESS", _LiveProc())
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", workspace.root)
    monkeypatch.setattr(manager, "_PROCESS_RUN_DIR", workspace.runs_dir / "a")
    monkeypatch.setattr(manager, "_PROCESS_DEPLOY_MODE", False)
    monkeypatch.setattr(manager, "stop_langgraph_dev", lambda *a: stopped.append(1))
    monkeypatch.setattr(manager, "_wait_for_port_release", lambda *a, **k: True)
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_kw: False)
    monkeypatch.setattr(
        manager, "start_langgraph_dev", lambda **kw: started.append(kw) or object()
    )
    monkeypatch.setattr(manager.atexit, "register", lambda *a, **k: None)
    cfg = manager.EvoScientistConfig()
    cfg.enable_async_subagents = True

    manager.ensure_langgraph_dev(
        cfg, workspace_dir=workspace.root, run_dir=workspace.runs_dir / "b"
    )

    assert stopped == [1]
    assert started[0]["workspace_dir"] == workspace.root
    assert started[0]["run_dir"] == workspace.runs_dir / "b"


def test_server_graphs_work_in_the_run_folder_and_crons_in_the_root(
    run_dirs, monkeypatch
):
    """Deployed sub-agents work where the session works; scheduled tasks
    belong to the workspace and work in its root."""
    import EvoScientist.memory.agents as memory_agents
    import EvoScientist.subagents._factory as factory
    import EvoScientist.subagents.expert_container_async as expert_async
    from EvoScientist.paths import process_session_dirs, process_workspace

    built: dict[str, dict] = {}

    def _record(name):
        def _build(*args, **kwargs):
            built[args[0] if args and isinstance(args[0], str) else name] = kwargs
            return object()

        return _build

    monkeypatch.setattr(factory, "build_async_subagent_graph", _record("async"))
    monkeypatch.setattr(
        expert_async, "build_expert_container_async_graph", _record("expert")
    )
    for name in (
        "build_memory_worker_graph",
        "build_observation_linker_graph",
        "build_autoskills_graph",
    ):
        monkeypatch.setattr(memory_agents, name, lambda *a, **k: object())
    monkeypatch.setenv("EVOSCIENTIST_WORKSPACE_DIR", str(run_dirs.workspace.root))
    monkeypatch.setenv("EVOSCIENTIST_RUN_DIR", str(run_dirs.run_dir))
    process_workspace.cache_clear()
    process_session_dirs.cache_clear()
    monkeypatch.delitem(sys.modules, "EvoScientist.langgraph_dev.graphs", raising=False)

    importlib.import_module("EvoScientist.langgraph_dev.graphs")

    assert built["writing-agent"]["work_dir"] == run_dirs.run_dir
    assert built["expert"]["work_dir"] == run_dirs.run_dir
    assert built["scheduler"] == {"workspace": run_dirs.workspace}


# ---------------------------------------------------------------------------
# Folders stored before run_dir are upgraded once
# ---------------------------------------------------------------------------


def _write_rows(db: Path, rows: list[tuple[str, str, dict]]) -> None:
    """Checkpoint rows as the saver writes them (metadata as a JSON BLOB)."""
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE IF NOT EXISTS checkpoints (thread_id TEXT, "
        "checkpoint_ns TEXT, checkpoint_id TEXT PRIMARY KEY, "
        "parent_checkpoint_id TEXT, type TEXT, checkpoint BLOB, metadata BLOB)"
    )
    for thread_id, checkpoint_id, meta in rows:
        con.execute(
            "INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?)",
            (
                thread_id,
                "",
                checkpoint_id,
                None,
                "empty",
                b"",
                json.dumps(meta).encode(),
            ),
        )
    con.commit()
    con.close()


def _stored(db: Path) -> dict[str, tuple[str, dict]]:
    """``checkpoint_id -> (SQLite type of metadata, metadata)``."""
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT checkpoint_id, typeof(metadata), metadata FROM checkpoints"
    ).fetchall()
    con.close()
    return {cid: (kind, json.loads(meta)) for cid, kind, meta in rows}


def _upgrade_recorded(db: Path) -> bool:
    """True if the upgrade's bit of ``user_version`` is set."""
    from EvoScientist.sessions import _STORED_DIRS_VERSION

    con = sqlite3.connect(db)
    version = con.execute("PRAGMA user_version").fetchone()[0]
    con.close()
    return bool(version & _STORED_DIRS_VERSION)


@pytest.fixture
def sessions_db(tmp_path, monkeypatch):
    import EvoScientist.paths as paths
    import EvoScientist.sessions as sessions

    db = tmp_path / "sessions.db"
    monkeypatch.setattr(sessions, "get_db_path", lambda: db)
    monkeypatch.setattr(paths, "MEMORIES_DIR", tmp_path / "memories")
    return db


def test_upgrade_splits_generated_run_folders_once(workspace, sessions_db):
    from EvoScientist.sessions import _upgrade_stored_dirs

    run = workspace.runs_dir / TS_RUN
    named = workspace.runs_dir / "exp"
    base = {"agent_name": "EvoScientist", "model": "m"}
    _write_rows(
        sessions_db,
        [
            ("t1", "old-run", {**base, "workspace_dir": str(run)}),
            ("t2", "named-run", {**base, "workspace_dir": str(named)}),
            ("t3", "daemon", {**base, "workspace_dir": f"{workspace.root}/"}),
            ("t4", "new", {**base, **SessionDirs(workspace, run).metadata()}),
        ],
    )

    asyncio.run(_upgrade_stored_dirs())

    stored = _stored(sessions_db)
    assert stored["old-run"] == (
        "blob",
        {**base, **SessionDirs(workspace, run).metadata()},
    )
    # A project can live in a folder called ``runs``: named runs stay roots.
    assert stored["named-run"][1] == {**base, "workspace_dir": str(named)}
    # Rows that gain no ``run_dir`` keep their spelling; readers normalise it.
    assert stored["daemon"][1] == {**base, "workspace_dir": f"{workspace.root}/"}
    assert stored["new"][1] == {**base, **SessionDirs(workspace, run).metadata()}
    assert _upgrade_recorded(sessions_db)


def test_rows_written_after_the_upgrade_are_never_split(workspace, sessions_db):
    """Continuing in an old run folder in daemon mode makes it a root."""
    from EvoScientist.sessions import _upgrade_stored_dirs

    folder = workspace.runs_dir / TS_RUN
    _write_rows(sessions_db, [("t1", "c1", {"workspace_dir": workspace.key})])
    asyncio.run(_upgrade_stored_dirs())
    _write_rows(sessions_db, [("t2", "c2", SessionDirs(Workspace(folder)).metadata())])

    asyncio.run(_upgrade_stored_dirs())

    assert _stored(sessions_db)["c2"][1] == {"workspace_dir": folder.as_posix()}


def test_upgrade_of_a_new_db_only_records_the_version(sessions_db):
    from EvoScientist.sessions import _upgrade_stored_dirs

    asyncio.run(_upgrade_stored_dirs())
    assert _upgrade_recorded(sessions_db)


def test_failed_upgrade_changes_nothing_and_retries(
    workspace, sessions_db, monkeypatch
):
    import EvoScientist.memory.autoskills.proposals as proposals
    from EvoScientist.sessions import (
        _upgrade_stored_dirs,
        _upgrade_stored_dirs_safely,
    )

    run = workspace.runs_dir / TS_RUN
    _write_rows(sessions_db, [("t1", "c1", {"workspace_dir": str(run)})])

    real_upgrade = proposals.upgrade_proposal_workspaces

    def _fail(_memory_dir):
        raise OSError("disk full")

    monkeypatch.setattr(proposals, "upgrade_proposal_workspaces", _fail)
    asyncio.run(_upgrade_stored_dirs_safely())

    assert _stored(sessions_db)["c1"][1] == {"workspace_dir": str(run)}
    assert not _upgrade_recorded(sessions_db)

    monkeypatch.setattr(proposals, "upgrade_proposal_workspaces", real_upgrade)
    asyncio.run(_upgrade_stored_dirs())
    assert _stored(sessions_db)["c1"][1] == SessionDirs(workspace, run).metadata()


def test_upgrade_keeps_bloat_sweep_gate(workspace, sessions_db, monkeypatch):
    """The legacy-bloat sweep and the folder upgrade are gated independently."""
    import EvoScientist.sessions as sessions
    from EvoScientist.sessions import _needs_migration, _upgrade_stored_dirs

    _write_rows(sessions_db, [("t1", "c1", {"workspace_dir": workspace.key})])
    monkeypatch.setattr(sessions, "_MIGRATION_THRESHOLD_BYTES", 0)
    assert asyncio.run(_needs_migration()) is True

    asyncio.run(_upgrade_stored_dirs())

    assert _upgrade_recorded(sessions_db)
    assert asyncio.run(_needs_migration()) is True


def test_upgrade_skips_unreadable_rows(workspace, sessions_db, tmp_path):
    """Unreadable rows leave the others upgraded and the upgrade done."""
    from EvoScientist.sessions import _upgrade_stored_dirs

    run = workspace.runs_dir / TS_RUN
    unresolvable = f"{workspace.root}/evil\x00x"
    _write_rows(
        sessions_db,
        [
            ("t1", "good", {"workspace_dir": str(run)}),
            ("t2", "nul", {"workspace_dir": unresolvable}),
        ],
    )
    con = sqlite3.connect(sessions_db)
    con.execute(
        "INSERT INTO checkpoints VALUES ('t3', '', 'broken', NULL, 'empty', NULL, "
        "CAST('{not json' AS BLOB))"
    )
    con.commit()
    _write_proposal(tmp_path / "memories", "nul", unresolvable)

    asyncio.run(_upgrade_stored_dirs())

    rows = dict(
        con.execute(
            "SELECT checkpoint_id, CAST(metadata AS TEXT) FROM checkpoints"
        ).fetchall()
    )
    con.close()
    assert json.loads(rows["good"]) == SessionDirs(workspace, run).metadata()
    assert json.loads(rows["nul"]) == {"workspace_dir": unresolvable}
    assert rows["broken"] == "{not json"
    assert _upgrade_recorded(sessions_db)


def test_thread_readers_return_stored_run_dir(workspace, sessions_db):
    from EvoScientist.sessions import get_thread_metadata, list_threads

    run = workspace.runs_dir / TS_RUN
    base = {
        "agent_name": "EvoScientist",
        "model": "m",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    _write_rows(
        sessions_db,
        [
            ("t-run", "c1", {**base, **SessionDirs(workspace, run).metadata()}),
            ("t-daemon", "c2", {**base, **SessionDirs(workspace).metadata()}),
        ],
    )

    by_id = {t["thread_id"]: t for t in asyncio.run(list_threads())}
    assert by_id["t-run"]["run_dir"] == run.resolve().as_posix()
    assert by_id["t-daemon"]["run_dir"] is None
    assert asyncio.run(get_thread_metadata("t-run"))["run_dir"] == (
        run.resolve().as_posix()
    )
    assert asyncio.run(get_thread_metadata("t-daemon"))["run_dir"] is None


def _write_proposal(memory_dir: Path, name: str, workspace_dir: str) -> Path:
    from EvoScientist.memory.autoskills.proposals import _proposal_root

    manifest_dir = _proposal_root(memory_dir) / name
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "proposal_id": name,
                "skill_name": name,
                "description": "d",
                "status": "pending",
                "operation": "create",
                "created_at": "2026-09-01T00:00:00+00:00",
                "updated_at": "2026-09-01T00:00:00+00:00",
                "cluster_hash": "h",
                "source_observation_ids": [],
                "workspace_dir": workspace_dir,
            }
        )
    )
    (manifest_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\n")
    return manifest


def test_upgrade_moves_old_run_mode_proposals_to_their_workspace(
    workspace, sessions_db, tmp_path
):
    from EvoScientist.memory.autoskills.proposals import list_skill_proposals
    from EvoScientist.sessions import _upgrade_stored_dirs

    memory_dir = tmp_path / "memories"
    _write_proposal(memory_dir, "demo", str(workspace.runs_dir / TS_RUN))

    asyncio.run(_upgrade_stored_dirs())

    listed = list_skill_proposals(memory_dir, workspace_dir=workspace.root)
    assert [p.skill_name for p in listed] == ["demo"]


async def test_cli_and_server_upgrade_before_reading(
    workspace, sessions_db, monkeypatch
):
    """Opening the DB from the CLI or the server upgrades old rows first, so
    the WebUI lists an old run-mode thread with its run folder."""
    import EvoScientist.sessions as sessions

    run = workspace.runs_dir / TS_RUN
    cli_tid, server_tid = (str(uuid.uuid4()) for _ in range(2))
    base = {"graph_id": "EvoScientist", "updated_at": "2026-09-01T00:00:00+00:00"}
    run_meta = SessionDirs(workspace, run).metadata()

    _write_rows(sessions_db, [(cli_tid, "c1", {**base, "workspace_dir": str(run)})])
    async with sessions.get_checkpointer():
        pass
    assert _stored(sessions_db)["c1"][1] == {**base, **run_meta}

    sessions_db.unlink()
    _write_rows(sessions_db, [(server_tid, "c2", {**base, "workspace_dir": str(run)})])
    store: dict = {"threads": []}
    monkeypatch.setitem(
        sys.modules,
        "langgraph_runtime_inmem.database",
        SimpleNamespace(GLOBAL_STORE=store),
    )
    monkeypatch.setattr(sessions, "_api_session_dirs", lambda: SessionDirs(workspace))
    async with sessions.create_checkpointer_for_langgraph_api():
        pass
    (restored,) = store["threads"]
    assert str(restored["thread_id"]) == server_tid
    assert restored["metadata"]["workspace_dir"] == run_meta["workspace_dir"]
    assert restored["metadata"]["run_dir"] == run_meta["run_dir"]


def _stamped(run_dirs: SessionDirs, monkeypatch, graph_id: str) -> dict:
    """Metadata the server's checkpointer writes for an unstamped row."""
    import EvoScientist.sessions as sessions

    captured: dict = {}

    async def _fake_super_aput(self, config, checkpoint, metadata, new_versions):
        captured.update(metadata)

    monkeypatch.setattr(sessions, "_api_session_dirs", lambda: run_dirs)
    monkeypatch.setattr(sessions.PruningCheckpointer, "aput", _fake_super_aput)
    saver = object.__new__(sessions._ApiPruningCheckpointer)
    asyncio.run(saver.aput({}, {}, {"graph_id": graph_id}, {}))
    return captured


def test_server_stamps_its_run_folder_on_unstamped_rows(run_dirs, monkeypatch):
    stamped = _stamped(run_dirs, monkeypatch, "writing-agent")

    assert stamped["workspace_dir"] == run_dirs.workspace.key
    assert stamped["run_dir"] == run_dirs.run_dir.as_posix()


@pytest.mark.parametrize("graph_id", ["scheduler", "evomemory-turn-worker"])
def test_server_stamps_root_graph_rows_without_run_dir(run_dirs, monkeypatch, graph_id):
    """Scheduled tasks and memory workers work in the workspace root."""
    stamped = _stamped(run_dirs, monkeypatch, graph_id)

    assert stamped["workspace_dir"] == run_dirs.workspace.key
    assert "run_dir" not in stamped


# ---------------------------------------------------------------------------
# Channel media from a run folder
# ---------------------------------------------------------------------------


def _print_file_cmd(path: Path) -> str:
    """Cross-platform command that prints the file at *path*."""
    return f"type {path}" if sys.platform == "win32" else f"cat {path}"


@pytest.fixture
def media_file(workspace) -> Path:
    workspace.media_dir.mkdir(parents=True)
    path = workspace.media_dir / "paper.pdf"
    path.write_text("attachment")
    return path


@pytest.fixture
def _plain_config(monkeypatch):
    import EvoScientist.EvoScientist as evo

    monkeypatch.setattr(
        evo,
        "_ensure_config",
        lambda *_a: SimpleNamespace(sandbox_execute_timeout=30, dangerous_mode=False),
    )


@pytest.mark.usefixtures("_plain_config")
def test_daemon_sandbox_mounts_nothing_extra(workspace, media_file):
    from EvoScientist.EvoScientist import _get_default_backend

    backend = _get_default_backend(workspace)
    assert backend.default._media_dir is None
    assert backend.read(str(media_file)).error is None


@pytest.mark.usefixtures("_plain_config")
def test_run_mode_sandbox_reaches_channel_media(run_dirs, media_file):
    from EvoScientist.EvoScientist import _get_default_backend

    run_dirs.run_dir.mkdir(parents=True)
    backend = _get_default_backend(run_dirs.workspace, work_dir=run_dirs.work_dir)

    assert backend.read(str(media_file)).error is None
    assert backend.execute(_print_file_cmd(media_file)).output.strip() == "attachment"


@pytest.mark.usefixtures("_plain_config")
def test_run_mode_reaches_media_through_a_symlinked_folder(run_dirs):
    """Channels reference attachments through ``<root>/media`` even when that
    folder is a symlink."""
    from EvoScientist.EvoScientist import _get_default_backend

    uploads = run_dirs.workspace.root / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "paper.pdf").write_text("attachment")
    run_dirs.workspace.media_dir.symlink_to(uploads, target_is_directory=True)
    run_dirs.run_dir.mkdir(parents=True)
    backend = _get_default_backend(run_dirs.workspace, work_dir=run_dirs.work_dir)
    referenced = run_dirs.workspace.media_dir / "paper.pdf"

    assert backend.read(str(referenced)).error is None
    assert backend.execute(_print_file_cmd(referenced)).output.strip() == "attachment"
    assert backend.write(str(run_dirs.workspace.media_dir / "new.pdf"), "x").error
    # The resolved spelling (what ``ls -l`` or ``realpath`` would show) works
    # for the file tools and for commands alike.
    resolved = uploads / "paper.pdf"
    assert backend.read(str(resolved)).error is None
    assert backend.execute(_print_file_cmd(resolved)).output.strip() == "attachment"


@pytest.mark.usefixtures("_plain_config")
def test_run_mode_file_tools_cannot_change_channel_media(run_dirs, media_file):
    """Attachments are shared by the workspace; a run writes in its own folder,
    including its own ``/media``."""
    from EvoScientist.EvoScientist import _get_default_backend

    run_dirs.run_dir.mkdir(parents=True)
    backend = _get_default_backend(run_dirs.workspace, work_dir=run_dirs.work_dir)

    assert backend.write(str(run_dirs.workspace.media_dir / "new.pdf"), "x").error
    assert backend.edit(str(media_file), "attachment", "changed").error
    assert backend.delete(str(media_file)).error
    assert media_file.read_text() == "attachment"
    assert not (run_dirs.workspace.media_dir / "new.pdf").exists()

    assert backend.write("/media/plot.png", "x").error is None
    assert (run_dirs.run_dir / "media" / "plot.png").exists()


@pytest.mark.usefixtures("_plain_config")
def test_run_mode_media_mount_does_not_open_the_rest_of_the_workspace(
    run_dirs, media_file
):
    from EvoScientist.EvoScientist import _get_default_backend

    run_dirs.run_dir.mkdir(parents=True)
    (run_dirs.workspace.root / "secret.txt").write_text("no")
    backend = _get_default_backend(run_dirs.workspace, work_dir=run_dirs.work_dir)

    # Refused as a read error the agent can act on, not as a traceback.
    result = backend.read(f"{run_dirs.workspace.media_dir}/../secret.txt")
    assert result.error is not None
    assert result.file_data is None


def _tool_path(path: Path) -> str:
    """*path* as the ``ls``/``glob``/``grep`` tools hand it to the backend."""
    from deepagents.backends.utils import validate_path

    return validate_path(str(path))


@pytest.mark.usefixtures("_plain_config")
def test_run_mode_file_tools_list_channel_media(run_dirs, media_file):
    """An agent finds attachments by listing or searching ``<root>/media``."""
    from EvoScientist.EvoScientist import _get_default_backend

    run_dirs.run_dir.mkdir(parents=True)
    backend = _get_default_backend(run_dirs.workspace, work_dir=run_dirs.work_dir)
    media = _tool_path(run_dirs.workspace.media_dir)

    listed = [entry["path"] for entry in backend.ls(media).entries]
    globbed = [match["path"] for match in backend.glob("*.pdf", media).matches]
    grepped = [match["path"] for match in backend.grep("attachment", media).matches]
    assert listed == globbed == grepped == [media_file.as_posix()]
    assert backend.read(listed[0]).error is None


@pytest.mark.usefixtures("_plain_config")
@pytest.mark.parametrize("folder", ["media", "uploads"])
def test_run_mode_lists_media_through_symlinked_folder(run_dirs, folder):
    """Either spelling of a symlinked media folder lists the attachments
    under the path channels reference them by."""
    from EvoScientist.EvoScientist import _get_default_backend

    uploads = run_dirs.workspace.root / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "paper.pdf").write_text("attachment")
    run_dirs.workspace.media_dir.symlink_to(uploads, target_is_directory=True)
    run_dirs.run_dir.mkdir(parents=True)
    backend = _get_default_backend(run_dirs.workspace, work_dir=run_dirs.work_dir)
    listed = _tool_path(run_dirs.workspace.root / folder)

    referenced = (run_dirs.workspace.media_dir / "paper.pdf").as_posix()
    assert [entry["path"] for entry in backend.ls(listed).entries] == [referenced]
    assert [m["path"] for m in backend.glob("*.pdf", listed).matches] == [referenced]


@pytest.mark.usefixtures("_plain_config")
def test_run_mode_media_listing_stays_inside_media_folder(run_dirs, media_file):
    from EvoScientist.EvoScientist import _get_default_backend

    root = run_dirs.workspace.root
    (root / "secret.txt").write_text("attachment")
    (root / "media" / "leak.txt").symlink_to(root / "secret.txt")
    (root / "media" / "root").symlink_to(root, target_is_directory=True)
    run_dirs.run_dir.mkdir(parents=True)
    backend = _get_default_backend(run_dirs.workspace, work_dir=run_dirs.work_dir)
    media = _tool_path(run_dirs.workspace.media_dir)

    expected = [media_file.as_posix()]
    assert [entry["path"] for entry in backend.ls(media).entries] == expected
    assert [match["path"] for match in backend.glob("*", media).matches] == expected
    matches = backend.grep("attachment", media).matches
    assert [match["path"] for match in matches] == expected
    linked = backend.ls(_tool_path(run_dirs.workspace.media_dir / "root"))
    assert linked.error is not None
    assert linked.entries is None


async def test_channel_attachments_follow_resume_into_another_workspace(
    workspace, tmp_path, monkeypatch
):
    """After ``/resume`` into another workspace, new attachments land where
    that workspace's sandbox can read them."""
    from unittest.mock import AsyncMock, patch

    import EvoScientist.cli.channel as channel_mod
    from EvoScientist.channels.bus import MessageBus
    from EvoScientist.channels.channel_manager import ChannelManager
    from EvoScientist.cli.commands import (
        ServeRuntimeState,
        _make_serve_handle_session_resume_cb,
    )
    from tests.fakes import StubChannel

    other = Workspace(tmp_path / "other")
    manager = ChannelManager(MessageBus(), media_dir=workspace.media_dir)
    channel = StubChannel()
    manager.register(channel)
    monkeypatch.setattr(channel_mod, "_manager", manager)
    state = ServeRuntimeState(
        agent=MagicMock(),
        thread_id="t1",
        dirs=SessionDirs(workspace),
        config=MagicMock(),
        runtime_gateways=MagicMock(),
        async_runtime=MagicMock(),
    )
    resume = _make_serve_handle_session_resume_cb(state, None, config=state.config)

    with (
        patch(
            "EvoScientist.cli.commands._sync_background_agent_server_workspace",
            new=AsyncMock(),
        ),
        patch("EvoScientist.cli.commands._load_agent", return_value=MagicMock()),
    ):
        await resume("t2", SessionDirs(other))

    assert channel._media_path("photo.jpg") == other.media_dir / "photo.jpg"
