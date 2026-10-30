"""The server home: one server per data folder, and the runtime store's backup."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from EvoScientist import paths
from EvoScientist.langgraph_dev import server_home


@pytest.fixture
def home(monkeypatch, tmp_path):
    """An empty server home this test's process holds nothing in."""
    folder = tmp_path / "server"
    monkeypatch.setattr(paths, "SERVER_HOME", folder)
    monkeypatch.setattr(server_home, "_held", None)
    monkeypatch.setattr(server_home, "LOCK_WAIT", 0.1)
    return folder


def _other_server(home):
    """The lock held as another server would."""
    from filelock import FileLock

    home.mkdir(mode=0o700, exist_ok=True)
    lock = FileLock(str(home / server_home.LOCK_FILE))
    lock.acquire()
    return lock


def test_server_lock_is_held_once(home):
    server_home.hold_server_lock()
    server_home.hold_server_lock()  # already held: nothing more

    assert server_home.server_running()
    if os.name == "posix":
        assert home.stat().st_mode & 0o777 == 0o700


def test_server_lock_refuses_second_server(home):
    lock = _other_server(home)
    try:
        with pytest.raises(server_home.ServerRunningError, match="already uses"):
            server_home.hold_server_lock()
        assert server_home.server_running()
    finally:
        lock.release()

    assert not server_home.server_running()
    server_home.hold_server_lock()


def test_server_running_is_false_without_server_home(home):
    assert not server_home.server_running()
    assert not home.exists()


@pytest.fixture
def served_from_home(home, monkeypatch):
    home.mkdir()
    monkeypatch.chdir(home)
    (home / server_home.store_file()).parent.mkdir(exist_ok=True)
    return home


def test_store_backup_is_dropped_once_loaded(served_from_home):
    store = served_from_home / server_home.store_file()
    store.write_bytes(b"store")

    server_home.back_up_store()
    server_home.check_store_loaded()

    assert sorted(p.name for p in store.parent.iterdir()) == [store.name]


def test_store_runtime_could_not_load_is_kept(served_from_home, caplog):
    """The runtime deletes a store it cannot load."""
    store = served_from_home / server_home.store_file()
    store.write_bytes(b"store")

    server_home.back_up_store()
    store.unlink()
    server_home.check_store_loaded()

    (kept,) = store.parent.iterdir()
    assert kept.name.startswith(f"{store.name}.unloadable-")
    assert kept.read_bytes() == b"store"
    assert str(kept.resolve()) in caplog.text


def test_process_elsewhere_is_not_server(home, monkeypatch, tmp_path):
    """The agent's commands run in project folders."""
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    assert not server_home.serves_from_home()
    monkeypatch.chdir(home)
    assert server_home.serves_from_home()


def test_store_backup_failure_never_stops_server(served_from_home, monkeypatch, caplog):
    """The backup is a safety net: the store itself may be fine."""
    store = served_from_home / server_home.store_file()
    store.write_bytes(b"store")

    def fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(server_home.shutil, "copy2", fail)
    server_home.back_up_store()
    server_home.check_store_loaded()

    assert "Could not back the server's store up" in caplog.text
    assert store.read_bytes() == b"store"


def test_failed_backup_leaves_none(served_from_home, monkeypatch):
    """A stale or partial backup would be kept as the store the runtime lost."""
    store = served_from_home / server_home.store_file()
    store.write_bytes(b"store")
    server_home.back_up_store()  # an earlier start's backup

    def partial(_src, dst):
        Path(dst).write_bytes(b"sto")
        raise OSError("disk full")

    monkeypatch.setattr(server_home.shutil, "copy2", partial)
    server_home.back_up_store()

    assert sorted(p.name for p in store.parent.iterdir()) == [store.name]
