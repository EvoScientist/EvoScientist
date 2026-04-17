"""Tests for EvoScientist.paths — set_workspace_root and ensure_dirs."""

import os
from unittest import mock

import pytest

from EvoScientist import paths


@pytest.fixture(autouse=True)
def _restore_paths():
    """Snapshot module-level path globals and restore after each test."""
    orig = {
        "WORKSPACE_ROOT": paths.WORKSPACE_ROOT,
        "RUNS_DIR": paths.RUNS_DIR,
        "DATA_DIR": paths.DATA_DIR,
        "MEMORIES_DIR": paths.MEMORIES_DIR,
        "MEMORY_DIR": paths.MEMORY_DIR,
        "GLOBAL_SKILLS_DIR": paths.GLOBAL_SKILLS_DIR,
        "GLOBAL_MEMORIES_DIR": paths.GLOBAL_MEMORIES_DIR,
        "USER_SKILLS_DIR": paths.USER_SKILLS_DIR,
        "_active_workspace": paths._active_workspace,
    }
    yield
    paths.WORKSPACE_ROOT = orig["WORKSPACE_ROOT"]
    paths.RUNS_DIR = orig["RUNS_DIR"]
    paths.DATA_DIR = orig["DATA_DIR"]
    paths.MEMORIES_DIR = orig["MEMORIES_DIR"]
    paths.MEMORY_DIR = orig["MEMORY_DIR"]
    paths.GLOBAL_SKILLS_DIR = orig["GLOBAL_SKILLS_DIR"]
    paths.GLOBAL_MEMORIES_DIR = orig["GLOBAL_MEMORIES_DIR"]
    paths.USER_SKILLS_DIR = orig["USER_SKILLS_DIR"]
    paths._active_workspace = orig["_active_workspace"]


class TestSetWorkspaceRoot:
    """Tests for set_workspace_root()."""

    def test_updates_derived_dirs(self, tmp_path, monkeypatch):
        """set_workspace_root should update WORKSPACE_ROOT and all derived dirs."""
        monkeypatch.delenv("EVOSCIENTIST_MEMORIES_DIR", raising=False)
        monkeypatch.delenv("EVOSCIENTIST_MEMORY_DIR", raising=False)
        monkeypatch.delenv("EVOSCIENTIST_RUNS_DIR", raising=False)
        monkeypatch.delenv("EVOSCIENTIST_SKILLS_DIR", raising=False)

        new_root = tmp_path / "my_workspace"
        new_root.mkdir()

        paths.set_workspace_root(new_root)

        assert paths.WORKSPACE_ROOT == new_root.resolve()
        assert paths.RUNS_DIR == new_root.resolve() / "runs"
        # MEMORIES_DIR is global — not derived from workspace root
        assert paths.MEMORIES_DIR == paths.GLOBAL_MEMORIES_DIR
        assert paths.USER_SKILLS_DIR == new_root.resolve() / "skills"

    def test_resets_active_workspace(self, tmp_path):
        """set_workspace_root should reset _active_workspace to new root."""
        new_root = tmp_path / "ws"
        new_root.mkdir()

        # Set active workspace to something different first
        paths._active_workspace = tmp_path / "other"

        paths.set_workspace_root(new_root)

        assert paths._active_workspace == new_root.resolve()

    def test_preserves_env_overrides(self, tmp_path):
        """Dirs set via env vars should NOT be overwritten by set_workspace_root."""
        custom_mem = tmp_path / "custom_memory"
        custom_skills = tmp_path / "custom_skills"
        custom_runs = tmp_path / "custom_runs"

        env = {
            "EVOSCIENTIST_MEMORIES_DIR": str(custom_mem),
            "EVOSCIENTIST_SKILLS_DIR": str(custom_skills),
            "EVOSCIENTIST_RUNS_DIR": str(custom_runs),
        }

        new_root = tmp_path / "ws"
        new_root.mkdir()

        with mock.patch.dict(os.environ, env):
            paths.set_workspace_root(new_root)

            # WORKSPACE_ROOT and _active_workspace should still update
            assert paths.WORKSPACE_ROOT == new_root.resolve()
            assert paths._active_workspace == new_root.resolve()

            # MEMORIES_DIR respects env override
            assert paths.MEMORIES_DIR == custom_mem.expanduser()
            assert paths.USER_SKILLS_DIR == custom_skills.expanduser()
            assert paths.RUNS_DIR == custom_runs.expanduser()

    def test_accepts_string_path(self, tmp_path):
        """set_workspace_root should accept str as well as Path."""
        new_root = tmp_path / "str_ws"
        new_root.mkdir()

        paths.set_workspace_root(str(new_root))

        assert paths.WORKSPACE_ROOT == new_root.resolve()


class TestEnsureDirsUsesUpdatedPaths:
    """ensure_dirs should create dirs at the currently set paths."""

    def test_ensure_dirs_uses_updated_paths(self, tmp_path):
        """ensure_dirs creates the global memories dir (not workspace-local)."""
        new_root = tmp_path / "workspace"
        new_root.mkdir()

        custom_mem = tmp_path / "memories"

        with mock.patch.dict(
            os.environ, {"EVOSCIENTIST_MEMORIES_DIR": str(custom_mem)}
        ):
            paths.set_workspace_root(new_root)
            paths.ensure_dirs()

            assert custom_mem.is_dir()
            assert not (new_root / "memory").exists()  # no longer workspace-local
            assert not (new_root / "memories").exists()
            assert not (
                new_root / "skills"
            ).exists()  # skills created on demand by install_skill()


class TestDataDir:
    """Tests for DATA_DIR and global data-dir helpers."""

    def test_global_skills_dir_under_data_dir(self):
        """GLOBAL_SKILLS_DIR must live under DATA_DIR."""
        assert paths.GLOBAL_SKILLS_DIR == paths.DATA_DIR / "skills"

    def test_global_memories_dir_under_data_dir(self):
        """GLOBAL_MEMORIES_DIR must live under DATA_DIR."""
        assert paths.GLOBAL_MEMORIES_DIR == paths.DATA_DIR / "memories"


class TestLegacySessionsDbMigration:
    """Tests for migrate_legacy_sessions_db() — transitional helper.

    Tests redirect ``DATA_DIR`` and ``Path.home()`` so the real user home
    is never touched.
    """

    def _setup(self, tmp_path, monkeypatch):
        """Redirect data dir to tmp_path/new_data and Path.home() to
        tmp_path/fake_home so legacy resolves to tmp_path/fake_home/.config/evoscientist.
        """
        data_dir = tmp_path / "new_data"
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        legacy_dir = fake_home / ".config" / "evoscientist"
        monkeypatch.setattr(paths, "DATA_DIR", data_dir)
        monkeypatch.setattr(paths.Path, "home", classmethod(lambda cls: fake_home))
        return data_dir, legacy_dir

    def test_copies_sqlite_trio_when_legacy_exists(self, tmp_path, monkeypatch):
        """All three SQLite files should be copied to the new location."""
        data_dir, legacy_dir = self._setup(tmp_path, monkeypatch)
        legacy_dir.mkdir(parents=True)
        for name in ("sessions.db", "sessions.db-wal", "sessions.db-shm"):
            (legacy_dir / name).write_bytes(b"stub-" + name.encode())

        paths.migrate_legacy_sessions_db()

        for name in ("sessions.db", "sessions.db-wal", "sessions.db-shm"):
            assert (data_dir / name).read_bytes() == b"stub-" + name.encode()
            # Legacy files must remain (copy, not move)
            assert (legacy_dir / name).exists()
        assert (data_dir / ".migrated").exists()

    def test_idempotent_via_marker(self, tmp_path, monkeypatch):
        """Once .migrated exists, migration should be a no-op."""
        data_dir, legacy_dir = self._setup(tmp_path, monkeypatch)
        data_dir.mkdir()
        (data_dir / ".migrated").touch()
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "sessions.db").write_bytes(b"should-not-copy")

        paths.migrate_legacy_sessions_db()

        assert not (data_dir / "sessions.db").exists()

    def test_no_legacy_just_creates_marker(self, tmp_path, monkeypatch):
        """When legacy dir doesn't exist, only the marker is created."""
        data_dir, _ = self._setup(tmp_path, monkeypatch)

        paths.migrate_legacy_sessions_db()

        assert data_dir.is_dir()
        assert (data_dir / ".migrated").exists()
        assert not (data_dir / "sessions.db").exists()

    def test_does_not_overwrite_existing_files(self, tmp_path, monkeypatch):
        """If new location already has a file, don't overwrite it."""
        data_dir, legacy_dir = self._setup(tmp_path, monkeypatch)
        data_dir.mkdir()
        (data_dir / "sessions.db").write_bytes(b"new-content-keep")
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "sessions.db").write_bytes(b"legacy-content")

        paths.migrate_legacy_sessions_db()

        assert (data_dir / "sessions.db").read_bytes() == b"new-content-keep"

    def test_respects_xdg_config_home(self, tmp_path, monkeypatch):
        """Legacy source must honor XDG_CONFIG_HOME so users who customize it
        don't silently get skipped by the migration."""
        data_dir = tmp_path / "new_data"
        xdg = tmp_path / "xdg"
        legacy_dir = xdg / "evoscientist"
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "sessions.db").write_bytes(b"xdg-db")

        monkeypatch.setattr(paths, "DATA_DIR", data_dir)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

        paths.migrate_legacy_sessions_db()

        assert (data_dir / "sessions.db").read_bytes() == b"xdg-db"

    def test_marker_not_written_on_partial_failure(self, tmp_path, monkeypatch):
        """If any copy fails, the .migrated marker must not be written."""
        data_dir, legacy_dir = self._setup(tmp_path, monkeypatch)
        legacy_dir.mkdir(parents=True)
        for name in ("sessions.db", "sessions.db-wal"):
            (legacy_dir / name).write_bytes(b"ok")

        real_copy2 = paths.shutil.copy2

        def flaky_copy2(src, dst, *args, **kwargs):
            if str(src).endswith("sessions.db-wal"):
                raise OSError("simulated I/O failure")
            return real_copy2(src, dst, *args, **kwargs)

        with mock.patch.object(paths.shutil, "copy2", side_effect=flaky_copy2):
            paths.migrate_legacy_sessions_db()

        assert (data_dir / "sessions.db").exists()  # main db copied
        assert not (data_dir / ".migrated").exists()  # retry allowed
