"""Tests for EvoScientist.paths — the Workspace type, start resolution, ensure_dirs."""

import os
from pathlib import Path
from unittest import mock

import pytest

from EvoScientist import paths
from EvoScientist.paths import (
    Workspace,
    normalize_path,
    process_workspace,
    resolve_virtual_path,
    start_workspace_path,
)


@pytest.fixture(autouse=True)
def _restore_paths():
    """Snapshot module-level path globals and restore after each test."""
    orig = {
        "DATA_DIR": paths.DATA_DIR,
        "MEMORIES_DIR": paths.MEMORIES_DIR,
        "MEMORY_DIR": paths.MEMORY_DIR,
        "GLOBAL_SKILLS_DIR": paths.GLOBAL_SKILLS_DIR,
        "GLOBAL_MEMORIES_DIR": paths.GLOBAL_MEMORIES_DIR,
    }
    yield
    for name, value in orig.items():
        setattr(paths, name, value)


class TestWorkspace:
    """The Workspace value object."""

    def test_derived_dirs(self, tmp_path):
        ws = Workspace(tmp_path)
        root = tmp_path.resolve()
        assert ws.root == root
        assert ws.skills_dir == root / "skills"
        assert ws.runs_dir == root / "runs"
        assert ws.media_dir == root / "media"

    def test_accepts_string_path(self, tmp_path):
        assert Workspace(str(tmp_path)).root == tmp_path.resolve()

    def test_resolves_symlinks(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        assert Workspace(link) == Workspace(real)
        assert Workspace(link).root == real.resolve()

    def test_expands_user(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert Workspace("~/proj").root == (tmp_path / "proj").resolve()

    def test_does_not_create_the_folder(self, tmp_path):
        missing = tmp_path / "not-there"
        ws = Workspace(missing)
        assert ws.root == missing.resolve()
        assert not missing.exists()

    def test_key_is_posix_without_trailing_slash(self, tmp_path):
        ws = Workspace(f"{tmp_path}/proj/")
        assert ws.key == (tmp_path / "proj").resolve().as_posix()
        assert not ws.key.endswith("/")

    def test_is_immutable_and_hashable(self, tmp_path):
        ws = Workspace(tmp_path)
        with pytest.raises(AttributeError):
            ws.root = tmp_path / "other"  # type: ignore[misc]
        assert {ws: 1}[Workspace(tmp_path)] == 1

    def test_ignores_removed_dir_overrides(self, tmp_path, monkeypatch):
        """The per-workspace folders always derive from the root."""
        monkeypatch.setenv("EVOSCIENTIST_SKILLS_DIR", str(tmp_path / "s"))
        monkeypatch.setenv("EVOSCIENTIST_RUNS_DIR", str(tmp_path / "r"))
        monkeypatch.setenv("EVOSCIENTIST_MEDIA_DIR", str(tmp_path / "m"))
        ws = Workspace(tmp_path / "proj")
        assert ws.skills_dir == ws.root / "skills"
        assert ws.runs_dir == ws.root / "runs"
        assert ws.media_dir == ws.root / "media"


class TestNormalizePath:
    def test_equal_spellings_normalise_equal(self, tmp_path):
        a = normalize_path(f"{tmp_path}/x/../y")
        b = normalize_path(tmp_path / "y")
        assert a == b


class TestStartWorkspacePath:
    """Which folder a process starts in: workdir, then default_workdir, then cwd."""

    def test_workdir_wins(self, tmp_path):
        got = start_workspace_path(tmp_path / "a", tmp_path / "b")
        assert got == tmp_path / "a"

    def test_default_workdir_when_no_workdir(self, tmp_path):
        assert start_workspace_path(None, tmp_path / "b") == tmp_path / "b"

    def test_cwd_when_neither(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert start_workspace_path() == Path(os.getcwd())

    def test_expands_user_and_is_absolute(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        got = start_workspace_path("~/proj")
        assert got == tmp_path / "proj"
        assert got.is_absolute()

    def test_relative_workdir_is_made_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert start_workspace_path("rel") == tmp_path / "rel"

    def test_empty_strings_fall_through(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert start_workspace_path("", "") == Path(os.getcwd())


class TestProcessWorkspace:
    def test_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EVOSCIENTIST_WORKSPACE_DIR", str(tmp_path))
        assert process_workspace() == Workspace(tmp_path)

    def test_cwd_without_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EVOSCIENTIST_WORKSPACE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        assert process_workspace() == Workspace(tmp_path)

    def test_read_once_per_process(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EVOSCIENTIST_WORKSPACE_DIR", str(tmp_path / "first"))
        first = process_workspace()
        monkeypatch.setenv("EVOSCIENTIST_WORKSPACE_DIR", str(tmp_path / "second"))
        assert process_workspace() == first


class TestReloadEnvDirs:
    """An override merged into the environment after import (project .env)."""

    def test_picks_up_a_late_memories_override(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EVOSCIENTIST_MEMORY_DIR", raising=False)
        monkeypatch.setenv("EVOSCIENTIST_MEMORIES_DIR", str(tmp_path / "mem"))
        paths.reload_env_dirs()
        assert paths.MEMORIES_DIR == tmp_path / "mem"
        assert paths.MEMORY_DIR == paths.MEMORIES_DIR

    def test_falls_back_to_global_memories(self, monkeypatch):
        monkeypatch.delenv("EVOSCIENTIST_MEMORIES_DIR", raising=False)
        monkeypatch.delenv("EVOSCIENTIST_MEMORY_DIR", raising=False)
        paths.reload_env_dirs()
        assert paths.MEMORIES_DIR == paths.GLOBAL_MEMORIES_DIR


class TestResolveVirtualPath:
    def test_absolute_virtual_path(self, tmp_path):
        assert (
            resolve_virtual_path(tmp_path, "/a/b.png")
            == (tmp_path / "a/b.png").resolve()
        )

    def test_relative_virtual_path(self, tmp_path):
        assert resolve_virtual_path(tmp_path, "a.png") == (tmp_path / "a.png").resolve()

    def test_root(self, tmp_path):
        assert resolve_virtual_path(tmp_path, "/") == tmp_path.resolve()

    def test_depends_only_on_the_given_work_dir(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        assert resolve_virtual_path(a, "/f") != resolve_virtual_path(b, "/f")


class TestEnsureDirs:
    """ensure_dirs creates the global data and memories folders only."""

    def test_creates_global_dirs_not_workspace_dirs(self, tmp_path, monkeypatch):
        data = tmp_path / "data"
        memories = tmp_path / "memories"
        ws_root = tmp_path / "workspace"
        ws_root.mkdir()
        monkeypatch.setattr(paths, "DATA_DIR", data)
        monkeypatch.setattr(paths, "MEMORIES_DIR", memories)
        monkeypatch.chdir(ws_root)

        paths.ensure_dirs()

        assert data.is_dir()
        assert memories.is_dir()
        assert not (ws_root / "memories").exists()
        assert not (ws_root / "skills").exists()  # created on demand by install_skill()


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

        Clears XDG_CONFIG_HOME so the legacy resolver deterministically uses
        the Path.home() fallback. Tests that want the XDG branch set the
        env var explicitly.
        """
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
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
