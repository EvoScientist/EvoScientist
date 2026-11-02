"""Tests for EvoScientist.paths — the Workspace type, start resolution, ensure_dirs."""

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from EvoScientist import paths
from EvoScientist.paths import (
    SessionDirs,
    Workspace,
    normalize_path,
    process_session_dirs,
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
        "GLOBAL_SKILLS_DIR": paths.GLOBAL_SKILLS_DIR,
        "GLOBAL_MEMORIES_DIR": paths.GLOBAL_MEMORIES_DIR,
    }
    yield
    for name, value in orig.items():
        setattr(paths, name, value)


def _fake_home(monkeypatch, home: Path) -> None:
    """Point ``~`` at *home* (Windows reads USERPROFILE, not HOME)."""
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


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
        _fake_home(monkeypatch, tmp_path)
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

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            ("proj/", "proj"),
            ("proj/./sub/..", "proj"),
            ("with space", "with space"),
            ("項目 ü", "項目 ü"),
        ],
    )
    def test_key_normalises_shared_spellings(self, tmp_path, spelling, expected):
        """Spellings the WebUI normalises to the same key as the core."""
        ws = Workspace(f"{tmp_path}/{spelling}")
        assert ws.key == (tmp_path.resolve() / expected).as_posix()

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

    def test_nul_byte_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="NUL byte"):
            normalize_path(f"{tmp_path}/evil\x00x")

    def test_darwin_canonicalises_stored_case_via_f_getpath(
        self, tmp_path, monkeypatch
    ):
        """#692: on macOS, ``Workspace('~/Project')`` and ``Workspace('~/project')``
        must resolve to the on-disk spelling returned by ``fcntl(fd, F_GETPATH)``."""
        import types

        stored_dir = tmp_path.resolve() / "Project"
        typed_dir = tmp_path.resolve() / "project"

        real_open = os.open
        real_close = os.close
        sentinel_fd = 99991

        def fake_open(target, flags, *args, **kwargs):
            p = Path(target)
            assert flags & os.O_NONBLOCK
            if p in (stored_dir, typed_dir):
                return sentinel_fd
            return real_open(target, flags, *args, **kwargs)

        def fake_close(fd):
            if fd == sentinel_fd:
                return
            real_close(fd)

        fake_fcntl_mod = types.SimpleNamespace(F_GETPATH=50)

        def fake_fcntl(fd, cmd, arg=0):
            assert cmd == fake_fcntl_mod.F_GETPATH
            if fd == sentinel_fd:
                raw = os.fsencode(str(stored_dir)) + b"\x00"
                return raw.ljust(1024, b"\x00")
            raise OSError("unexpected fd")

        fake_fcntl_mod.fcntl = fake_fcntl
        # Simulate the Darwin open flag too: Windows does not expose it.
        monkeypatch.setattr(
            paths.os, "O_NONBLOCK", getattr(os, "O_NONBLOCK", 0x800), raising=False
        )
        monkeypatch.setattr(paths.sys, "platform", "darwin")
        monkeypatch.setitem(sys.modules, "fcntl", fake_fcntl_mod)
        monkeypatch.setattr(paths.os, "open", fake_open)
        monkeypatch.setattr(paths.os, "close", fake_close)

        ws_upper = Workspace(stored_dir)
        ws_lower = Workspace(typed_dir)
        assert ws_lower == ws_upper
        assert ws_lower.key == ws_upper.key == stored_dir.as_posix()

        # Non-existent child folders under a differently-cased existing parent
        # still inherit the parent's stored on-disk case without creating dirs.
        child = Workspace(typed_dir / "runs" / "20260930_120000")
        assert child.key == (stored_dir / "runs" / "20260930_120000").as_posix()
        assert not (stored_dir / "runs").exists()


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
        _fake_home(monkeypatch, tmp_path)
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


class TestSessionDirs:
    def test_daemon_session_works_in_the_root(self, tmp_path):
        dirs = SessionDirs(Workspace(tmp_path))
        assert dirs.run_dir is None
        assert dirs.work_dir == tmp_path.resolve()
        assert dirs.metadata() == {"workspace_dir": tmp_path.resolve().as_posix()}

    def test_run_session_works_in_its_run_folder(self, tmp_path):
        ws = Workspace(tmp_path)
        dirs = SessionDirs(ws, ws.runs_dir / "exp")
        assert dirs.work_dir == ws.runs_dir / "exp"
        assert dirs.metadata() == {
            "workspace_dir": ws.key,
            "run_dir": (ws.runs_dir / "exp").as_posix(),
        }

    def test_run_dir_equal_to_root_is_no_run_dir(self, tmp_path):
        dirs = SessionDirs(Workspace(tmp_path), tmp_path / "." / "")
        assert dirs.run_dir is None
        assert "run_dir" not in dirs.metadata()


class TestWorkspaceIsRunFolder:
    def test_folder_under_runs_is_run_folder(self, tmp_path):
        assert Workspace(tmp_path).is_run_folder(tmp_path / "runs" / "a")

    def test_other_spellings_are_not_run_folders(self, tmp_path):
        workspace = Workspace(tmp_path)
        for path in (
            tmp_path / "runs",
            tmp_path / "runs" / "a" / "..",
            tmp_path / "runs" / "a" / "b",
            tmp_path / "other" / "runs" / "a",
        ):
            assert not workspace.is_run_folder(path)


class TestSessionDirsFromStored:
    def test_nothing_stored(self):
        assert SessionDirs.from_stored(None) is None
        assert SessionDirs.from_stored("") is None

    def test_a_timestamp_named_root_stays_a_root(self, tmp_path):
        """New rows carry ``run_dir`` themselves; reading never guesses."""
        folder = tmp_path / "runs" / "20260930_120000"
        assert SessionDirs.from_stored(str(folder)) == SessionDirs(Workspace(folder))

    def test_unresolved_values_are_normalised(self, tmp_path, monkeypatch):
        _fake_home(monkeypatch, tmp_path)
        assert SessionDirs.from_stored("~/proj/") == SessionDirs(
            Workspace(tmp_path / "proj")
        )

    def test_run_dir_outside_workspace_is_dropped(self, tmp_path):
        """Run folders live under ``<root>/runs``; anything else is not one."""
        dirs = SessionDirs.from_stored(
            str(tmp_path / "a"), str(tmp_path / "b" / "runs" / "20260930_120000")
        )
        assert dirs == SessionDirs(Workspace(tmp_path / "a"))

    def test_other_folder_inside_workspace_is_dropped(self, tmp_path):
        workspace = Workspace(tmp_path)
        for run_dir in (tmp_path / "memories", tmp_path / "runs" / "a" / "b"):
            assert SessionDirs.from_stored(str(tmp_path), str(run_dir)) == (
                SessionDirs(workspace)
            )
        run = tmp_path / "runs" / "a"
        assert SessionDirs.from_stored(str(tmp_path), str(run)) == SessionDirs(
            workspace, run
        )

    def test_runs_folder_linked_outside_workspace_is_dropped(self, tmp_path):
        """Not supported: the run folder resolves outside the workspace."""
        root, disk = tmp_path / "ws", tmp_path / "disk"
        root.mkdir()
        disk.mkdir()
        (root / "runs").symlink_to(disk, target_is_directory=True)
        dirs = SessionDirs.from_stored(str(root), str(root / "runs" / "a"))
        assert dirs == SessionDirs(Workspace(root))

    def test_unreadable_value_reads_as_nothing_stored(self):
        assert SessionDirs.from_stored("/tmp/evil\x00x") is None

    def test_non_string_value_reads_as_nothing_stored(self, tmp_path):
        """Metadata is caller-supplied JSON, so a folder can be stored as a
        number or a list; one bad row must not break listing the rest."""
        assert SessionDirs.from_stored(123) is None
        assert SessionDirs.from_stored(str(tmp_path), ["runs", "x"]) is None


class TestSessionDirsFromLegacy:
    def test_generated_run_folder_is_split_into_workspace_and_run_dir(self, tmp_path):
        run = tmp_path / "runs" / "20260930_120000"
        assert SessionDirs.from_legacy(str(run)) == SessionDirs(
            Workspace(tmp_path), run
        )

    def test_named_run_folder_is_left_as_its_own_workspace(self, tmp_path):
        """A project can live in a folder called ``runs``; only generated
        names are known to be run folders."""
        folder = tmp_path / "runs" / "exp"
        assert SessionDirs.from_legacy(str(folder)) == SessionDirs(Workspace(folder))

    def test_names_the_cli_does_not_generate_are_not_split(self, tmp_path):
        for name in (
            "20260930_120000_1",
            "2026093_120000",
            "x20260930_120000",
            "99999999_999999",
        ):
            folder = tmp_path / "runs" / name
            assert SessionDirs.from_legacy(str(folder)).run_dir is None, name

    def test_timestamp_folder_outside_runs_is_not_split(self, tmp_path):
        folder = tmp_path / "other" / "20260930_120000"
        assert SessionDirs.from_legacy(str(folder)) == SessionDirs(Workspace(folder))

    def test_generated_names_round_trip(self):
        from datetime import datetime

        name = datetime(2026, 9, 30, 12, 0, 0).strftime(paths.RUN_NAME_FORMAT)
        assert paths.is_generated_run_name(name)


class TestProcessSessionDirs:
    def test_with_run_dir(self, tmp_path, monkeypatch):
        run = tmp_path / "runs" / "exp"
        monkeypatch.setenv("EVOSCIENTIST_WORKSPACE_DIR", str(tmp_path))
        monkeypatch.setenv("EVOSCIENTIST_RUN_DIR", str(run))
        assert process_session_dirs() == SessionDirs(Workspace(tmp_path), run)


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
        """Redirect the data folder to tmp_path/new_data and the config folder,
        the legacy source, to tmp_path/config/evoscientist."""
        data_dir = tmp_path / "new_data"
        legacy_dir = tmp_path / "config" / "evoscientist"
        monkeypatch.setattr(paths, "DATA_DIR", data_dir)
        monkeypatch.setattr(paths, "CONFIG_DIR", legacy_dir)
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


class TestSharedDirFromEnv:
    """The data and config folders, which other programs locate too, follow one
    path rule."""

    @pytest.mark.parametrize("value", ["", "   "])
    def test_shared_dir_from_env_blank_is_unset(self, value, monkeypatch):
        monkeypatch.setenv("EVOSCIENTIST_DATA_DIR", value)
        assert paths._shared_dir_from_env("EVOSCIENTIST_DATA_DIR") is None

    @pytest.mark.parametrize(("value", "suffix"), [("~", ""), ("~/evo", "evo")])
    def test_shared_dir_from_env_expands_a_leading_tilde(
        self, value, suffix, tmp_path, monkeypatch
    ):
        _fake_home(monkeypatch, tmp_path)
        monkeypatch.setenv("EVOSCIENTIST_DATA_DIR", value)
        assert paths._shared_dir_from_env("EVOSCIENTIST_DATA_DIR") == tmp_path / suffix

    @pytest.mark.parametrize("value", ["~nobody/evo", "$HOME/evo"])
    def test_shared_dir_from_env_expands_nothing_else(
        self, value, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("EVOSCIENTIST_DATA_DIR", value)
        assert paths._shared_dir_from_env("EVOSCIENTIST_DATA_DIR") == tmp_path / value

    def test_shared_dir_from_env_resolves_a_relative_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("EVOSCIENTIST_DATA_DIR", "data")
        assert paths._shared_dir_from_env("EVOSCIENTIST_DATA_DIR") == tmp_path / "data"

    @pytest.mark.parametrize("value", [None, "data"])
    def test_data_dir_follows_evoscientist_data_dir(self, value, tmp_path, monkeypatch):
        _fake_home(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        if value is None:
            monkeypatch.delenv("EVOSCIENTIST_DATA_DIR", raising=False)
            expected = tmp_path / ".evoscientist"
        else:
            monkeypatch.setenv("EVOSCIENTIST_DATA_DIR", value)
            expected = tmp_path / value
        assert paths._data_dir() == expected

    @pytest.mark.parametrize("xdg", [None, "cfg"])
    def test_config_dir_follows_xdg_config_home(self, xdg, tmp_path, monkeypatch):
        _fake_home(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        if xdg is None:
            monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
            expected = tmp_path / ".config" / "evoscientist"
        else:
            monkeypatch.setenv("XDG_CONFIG_HOME", xdg)
            expected = tmp_path / xdg / "evoscientist"
        assert paths._config_dir() == expected

    @pytest.mark.parametrize("xdg_set", [True, False])
    def test_dir_env_gives_children_this_process_folders(
        self, xdg_set, tmp_path, monkeypatch
    ):
        """Children run in other folders and load their own ``.env``;
        ``XDG_CONFIG_HOME`` is passed only when this process started with it."""
        monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path / "cfg" / "evoscientist")
        monkeypatch.setattr(paths, "_XDG_CONFIG_HOME_SET", xdg_set)
        env = paths.dir_env(
            {"EVOSCIENTIST_DATA_DIR": "rel", "XDG_CONFIG_HOME": "rel", "OTHER": "x"}
        )
        expected = {"EVOSCIENTIST_DATA_DIR": str(tmp_path / "data"), "OTHER": "x"}
        if xdg_set:
            expected["XDG_CONFIG_HOME"] = str(tmp_path / "cfg")
        assert env == expected

    def test_dir_env_gives_children_memories_override_as_absolute(
        self, tmp_path, monkeypatch
    ):
        """The server works in the server home, so a relative override would
        name another folder there."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(paths, "MEMORIES_DIR", Path("memories"))
        env = paths.dir_env(
            {"EVOSCIENTIST_MEMORIES_DIR": "memories", "EVOSCIENTIST_MEMORY_DIR": "old"}
        )
        assert env["EVOSCIENTIST_MEMORIES_DIR"] == str(tmp_path / "memories")
        assert "EVOSCIENTIST_MEMORY_DIR" not in env

    def test_launcher_workspace_unreadable_is_none(self, monkeypatch):
        def unreadable(_path):
            raise OSError("loop")

        monkeypatch.setenv(paths.LAUNCHER_WORKSPACE_ENV, "/proj")
        monkeypatch.setattr(paths, "Workspace", unreadable)
        paths.launcher_workspace.cache_clear()
        assert paths.launcher_workspace() is None


def test_server_marker_read_once_and_removed(tmp_path):
    """The agent's commands and MCP servers must not take themselves for the
    server."""
    import os
    import subprocess
    import sys

    code = (
        "import os, EvoScientist.paths as p; "
        "print(p.SERVER_PROCESS, 'EVOSCIENTIST_SERVER_PROCESS' in os.environ)"
    )
    env = {**os.environ, "HOME": str(tmp_path), "EVOSCIENTIST_SERVER_PROCESS": "1"}
    env.pop("EVOSCIENTIST_DATA_DIR", None)
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert out.stdout.split() == ["True", "False"], out.stderr


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell quoting")
@pytest.mark.parametrize(
    ("path", "shown"),
    [
        ("/home/me/proj", "/home/me/proj"),
        ("/home/me/My Projects/app", "'/home/me/My Projects/app'"),
        ("/home/me/R&D", "'/home/me/R&D'"),
        ("/home/me/it's", """'/home/me/it'"'"'s'"""),
    ],
)
def test_command_arg_quotes_path_for_posix_shell(path, shown):
    """A printed command the user copies must keep the path one argument."""
    from EvoScientist.paths import command_arg

    assert command_arg(path) == shown


@pytest.mark.skipif(os.name != "nt", reason="Windows quoting")
@pytest.mark.parametrize(
    ("path", "shown"),
    [
        (r"C:\Users\me\proj", r"C:\Users\me\proj"),
        (r"C:\Users\me\My Projects", r'"C:\Users\me\My Projects"'),
    ],
)
def test_command_arg_quotes_path_with_whitespace_on_windows(path, shown):
    from EvoScientist.paths import command_arg

    assert command_arg(path) == shown


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS filesystem")
def test_darwin_stored_case_and_run_directories(tmp_path):
    """Real APFS spellings preserve project identity and resumed run folders."""
    (tmp_path / "Project" / "Runs").mkdir(parents=True)
    if not (tmp_path / "project" / "runs").exists():
        pytest.skip("case-sensitive volume")
    typed = Workspace(tmp_path / "project")
    assert typed.key == Workspace(tmp_path / "Project").key
    assert typed.root.name == "Project"
    run = typed.runs_dir / "20260930_120000"
    assert typed.is_run_folder(run)
    assert SessionDirs.from_stored(typed.key, run).run_dir is not None
    assert SessionDirs.from_legacy(run).run_dir is not None


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS F_GETPATH")
@pytest.mark.parametrize("child", [False, True])
def test_darwin_fifo_path_does_not_block(tmp_path, child):
    """A real FIFO or a missing child under it must not wait for a writer."""
    fifo = tmp_path / "Pipe"
    os.mkfifo(fifo)
    candidate = fifo / "missing" if child else fifo
    assert normalize_path(candidate) == candidate.resolve()
