"""Tests for the Git for Windows setup stage.

The download, the self-extractor and every ``git`` / ``bash`` probe are faked,
so these run on any OS.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from EvoScientist.setup import git
from EvoScientist.setup.protocol import HEARTBEAT_INTERVAL, StageError

# The env fixture pins _arch() to x64; the platform tests restore this one.
_REAL_ARCH = git._arch
DATA = b"fake PortableGit archive"
ASSET = git.ASSETS["x64"][0]
PINNED = hashlib.sha256(DATA).hexdigest()
WINDOWS_VERSION = "git version 2.56.0.windows.1\n"


def _layout(root: Path, *, bash: bool = True) -> Path:
    """A Git for Windows tree: ``cmd\\git.exe`` and (optionally) ``bin\\bash.exe``."""
    (root / "cmd").mkdir(parents=True, exist_ok=True)
    (root / "cmd" / "git.exe").write_bytes(b"")
    if bash:
        (root / "bin").mkdir(parents=True, exist_ok=True)
        (root / "bin" / "bash.exe").write_bytes(b"")
    return root


class FakeRun:
    """Fake probe results keyed by (executable, first argument).

    Anything under the tools dir answers like a working PortableGit unless a
    result is set for it explicitly; a value of None makes the probe fail, and
    :data:`BLOCKED` makes it behave as if a Windows policy refused to start it.
    """

    def __init__(self, tools: Path) -> None:
        self.tools = tools
        self.results: dict[tuple[str, str], str | None] = {}
        self.calls: list[list[str]] = []
        self.blocked: set[str] = set()

    def set(self, exe: Path, arg: str, stdout: str | None) -> None:
        self.results[(os.path.normcase(str(exe)), arg)] = stdout

    def block(self, exe: Path) -> None:
        self.blocked.add(os.path.normcase(str(exe)))

    def __call__(self, argv: list[str]):
        self.calls.append(argv)
        if os.path.normcase(argv[0]) in self.blocked:
            raise git._PolicyBlocked(argv[0])
        key = (os.path.normcase(argv[0]), argv[1])
        if key in self.results:
            stdout = self.results[key]
        elif git.is_under(Path(argv[0]), self.tools):
            exe = Path(argv[0])
            stdout = {
                "--version": WINDOWS_VERSION if exe.name == "git.exe" else "bash 5",
                "--exec-path": str(exe.parent.parent / "ucrt64/libexec/git-core"),
            }[argv[1]]
        else:
            stdout = None
        if stdout is None:
            return None
        return subprocess.CompletedProcess(argv, 0, stdout, "")


class FakeNet:
    def __init__(self) -> None:
        self.urls: list[str] = []
        self.data = DATA

    def download(self, url, dest: Path, progress=None) -> str:
        self.urls.append(url)
        dest.write_bytes(self.data)
        if progress:
            progress(1.0)
        return hashlib.sha256(self.data).hexdigest()


class FakeSfx:
    """Stands in for ``git._launch`` (the hidden-desktop process) of the
    self-extractor."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.exit_code = 0
        self.leftovers: list[str] = []
        self.hang = False  # never finishes on its own
        self.busy_polls = 0  # polls that report "still running" before it ends
        self.wait_timeouts: list[float] = []
        self.extract = True
        self.closed = 0
        self.launched_sha: str | None = None
        self.part_left: bool | None = None
        self.killed = False  # set by the fake _kill_tree
        self.survives_kill = False  # the tree does not exit when killed
        self.wait_error: BaseException | None = None  # raised by the first wait

    def __call__(self, argv):
        self.calls.append(argv)
        # Like the real launch: only an existing file starts. Record what ran,
        # so tests can pin that it is the hashed download.
        exe = Path(argv[0])
        if not exe.is_file():
            raise FileNotFoundError(2, "The system cannot find the file", argv[0])
        self.launched_sha = hashlib.sha256(exe.read_bytes()).hexdigest()
        self.part_left = exe.with_name(exe.name + ".part").exists()
        sfx = self

        class Proc:
            pid = 4242

            def __init__(self) -> None:
                self.waited = 0

            def wait(self, timeout):
                self.waited += 1
                sfx.wait_timeouts.append(timeout)
                if sfx.killed:
                    return None if sfx.survives_kill else 1
                if sfx.wait_error is not None and self.waited == 1:
                    raise sfx.wait_error
                if sfx.hang or self.waited <= sfx.busy_polls:
                    return None  # still running
                return sfx.exit_code

            def close(self) -> None:
                sfx.closed += 1

        if self.extract:
            out = Path(argv[2][2:])
            _layout(out)
            for leftover in self.leftovers:
                path = out / leftover
                path.parent.mkdir(parents=True, exist_ok=True)
                path.mkdir() if leftover.endswith("post-install") else path.write_text(
                    ""
                )
        return Proc()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated DATA_DIR with a space and non-ASCII characters, no git on PATH."""
    from EvoScientist import paths

    data = tmp_path.resolve() / "Jan Kowalski ąę" / ".evoscientist"
    tools = data / "tools"
    monkeypatch.setattr(paths, "DATA_DIR", data)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.setattr(git, "_arch", lambda: "x64")
    monkeypatch.setitem(git.ASSETS, "x64", (ASSET, PINNED))
    net = FakeNet()
    monkeypatch.setattr(git, "download", net.download)
    run = FakeRun(tools)
    monkeypatch.setattr(git, "_run", run)
    sfx = FakeSfx()
    monkeypatch.setattr(git, "_launch", sfx)
    killed: list[int] = []

    def kill_tree(pid):
        killed.append(pid)
        sfx.killed = True

    monkeypatch.setattr(git, "_kill_tree", kill_tree)
    # The real check would read the test host's free space.
    plenty = type("Usage", (), {"free": git.MIN_FREE_BYTES * 10})
    monkeypatch.setattr(git.shutil, "disk_usage", lambda _p: plenty)

    def which(name, path=None):
        # Like shutil.which: PATH when no path= is given.
        for d in (path or os.environ.get("PATH", "")).split(os.pathsep):
            candidate = Path(d) / f"{name}.exe"
            if d and candidate.is_file():
                return str(candidate)
        return None

    monkeypatch.setattr(git.shutil, "which", which)
    return {
        "data": data,
        "tools": tools,
        "net": net,
        "run": run,
        "sfx": sfx,
        "killed": killed,
        "tmp": tmp_path,
    }


def _record(env) -> dict | None:
    path = env["tools"] / "git.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _system_git(env, monkeypatch, *, bash: bool = True, version=WINDOWS_VERSION):
    root = _layout(env["tmp"] / "Program Files" / "Git", bash=bash)
    monkeypatch.setenv("PATH", str(root / "cmd"))
    exe = root / "cmd" / "git.exe"
    env["run"].set(exe, "--version", version)
    env["run"].set(exe, "--exec-path", f"{root.as_posix()}/mingw64/libexec/git-core")
    env["run"].set(root / "bin" / "bash.exe", "--version", "bash 5")
    return root


# --------------------------------------------------------------------------- #
# System Git
# --------------------------------------------------------------------------- #
def test_system_git_for_windows_is_used_without_download(env, monkeypatch):
    root = _system_git(env, monkeypatch)
    info = git.ensure_git()
    assert info.source == "system"
    assert info.version == "2.56.0.windows.1"
    assert info.bash == root / "bin" / "bash.exe"
    assert env["net"].urls == []
    assert env["sfx"].calls == []
    assert _record(env) == {
        "version": "2.56.0.windows.1",
        "git": str(root / "cmd" / "git.exe"),
        "bash": str(root / "bin" / "bash.exe"),
        "source": "system",
    }


def test_system_git_without_bash_leads_to_portablegit(env, monkeypatch, caplog):
    root = _system_git(env, monkeypatch, bash=False)
    with caplog.at_level("INFO", logger=git.__name__):
        assert git.ensure_git().source == "portablegit"
    assert env["net"].urls
    # Says why the system Git was not used, before 60 MB are downloaded.
    assert f"{root} has no cmd\\git.exe or no working bin\\bash.exe" in caplog.text


def test_no_git_leads_to_portablegit(env):
    info = git.ensure_git()
    assert info.source == "portablegit"
    assert info.git == env["tools"] / f"git-{git.GIT_VERSION}" / "cmd" / "git.exe"


def test_non_windows_git_does_not_count(env, monkeypatch, caplog):
    """MSYS2's and Cygwin's own git report no .windows.N."""
    _system_git(env, monkeypatch, version="git version 2.56.0\n")
    with caplog.at_level("INFO", logger=git.__name__):
        assert git.ensure_git().source == "portablegit"
    assert "not a Git for Windows build" in caplog.text


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (subprocess.TimeoutExpired("git", 30), "did not finish within 30 s"),
        (OSError(13, "Access is denied"), "could not run: [Errno 13] Access is denied"),
        (
            subprocess.CompletedProcess([], 128, "", "fatal: broken\nmore\n"),
            "exited with code 128: fatal: broken",
        ),
    ],
)
def test_failed_probe_logs_why(monkeypatch, caplog, outcome, expected):
    def fake(argv, **kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(git.subprocess, "run", fake)
    with caplog.at_level("INFO", logger=git.__name__):
        assert git._run(["C:/Git/cmd/git.exe", "--version"]) is None
    assert f"C:/Git/cmd/git.exe --version {expected}" in caplog.text


def test_system_record_is_written_under_the_install_lock(env, monkeypatch):
    _system_git(env, monkeypatch)
    held: list[bool] = []
    real_write = git._write_record

    def write(info):
        from filelock import FileLock, Timeout

        # A second lock on the same file cannot be taken while ensure_git holds it.
        other = FileLock(str(env["tools"] / "git.lock"), timeout=0)
        try:
            other.acquire()
        except Timeout:
            held.append(True)
        else:
            other.release()
            held.append(False)
        real_write(info)

    monkeypatch.setattr(git, "_write_record", write)
    git.ensure_git()
    assert held == [True]


def test_shim_on_path_resolves_the_real_install(env, monkeypatch):
    """A launcher shim (Scoop) has no install tree around it; --exec-path does."""
    real = _layout(env["tmp"] / "scoop" / "apps" / "git" / "current")
    shims = env["tmp"] / "scoop" / "shims"
    shims.mkdir(parents=True)
    shim = shims / "git.exe"
    shim.write_bytes(b"")
    monkeypatch.setenv("PATH", str(shims))
    env["run"].set(shim, "--version", WINDOWS_VERSION)
    env["run"].set(shim, "--exec-path", f"{real.as_posix()}/ucrt64/libexec/git-core")
    env["run"].set(real / "cmd" / "git.exe", "--version", WINDOWS_VERSION)
    env["run"].set(real / "bin" / "bash.exe", "--version", "bash 5")
    info = git.ensure_git()
    assert info.source == "system"
    assert info.git == real / "cmd" / "git.exe"
    assert info.bash == real / "bin" / "bash.exe"


def test_own_portablegit_on_path_is_not_reported_as_system(env, monkeypatch):
    """activate_runtime() runs before `EvoSci setup`, so our cmd dir is on PATH."""
    git.ensure_git()
    own_cmd = env["tools"] / f"git-{git.GIT_VERSION}" / "cmd"
    monkeypatch.setenv("PATH", str(own_cmd))
    assert git.ensure_git().source == "portablegit"
    assert _record(env)["source"] == "portablegit"


def test_exec_path_runs_without_inherited_git_exec_path(monkeypatch):
    seen = {}

    def fake(argv, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setenv("GIT_EXEC_PATH", "/somewhere/else")
    monkeypatch.setattr(git.subprocess, "run", fake)
    git._run(["git", "--exec-path"])
    assert "GIT_EXEC_PATH" not in {k.upper() for k in seen["env"]}
    assert seen["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)


# --------------------------------------------------------------------------- #
# PortableGit install
# --------------------------------------------------------------------------- #
def test_download_and_sfx_switches(env):
    git.ensure_git()
    assert env["net"].urls == [
        f"{git.SOURCES['default']}/{git.GIT_TAG}/{ASSET}",
    ]
    (argv,) = env["sfx"].calls
    assert Path(argv[0]).name == ASSET  # renamed to .exe only after the check
    assert argv[1] == "-y"
    assert argv[2].startswith("-o")
    assert not argv[2].endswith((os.sep, "/"))
    assert len(argv) == 3  # anything more is appended to the post-install command
    out = Path(argv[2][2:])
    assert out.parent.parent == env["tools"]  # a temp dir inside tools\
    # The file that ran is the hashed download, with no .part left beside it.
    assert env["sfx"].launched_sha == PINNED
    assert env["sfx"].part_left is False


def test_cn_mirror_downloads_from_npmmirror(env):
    git.ensure_git("cn")
    assert env["net"].urls == [
        f"https://registry.npmmirror.com/-/binary/git-for-windows/{git.GIT_TAG}/{ASSET}",
    ]


def test_sfx_process_is_always_closed(env):
    env["sfx"].exit_code = 1
    with pytest.raises(StageError):
        git.ensure_git()
    assert env["sfx"].closed == 1


_windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="the hidden desktop is a Windows API"
)


@_windows_only
def test_hidden_desktop_process_reports_the_exit_code():
    proc = git._HiddenDesktopProcess(["cmd.exe", "/c", "exit 3"])
    try:
        assert proc.wait(30) == 3
    finally:
        proc.close()


@_windows_only
def test_hidden_desktop_process_wait_times_out_and_can_be_killed():
    proc = git._HiddenDesktopProcess(["ping.exe", "-n", "30", "127.0.0.1"])
    try:
        assert proc.wait(0.5) is None
        git._kill_tree(proc.pid)
        assert proc.wait(10) is not None
    finally:
        proc.close()


def test_checksum_mismatch_deletes_the_file_without_running_it(env):
    previous = {"version": "x", "git": "y", "bash": "z", "source": "system"}
    env["tools"].mkdir(parents=True)
    (env["tools"] / "git.json").write_text(json.dumps(previous))
    env["net"].data = b"tampered"
    with pytest.raises(StageError) as ei:
        git.ensure_git()
    assert ei.value.code == "checksum_mismatch"
    assert env["sfx"].calls == []
    assert not list(env["tools"].glob(".git-*"))
    assert _record(env) == previous


@pytest.mark.parametrize("leftover", ["post-install.bat", "etc/post-install"])
def test_unfinished_post_install_is_probe_failed(env, leftover):
    env["sfx"].leftovers = [leftover]
    with pytest.raises(StageError) as ei:
        git.ensure_git()
    assert ei.value.code == "probe_failed"
    assert _record(env) is None
    assert not (env["tools"] / f"git-{git.GIT_VERSION}").exists()


def test_sfx_exit_code_is_install_failed(env):
    env["sfx"].exit_code = 1
    with pytest.raises(StageError) as ei:
        git.ensure_git()
    assert ei.value.code == "install_failed"
    assert _record(env) is None


def test_stalled_sfx_is_stopped_and_is_install_failed(env, monkeypatch):
    """No growth of the extraction dir for the silence limit stops the tree."""
    monkeypatch.setattr(git, "SFX_SILENCE_LIMIT", 0)
    env["sfx"].hang = True
    with pytest.raises(StageError) as ei:
        git.ensure_git()
    assert ei.value.code == "install_failed"
    assert "made no progress" in ei.value.message
    assert env["killed"] == [4242]
    assert env["sfx"].closed == 1  # process handle and desktop released
    assert not list(env["tools"].glob(".git-*"))


def test_stalled_sfx_that_survives_the_kill_keeps_its_temp_dir(env, monkeypatch):
    """Its tree may still write there, so the dir is left for the next run."""
    monkeypatch.setattr(git, "SFX_SILENCE_LIMIT", 0)
    env["sfx"].hang = True
    env["sfx"].survives_kill = True
    with pytest.raises(StageError) as ei:
        git.ensure_git()
    assert ei.value.code == "install_failed"
    assert "could not be stopped" in ei.value.message
    assert env["sfx"].closed == 1
    assert list(env["tools"].glob(".git-*"))
    assert _record(env) is None


@pytest.mark.parametrize(
    "error",
    [
        KeyboardInterrupt(),
        BrokenPipeError(32, "Broken pipe"),
        OSError(6, "WAIT_FAILED"),
    ],
    ids=["ctrl-c", "closed-stdout", "wait-failed"],
)
def test_interrupted_wait_stops_the_extractor_tree(env, error):
    """Whatever ends the wait, the unseen extractor is not left running."""
    env["sfx"].wait_error = error
    with pytest.raises((KeyboardInterrupt, StageError)) as ei:
        git.ensure_git()
    if isinstance(error, OSError):
        assert ei.value.code == "install_failed"
    assert env["killed"] == [4242]
    assert env["sfx"].closed == 1
    assert not list(env["tools"].glob(".git-*"))


@pytest.mark.parametrize(
    "error",
    [
        KeyboardInterrupt(),
        BrokenPipeError(32, "Broken pipe"),
        OSError(6, "WAIT_FAILED"),
    ],
    ids=["ctrl-c", "closed-stdout", "wait-failed"],
)
def test_interrupted_wait_keeps_the_folder_of_a_tree_that_survives(env, error):
    """The keep-the-folder rule holds on every early end, not only the stall."""
    env["sfx"].wait_error = error
    env["sfx"].survives_kill = True
    with pytest.raises((KeyboardInterrupt, StageError)):
        git.ensure_git()
    assert env["killed"] == [4242]
    assert env["sfx"].closed == 1
    assert list(env["tools"].glob(".git-*"))
    assert _record(env) is None


def test_silent_sfx_emits_heartbeats_until_it_ends(env):
    env["sfx"].busy_polls = 3
    events: list[dict] = []
    git.run_stage(events.append, "default")
    unpacking = [e for e in events if e.get("message") == "Unpacking"]
    # The first "Unpacking" line plus one heartbeat per empty poll.
    assert len(unpacking) == 1 + 3
    assert all(e["status"] == "running" and e["progress"] == 0.8 for e in unpacking)
    assert set(env["sfx"].wait_timeouts[:3]) == {HEARTBEAT_INTERVAL}


def test_tree_signature_changes_as_files_are_written(tmp_path):
    root = tmp_path / "out"
    assert git._tree_signature(root) == (0, 0)  # not created yet
    (root / "a").mkdir(parents=True)
    (root / "a" / "f").write_bytes(b"123")
    first = git._tree_signature(root)
    (root / "a" / "g").write_bytes(b"45")
    assert first == (1, 3)
    assert git._tree_signature(root) == (2, 5)


def test_too_little_disk_space_is_install_failed_before_download(env, monkeypatch):
    usage = type("Usage", (), {"free": git.MIN_FREE_BYTES - 1})
    monkeypatch.setattr(git.shutil, "disk_usage", lambda _p: usage)
    with pytest.raises(StageError) as ei:
        git.ensure_git()
    assert ei.value.code == "install_failed"
    assert env["net"].urls == []


def test_record_is_written_only_after_both_probes(env):
    final = env["tools"] / f"git-{git.GIT_VERSION}"
    env["run"].set(final / "bin" / "bash.exe", "--version", None)
    with pytest.raises(StageError) as ei:
        git.ensure_git()
    assert ei.value.code == "probe_failed"
    assert _record(env) is None
    assert not final.exists()


def test_failed_git_probe_after_install_is_probe_failed(env):
    final = env["tools"] / f"git-{git.GIT_VERSION}"
    env["run"].set(final / "cmd" / "git.exe", "--version", None)
    with pytest.raises(StageError) as ei:
        git.ensure_git()
    assert ei.value.code == "probe_failed"
    assert _record(env) is None


def test_stale_temp_dirs_are_removed_before_an_install(env):
    stale = env["tools"] / ".git-killed"
    stale.mkdir(parents=True)
    git.ensure_git()
    assert not stale.exists()


def test_unsupported_architecture(monkeypatch):
    monkeypatch.setattr(git.platform, "machine", lambda: "x86")
    with pytest.raises(StageError) as ei:
        git._arch()
    assert ei.value.code == "unsupported_platform"


def test_arm64_without_system_git_is_refused_before_any_download(env, monkeypatch):
    monkeypatch.setattr(git, "_arch", _REAL_ARCH)
    monkeypatch.setattr(git.platform, "machine", lambda: "ARM64")
    with pytest.raises(StageError) as ei:
        git.ensure_git()
    assert ei.value.code == "unsupported_platform"
    assert "Windows on arm64 is not supported" in ei.value.message
    assert env["net"].urls == []
    assert _record(env) is None


def test_arm64_still_uses_a_system_git(env, monkeypatch):
    monkeypatch.setattr(git, "_arch", _REAL_ARCH)
    monkeypatch.setattr(git.platform, "machine", lambda: "ARM64")
    _system_git(env, monkeypatch)
    assert git.ensure_git().source == "system"


# --------------------------------------------------------------------------- #
# Blocked by a Windows policy (AppLocker / WDAC)
# --------------------------------------------------------------------------- #
def _policy_oserror(winerror: int) -> OSError:
    exc = OSError(13, "This program is blocked by group policy")
    exc.winerror = winerror
    return exc


@pytest.mark.parametrize("winerror", [1260, 4551])
def test_run_reports_a_policy_block(monkeypatch, winerror):
    def fake(argv, **kwargs):
        raise _policy_oserror(winerror)

    monkeypatch.setattr(git.subprocess, "run", fake)
    with pytest.raises(git._PolicyBlocked) as ei:
        git._run(["C:/pg/bin/bash.exe", "--version"])
    assert ei.value.path == "C:/pg/bin/bash.exe"


def test_run_treats_other_start_errors_as_a_failed_probe(monkeypatch):
    def fake(argv, **kwargs):
        raise _policy_oserror(2)  # ERROR_FILE_NOT_FOUND

    monkeypatch.setattr(git.subprocess, "run", fake)
    assert git._run(["C:/pg/bin/bash.exe", "--version"]) is None


def test_blocked_system_git_falls_back_to_portablegit(env, monkeypatch):
    root = _system_git(env, monkeypatch)
    env["run"].block(root / "bin" / "bash.exe")
    assert git.ensure_git().source == "portablegit"


def test_blocked_fresh_portablegit_is_probe_failed_without_a_new_download(env):
    final = env["tools"] / f"git-{git.GIT_VERSION}"
    env["run"].block(final / "bin" / "bash.exe")
    with pytest.raises(StageError) as ei:
        git.ensure_git()
    assert ei.value.code == "probe_failed"
    assert "Windows policy (AppLocker or WDAC) blocked" in ei.value.message
    assert "bash.exe" in ei.value.message
    assert _record(env) is None
    assert final.is_dir()  # kept, so the next run does not download it again

    env["net"].urls.clear()
    with pytest.raises(StageError) as ei:
        git.ensure_git()
    assert ei.value.code == "probe_failed"
    assert env["net"].urls == []


def test_blocked_recorded_portablegit_is_probe_failed_without_a_new_download(env):
    git.ensure_git()
    env["run"].block(env["tools"] / f"git-{git.GIT_VERSION}" / "cmd" / "git.exe")
    env["net"].urls.clear()
    with pytest.raises(StageError) as ei:
        git.ensure_git()
    assert ei.value.code == "probe_failed"
    assert env["net"].urls == []
    assert len(env["sfx"].calls) == 1


def test_blocked_sfx_is_probe_failed(env, monkeypatch):
    def blocked_launch(argv):
        raise _policy_oserror(1260)

    monkeypatch.setattr(git, "_launch", blocked_launch)
    with pytest.raises(StageError) as ei:
        git.ensure_git()
    assert ei.value.code == "probe_failed"
    assert ASSET in ei.value.message
    assert _record(env) is None


# --------------------------------------------------------------------------- #
# Idempotence and offline
# --------------------------------------------------------------------------- #
def test_second_run_downloads_nothing(env):
    git.ensure_git()
    env["net"].urls.clear()
    assert git.ensure_git().source == "portablegit"
    assert env["net"].urls == []
    assert len(env["sfx"].calls) == 1


def test_intact_install_is_recorded_again_without_download(env, monkeypatch):
    """A system Git replaced the record; once it is gone, PortableGit returns
    from disk (works offline)."""
    git.ensure_git()
    _system_git(env, monkeypatch)
    assert git.ensure_git().source == "system"
    monkeypatch.setenv("PATH", str(env["tmp"] / "empty-bin"))
    env["net"].urls.clear()
    assert git.ensure_git().source == "portablegit"
    assert env["net"].urls == []
    assert _record(env)["source"] == "portablegit"


def test_broken_recorded_install_is_replaced(env):
    git.ensure_git()
    final = env["tools"] / f"git-{git.GIT_VERSION}"
    env["run"].set(final / "cmd" / "git.exe", "--version", None)
    env["net"].urls.clear()
    with pytest.raises(StageError):
        git.ensure_git()  # reinstall attempted; the fake probe still fails
    assert env["net"].urls


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
def test_activate_runtime_prepends_only_the_cmd_dir(env, monkeypatch):
    git.ensure_git()
    monkeypatch.setattr(git.sys, "platform", "win32")
    monkeypatch.setenv("PATH", os.pathsep.join(["/a", "/b"]))
    cmd = git.activate_runtime()
    assert cmd == (env["tools"] / f"git-{git.GIT_VERSION}" / "cmd").resolve()
    assert os.environ["PATH"].split(os.pathsep) == [str(cmd), "/a", "/b"]


def test_activate_runtime_leaves_a_system_git_alone(env, monkeypatch):
    _system_git(env, monkeypatch)
    git.ensure_git()
    monkeypatch.setattr(git.sys, "platform", "win32")
    monkeypatch.setenv("PATH", "/a")
    assert git.activate_runtime() is None
    assert os.environ["PATH"] == "/a"


def test_activate_runtime_does_nothing_outside_windows(env, monkeypatch):
    git.ensure_git()
    monkeypatch.setattr(git.sys, "platform", "linux")
    monkeypatch.setenv("PATH", "/a")
    assert git.activate_runtime() is None
    assert os.environ["PATH"] == "/a"


def test_activate_runtime_ignores_a_missing_install(env, monkeypatch):
    env["tools"].mkdir(parents=True)
    (env["tools"] / "git.json").write_text(
        json.dumps(
            {
                "version": "2.56.0.windows.1",
                "git": str(env["tools"] / "gone" / "cmd" / "git.exe"),
                "bash": str(env["tools"] / "gone" / "bin" / "bash.exe"),
                "source": "portablegit",
            }
        )
    )
    monkeypatch.setattr(git.sys, "platform", "win32")
    monkeypatch.setenv("PATH", "/a")
    assert git.activate_runtime() is None


def test_recorded_git_reads_the_portablegit_record(env, monkeypatch):
    git.ensure_git()
    monkeypatch.setattr(git.sys, "platform", "win32")
    info = git.recorded_git()
    root = (env["tools"] / f"git-{git.GIT_VERSION}").resolve()
    assert info == git.GitInfo(
        "portablegit",
        git.GIT_VERSION,
        root / "cmd" / "git.exe",
        root / "bin" / "bash.exe",
    )


def test_recorded_git_reads_a_system_git_record(env, monkeypatch):
    _system_git(env, monkeypatch)
    git.ensure_git()
    monkeypatch.setattr(git.sys, "platform", "win32")
    info = git.recorded_git()
    assert info is not None
    assert info.source == "system"
    assert info.bash.name == "bash.exe"


def test_recorded_git_is_none_without_a_record_or_outside_windows(env, monkeypatch):
    monkeypatch.setattr(git.sys, "platform", "win32")
    assert git.recorded_git() is None
    git.ensure_git()
    monkeypatch.setattr(git.sys, "platform", "linux")
    assert git.recorded_git() is None


def test_recorded_git_is_none_when_the_bash_is_gone(env, monkeypatch):
    git.ensure_git()
    monkeypatch.setattr(git.sys, "platform", "win32")
    (env["tools"] / f"git-{git.GIT_VERSION}" / "bin" / "bash.exe").unlink()
    assert git.recorded_git() is None


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
def test_run_stage_done_detail(env):
    events: list[dict] = []
    result = git.run_stage(events.append, "default")
    assert result.status == "done"
    assert set(result.detail) == {"source", "version", "bash"}
    assert result.detail["source"] == "portablegit"
    assert result.detail["version"] == "2.56.0.windows.1"
    assert all(e["status"] == "running" for e in events)


# --------------------------------------------------------------------------- #
# The git that run_git uses
# --------------------------------------------------------------------------- #
def _recorded_portablegit(env) -> Path:
    root = _layout(env["tools"] / f"git-{git.GIT_VERSION}")
    git._write_record(
        git.GitInfo(
            "portablegit",
            git.GIT_VERSION,
            root / "cmd" / "git.exe",
            root / "bin" / "bash.exe",
        )
    )
    return root


def test_private_git_on_path_is_the_recorded_portablegit(env, monkeypatch):
    root = _recorded_portablegit(env)
    monkeypatch.setenv("PATH", str(root / "cmd"))
    assert git.private_git_on_path()


def test_private_git_on_path_is_false_for_a_system_git_ahead_of_it(env, monkeypatch):
    root = _recorded_portablegit(env)
    system = _layout(env["tmp"] / "Program Files" / "Git")
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(system / "cmd"), str(root / "cmd")])
    )
    assert not git.private_git_on_path()


def test_private_git_on_path_is_false_for_a_recorded_system_git(env, monkeypatch):
    _system_git(env, monkeypatch)
    git.ensure_git()
    assert _record(env)["source"] == "system"
    assert not git.private_git_on_path()


def test_private_git_on_path_is_false_without_a_record(env, monkeypatch):
    assert not git.private_git_on_path()


def test_probe_output_is_decoded_as_utf8(monkeypatch):
    """Git for Windows writes paths as UTF-8; the ANSI code page would garble
    or reject a profile name such as ``Łukasz``."""
    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs)
        exec_path = "C:/Users/Łukasz/Git/ucrt64/libexec/git-core\n"
        return subprocess.CompletedProcess(argv, 0, exec_path, "")

    monkeypatch.setattr(git.subprocess, "run", fake_run)
    root = git._root_from_exec_path(Path("C:/Users/Łukasz/Git/cmd/git.exe"))
    assert root == Path("C:/Users/Łukasz/Git")
    assert (seen["encoding"], seen["errors"]) == ("utf-8", "replace")


def test_verified_download_rename_retries_a_scanner_lock(env, monkeypatch):
    """A scanner holding the just-closed download makes its first rename fail."""
    from EvoScientist.setup import _install

    real_replace = os.replace
    denied: list[str] = []

    def flaky_replace(src, dst):
        if str(src).endswith(".part") and not denied:
            denied.append(str(src))
            raise PermissionError(13, "Access is denied")
        real_replace(src, dst)

    monkeypatch.setattr(_install.os, "replace", flaky_replace)
    monkeypatch.setattr(_install.time, "sleep", lambda _s: None)
    info = git.ensure_git()
    assert info.source == "portablegit"
    assert denied
    assert len(env["net"].urls) == 1
