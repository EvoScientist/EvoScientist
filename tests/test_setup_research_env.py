"""Tests for the research environment stage. ``venv``, pip and every probe are mocked."""

from __future__ import annotations

import configparser
import logging
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from EvoScientist.setup import research_env as re_env
from EvoScientist.setup.protocol import StageError

# Captured at import, before conftest's autouse stub replaces them per test.
_REAL_AGENT_PYTHON = re_env._agent_python
_REAL_LOG_MISSING_PYTHON_HINT = re_env._log_missing_python_hint


@pytest.fixture(autouse=True)
def _real_agent_python_decision(monkeypatch):
    """This module tests the decision itself; undo conftest's stub."""
    monkeypatch.setattr(re_env, "_agent_python", _REAL_AGENT_PYTHON)
    monkeypatch.setattr(
        re_env, "_log_missing_python_hint", _REAL_LOG_MISSING_PYTHON_HINT
    )


class FakeRunner:
    """Stands in for ``research_env._run``: venv, pip and the probes."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        # Commands as run, with ``-I``; ``calls`` drops it so matching by
        # position works.
        self.raw: list[list[str]] = []
        self.venv_ok = True
        self.venv_output = ""
        self.pip_ok = True
        self.env_starts = True
        self.imports_ok = True
        # Results for the next import checks, used before ``imports_ok``.
        self.imports_script: list[bool] = []
        # Packages a failing import check reports (all of them when empty).
        self.failed_packages: list[str] = []
        # stderr the environment's Python writes before the check's own lines.
        self.import_noise = ""
        # Packages whose pinned version has no wheel for this interpreter.
        self.no_wheel: set[str] = set()
        # The index is not reached: pip warns about the connection, then
        # every requirement reports "from versions: none".
        self.index_down = False
        # Packages with no wheel at all for this interpreter: "from versions:
        # none" without a connection warning.
        self.no_wheel_at_all: set[str] = set()
        # Keyed by normcase(exe): False makes the system probe fail.
        self.system_ok: dict[str, bool] = {}
        # Keyed by normcase(exe): the system probe's output (version, prefix,
        # PEP 668 flag); a usable Python 3.12 by default.
        self.system_output: dict[str, str] = {}

    def pip_requirements(self) -> list[list[str]]:
        """The requirements of each ``pip install`` call, in order."""
        return [
            [part for part in cmd[cmd.index("--no-input") + 1 :] if "--" not in part]
            for cmd in self.calls
            if self._kind(cmd) == "pip"
        ]

    def kinds(self) -> list[str]:
        return [self._kind(cmd) for cmd in self.calls]

    @staticmethod
    def _kind(cmd: list[str]) -> str:
        cmd = [part for part in cmd if part != "-I"]
        if cmd[1:3] == ["-m", "venv"]:
            return "venv"
        if cmd[1:4] == ["-m", "pip", "install"]:
            return "pip"
        if cmd[1:5] == ["-m", "pip", "config", "--site"]:
            return "config"
        if cmd[1:] == ["-c", re_env._IMPORT_CHECK]:
            return "imports"
        return "starts"

    def __call__(self, cmd, timeout):
        self.raw.append(list(cmd))
        cmd = [part for part in cmd if part != "-I"]
        self.calls.append(cmd)
        kind = self._kind(cmd)
        if kind == "venv":
            if self.venv_ok:
                python = re_env._env_python(Path(cmd[3]))
                python.parent.mkdir(parents=True)
                python.write_text("")
                self.env_starts = True
            return subprocess.CompletedProcess(
                cmd, 0 if self.venv_ok else 1, self.venv_output
            )
        if kind == "pip":
            return self._pip_install(cmd)
        if kind == "config":
            return self._pip_config(cmd)
        if kind == "imports":
            scripted = self.imports_script.pop(0) if self.imports_script else None
            ok = self.env_starts and (self.imports_ok if scripted is None else scripted)
            if not self.env_starts:
                return subprocess.CompletedProcess(cmd, 1, "")
            failed = [] if ok else (self.failed_packages or list(re_env.PACKAGES))
            output = f"{self.import_noise}version 3.12.9\n" + "".join(
                f"failed {name}: ModuleNotFoundError({name!r})\n" for name in failed
            )
            return subprocess.CompletedProcess(cmd, 0 if ok else 1, output)
        # Keyed by normcase: on Windows shutil.which returns the PATHEXT
        # spelling (python.EXE).
        if cmd[1:] == ["-c", re_env._SYSTEM_PYTHON_PROBE]:
            exe = os.path.normcase(cmd[0])
            if not self.system_ok.get(exe, True):
                return subprocess.CompletedProcess(cmd, 1, "")
            output = self.system_output.get(exe, "3.12\n/opt/conda\n0\n")
            return subprocess.CompletedProcess(cmd, 0, output)
        return subprocess.CompletedProcess(cmd, 0 if self.env_starts else 1, "")

    def _pip_install(self, cmd: list[str]) -> subprocess.CompletedProcess:
        """pip's real wording (checked with pip 24.3 and 26.2) for a pin
        without a wheel, a package without any wheel for this interpreter and
        an unreachable https index; it stops at the first requirement it
        cannot match."""
        if not self.pip_ok:
            return subprocess.CompletedProcess(cmd, 1, "ERROR: offline")
        for req in cmd[cmd.index("--no-input") + 1 :]:
            if req.startswith("--") or "://" in req:
                continue
            name = req.split("==")[0]
            warning = ""
            if self.index_down:
                versions = "none"
                warning = (
                    "WARNING: Retrying (Retry(total=4, connect=None, read=None,"
                    " redirect=None, status=None)) after connection broken by"
                    " 'NewConnectionError(...: Failed to resolve host)': /simple/"
                    f"{name}/\n"
                )
            elif "==" in req and name in self.no_wheel:
                versions = "1.18.0, 1.18.1"
            elif name in self.no_wheel_at_all:
                versions = "none"
            else:
                continue
            return subprocess.CompletedProcess(
                cmd,
                1,
                f"{warning}ERROR: Could not find a version that satisfies the"
                f" requirement {req} (from versions: {versions})\n"
                f"ERROR: No matching distribution found for {req}\n",
            )
        return subprocess.CompletedProcess(cmd, 0, "")

    @staticmethod
    def _pip_config(cmd: list[str]) -> subprocess.CompletedProcess:
        """``pip config --site set|unset global.<key>`` on the environment's file."""
        env = Path(cmd[0]).parent.parent
        path = re_env._pip_config(env)
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(path, encoding="utf-8")
        action, key = cmd[5], cmd[6].removeprefix("global.")
        if action == "set":
            if not parser.has_section("global"):
                parser.add_section("global")
            parser.set("global", key, cmd[7])
        else:
            removed = parser.has_section("global") and parser.remove_option(
                "global", key
            )
            if not removed:
                return subprocess.CompletedProcess(
                    cmd, 1, f"ERROR: No such key - {cmd[6]}"
                )
        with path.open("w", encoding="utf-8") as fh:
            parser.write(fh)
        return subprocess.CompletedProcess(cmd, 0, f"Writing to {path}\n")


def _same_path(found: str | None, exe: Path) -> bool:
    """On Windows shutil.which returns the PATHEXT spelling (python.EXE)."""
    return found is not None and os.path.normcase(found) == os.path.normcase(str(exe))


def _forget_decision() -> None:
    """Start the once-per-process decision and hint over."""
    re_env._agent_python.cache_clear()
    re_env._log_missing_python_hint.cache_clear()


def _fake_python(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / ("python.exe" if os.name == "nt" else "python")
    exe.write_text("")
    exe.chmod(0o755)
    return exe


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated DATA_DIR with a space and non-ASCII characters, no python on PATH."""
    from EvoScientist import paths

    data = tmp_path / "Jan Kowalski ąę" / ".evoscientist"
    monkeypatch.setattr(paths, "DATA_DIR", data)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    runner = FakeRunner()
    monkeypatch.setattr(re_env, "_run", runner)
    _forget_decision()
    yield {
        "data": data,
        "run": runner,
        "tmp": tmp_path,
        "env": data / "envs" / "default",
    }
    _forget_decision()


def _events():
    events: list[dict] = []
    return events, events.append


def _extra_package(env_path: Path) -> Path:
    """Something the agent installed on its own."""
    extra = env_path / "lib" / "seaborn"
    extra.mkdir(parents=True)
    return extra


# --------------------------------------------------------------------------- #
# find_usable_python
# --------------------------------------------------------------------------- #
def test_find_usable_python_uses_a_working_python(env, monkeypatch):
    exe = _fake_python(env["tmp"] / "conda" / "bin")
    monkeypatch.setenv("PATH", str(exe.parent))
    assert _same_path(re_env.find_usable_python(), exe)


def test_find_usable_python_none_without_python(env):
    assert re_env.find_usable_python() is None
    assert env["run"].calls == []


@pytest.mark.parametrize(
    "package",
    [
        "PythonSoftwareFoundation.PythonManager_3847v3x7pw1km",
        "PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0",
    ],
)
def test_find_usable_python_uses_a_python_software_foundation_alias(
    env, monkeypatch, package
):
    """The Python install manager's and Store CPython's ``python.exe`` in
    WindowsApps are real Pythons."""
    exe = _fake_python(env["tmp"] / "AppData" / "Local" / "Microsoft" / "WindowsApps")
    monkeypatch.setenv("PATH", str(exe.parent))
    monkeypatch.setattr(re_env, "_app_exec_link_package", lambda _path: package)
    assert _same_path(re_env.find_usable_python(), exe)


@pytest.mark.parametrize(
    "package",
    [
        # The Store's prompt: opens the Microsoft Store when run.
        "Microsoft.DesktopAppInstaller_8wekyb3d8bbwe",
        # Another publisher's alias named python.exe.
        "Contoso.Tools_1a2b3c4d5e6f7",
        # The right name with another publisher id.
        "PythonSoftwareFoundation.PythonManager_1a2b3c4d5e6f7",
        # An alias whose package cannot be read.
        None,
    ],
)
def test_find_usable_python_never_runs_any_other_alias(env, monkeypatch, package):
    exe = _fake_python(env["tmp"] / "AppData" / "Local" / "Microsoft" / "WindowsApps")
    monkeypatch.setenv("PATH", str(exe.parent))
    monkeypatch.setattr(re_env, "_app_exec_link_package", lambda _path: package)
    assert re_env.find_usable_python() is None
    assert env["run"].calls == []


def test_find_usable_python_returns_an_absolute_path(env, monkeypatch):
    exe = _fake_python(env["tmp"] / "venv" / "bin")
    monkeypatch.chdir(env["tmp"])
    monkeypatch.setenv("PATH", os.path.join("venv", "bin"))
    found = re_env.find_usable_python()
    assert os.path.isabs(found)
    assert _same_path(found, exe)


def test_parse_app_exec_link_reads_the_package_family_name():
    """Bytes as ``fsutil reparsepoint query`` showed them for the Python
    install manager's ``python.exe`` alias."""
    family = "PythonSoftwareFoundation.PythonManager_3847v3x7pw1km"
    strings = "\0".join(
        [
            family,
            f"{family}!Python.Exe",
            r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.PythonManager"
            r"_26.3.240.0_x64__3847v3x7pw1km\python.exe",
            "0",
            "",
        ]
    )
    payload = struct.pack("<I", 3) + strings.encode("utf-16-le")
    data = struct.pack("<IHH", 0x8000001B, len(payload), 0) + payload
    assert len(payload) == 0x1CC
    assert re_env._parse_app_exec_link(data) == family


def test_parse_app_exec_link_rejects_other_reparse_points():
    payload = struct.pack("<I", 3) + "x\0".encode("utf-16-le")
    symlink = struct.pack("<IHH", 0xA000000C, len(payload), 0) + payload
    assert re_env._parse_app_exec_link(symlink) is None
    assert re_env._parse_app_exec_link(b"") is None


@pytest.mark.skipif(os.name != "nt", reason="runs the real Win32 calls")
def test_app_exec_link_package_is_none_for_a_regular_file(tmp_path):
    """Runs the CreateFileW / DeviceIoControl declarations on the Windows CI
    jobs without needing a Store alias: a regular file is no reparse point."""
    regular = tmp_path / "python.exe"
    regular.write_bytes(b"")
    assert re_env._app_exec_link_package(str(regular)) is None


def test_find_usable_python_rejects_a_shim_that_fails_the_probe(env, monkeypatch):
    exe = _fake_python(env["tmp"] / "pyenv" / "shims")
    monkeypatch.setenv("PATH", str(exe.parent))
    env["run"].system_ok[os.path.normcase(str(exe))] = False
    assert re_env.find_usable_python() is None


def _exec_probe(capsys) -> list[str]:
    exec(re_env._SYSTEM_PYTHON_PROBE, {})
    return capsys.readouterr().out.splitlines()


def test_system_python_probe_reports_python_2_without_python_3_calls(
    monkeypatch, capsys
):
    """Under Python 2 the probe prints its version and skips the PEP 668 check,
    whose sysconfig call differs there; the 3.9 floor then rejects it."""
    import sysconfig

    def python_3_only(_name):
        raise AssertionError("PEP 668 check ran under Python 2")

    monkeypatch.setattr(sys, "version_info", (2, 7, 18, "final", 0))
    monkeypatch.setattr(sysconfig, "get_path", python_3_only)
    lines = _exec_probe(capsys)
    assert lines == ["2.7", sys.prefix, "0"]


def test_system_python_probe_runs_on_python_3():
    """A real subprocess; the test interpreter is a venv, so a PEP 668 marker
    on its base does not count."""
    result = subprocess.run(
        [sys.executable, "-c", re_env._SYSTEM_PYTHON_PROBE],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        f"{sys.version_info[0]}.{sys.version_info[1]}",
        sys.prefix,
        "0",
    ]


@pytest.mark.parametrize(("in_venv", "flag"), [(False, "1"), (True, "0")])
def test_system_python_probe_flags_an_externally_managed_python(
    monkeypatch, tmp_path, capsys, in_venv, flag
):
    """PEP 668 applies only outside a venv."""
    import sysconfig

    (tmp_path / "EXTERNALLY-MANAGED").write_text("[externally-managed]\n")
    monkeypatch.setattr(sysconfig, "get_path", lambda _name: str(tmp_path))
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/opt/base" if in_venv else "/usr")
    assert _exec_probe(capsys)[2] == flag


@pytest.mark.parametrize(
    ("output", "usable"),
    [
        ("3.12\n/opt/conda\n0\n", True),
        ("3.9\n/opt/conda\n0\n", True),
        ("3.8\n/opt/conda\n0\n", False),
        ("2.7\n/usr\n0\n", False),
        ("3.12\n/usr\n1\n", False),  # PEP 668 outside a venv
        ("", False),
        ("Python 3.12\n", False),
        # Startup warnings on stderr share the pipe and come first.
        (
            "Error processing line 1 of /env/site-packages/broken.pth:\n"
            "  ModuleNotFoundError: No module named 'gone'\n"
            "Remainder of file ignored\n"
            "3.12\n/opt/conda\n0\n",
            True,
        ),
    ],
)
def test_unusable_reason(output, usable):
    assert (re_env._unusable_reason(output) is None) is usable


@pytest.mark.parametrize(
    ("marker", "usable"),
    [("uv-receipt.toml", False), (".evoscientist-managed", False), (None, True)],
)
def test_evoscientists_own_environment(monkeypatch, tmp_path, marker, usable):
    """The environment EvoScientist runs from is unusable only when its
    installer made it (uv tool install, the Docker image); a conda env or venv
    the user made stays usable."""
    if marker is not None:
        (tmp_path / marker).write_text("")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    output = f"3.12\n{tmp_path}{os.sep}\n0\n"  # another spelling of the prefix
    assert (re_env._unusable_reason(output) is None) is usable


def test_a_marker_in_another_prefix_does_not_count(monkeypatch, tmp_path):
    (tmp_path / "uv-receipt.toml").write_text("")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "evoscientist"))
    assert re_env._unusable_reason(f"3.12\n{tmp_path}\n0\n") is None


def test_find_usable_python_runs_the_python_3_probe(env, monkeypatch):
    exe = _fake_python(env["tmp"] / "conda" / "bin")
    monkeypatch.setenv("PATH", str(exe.parent))
    found = re_env.find_usable_python()
    assert env["run"].calls == [[found, "-c", re_env._SYSTEM_PYTHON_PROBE]]


def _own_env_before_a_system_python(env, monkeypatch) -> Path:
    """Our environment first on PATH (activated by hand, or a nested EvoSci in
    the agent's shell), a working system python after it."""
    re_env.ensure_research_env("default")
    re_env._env_python(env["env"]).chmod(0o755)  # the fake venv wrote a plain file
    system = _fake_python(env["tmp"] / "conda" / "bin")
    monkeypatch.setenv(
        "PATH", f"{re_env._bin_dir(env['env'])}{os.pathsep}{system.parent}"
    )
    _forget_decision()
    return system


def test_own_environment_first_on_path_is_not_a_system_python(env, monkeypatch):
    """The shell runs the first python, so a later system python must not be
    reported in its place."""
    _own_env_before_a_system_python(env, monkeypatch)
    assert re_env.find_usable_python() is None
    assert re_env.run_stage(lambda event: None, "default").status == "done"


def test_working_own_environment_first_on_path_is_injected(env, monkeypatch):
    _own_env_before_a_system_python(env, monkeypatch)
    overrides = re_env.research_env_overrides()
    assert overrides is not None
    assert overrides["VIRTUAL_ENV"] == str(env["env"])
    assert _same_path(re_env.agent_python(), re_env._env_python(env["env"]))


def test_broken_own_environment_first_on_path_gives_the_hint(env, monkeypatch):
    """A later system python does not hide a broken environment the shell
    would run first: the hint points at `EvoSci setup`, which rebuilds it."""
    _own_env_before_a_system_python(env, monkeypatch)
    env["run"].env_starts = False
    assert re_env.research_env_overrides() is None
    assert re_env.missing_python_hint() == re_env.MISSING_PYTHON_HINT


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
def test_stage_skipped_when_a_usable_python_exists(env, monkeypatch):
    exe = _fake_python(env["tmp"] / "conda" / "bin")
    monkeypatch.setenv("PATH", str(exe.parent))
    result = re_env.run_stage(lambda event: None, "default")
    assert result.status == "skipped"
    assert result.detail["reason"] == "system_python"
    assert _same_path(result.detail["python"], exe)
    assert not env["env"].exists()


def test_stage_builds_the_environment(env):
    result = re_env.run_stage(lambda event: None, "default")
    assert result.status == "done"
    assert result.detail == {
        "source": "venv",
        "path": str(env["env"]),
        "python": "3.12.9",
    }
    assert re_env.is_ready(env["env"])
    assert env["run"].kinds() == ["venv", "pip", "imports"]
    venv_cmd, pip_cmd = env["run"].calls[0], env["run"].calls[1]
    assert venv_cmd[0] == sys.executable
    assert pip_cmd[0] == str(re_env._env_python(env["env"]))
    assert pip_cmd[pip_cmd.index("--only-binary") + 1] == ":all:"
    assert env["run"].pip_requirements() == [
        [f"{name}=={version}" for name, version in re_env.PINS.items()]
    ]
    assert "--upgrade" not in pip_cmd
    assert "--index-url" not in pip_cmd
    assert not re_env._pip_config(env["env"]).exists()


def test_stage_reports_progress_per_step_in_order(env):
    events, emit = _events()
    re_env.run_stage(emit, "default")
    assert all(e["status"] == "running" for e in events)
    progress = [e["progress"] for e in events]
    assert progress == sorted(progress)
    assert [e["message"] for e in events][:2] == [
        "Checking for a usable python",
        "Creating the virtual environment",
    ]


def test_progress_keeps_going_forward_when_a_repair_falls_back_to_a_rebuild(env):
    re_env.ensure_research_env("default")
    env["run"].imports_script = [False, False]
    events, emit = _events()
    re_env.run_stage(emit, "default")
    assert "Creating the virtual environment" in [e["message"] for e in events]
    progress = [e["progress"] for e in events]
    assert progress == sorted(progress)


def test_pin_without_a_wheel_falls_back_to_the_newest_wheel(env):
    """pip lists other versions: the pin has no wheel for this Python."""
    env["run"].no_wheel = {"scipy"}
    result = re_env.run_stage(lambda event: None, "default")
    assert result.status == "done"
    assert result.detail["unpinned"] == ["scipy"]
    first, second = env["run"].pip_requirements()
    assert f"scipy=={re_env.PINS['scipy']}" in first
    assert "scipy" in second
    assert f"numpy=={re_env.PINS['numpy']}" in second


def test_unpinned_packages_are_reported_on_every_later_run(env):
    """Untested versions are a fact about the environment, not one run."""
    env["run"].no_wheel = {"scipy"}
    re_env.run_stage(lambda event: None, "default")
    env["run"].calls.clear()
    result = re_env.run_stage(lambda event: None, "default")
    assert result.detail["unpinned"] == ["scipy"]
    assert env["run"].kinds() == ["imports"]  # nothing installed


def test_repair_at_the_pin_clears_an_unpinned_package(env):
    env["run"].no_wheel = {"scipy"}
    re_env.ensure_research_env("default")
    env["run"].no_wheel = set()  # e.g. after a pin update
    env["run"].imports_script = [False]
    env["run"].failed_packages = ["scipy"]
    assert re_env.ensure_research_env("default").unpinned == []
    assert re_env.ensure_research_env("default").unpinned == []


def test_marker_from_an_older_version_still_counts_as_ready(env):
    """#542 wrote only the Python version into the marker."""
    re_env.ensure_research_env("default")
    (env["env"] / re_env._READY_MARKER).write_text("3.12.9", encoding="utf-8")
    env["run"].calls.clear()
    result = re_env.ensure_research_env("default")
    assert result == re_env.EnvResult("3.12.9", [])
    assert env["run"].kinds() == ["imports"]


def test_stage_without_a_fallback_has_no_unpinned_detail(env):
    result = re_env.run_stage(lambda event: None, "default")
    assert "unpinned" not in result.detail


@pytest.mark.parametrize(("mirror", "hinted"), [("default", True), ("cn", False)])
def test_unreachable_index_is_install_failed_without_a_fallback(env, mirror, hinted):
    """pip warns about the connection, then says "from versions: none"."""
    env["run"].index_down = True
    with pytest.raises(StageError) as exc:
        re_env.ensure_research_env(mirror)
    assert exc.value.code == "install_failed"
    assert "could not be reached" in exc.value.message
    assert (re_env.CN_MIRROR_HINT in exc.value.message) is hinted
    assert len(env["run"].pip_requirements()) == 1
    assert not re_env.is_ready(env["env"])


def test_no_wheel_for_this_python_gets_no_mirror_hint(env):
    """ "from versions: none" without a connection warning: the index was
    reached but has no wheel for this interpreter (e.g. a brand-new Python),
    which the mirror does not change."""
    env["run"].no_wheel_at_all = {"numpy"}
    with pytest.raises(StageError) as exc:
        re_env.ensure_research_env("default")
    assert exc.value.code == "install_failed"
    assert "no wheel of numpy for this Python" in exc.value.message
    assert re_env.CN_MIRROR_HINT not in exc.value.message


def test_no_wheel_at_all_after_the_unpinned_fallback_is_named(env):
    """The second "none" is for the bare name, without ``==version``."""
    env["run"].no_wheel = {"scipy"}  # the pinned request lists other versions
    env["run"].no_wheel_at_all = {"scipy"}  # the bare one finds none
    with pytest.raises(StageError) as exc:
        re_env.ensure_research_env("default")
    assert "no wheel of scipy for this Python" in exc.value.message
    assert [reqs[-1] for reqs in env["run"].pip_requirements()] == [
        f"scipy=={re_env.PINS['scipy']}",
        "scipy",
    ]


def test_import_check_reads_its_own_lines_past_startup_warnings(tmp_path, monkeypatch):
    """A real subprocess in a throwaway venv whose broken .pth writes to
    stderr, which shares the pipe, before the check's lines. ``json`` and
    ``numpy`` (absent there) stand in for the four packages."""
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True
    )
    (site_packages,) = venv.glob("**/site-packages")
    (site_packages / "broken.pth").write_text("import no_such_module_for_pth\n")
    monkeypatch.setattr(
        re_env,
        "_IMPORT_CHECK",
        re_env._IMPORT_CHECK.replace(repr(re_env.PACKAGES), "('json', 'numpy')"),
    )
    check = re_env._import_check(venv)
    assert check == re_env.ImportCheck(
        ".".join(map(str, sys.version_info[:3])), ("numpy",)
    )


def test_import_check_ignores_warnings_before_its_version_line(env):
    re_env.ensure_research_env("default")
    env[
        "run"
    ].import_noise = (
        "UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr'\n"
    )
    assert re_env._import_check(env["env"]) == re_env.ImportCheck("3.12.9", ())


def test_import_check_without_a_version_line_reinstalls_everything(env, monkeypatch):
    """Exit 0 but no version line: never an empty repair ``pip install``."""
    re_env.ensure_research_env("default")

    def run(cmd, timeout):
        if FakeRunner._kind(list(cmd)) == "imports":
            return subprocess.CompletedProcess(cmd, 0, "\n")
        return env["run"](cmd, timeout)

    monkeypatch.setattr(re_env, "_run", run)
    assert re_env._import_check(env["env"]) == re_env.ImportCheck(None, re_env.PACKAGES)


def test_import_check_names_the_failed_packages(env, monkeypatch):
    re_env.ensure_research_env("default")
    env["run"].imports_ok = False
    env["run"].failed_packages = ["scipy"]
    assert re_env._import_check(env["env"]) == re_env.ImportCheck("3.12.9", ("scipy",))


def test_failed_pip_leaves_no_ready_marker(env):
    env["run"].pip_ok = False
    with pytest.raises(StageError) as exc:
        re_env.ensure_research_env("default")
    assert exc.value.code == "install_failed"
    assert "ERROR: offline" in exc.value.message
    assert not re_env.is_ready(env["env"])


def test_failed_pip_logs_its_whole_output_at_warning(env, monkeypatch, caplog):
    """Offline, pip's last line reads like a packaging problem; the network
    errors before it must reach the user (WARNING is shown by default)."""
    pip_output = (
        "WARNING: Retrying ... NewConnectionError: Connection refused\n"
        "ERROR: No matching distribution found for numpy\n"
    )

    def run(cmd, timeout):
        if FakeRunner._kind(list(cmd)) == "pip":
            return subprocess.CompletedProcess(cmd, 1, pip_output)
        return env["run"](cmd, timeout)

    monkeypatch.setattr(re_env, "_run", run)
    with caplog.at_level(logging.WARNING, logger=re_env.__name__):
        with pytest.raises(StageError):
            re_env.ensure_research_env("default")
    assert "NewConnectionError" in caplog.text


def test_failed_venv_logs_its_whole_output_at_warning(env, caplog):
    env["run"].venv_ok = False
    env["run"].venv_output = "first cause line\nError: last line\n"
    with caplog.at_level(logging.WARNING, logger=re_env.__name__):
        with pytest.raises(StageError):
            re_env.ensure_research_env("default")
    assert "first cause line" in caplog.text


def test_missing_ensurepip_names_the_venv_package(env):
    """The output Debian's venv prints when ensurepip is missing."""
    env["run"].venv_ok = False
    env["run"].venv_output = (
        "The virtual environment was not created successfully because ensurepip"
        " is not\navailable.  On Debian/Ubuntu systems, you need to install the"
        " python3-venv\npackage using the following command.\n\n"
        "    apt install python3.12-venv\n\n"
        "Failing command: /x/bin/python3\n"
    )
    with pytest.raises(StageError) as exc:
        re_env.ensure_research_env("default")
    assert exc.value.code == "install_failed"
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert f"python{version}-venv" in exc.value.message


def test_other_ensurepip_failures_get_no_debian_hint(env):
    """Any failed pip bootstrap names ensurepip, e.g. a read-only
    site-packages on macOS."""
    env["run"].venv_ok = False
    env["run"].venv_output = (
        "Error: Command '['/x/bin/python3', '-m', 'ensurepip', '--upgrade', "
        "'--default-pip']' returned non-zero exit status 1."
    )
    with pytest.raises(StageError) as exc:
        re_env.ensure_research_env("default")
    assert "Debian" not in exc.value.message


def test_failed_import_check_is_probe_failed(env):
    env["run"].imports_ok = False
    with pytest.raises(StageError) as exc:
        re_env.ensure_research_env("default")
    assert exc.value.code == "probe_failed"
    assert not re_env.is_ready(env["env"])


def test_failed_import_check_logs_its_output_at_warning(env, monkeypatch, caplog):
    def run(cmd, timeout):
        if FakeRunner._kind(list(cmd)) == "imports":
            return subprocess.CompletedProcess(cmd, 1, "ImportError: libgfortran\n")
        return env["run"](cmd, timeout)

    monkeypatch.setattr(re_env, "_run", run)
    with caplog.at_level(logging.WARNING, logger=re_env.__name__):
        with pytest.raises(StageError):
            re_env.ensure_research_env("default")
    assert "ImportError: libgfortran" in caplog.text


def test_import_check_ignores_the_callers_cwd_and_pythonpath(tmp_path, monkeypatch):
    """A real subprocess, with ``json`` standing in for the four packages."""
    (tmp_path / "json.py").write_text("raise ImportError('shadowed')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(re_env, "_env_python", lambda _env: Path(sys.executable))
    monkeypatch.setattr(re_env, "_IMPORT_CHECK", "import json; print('version 3.12.9')")
    assert re_env._import_check(tmp_path) == re_env.ImportCheck("3.12.9", ())


def test_setup_commands_run_isolated(env):
    re_env.ensure_research_env("cn")
    env["run"].imports_script = [False]
    re_env.ensure_research_env("cn")
    kinds = {FakeRunner._kind(cmd) for cmd in env["run"].raw}
    assert kinds == {"venv", "config", "pip", "imports", "starts"}
    assert all(cmd[1] == "-I" for cmd in env["run"].raw)


def test_unwritable_data_dir_is_install_failed(env, monkeypatch):
    env["data"].parent.mkdir(parents=True)
    env["data"].write_text("a file where the data dir should be")
    with pytest.raises(StageError) as exc:
        re_env.ensure_research_env("default")
    assert exc.value.code == "install_failed"


# --------------------------------------------------------------------------- #
# Mirror
# --------------------------------------------------------------------------- #
def test_cn_mirror_adds_the_index_and_writes_the_pip_config(env):
    re_env.ensure_research_env("cn")
    assert env["run"].kinds() == ["venv", "config", "pip", "imports"]
    pip_cmd = env["run"].calls[2]
    assert pip_cmd[pip_cmd.index("--index-url") + 1] == re_env.CN_INDEX_URL
    config = re_env._pip_config(env["env"]).read_text(encoding="utf-8")
    assert f"index-url = {re_env.CN_INDEX_URL}" in config


def test_pip_config_follows_the_mirror_and_keeps_other_entries(env):
    re_env.ensure_research_env("default")
    config = re_env._pip_config(env["env"])
    config.write_text("[global]\ntimeout = 60\n", encoding="utf-8")
    env["run"].calls.clear()
    re_env.ensure_research_env("cn")
    assert re_env._site_index_url(env["env"]) == re_env.CN_INDEX_URL
    re_env.ensure_research_env("default")
    assert re_env._site_index_url(env["env"]) is None
    assert "timeout = 60" in config.read_text(encoding="utf-8")
    # A ready environment needs no pip install to follow the mirror.
    assert "pip" not in env["run"].kinds()


def test_pip_config_runs_pip_only_when_the_entry_changes(env):
    re_env.ensure_research_env("cn")
    env["run"].calls.clear()
    re_env.ensure_research_env("cn")
    assert env["run"].kinds() == ["imports"]


def test_default_mirror_keeps_an_index_the_user_set(env):
    re_env.ensure_research_env("default")
    config = re_env._pip_config(env["env"])
    config.write_text("[global]\nindex-url = https://corp/simple\n", encoding="utf-8")
    re_env.ensure_research_env("default")
    assert re_env._site_index_url(env["env"]) == "https://corp/simple"


def test_failed_repair_still_brings_the_pip_config_in_line(env):
    re_env.ensure_research_env("default")
    env["run"].imports_ok = False
    env["run"].pip_ok = False
    with pytest.raises(StageError):
        re_env.ensure_research_env("cn")
    assert re_env._site_index_url(env["env"]) == re_env.CN_INDEX_URL


def test_skipped_stage_leaves_the_pip_config_untouched(env, monkeypatch):
    re_env.ensure_research_env("cn")
    exe = _fake_python(env["tmp"] / "conda" / "bin")
    monkeypatch.setenv("PATH", str(exe.parent))
    assert re_env.run_stage(lambda event: None, "default").status == "skipped"
    assert re_env._site_index_url(env["env"]) == re_env.CN_INDEX_URL


# --------------------------------------------------------------------------- #
# Re-runs
# --------------------------------------------------------------------------- #
def test_ready_environment_reruns_without_network_and_keeps_agent_installs(env):
    re_env.ensure_research_env("default")
    extra = _extra_package(env["env"])
    env["run"].calls.clear()
    env["run"].pip_ok = False  # offline: any pip call would fail
    assert re_env.ensure_research_env("default").version == "3.12.9"
    assert env["run"].kinds() == ["imports"]
    assert extra.exists()


def test_missing_package_is_reinstalled_in_place(env):
    re_env.ensure_research_env("default")
    extra = _extra_package(env["env"])
    env["run"].calls.clear()
    env["run"].imports_script = [False]  # pandas gone, back after pip
    env["run"].failed_packages = ["pandas"]
    assert re_env.ensure_research_env("default").version == "3.12.9"
    assert env["run"].kinds() == ["imports", "starts", "pip", "imports"]
    assert extra.exists()
    assert re_env.is_ready(env["env"])


def test_repair_reinstalls_only_the_failed_package_at_its_pin(env):
    """Passing only pandas keeps a newer numpy that the agent installed:
    without --upgrade pip leaves packages it was not asked for alone."""
    re_env.ensure_research_env("default")
    env["run"].calls.clear()
    env["run"].imports_script = [False]
    env["run"].failed_packages = ["pandas"]
    re_env.ensure_research_env("default")
    assert env["run"].pip_requirements() == [[f"pandas=={re_env.PINS['pandas']}"]]


def test_still_broken_after_repair_is_rebuilt(env):
    re_env.ensure_research_env("default")
    extra = _extra_package(env["env"])
    env["run"].calls.clear()
    env["run"].imports_ok = False
    with pytest.raises(StageError) as exc:
        re_env.ensure_research_env("default")
    assert exc.value.code == "probe_failed"
    assert env["run"].kinds() == [
        "imports",
        "starts",
        "pip",
        "imports",
        "venv",
        "pip",
        "imports",
    ]
    assert not extra.exists()


def test_environment_whose_python_does_not_start_is_rebuilt(env):
    re_env.ensure_research_env("default")
    extra = _extra_package(env["env"])
    env["run"].calls.clear()
    env["run"].env_starts = False  # base interpreter removed; a new venv fixes it
    assert re_env.ensure_research_env("default").version == "3.12.9"
    assert env["run"].kinds() == ["imports", "starts", "venv", "pip", "imports"]
    assert not extra.exists()
    assert re_env.is_ready(env["env"])


def test_failed_repair_keeps_the_environment_and_its_marker(env):
    re_env.ensure_research_env("default")
    extra = _extra_package(env["env"])
    env["run"].imports_ok = False
    env["run"].pip_ok = False
    with pytest.raises(StageError) as exc:
        re_env.ensure_research_env("default")
    assert exc.value.code == "install_failed"
    assert extra.exists()
    assert re_env.is_ready(env["env"])


# --------------------------------------------------------------------------- #
# Runtime overrides
# --------------------------------------------------------------------------- #
def test_overrides_none_without_ready_marker(env):
    assert re_env.research_env_overrides() is None


def test_overrides_none_when_a_usable_python_exists(env, monkeypatch):
    re_env.ensure_research_env("default")
    exe = _fake_python(env["tmp"] / "conda" / "bin")
    monkeypatch.setenv("PATH", str(exe.parent))
    _forget_decision()
    assert re_env.research_env_overrides() is None
    assert _same_path(re_env.agent_python(), exe)


def test_ready_environment_whose_python_does_not_start_is_not_injected(env):
    """On Windows the venv's python.exe survives the removal of its base
    interpreter; the agent then gets the setup hint instead of a dead python."""
    re_env.ensure_research_env("default")
    env["run"].env_starts = False
    _forget_decision()
    assert re_env.research_env_overrides() is None
    assert re_env.missing_python_hint() == re_env.MISSING_PYTHON_HINT


def test_overrides_prepend_the_environment_and_follow_path(env, monkeypatch):
    re_env.ensure_research_env("default")
    _forget_decision()
    overrides = re_env.research_env_overrides()
    bin_dir = str(re_env._bin_dir(env["env"]))
    assert overrides == {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "VIRTUAL_ENV": str(env["env"]),
    }
    assert re_env.agent_python() == str(re_env._env_python(env["env"]))
    # A PATH change after the decision, e.g. activate_runtime(), is kept.
    monkeypatch.setenv("PATH", f"/private/node/bin{os.pathsep}{os.environ['PATH']}")
    assert re_env.research_env_overrides()["PATH"].startswith(
        f"{bin_dir}{os.pathsep}/private/node/bin"
    )


def test_bin_dir_is_scripts_on_windows(monkeypatch):
    # Built before os.name is patched: Path() picks its flavour from it.
    env_path, scripts, bin_ = Path("env"), Path("env") / "Scripts", Path("env") / "bin"
    monkeypatch.setattr(re_env.os, "name", "nt")
    assert re_env._bin_dir(env_path) == scripts
    monkeypatch.setattr(re_env.os, "name", "posix")
    assert re_env._bin_dir(env_path) == bin_


def test_decision_is_logged_once_with_a_setup_hint(env, caplog):
    with caplog.at_level(logging.INFO, logger=re_env.__name__):
        assert re_env.research_env_overrides() is None
        assert re_env.research_env_overrides() is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Run `EvoSci setup`, then restart EvoScientist" in warnings[0].getMessage()


@pytest.mark.parametrize(
    ("sidecar", "own", "shown"),
    [
        # A reused server whose agents have a python: no hint, even if we lack one.
        ({re_env.SIDECAR_KEY: "/conda/bin/python"}, None, False),
        # A reused server whose agents have none: hint when we have none too;
        # when we have one, the drift warning names the fix.
        ({re_env.SIDECAR_KEY: None}, None, True),
        ({re_env.SIDECAR_KEY: None}, "/usr/bin/python", False),
        # No record (older server) or no reuse: our own decision.
        ({"workspace": "/w"}, None, True),
        (None, None, True),
        (None, "/usr/bin/python", False),
    ],
)
def test_server_missing_python_hint(monkeypatch, sidecar, own, shown):
    monkeypatch.setattr(re_env, "agent_python", lambda: own)
    assert (re_env.server_missing_python_hint(sidecar) is not None) is shown


def test_missing_python_hint_is_returned_but_not_logged(env, caplog):
    with caplog.at_level(logging.INFO, logger=re_env.__name__):
        assert re_env.missing_python_hint() == re_env.MISSING_PYTHON_HINT
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    re_env.ensure_research_env("default")
    _forget_decision()
    assert re_env.missing_python_hint() is None


@pytest.mark.parametrize(
    ("recorded", "current", "warns"),
    [
        ("/env/bin/python", "/env/bin/python", False),
        ("/conda/bin/python", "/env/bin/python", True),
        (None, "/env/bin/python", True),
        ("/env/bin/python", None, True),
        (None, None, False),
        # Windows: the same file spelled with another case.
        (r"C:\Users\A\python.EXE", r"c:\users\a\python.exe", False),
    ],
)
def test_python_drift_message(monkeypatch, recorded, current, warns):
    monkeypatch.setattr(re_env, "agent_python", lambda: current)
    # Windows-style normcase, so the case rule is tested on every platform.
    monkeypatch.setattr(re_env.os.path, "normcase", str.lower)
    message = re_env.python_drift_message({re_env.SIDECAR_KEY: recorded})
    assert (message is not None) is warns


def test_relative_data_dir_puts_an_absolute_path_on_path(env, monkeypatch):
    from EvoScientist import paths

    monkeypatch.chdir(env["tmp"])
    monkeypatch.setattr(paths, "DATA_DIR", Path("rel-data"))
    re_env.ensure_research_env("default")
    _forget_decision()
    overrides = re_env.research_env_overrides()
    assert os.path.isabs(overrides["PATH"].split(os.pathsep)[0])
    assert os.path.isabs(overrides["VIRTUAL_ENV"])


def test_python_drift_message_none_without_a_record():
    """A server started by an older version has no record."""
    assert re_env.python_drift_message({"workspace": "/w"}) is None


def test_python_used_is_logged_once(env, caplog):
    re_env.ensure_research_env("default")
    _forget_decision()
    with caplog.at_level(logging.INFO, logger=re_env.__name__):
        re_env.research_env_overrides()
        re_env.research_env_overrides()
    lines = [r for r in caplog.records if "Agent shell python" in r.getMessage()]
    assert len(lines) == 1
    assert str(env["env"]) in lines[0].getMessage()
