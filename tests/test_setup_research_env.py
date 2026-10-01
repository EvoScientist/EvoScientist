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


class FakeRunner:
    """Stands in for ``research_env._run``: venv, pip and the probes."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.venv_ok = True
        self.venv_output = ""
        self.pip_ok = True
        self.env_starts = True
        self.imports_ok = True
        # Results for the next import checks, used before ``imports_ok``.
        self.imports_script: list[bool] = []
        self.system_ok: dict[str, bool] = {}

    def kinds(self) -> list[str]:
        return [self._kind(cmd) for cmd in self.calls]

    @staticmethod
    def _kind(cmd: list[str]) -> str:
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
        cmd = list(cmd)
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
            return subprocess.CompletedProcess(
                cmd, 0 if self.pip_ok else 1, "" if self.pip_ok else "ERROR: offline"
            )
        if kind == "config":
            return self._pip_config(cmd)
        if kind == "imports":
            scripted = self.imports_script.pop(0) if self.imports_script else None
            ok = self.env_starts and (self.imports_ok if scripted is None else scripted)
            return subprocess.CompletedProcess(cmd, 0 if ok else 1, "3.12.9\n")
        # Keyed by normcase: on Windows shutil.which returns the PATHEXT
        # spelling (python.EXE).
        system_ok = self.system_ok.get(os.path.normcase(cmd[0]))
        if system_ok is not None:
            return subprocess.CompletedProcess(cmd, 0 if system_ok else 1, "")
        return subprocess.CompletedProcess(cmd, 0 if self.env_starts else 1, "")

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


def test_find_usable_python_rejects_a_shim_that_fails_the_probe(env, monkeypatch):
    exe = _fake_python(env["tmp"] / "pyenv" / "shims")
    monkeypatch.setenv("PATH", str(exe.parent))
    env["run"].system_ok[os.path.normcase(str(exe))] = False
    assert re_env.find_usable_python() is None


def test_system_python_probe_rejects_python_2(monkeypatch):
    """The probe that decides "usable" exits non-zero under Python 2."""
    monkeypatch.setattr(sys, "version_info", (2, 7, 18, "final", 0))
    with pytest.raises(SystemExit) as exc:
        exec(re_env._SYSTEM_PYTHON_PROBE, {})
    assert exc.value.code


def test_system_python_probe_accepts_python_3():
    result = subprocess.run([sys.executable, "-c", re_env._SYSTEM_PYTHON_PROBE])
    assert result.returncode == 0


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
    assert set(re_env.PACKAGES) <= set(pip_cmd)
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


def test_missing_ensurepip_names_python3_venv(env):
    env["run"].venv_ok = False
    env["run"].venv_output = (
        "Error: Command '['/x/bin/python3', '-m', 'ensurepip', '--upgrade', "
        "'--default-pip']' returned non-zero exit status 1."
    )
    with pytest.raises(StageError) as exc:
        re_env.ensure_research_env("default")
    assert exc.value.code == "install_failed"
    assert "python3-venv" in exc.value.message


def test_failed_import_check_is_probe_failed(env):
    env["run"].imports_ok = False
    with pytest.raises(StageError) as exc:
        re_env.ensure_research_env("default")
    assert exc.value.code == "probe_failed"
    assert not re_env.is_ready(env["env"])


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
    assert re_env.ensure_research_env("default") == "3.12.9"
    assert env["run"].kinds() == ["imports"]
    assert extra.exists()


def test_missing_package_is_reinstalled_in_place(env):
    re_env.ensure_research_env("default")
    extra = _extra_package(env["env"])
    env["run"].calls.clear()
    env["run"].imports_script = [False]  # pandas gone, back after pip
    assert re_env.ensure_research_env("default") == "3.12.9"
    assert env["run"].kinds() == ["imports", "starts", "pip", "imports"]
    assert extra.exists()
    assert re_env.is_ready(env["env"])


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
    assert re_env.ensure_research_env("default") == "3.12.9"
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
        # A reused server whose agents have none: hint, even if we have one.
        ({re_env.SIDECAR_KEY: None}, "/usr/bin/python", True),
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
