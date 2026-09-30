"""Tests for the research environment stage. ``venv``, pip and every probe are mocked."""

from __future__ import annotations

import logging
import os
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
        if kind == "imports":
            scripted = self.imports_script.pop(0) if self.imports_script else None
            ok = self.env_starts and (self.imports_ok if scripted is None else scripted)
            return subprocess.CompletedProcess(cmd, 0 if ok else 1, "3.12.9\n")
        if cmd[0] in self.system_ok:
            return subprocess.CompletedProcess(
                cmd, 0 if self.system_ok[cmd[0]] else 1, ""
            )
        return subprocess.CompletedProcess(cmd, 0 if self.env_starts else 1, "")


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
    re_env._agent_python.cache_clear()
    yield {
        "data": data,
        "run": runner,
        "tmp": tmp_path,
        "env": data / "envs" / "default",
    }
    re_env._agent_python.cache_clear()


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
    assert re_env.find_usable_python() == str(exe)


def test_find_usable_python_none_without_python(env):
    assert re_env.find_usable_python() is None
    assert env["run"].calls == []


def test_find_usable_python_ignores_the_store_alias_without_running_it(
    env, monkeypatch
):
    exe = _fake_python(env["tmp"] / "AppData" / "Local" / "Microsoft" / "WindowsApps")
    monkeypatch.setenv("PATH", str(exe.parent))
    assert re_env.find_usable_python() is None
    assert env["run"].calls == []


def test_find_usable_python_rejects_a_shim_that_fails_the_probe(env, monkeypatch):
    exe = _fake_python(env["tmp"] / "pyenv" / "shims")
    monkeypatch.setenv("PATH", str(exe.parent))
    env["run"].system_ok[str(exe)] = False
    assert re_env.find_usable_python() is None


def test_find_usable_python_leaves_out_our_own_environment(env, monkeypatch):
    re_env.ensure_research_env("default")
    monkeypatch.setenv("PATH", str(re_env._bin_dir(env["env"])))
    assert re_env.find_usable_python() is None


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
def test_stage_skipped_when_a_usable_python_exists(env, monkeypatch):
    exe = _fake_python(env["tmp"] / "conda" / "bin")
    monkeypatch.setenv("PATH", str(exe.parent))
    result = re_env.run_stage(lambda event: None, "default")
    assert result.status == "skipped"
    assert result.detail == {"reason": "system_python", "python": str(exe)}
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
    pip_cmd = env["run"].calls[1]
    assert pip_cmd[pip_cmd.index("--index-url") + 1] == re_env.CN_INDEX_URL
    config = re_env._pip_config(env["env"]).read_text(encoding="utf-8")
    assert f"index-url = {re_env.CN_INDEX_URL}" in config


def test_pip_config_follows_the_mirror_on_a_ready_environment(env):
    re_env.ensure_research_env("default")
    env["run"].calls.clear()
    re_env.ensure_research_env("cn")
    assert re_env._pip_config(env["env"]).exists()
    re_env.ensure_research_env("default")
    assert not re_env._pip_config(env["env"]).exists()
    assert "pip" not in env["run"].kinds()


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
    re_env._agent_python.cache_clear()
    assert re_env.research_env_overrides() is None
    assert re_env.agent_python() == str(exe)


def test_overrides_prepend_the_environment_and_follow_path(env, monkeypatch):
    re_env.ensure_research_env("default")
    re_env._agent_python.cache_clear()
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
    assert "EvoSci setup --stage research-env" in warnings[0].getMessage()


def test_python_used_is_logged_once(env, caplog):
    re_env.ensure_research_env("default")
    re_env._agent_python.cache_clear()
    with caplog.at_level(logging.INFO, logger=re_env.__name__):
        re_env.research_env_overrides()
        re_env.research_env_overrides()
    lines = [r for r in caplog.records if "Agent shell python" in r.getMessage()]
    assert len(lines) == 1
    assert str(env["env"]) in lines[0].getMessage()
