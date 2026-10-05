"""Tests for EvoScientist.agent_shell: the shell the agent's commands run in.

The unit tests run anywhere: a "recorded bash" is faked with ``/bin/bash`` on
POSIX, which runs the script file the same way Git Bash does. The tests marked
``windows_bash`` run the real Git for Windows bash and need a Windows host with
Git for Windows in ``C:\\Program Files\\Git`` (the ``windows-latest`` CI image
has one).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from EvoScientist import agent_shell
from EvoScientist import background as bg
from EvoScientist.backends import CustomSandboxBackend
from EvoScientist.setup.git import GitInfo

# Saved at import, before the autouse fixture in conftest pins it to None.
_REAL_AGENT_BASH = agent_shell.agent_bash

_SYSTEM_GIT = Path(r"C:\Program Files\Git")
windows_bash = pytest.mark.skipif(
    sys.platform != "win32" or not (_SYSTEM_GIT / "bin" / "bash.exe").is_file(),
    reason="needs Git for Windows in C:\\Program Files\\Git",
)


def _info(bash: Path, source: str = "system") -> GitInfo:
    return GitInfo(
        source, "2.56.0.windows.1", bash.parent.parent / "cmd" / "git.exe", bash
    )


@pytest.fixture
def scripts(tmp_path, monkeypatch) -> Path:
    """Script files go to a test directory, swept only when a test asks."""
    directory = tmp_path / "scripts"
    monkeypatch.setattr(agent_shell, "_script_dir", lambda: directory)
    monkeypatch.setattr(agent_shell, "_sweep_stale_scripts", lambda: None)
    return directory


@pytest.fixture
def fake_bash(scripts, monkeypatch) -> GitInfo:
    """A recorded bash that is really /bin/bash (POSIX hosts)."""
    bash = shutil.which("bash")
    if bash is None or sys.platform == "win32":
        pytest.skip("needs a POSIX bash")
    info = _info(Path(bash))
    monkeypatch.setattr(agent_shell, "agent_bash", lambda: info)
    return info


@pytest.fixture
def real_bash(scripts, monkeypatch) -> GitInfo:
    info = _info(_SYSTEM_GIT / "bin" / "bash.exe")
    monkeypatch.setattr(agent_shell, "agent_bash", lambda: info)
    return info


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #
def test_agent_bash_is_the_recorded_git(monkeypatch, caplog):
    info = _info(Path(r"C:\Git\bin\bash.exe"))
    monkeypatch.setattr("EvoScientist.setup.git.recorded_git", lambda: info)
    with caplog.at_level("INFO", logger="EvoScientist.agent_shell"):
        assert _REAL_AGENT_BASH.__wrapped__() == info
    assert "Agent shell: " in caplog.text


def test_agent_bash_is_none_without_a_record(monkeypatch):
    monkeypatch.setattr("EvoScientist.setup.git.recorded_git", lambda: None)
    assert _REAL_AGENT_BASH.__wrapped__() is None


# --------------------------------------------------------------------------- #
# prepare()
# --------------------------------------------------------------------------- #
def test_without_bash_the_command_goes_to_the_default_shell(scripts):
    env = {"A": "1"}
    launch = agent_shell.prepare("echo hi", env)
    assert launch.args == "echo hi"
    assert launch.shell is True
    assert launch.env is env
    assert launch.script is None
    assert launch.creationflags == 0
    assert launch.text_options == {}
    launch.cleanup()
    assert not scripts.exists()


def test_with_bash_the_command_goes_into_a_script_file(fake_bash, scripts):
    command = "printf '%s\\n' 'a\\\\b' && echo café 中文\n"
    launch = agent_shell.prepare(command, {"A": "1"})
    assert launch.shell is False
    assert launch.args[0] == str(fake_bash.bash)
    script = Path(launch.args[1])
    assert launch.args[1] == launch.script.as_posix()
    assert script.parent == scripts
    assert script.read_bytes() == command.encode("utf-8")
    assert launch.text_options == {"encoding": "utf-8", "errors": "replace"}
    launch.cleanup()
    assert not script.exists()
    launch.cleanup()  # idempotent


def test_bash_env_turns_off_path_conversion_and_asks_python_for_utf8(fake_bash):
    launch = agent_shell.prepare("true", {"A": "1"})
    assert launch.env == {
        "A": "1",
        "MSYS_NO_PATHCONV": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    launch.cleanup()


def test_bash_env_keeps_the_users_python_encoding(fake_bash):
    launch = agent_shell.prepare("true", {"pythonioencoding": "cp1252"})
    assert launch.env["pythonioencoding"] == "cp1252"
    assert "PYTHONIOENCODING" not in launch.env
    launch.cleanup()


def test_bash_env_inherits_ours_when_none_is_given(fake_bash, monkeypatch):
    monkeypatch.setenv("EVOSCI_TEST_MARKER", "x")
    launch = agent_shell.prepare("true", None)
    assert launch.env["EVOSCI_TEST_MARKER"] == "x"
    assert launch.env["MSYS_NO_PATHCONV"] == "1"
    launch.cleanup()


def test_bash_gets_no_console_window(fake_bash, monkeypatch):
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    launch = agent_shell.prepare("true", None)
    assert launch.creationflags == 0x08000000
    launch.cleanup()


def test_sweep_removes_only_stale_scripts(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_shell, "_script_dir", lambda: tmp_path)
    old = tmp_path / "cmd-old.sh"
    fresh = tmp_path / "cmd-fresh.sh"
    other = tmp_path / "keep.txt"
    for f in (old, fresh, other):
        f.write_text("x")
    long_ago = time.time() - agent_shell._STALE_SCRIPT_SECONDS - 60
    os.utime(old, (long_ago, long_ago))
    os.utime(other, (long_ago, long_ago))
    agent_shell._sweep_stale_scripts.__wrapped__()
    assert not old.exists()
    assert fresh.exists()
    assert other.exists()


def test_sweep_without_a_directory_is_quiet(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_shell, "_script_dir", lambda: tmp_path / "missing")
    agent_shell._sweep_stale_scripts.__wrapped__()


# --------------------------------------------------------------------------- #
# The two spawn sites, with a bash (POSIX stand-in)
# --------------------------------------------------------------------------- #
def test_execute_runs_the_script_and_deletes_it(fake_bash, scripts, tmp_path):
    backend = CustomSandboxBackend(root_dir=str(tmp_path / "ws"))
    resp = backend.execute(
        "printf '%s\\n' 'a\\\\b'; echo \"$MSYS_NO_PATHCONV\"; echo café 中文"
    )
    assert resp.exit_code == 0, resp.output
    assert resp.output.splitlines() == ["a\\\\b", "1", "café 中文"]
    assert list(scripts.iterdir()) == []


def test_execute_deletes_the_script_after_a_timeout(fake_bash, scripts, tmp_path):
    backend = CustomSandboxBackend(root_dir=str(tmp_path / "ws"), timeout=1)
    resp = backend.execute("sleep 10")
    assert resp.exit_code == 124
    assert list(scripts.iterdir()) == []


def test_background_runs_the_script_and_deletes_it_on_exit(
    fake_bash, scripts, tmp_path
):
    bg._PROCESSES.clear()
    pid = bg.launch('echo "$MSYS_NO_PATHCONV"', str(tmp_path))
    proc = bg._PROCESSES[pid]
    proc.popen.wait(timeout=10)
    deadline = time.time() + 5
    while any(scripts.iterdir()) and time.time() < deadline:
        time.sleep(0.05)
    assert proc.log_path.read_text().strip() == "1"
    assert list(scripts.iterdir()) == []
    bg._PROCESSES.clear()


# --------------------------------------------------------------------------- #
# Real Git for Windows bash
# --------------------------------------------------------------------------- #
def _py() -> str:
    return Path(sys.executable).as_posix()


def _marked(tag: str) -> list[psutil.Process]:
    found = []
    for p in psutil.process_iter(["cmdline"]):
        try:
            if any(tag in a for a in (p.info["cmdline"] or [])):
                found.append(p)
        except (psutil.Error, TypeError):
            pass
    return found


@windows_bash
def test_windows_execute_runs_in_bash(real_bash, tmp_path):
    backend = CustomSandboxBackend(root_dir=str(tmp_path / "ws"))
    resp = backend.execute("echo $((1+2)); printf '%s\\n' 'a\\\\b'")
    assert resp.exit_code == 0, resp.output
    assert resp.output.splitlines() == ["3", "a\\\\b"]


@windows_bash
def test_windows_slash_arguments_reach_native_programs_unchanged(real_bash, tmp_path):
    backend = CustomSandboxBackend(root_dir=str(tmp_path / "ws"))
    resp = backend.execute(f'"{_py()}" -c "import sys; print(sys.argv[1])" "/hi"')
    assert resp.output.strip() == "/hi", resp.output


@windows_bash
def test_windows_native_python_output_is_decoded(real_bash, tmp_path):
    backend = CustomSandboxBackend(root_dir=str(tmp_path / "ws"))
    resp = backend.execute(f'"{_py()}" -c "print(\'café\')"; echo café')
    assert resp.output.splitlines() == ["café", "café"], resp.output


@windows_bash
def test_windows_manual_background_recipe_returns_at_once(real_bash, tmp_path):
    tag = "evosci-c07-bg"
    backend = CustomSandboxBackend(root_dir=str(tmp_path / "ws"))
    started = time.monotonic()
    resp = backend.execute(
        f'"{_py()}" -c "import time; time.sleep(30)" {tag} > out.log 2>&1 &\n'
        'echo "PID: $!"'
    )
    try:
        assert time.monotonic() - started < 10
        pid = resp.output.split("PID:")[-1].strip()
        assert backend.execute(f"ps -p {pid}").exit_code == 0
        backend.execute(f"kill {pid}")
        deadline = time.time() + 10
        while _marked(tag) and time.time() < deadline:
            time.sleep(0.2)
        assert _marked(tag) == []
    finally:
        for p in _marked(tag):
            p.kill()


@windows_bash
def test_windows_timeout_stops_a_python_child(real_bash, tmp_path):
    tag = "evosci-c07-timeout"
    backend = CustomSandboxBackend(root_dir=str(tmp_path / "ws"), timeout=3)
    try:
        resp = backend.execute(f'"{_py()}" -c "import time; time.sleep(60)" {tag}')
        assert resp.exit_code == 124
        deadline = time.time() + 10
        while _marked(tag) and time.time() < deadline:
            time.sleep(0.2)
        assert _marked(tag) == []
    finally:
        for p in _marked(tag):
            p.kill()


@windows_bash
def test_windows_background_job_runs_in_bash_and_stops(real_bash, tmp_path):
    tag = "evosci-c07-job"
    bg._PROCESSES.clear()
    pid = bg.launch(
        f'echo "$BASH_VERSION"; "{_py()}" -c "import time; time.sleep(60)" {tag}',
        str(tmp_path),
    )
    try:
        deadline = time.time() + 10
        while not _marked(tag) and time.time() < deadline:
            time.sleep(0.2)
        assert _marked(tag)
        bg.stop(pid)
        deadline = time.time() + 10
        while _marked(tag) and time.time() < deadline:
            time.sleep(0.2)
        assert _marked(tag) == []
        assert bg._PROCESSES[pid].log_path.read_text().strip()
    finally:
        for p in _marked(tag):
            p.kill()
        bg._PROCESSES.clear()
