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
from EvoScientist.setup import research_env
from EvoScientist.setup.git import GitInfo

# Saved at import, before the autouse fixture in conftest pins them.
_REAL_AGENT_BASH = agent_shell.agent_bash
_REAL_BASH_MISSING = agent_shell._bash_missing
_REAL_LOG_SETUP_HINT = agent_shell.log_setup_hint

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
# Setup hint and server drift
# --------------------------------------------------------------------------- #
def test_bash_is_missing_only_on_windows_without_a_record(monkeypatch):
    monkeypatch.setattr(agent_shell, "_bash_missing", _REAL_BASH_MISSING)
    monkeypatch.setattr(agent_shell.sys, "platform", "win32")
    assert agent_shell._bash_missing() is True
    monkeypatch.setattr(agent_shell, "agent_bash", lambda: object())
    assert agent_shell._bash_missing() is False
    monkeypatch.setattr(agent_shell, "agent_bash", lambda: None)
    monkeypatch.setattr(agent_shell.sys, "platform", "linux")
    assert agent_shell._bash_missing() is False


@pytest.mark.parametrize(
    ("python", "bash_missing", "expected"),
    [
        ("/usr/bin/python", False, None),
        (None, False, research_env.MISSING_PYTHON_HINT),
        ("/usr/bin/python", True, agent_shell.MISSING_BASH_HINT),
        (None, True, "combined"),
    ],
)
def test_setup_hint_names_what_is_missing(monkeypatch, python, bash_missing, expected):
    monkeypatch.setattr(research_env, "agent_python", lambda: python)
    monkeypatch.setattr(agent_shell, "_bash_missing", lambda: bash_missing)
    hint = agent_shell.setup_hint()
    if expected == "combined":
        assert hint.count("Run `EvoSci setup`") == 1
        assert "no usable `python`" in hint
        assert "cmd.exe" in hint
    else:
        assert hint == expected


@pytest.mark.parametrize(
    ("sidecar", "own_missing", "shown"),
    [
        # The reused server's agents run in Git Bash: no hint, even if we lack it.
        ({agent_shell.SIDECAR_KEY: r"C:\Git\bin\bash.exe"}, True, False),
        # They run in cmd.exe: hint when we lack bash too; otherwise the drift
        # warning names the fix.
        ({agent_shell.SIDECAR_KEY: None}, True, True),
        ({agent_shell.SIDECAR_KEY: None}, False, False),
        # No record (older server) or no reuse: our own state.
        ({"workspace": "/w"}, True, True),
        (None, True, True),
        (None, False, False),
    ],
)
def test_server_setup_hint_for_bash(monkeypatch, sidecar, own_missing, shown):
    monkeypatch.setattr(research_env, "agent_python", lambda: "/usr/bin/python")
    monkeypatch.setattr(agent_shell, "_bash_missing", lambda: own_missing)
    hint = agent_shell.server_setup_hint(sidecar)
    assert (hint == agent_shell.MISSING_BASH_HINT) is shown
    assert hint is None or shown


@pytest.mark.parametrize(
    ("recorded", "current", "warns"),
    [
        (r"C:\Git\bin\bash.exe", r"C:\Git\bin\bash.exe", False),
        (None, None, False),
        (None, r"C:\Git\bin\bash.exe", True),
        (r"C:\Git\bin\bash.exe", None, True),
        # Windows: the same file spelled with another case.
        (r"C:\Git\bin\bash.exe", r"c:\git\BIN\bash.exe", False),
    ],
)
def test_shell_drift_message_for_bash(monkeypatch, recorded, current, warns):
    from types import SimpleNamespace

    info = None if current is None else SimpleNamespace(bash=current)
    monkeypatch.setattr(agent_shell, "agent_bash", lambda: info)
    monkeypatch.setattr(agent_shell.os.path, "normcase", str.lower)
    message = agent_shell.shell_drift_message({agent_shell.SIDECAR_KEY: recorded})
    assert (message is not None) is warns


def test_shell_drift_message_joins_python_and_bash(monkeypatch):
    monkeypatch.setattr(research_env, "agent_python", lambda: "/env/bin/python")
    sidecar = {
        research_env.SIDECAR_KEY: "/conda/bin/python",
        agent_shell.SIDECAR_KEY: "C:/b",
    }
    message = agent_shell.shell_drift_message(sidecar)
    assert "/conda/bin/python" in message
    assert "C:/b" in message
    assert agent_shell.shell_drift_message({"workspace": "/w"}) is None


def test_setup_hint_is_logged_once(monkeypatch, caplog):
    monkeypatch.setattr(agent_shell, "setup_hint", lambda: "Run `EvoSci setup`")
    log_once = _REAL_LOG_SETUP_HINT.__wrapped__
    with caplog.at_level("WARNING", logger="EvoScientist.agent_shell"):
        log_once()
    assert "Run `EvoSci setup`" in caplog.text


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


def test_bash_env_asks_python_for_utf8_and_keeps_path_conversion(fake_bash):
    launch = agent_shell.prepare("true", {"A": "1"})
    # No MSYS_NO_PATHCONV: $(pwd), ~ and $HOME must reach Windows programs
    # converted to C:/...
    assert launch.env == {"A": "1", "PYTHONIOENCODING": "utf-8"}
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
    assert launch.env["PYTHONIOENCODING"] == "utf-8"
    launch.cleanup()


@pytest.mark.parametrize(
    ("value", "stray"),
    [
        ("/hi", "/hi"),
        ("/api/v1", "/api/v1"),
        ("--root=/api", "/api"),
        ("/", "/"),
        # Drive paths ($(pwd), ~, $HOME expand to these) and Git Bash's own tree.
        ("/c/Users/me/x.py", None),
        ("/d", None),
        ("/usr/bin/env", None),
        ("/tmp/x", None),
        ("/ucrt64/bin/git", None),
        # Escaped, not a path, or a URL.
        ("//hi", None),
        ("hi", None),
        ("a/b", None),
        ("https://example.org/a", None),
        ("--url=https://example.org/a", None),
    ],
)
def test_converted_by_mistake(value, stray):
    assert agent_shell._converted_by_mistake(value) == stray


def test_path_conversion_note_names_each_stray_argument(fake_bash):
    note = agent_shell.path_conversion_note(
        'python x.py "/hi" --root=/api "$(pwd)/o" /c/Users/me ~/a //lit "/hi"'
    )
    root = fake_bash.bash.parent.parent.as_posix()
    assert note.startswith("Note: Git Bash passes `/hi`, `/api` to Windows programs")
    assert f"`/hi` as `{root}/hi`" in note
    assert "write `//hi` or start the command with `MSYS_NO_PATHCONV=1`" in note


def test_path_conversion_note_only_with_git_bash():
    # conftest pins "no bash": cmd.exe on Windows, /bin/sh elsewhere.
    assert agent_shell.path_conversion_note('python x.py "/hi"') is None


def test_bash_gets_no_console_window(fake_bash, monkeypatch):
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    launch = agent_shell.prepare("true", None)
    assert launch.creationflags == 0x08000000
    launch.cleanup()


# --------------------------------------------------------------------------- #
# The agent's git settings for PortableGit
# --------------------------------------------------------------------------- #
_SHIPPED_GITCONFIG = (
    "[core]\n\tautocrlf = true\n[credential]\n\thelper = helper-selector\n"
)


@pytest.fixture
def portablegit(scripts, tmp_path, monkeypatch) -> GitInfo:
    """A recorded PortableGit (bash is /bin/bash) under a path that needs quoting."""
    from EvoScientist import paths

    bash = shutil.which("bash")
    if bash is None or sys.platform == "win32":
        pytest.skip("needs a POSIX bash")
    data = tmp_path / "Jan #1; ąę" / ".evoscientist"
    root = data / "tools" / "git-2.56.0.windows.1"
    (root / "etc").mkdir(parents=True)
    (root / "etc" / "gitconfig").write_text(_SHIPPED_GITCONFIG)
    monkeypatch.setattr(paths, "DATA_DIR", data)
    agent_shell._agent_gitconfig.cache_clear()
    info = GitInfo(
        "portablegit", "2.56.0.windows.1", root / "cmd" / "git.exe", Path(bash)
    )
    monkeypatch.setattr(agent_shell, "agent_bash", lambda: info)
    yield info
    agent_shell._agent_gitconfig.cache_clear()


def _git_config(env: dict[str, str], *args: str) -> str:
    result = subprocess.run(
        ["git", "config", *args], env=env, capture_output=True, text=True
    )
    return result.stdout


def test_portablegit_gets_the_agent_gitconfig(portablegit):
    launch = agent_shell.prepare("true", {"A": "1"})
    launch.cleanup()
    path = Path(launch.env["GIT_CONFIG_SYSTEM"])
    assert path.parent == portablegit.git.parent.parent.parent.resolve()
    assert path.name == "git-agent.gitconfig"
    shipped = (portablegit.git.parent.parent / "etc" / "gitconfig").as_posix()
    assert f'path = "{shipped}"' in path.read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_agent_gitconfig_overrides_the_shipped_settings(portablegit, tmp_path):
    launch = agent_shell.prepare("true", {"GIT_CONFIG_GLOBAL": os.devnull})
    launch.cleanup()
    env = {**os.environ, **launch.env}
    assert _git_config(env, "--get", "core.autocrlf").strip() == "input"
    # The shipped file is included; the empty entry then resets the list, so
    # git asks only `manager`.
    assert _git_config(env, "--get-all", "credential.helper").splitlines() == [
        "helper-selector",
        "",
        "manager",
    ]
    # The user's own setting wins over a system-level file.
    user = tmp_path / "user.gitconfig"
    user.write_text("[core]\n\tautocrlf = false\n")
    env["GIT_CONFIG_GLOBAL"] = str(user)
    assert _git_config(env, "--get", "core.autocrlf").strip() == "false"


def test_users_own_system_config_is_kept(portablegit):
    for name in ("GIT_CONFIG_SYSTEM", "GIT_CONFIG_NOSYSTEM"):
        launch = agent_shell.prepare("true", {name: "x"})
        launch.cleanup()
        assert launch.env[name] == "x"
        assert set(launch.env) == {name, "PYTHONIOENCODING"}


def test_a_system_git_keeps_the_users_settings(fake_bash):
    launch = agent_shell.prepare("true", {})
    launch.cleanup()
    assert "GIT_CONFIG_SYSTEM" not in launch.env


def test_unwritable_agent_gitconfig_is_skipped_with_a_warning(
    portablegit, monkeypatch, tmp_path, caplog
):
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    monkeypatch.setattr(
        "EvoScientist.setup._install.tools_dir", lambda: blocker / "tools"
    )
    with caplog.at_level("WARNING", logger="EvoScientist.agent_shell"):
        launch = agent_shell.prepare("true", {})
    launch.cleanup()
    assert "GIT_CONFIG_SYSTEM" not in launch.env
    assert "Could not write" in caplog.text


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
        "printf '%s\\n' 'a\\\\b'; echo \"$PYTHONIOENCODING\"; echo café 中文"
    )
    assert resp.exit_code == 0, resp.output
    assert resp.output.splitlines() == ["a\\\\b", "utf-8", "café 中文"]
    assert list(scripts.iterdir()) == []


def test_execute_output_carries_the_path_conversion_note(fake_bash, tmp_path):
    backend = CustomSandboxBackend(root_dir=str(tmp_path / "ws"))
    resp = backend.execute('echo "/api/v1"')
    assert resp.output.startswith("/api/v1\n")
    assert "Note: Git Bash passes `/api/v1`" in resp.output
    assert "Note:" not in backend.execute("echo ok").output


def test_execute_deletes_the_script_after_a_timeout(fake_bash, scripts, tmp_path):
    backend = CustomSandboxBackend(root_dir=str(tmp_path / "ws"), timeout=1)
    resp = backend.execute("sleep 10")
    assert resp.exit_code == 124
    assert list(scripts.iterdir()) == []


def test_background_runs_the_script_and_deletes_it_on_exit(
    fake_bash, scripts, tmp_path
):
    bg._PROCESSES.clear()
    pid = bg.launch('echo "$PYTHONIOENCODING"', str(tmp_path))
    proc = bg._PROCESSES[pid]
    proc.popen.wait(timeout=10)
    deadline = time.time() + 5
    while any(scripts.iterdir()) and time.time() < deadline:
        time.sleep(0.05)
    assert proc.log_path.read_text().strip() == "utf-8"
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
def test_windows_bash_paths_reach_native_programs_converted(real_bash, tmp_path):
    """$(pwd), ~ and $HOME expand to /c/... in Git Bash; python must get C:/..."""
    backend = CustomSandboxBackend(root_dir=str(tmp_path / "ws"))
    resp = backend.execute(
        f'"{_py()}" -c "import os, sys; print([os.path.isdir(a) for a in sys.argv[1:]])" '
        '"$(pwd)" ~ "$HOME"'
    )
    assert resp.output.strip() == "[True, True, True]", resp.output


@windows_bash
def test_windows_literal_slash_argument_gets_a_note(real_bash, tmp_path):
    backend = CustomSandboxBackend(root_dir=str(tmp_path / "ws"))
    resp = backend.execute(f'"{_py()}" -c "import sys; print(sys.argv[1])" "/hi"')
    converted = (_SYSTEM_GIT / "hi").as_posix()
    assert resp.output.splitlines()[0].lower() == converted.lower(), resp.output
    assert "Note: Git Bash passes `/hi`" in resp.output
    resp = backend.execute(f'"{_py()}" -c "import sys; print(sys.argv[1])" "//hi"')
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
