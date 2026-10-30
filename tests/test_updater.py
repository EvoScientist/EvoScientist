"""Tests for EvoScientist.updater (``EvoSci update``). The network and
subprocesses are mocked."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
from pathlib import Path
from typing import ClassVar

import pytest

from EvoScientist import updater, uv_tool
from EvoScientist.setup.protocol import ERROR_CODES, JsonEmitter

RECEIPT = """\
[tool]
requirements = [
    { name = "evoscientist", extras = ["oauth"], specifier = "==0.3.6" },
    { name = "six", specifier = ">=1.10,<2" },
    { name = "idna", marker = "python_full_version >= '3.8'", specifier = ">=3" },
]
python = "3.12"
"""
VERSIONS = ["0.3.5", "0.3.6", "0.3.10", "0.4.0rc1", "0.3.7", "0.3.8"]
NEW_CODES = {
    "unknown_version",
    "no_constraints",
    "setup_failed",
    "manual_update",
    "server_running",
    "environment_busy",
}


class FakeConfig:
    mirror = "default"


@pytest.fixture
def uv_install(tmp_path, monkeypatch):
    """A uv tool install of 0.3.7 with the receipt above, an index listing
    VERSIONS, and constraints that download."""
    env = tmp_path / "tools" / "evoscientist"
    env.mkdir(parents=True)
    (env / "uv-receipt.toml").write_text(RECEIPT, encoding="utf-8")
    monkeypatch.setattr(sys, "prefix", str(env))
    # The macOS / Linux path (uv in-process); the Windows handover has its own
    # tests below.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(updater, "installed_version", lambda: "0.3.7")
    monkeypatch.setattr(uv_tool, "find_uv", lambda: "/opt/uv")
    monkeypatch.setattr("EvoScientist.config.load_config", lambda: FakeConfig())
    monkeypatch.setattr(
        "EvoScientist.update_check.fetch_versions", lambda index, timeout: VERSIONS
    )
    downloads: list[str] = []

    def fake_download(target, mirror, dest):
        downloads.append(updater.constraints_url(target, mirror))
        dest.write_text("six==1.17.0\n")

    monkeypatch.setattr(updater, "download_constraints", fake_download)
    monkeypatch.setattr(updater, "check_server", lambda opts, emit, confirm: None)
    monkeypatch.setattr(updater, "tool_bin_dir", lambda uv: tmp_path / "bin")
    # Nothing else runs from the environment (tests below set blockers).
    monkeypatch.setattr(updater, "running_from_env", lambda: [])
    monkeypatch.setattr(updater, "_server_tree", set)
    # The Windows handover creates the data dir's tools folder (its lock).
    monkeypatch.setattr("EvoScientist.paths.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("EvoScientist.update_helper.has_visible_desktop", lambda: True)
    return {"env": env, "downloads": downloads, "tmp": tmp_path}


class Runs:
    """Records subprocess.run calls; uv and setup succeed unless told."""

    def __init__(self, uv_rc=0, setup_rc=0):
        self.calls: list[dict] = []
        self.uv_rc = uv_rc
        self.setup_rc = setup_rc

    def __call__(self, cmd, **kwargs):
        self.calls.append({"cmd": list(cmd), **kwargs})
        rc = self.uv_rc if cmd[0] == "/opt/uv" else self.setup_rc
        return subprocess.CompletedProcess(cmd, rc)


class FakePopen:
    """``EvoSci setup --json`` writing *lines*, then exiting with *rc*."""

    calls: ClassVar[list[list[str]]] = []

    def __init__(self, lines, rc=0):
        self._lines, self._rc = lines, rc

    def __call__(self, cmd, **kwargs):
        FakePopen.calls.append(list(cmd))
        self.stdout = io.StringIO("".join(line + "\n" for line in self._lines))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def wait(self):
        return self._rc


def _run(opts, confirm=None):
    out = io.StringIO()
    emit = JsonEmitter(out)
    code = updater.run_update(
        opts, emit, confirm=confirm, emit_line=lambda line: out.write(line + "\n")
    )
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    return code, lines, out.getvalue()


def _terminal(lines):
    terminal = [
        e for e in lines if e["stage"] == "update" and e["status"] in ("done", "error")
    ]
    assert len(terminal) == 1, lines
    assert lines[-1] is terminal[0]
    return terminal[0]


# --------------------------------------------------------------------------- #
# Codes and help
# --------------------------------------------------------------------------- #
def test_every_new_code_is_registered():
    assert NEW_CODES <= ERROR_CODES


def test_unknown_code_is_rejected():
    with pytest.raises(ValueError, match="unknown setup error code"):
        updater.UpdateError("not_a_code", "x")


def test_version_help_describes_rollback():
    from typer.testing import CliRunner

    from EvoScientist.cli import app

    result = CliRunner().invoke(app, ["update", "--help"], terminal_width=200)
    assert result.exit_code == 0
    # CI forces colour: drop the styling codes Rich puts between the words.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    text = " ".join(plain.split())
    assert "config.yaml.bak-<current version>" in text
    assert "rolls back" in text


# --------------------------------------------------------------------------- #
# Version choice
# --------------------------------------------------------------------------- #
def test_newest_release_skips_prereleases():
    assert updater.choose_target(None, VERSIONS) == "0.3.10"


def test_requested_prerelease_is_allowed():
    assert updater.choose_target("0.4.0rc1", VERSIONS) == "0.4.0rc1"


def test_requested_version_must_be_listed():
    with pytest.raises(updater.UpdateError) as exc:
        updater.choose_target("0.9.0", VERSIONS)
    assert exc.value.code == "unknown_version"


@pytest.mark.parametrize("requested", ["0.3.5", "v0.3.5", "0.3.5rc1", "0.2.10"])
def test_releases_without_constraints_are_refused(requested):
    with pytest.raises(updater.UpdateError) as exc:
        updater.choose_target(requested, [*VERSIONS, "0.3.5rc1", "0.2.10"])
    assert exc.value.code == "no_constraints"
    assert "0.3.5 is the last release without constraints.txt" in exc.value.message


def test_older_release_with_constraints_is_accepted():
    assert updater.choose_target("0.3.6", VERSIONS) == "0.3.6"


def test_constraints_url_by_mirror():
    assert updater.constraints_url("0.3.7", "default") == (
        "https://raw.githubusercontent.com/EvoScientist/EvoScientist/v0.3.7/constraints.txt"
    )
    assert updater.constraints_url("0.3.7", "cn") == (
        "https://cdn.jsdelivr.net/gh/EvoScientist/EvoScientist@v0.3.7/constraints.txt"
    )


def test_missing_constraints_file_is_no_constraints(tmp_path, monkeypatch):
    def fake_urlopen(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(updater.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(updater.UpdateError) as exc:
        updater.download_constraints("0.3.7", "default", tmp_path / "c.txt")
    assert exc.value.code == "no_constraints"


def test_constraints_network_error_is_download_failed(tmp_path, monkeypatch):
    def fake_urlopen(url, timeout):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(updater.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(updater.UpdateError) as exc:
        updater.download_constraints("0.3.7", "default", tmp_path / "c.txt")
    assert exc.value.code == "download_failed"


# --------------------------------------------------------------------------- #
# The uv command
# --------------------------------------------------------------------------- #
def test_uv_command_keeps_python_extras_and_with(tmp_path):
    receipt = uv_tool.read_receipt(_write_receipt(tmp_path))
    cmd = updater.uv_install_command("/opt/uv", receipt, "0.3.10", tmp_path / "c.txt")
    assert cmd[:4] == ["/opt/uv", "tool", "install", "--force"]
    assert cmd[cmd.index("--python") + 1] == "3.12"
    assert "--managed-python" in cmd
    # By name: uv splits a -c path at spaces, so it runs from the file's dir.
    assert cmd[cmd.index("-c") + 1] == "c.txt"
    assert "EvoScientist[oauth]==0.3.10" in cmd
    with_args = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--with"]
    assert with_args == ["six>=1.10,<2", "idna>=3 ; python_full_version >= '3.8'"]
    assert "upgrade" not in cmd


def test_uv_command_without_python_key_uses_running_version(tmp_path):
    env = _write_receipt(
        tmp_path, '[tool]\nrequirements = [{ name = "evoscientist" }]\n'
    )
    cmd = updater.uv_install_command(
        "/opt/uv", uv_tool.read_receipt(env), "0.3.10", tmp_path / "c.txt"
    )
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert cmd[cmd.index("--python") + 1] == py
    assert "--managed-python" in cmd


def _write_receipt(tmp_path, text=RECEIPT):
    env = tmp_path / "env"
    env.mkdir()
    (env / "uv-receipt.toml").write_text(text, encoding="utf-8")
    return env


def test_exclude_newer_hint_names_the_config(tmp_path, monkeypatch):
    monkeypatch.setattr(updater.sys, "platform", "linux")
    monkeypatch.delenv("UV_EXCLUDE_NEWER", raising=False)
    monkeypatch.delenv("UV_NO_CONFIG", raising=False)
    monkeypatch.delenv("UV_CONFIG_FILE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    (tmp_path / "uv").mkdir()
    (tmp_path / "uv" / "uv.toml").write_text('exclude-newer = "7 days"\n')
    assert str(tmp_path / "uv" / "uv.toml") in updater.exclude_newer_hint()
    monkeypatch.setenv("UV_NO_CONFIG", "1")
    assert updater.exclude_newer_hint() == ""


# --------------------------------------------------------------------------- #
# A full uv tool update
# --------------------------------------------------------------------------- #
def test_json_update_forwards_setup_lines(uv_install, monkeypatch):
    runs = Runs()
    monkeypatch.setattr(updater.subprocess, "run", runs)
    setup_lines = [
        json.dumps({"protocol": 1, "stage": "node", "status": "done", "message": "ok"}),
        json.dumps(
            {"protocol": 1, "stage": "webui", "status": "skipped", "message": "skip"}
        ),
    ]
    FakePopen.calls = []
    monkeypatch.setattr(updater.subprocess, "Popen", FakePopen(setup_lines))
    code, lines, raw = _run(
        updater.UpdateOptions(json=True, skip=["webui"]), confirm=None
    )
    assert code == 0
    assert raw.isascii()
    assert all(e["protocol"] == 1 for e in lines)
    assert [e["stage"] for e in lines if e["stage"] != "update"] == ["node", "webui"]
    assert {"stage": "webui", "status": "skipped"}.items() <= next(
        e for e in lines if e["stage"] == "webui"
    ).items()
    done = _terminal(lines)
    assert done["status"] == "done"
    assert done["detail"] == {"version": "0.3.10"}
    # uv output goes to stderr, never stdout.
    uv_call = runs.calls[0]
    # The constraints file by name, from its temp dir (removed afterwards).
    assert uv_call["cmd"][uv_call["cmd"].index("-c") + 1] == "constraints.txt"
    assert Path(uv_call["cwd"]).name.startswith("evosci-update-")
    assert not Path(uv_call["cwd"]).exists()
    assert uv_call["cmd"][0] == "/opt/uv"
    assert uv_call["stdout"] is sys.stderr
    # --skip reaches the new setup, by absolute path in uv's bin dir.
    assert FakePopen.calls == [
        [
            str(uv_install["tmp"] / "bin" / "EvoSci"),
            "setup",
            "--skip",
            "webui",
            "--json",
        ]
    ]
    assert uv_install["downloads"] == [updater.constraints_url("0.3.10", "default")]


def test_terminal_update_runs_setup_on_the_console(uv_install, monkeypatch):
    runs = Runs()
    monkeypatch.setattr(updater.subprocess, "run", runs)
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.8"))
    assert code == 0
    assert runs.calls[1]["cmd"] == [str(uv_install["tmp"] / "bin" / "EvoSci"), "setup"]
    assert _terminal(lines)["detail"] == {"version": "0.3.8"}


def test_cn_mirror_update_uses_mirrors(uv_install, monkeypatch):
    class CnConfig:
        mirror = "cn"

    indexes: list[str] = []
    monkeypatch.setattr("EvoScientist.config.load_config", lambda: CnConfig())
    monkeypatch.setattr(
        "EvoScientist.update_check.fetch_versions",
        lambda index, timeout: indexes.append(index) or VERSIONS,
    )
    runs = Runs()
    monkeypatch.setattr(updater.subprocess, "run", runs)
    code, _, _ = _run(updater.UpdateOptions(version="0.3.8"))
    assert code == 0
    assert indexes == ["https://pypi.tuna.tsinghua.edu.cn/simple"]
    assert uv_install["downloads"][0].startswith("https://cdn.jsdelivr.net/")
    assert runs.calls[0]["env"]["UV_DEFAULT_INDEX"] == indexes[0]


def test_already_newest_changes_nothing(uv_install, monkeypatch):
    monkeypatch.setattr(updater, "installed_version", lambda: "0.3.10")
    runs = Runs()
    monkeypatch.setattr(updater.subprocess, "run", runs)
    code, lines, _ = _run(updater.UpdateOptions(json=True))
    assert code == 0
    assert runs.calls == []
    assert _terminal(lines)["status"] == "done"


def test_check_only_prints_versions(uv_install, monkeypatch):
    runs = Runs()
    monkeypatch.setattr(updater.subprocess, "run", runs)
    code, lines, _ = _run(updater.UpdateOptions(check=True, json=True))
    assert code == 0
    assert runs.calls == []
    assert _terminal(lines)["detail"] == {"version": "0.3.7", "latest": "0.3.10"}
    assert uv_install["downloads"] == []


def test_rollback_backs_up_the_config(uv_install, monkeypatch):
    config = uv_install["tmp"] / "config.yaml"
    config.write_text("newer_key: 1\n")
    monkeypatch.setattr("EvoScientist.config.settings.get_config_path", lambda: config)
    monkeypatch.setattr(updater.subprocess, "run", Runs())
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.6"))
    assert code == 0
    backup = uv_install["tmp"] / "config.yaml.bak-0.3.7"
    assert backup.read_text() == "newer_key: 1\n"
    assert any(str(backup) in e.get("message", "") for e in lines)


def test_no_backup_for_a_newer_target(uv_install, monkeypatch):
    config = uv_install["tmp"] / "config.yaml"
    config.write_text("a: 1\n")
    monkeypatch.setattr("EvoScientist.config.settings.get_config_path", lambda: config)
    monkeypatch.setattr(updater.subprocess, "run", Runs())
    _run(updater.UpdateOptions(version="0.3.8"))
    assert not list(uv_install["tmp"].glob("config.yaml.bak-*"))


# --------------------------------------------------------------------------- #
# Failures, each with its code and one terminal line
# --------------------------------------------------------------------------- #
def test_lookup_without_network(uv_install, monkeypatch):
    def offline(index, timeout):
        raise OSError("Network is unreachable")

    monkeypatch.setattr("EvoScientist.update_check.fetch_versions", offline)
    code, lines, _ = _run(updater.UpdateOptions(json=True))
    assert code == 1
    assert _terminal(lines)["code"] == "download_failed"


def test_unknown_version(uv_install):
    code, lines, _ = _run(updater.UpdateOptions(version="9.9.9", json=True))
    assert code == 1
    assert _terminal(lines)["code"] == "unknown_version"


def test_target_without_constraints(uv_install, monkeypatch):
    def missing(target, mirror, dest):
        raise updater.UpdateError("no_constraints", "none")

    monkeypatch.setattr(updater, "download_constraints", missing)
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.8", json=True))
    assert code == 1
    assert _terminal(lines)["code"] == "no_constraints"


def test_uv_failure_is_install_failed(uv_install, monkeypatch):
    runs = Runs(uv_rc=1)
    monkeypatch.setattr(updater.subprocess, "run", runs)
    code, lines, _ = _run(updater.UpdateOptions(json=True))
    assert code == 1
    assert _terminal(lines)["code"] == "install_failed"
    assert len(runs.calls) == 1  # no setup after a failed install


def test_failed_setup_stage_is_setup_failed(uv_install, monkeypatch):
    monkeypatch.setattr(updater.subprocess, "run", Runs())
    failed = json.dumps(
        {"protocol": 1, "stage": "webui", "status": "error", "code": "download_failed"}
    )
    monkeypatch.setattr(updater.subprocess, "Popen", FakePopen([failed], rc=1))
    code, lines, _ = _run(updater.UpdateOptions(json=True))
    end = _terminal(lines)
    assert code == 1
    assert end["code"] == "setup_failed"
    assert end["detail"] == {"stage": "webui", "code": "download_failed"}
    # The forwarded stage error comes before the update's terminal line.
    assert lines[-2]["stage"] == "webui"


def test_unexpected_exception_still_ends_with_one_terminal_line(
    uv_install, monkeypatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(updater.subprocess, "run", boom)
    code, lines, _ = _run(updater.UpdateOptions(json=True))
    assert code == 1
    assert _terminal(lines)["code"] == "install_failed"


# --------------------------------------------------------------------------- #
# Installs this command does not change: manual_update, no subprocess
# --------------------------------------------------------------------------- #
@pytest.fixture
def no_subprocess(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"unexpected subprocess: {args!r}")

    monkeypatch.setattr(updater.subprocess, "run", refuse)
    monkeypatch.setattr(updater.subprocess, "Popen", refuse)


def _plain_env(tmp_path, monkeypatch, *, editable=False):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(updater, "installed_version", lambda: "0.3.7")
    monkeypatch.setattr("EvoScientist.config.load_config", lambda: FakeConfig())
    monkeypatch.setattr(
        "EvoScientist.update_check.fetch_versions", lambda index, timeout: VERSIONS
    )
    monkeypatch.setattr(
        "EvoScientist.ccproxy_manager._is_editable_install", lambda: editable
    )


def test_pip_install_prints_the_command(tmp_path, monkeypatch, no_subprocess):
    _plain_env(tmp_path, monkeypatch)
    code, lines, _ = _run(updater.UpdateOptions(json=True))
    end = _terminal(lines)
    assert code == 1
    assert end["code"] == "manual_update"
    assert 'pip install --upgrade "EvoScientist==0.3.10"' in end["message"]
    assert updater.constraints_url("0.3.10", "default") in end["message"]


def test_editable_install_says_git(tmp_path, monkeypatch, no_subprocess):
    _plain_env(tmp_path, monkeypatch, editable=True)
    _code, lines, _ = _run(updater.UpdateOptions(json=True))
    end = _terminal(lines)
    assert end["code"] == "manual_update"
    assert "git pull" in end["message"]


def test_docker_image_says_pull(tmp_path, monkeypatch, no_subprocess):
    _plain_env(tmp_path, monkeypatch)
    (tmp_path / ".evoscientist-managed").write_text("")
    _code, lines, _ = _run(updater.UpdateOptions(json=True))
    end = _terminal(lines)
    assert end["code"] == "manual_update"
    assert "docker pull ghcr.io/evoscientist/evoscientist:0.3.10" in end["message"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("directory", "/opt/src/evo"),
        ("git", "https://github.com/EvoScientist/EvoScientist?branch=main"),
        ("url", "https://example.org/evoscientist-0.3.7-py3-none-any.whl"),
    ],
)
def test_source_install_is_not_moved_to_pypi(
    tmp_path, monkeypatch, no_subprocess, key, value
):
    _plain_env(tmp_path, monkeypatch)
    (tmp_path / "uv-receipt.toml").write_text(
        f'[tool]\nrequirements = [{{ name = "evoscientist", {key} = "{value}" }}]\n'
    )
    _code, lines, _ = _run(updater.UpdateOptions(json=True))
    end = _terminal(lines)
    assert end["code"] == "manual_update"
    assert value in end["message"]


def test_unreadable_receipt(tmp_path, monkeypatch, no_subprocess):
    _plain_env(tmp_path, monkeypatch)
    (tmp_path / "uv-receipt.toml").write_text("not [ toml")
    _code, lines, _ = _run(updater.UpdateOptions(json=True))
    end = _terminal(lines)
    assert end["code"] == "manual_update"
    assert "install script" in end["message"]


def test_missing_uv(uv_install, monkeypatch, no_subprocess):
    monkeypatch.setattr(uv_tool, "find_uv", lambda: None)
    _code, lines, _ = _run(updater.UpdateOptions(json=True))
    end = _terminal(lines)
    assert end["code"] == "manual_update"
    assert "No uv was found" in end["message"]


# --------------------------------------------------------------------------- #
# A running server
# --------------------------------------------------------------------------- #
@pytest.fixture
def server(monkeypatch):
    """A server answering on 6174; *owned* when its sidecar pid serves it."""
    import EvoScientist.langgraph_dev.manager as manager

    state = {
        "owned": True,
        "running": True,
        "stopped": 0,
        "stop_fails": False,
        "jobs": ["train.py"],
        "busy": True,
    }
    monkeypatch.setattr(
        manager, "is_langgraph_dev_running", lambda **kw: state["running"]
    )
    monkeypatch.setattr(manager, "_read_workspace_sidecar", lambda: {"pid": 42})
    monkeypatch.setattr(manager, "_pid_serves_port", lambda pid, port: state["owned"])

    def stop():
        if state["stop_fails"] or not state["owned"]:
            return None
        state["stopped"] += 1
        state["running"] = False
        return 42

    monkeypatch.setattr(manager, "stop_recorded_server", stop)
    monkeypatch.setattr(
        "EvoScientist.deploy.busy.running_bg_process_names", lambda url: state["jobs"]
    )
    monkeypatch.setattr(
        "EvoScientist.deploy.busy.has_busy_thread", lambda url: state["busy"]
    )

    class Config:
        langgraph_dev_port = 6174
        langgraph_dev_host = "127.0.0.1"

    monkeypatch.setattr("EvoScientist.config.get_effective_config", lambda: Config())
    return state


def _emits():
    events: list[dict] = []
    return events.append


def test_json_without_yes_ends_with_server_running(server):
    emit = _emits()
    with pytest.raises(updater.UpdateError) as exc:
        updater.check_server(updater.UpdateOptions(json=True), emit, None)
    assert exc.value.code == "server_running"
    assert server["stopped"] == 0


def test_yes_stops_without_asking(server):
    asked: list[str] = []
    emit = _emits()
    updater.check_server(
        updater.UpdateOptions(yes=True), emit, lambda q: asked.append(q) or False
    )
    assert asked == []
    assert server["stopped"] == 1


def test_terminal_asks_and_names_busy_work(server):
    asked: list[str] = []
    emit = _emits()
    updater.check_server(
        updater.UpdateOptions(), emit, lambda q: asked.append(q) or True
    )
    assert server["stopped"] == 1
    assert "train.py" in asked[0]
    assert "a task is running" in asked[0]


def test_background_job_alone_is_not_called_a_task(server):
    server["busy"] = False
    asked: list[str] = []
    updater.check_server(
        updater.UpdateOptions(), _emits(), lambda q: asked.append(q) or True
    )
    assert "train.py" in asked[0]
    assert "a task is running" not in asked[0]


def test_owned_server_that_does_not_stop(server):
    server["stop_fails"] = True
    with pytest.raises(updater.UpdateError) as exc:
        updater.check_server(updater.UpdateOptions(yes=True), _emits(), None)
    assert exc.value.code == "server_running"
    assert "Could not stop" in exc.value.message
    assert "ownership record" not in exc.value.message


@pytest.mark.parametrize("stop_fails", [False, True])
def test_lasting_server_is_recorded_for_restart(server, monkeypatch, stop_fails):
    """The setup after the update starts it again, from the same project;
    a server that did not stop leaves no record."""
    import EvoScientist.langgraph_dev.manager as manager
    from EvoScientist.setup import server as server_stage

    monkeypatch.setattr(
        manager,
        "_read_workspace_sidecar",
        lambda: {"pid": 42, "workspace": "/proj", "lasting": True},
    )
    server["stop_fails"] = stop_fails
    recorded: list = []
    monkeypatch.setattr(server_stage, "remember_for_restart", recorded.append)
    monkeypatch.setattr(server_stage, "forget_restart", lambda: recorded.clear())

    if stop_fails:
        with pytest.raises(updater.UpdateError):
            updater.check_server(updater.UpdateOptions(yes=True), _emits(), None)
        assert recorded == []
    else:
        updater.check_server(updater.UpdateOptions(yes=True), _emits(), None)
        assert recorded == ["/proj"]


def test_has_busy_thread_asks_for_busy_threads_only(monkeypatch):
    from EvoScientist.deploy import busy

    payloads: list[dict] = []
    monkeypatch.setattr(
        busy,
        "_search_threads",
        lambda url, payload, timeout: payloads.append(payload) or [],
    )
    assert busy.has_busy_thread("http://x") is False
    assert payloads == [{"status": "busy", "limit": 1}]

    def broken(url, payload, timeout):
        raise OSError("down")

    monkeypatch.setattr(busy, "_search_threads", broken)
    assert busy.has_busy_thread("http://x") is False


def test_ctrl_c_at_the_question_cancels(uv_install, monkeypatch):
    import click

    def abort(opts, emit, confirm):
        raise click.exceptions.Abort()

    monkeypatch.setattr(updater, "check_server", abort)
    monkeypatch.setattr(updater.subprocess, "run", Runs())
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.8"))
    end = _terminal(lines)
    assert code == 1
    assert end["code"] == "server_running"
    assert end["message"] == "Update cancelled."


def test_answering_no_cancels(server):
    emit = _emits()
    with pytest.raises(updater.UpdateError) as exc:
        updater.check_server(updater.UpdateOptions(), emit, lambda q: False)
    assert exc.value.code == "server_running"
    assert server["stopped"] == 0


def test_unowned_server_is_never_stopped(server):
    server["owned"] = False
    emit = _emits()
    with pytest.raises(updater.UpdateError) as exc:
        updater.check_server(updater.UpdateOptions(json=True, yes=True), emit, None)
    assert exc.value.code == "server_running"
    assert server["stopped"] == 0
    assert "6174" in exc.value.message


def test_unowned_server_terminal_may_continue(server):
    server["owned"] = False
    emit = _emits()
    updater.check_server(updater.UpdateOptions(), emit, lambda q: True)
    assert server["stopped"] == 0


def test_stale_record_without_server_is_ignored(server):
    server["running"] = False
    emit = _emits()
    updater.check_server(updater.UpdateOptions(json=True), emit, None)
    assert server["stopped"] == 0


# --------------------------------------------------------------------------- #
# Windows: the helper takes over
# --------------------------------------------------------------------------- #
def test_windows_hands_over_to_the_helper(uv_install, monkeypatch):
    from EvoScientist import update_helper

    started: list[dict] = []
    monkeypatch.setattr(updater.sys, "platform", "win32")
    monkeypatch.setattr(
        update_helper,
        "start_helper",
        lambda tmp, config, json_mode, own_window: started.append(
            {"tmp": tmp, "config": config, "json": json_mode, "own": own_window}
        ),
    )
    runs = Runs()
    monkeypatch.setattr(updater.subprocess, "run", runs)
    code, lines, _ = _run(updater.UpdateOptions(json=True, skip=["webui"]))
    assert code == 0
    assert runs.calls == []  # uv runs in the helper, after EvoSci has exited
    # The helper writes the terminal line.
    assert all(e["status"] == "running" for e in lines)
    (call,) = started
    config = call["config"]
    assert call["json"] is True
    assert call["own"] is False
    assert call["tmp"].is_dir()  # the helper deletes it
    assert (call["tmp"] / "constraints.txt").read_text() == "six==1.17.0\n"
    assert config["env_dir"] == str(Path(sys.prefix).resolve())
    assert config["wait_pids"][0] == os.getpid()
    assert config["uv"] == "/opt/uv"
    assert config["uv_args"][:3] == ["tool", "install", "--force"]
    evosci = uv_install["tmp"] / "bin" / "EvoSci.exe"
    assert config["evosci"] == str(evosci)
    assert config["setup_command"] == f"& '{evosci}' setup"
    assert config["skip"] == ["webui"]
    assert (config["installed"], config["target"]) == ("0.3.7", "0.3.10")
    assert config["lock"] == str(uv_install["tmp"] / "data" / "tools" / "update.lock")
    assert (uv_install["tmp"] / "data" / "tools").is_dir()
    assert config["unattended"] is False
    assert "install_hint" in config


def test_windows_handover_passes_the_uv_environment(uv_install, monkeypatch):
    from EvoScientist import update_helper
    from EvoScientist.update_check import CN_SIMPLE

    class CnConfig:
        mirror = "cn"

    started: list[dict] = []
    monkeypatch.setattr(updater.sys, "platform", "win32")
    monkeypatch.setattr("EvoScientist.config.load_config", lambda: CnConfig())
    for name in ("UV_SYSTEM_CERTS", "UV_NATIVE_TLS", "UV_HTTP_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("UV_DEFAULT_INDEX", raising=False)
    monkeypatch.delenv("UV_PYTHON_INSTALL_MIRROR", raising=False)
    monkeypatch.setattr(
        update_helper, "start_helper", lambda tmp, config, **kw: started.append(config)
    )
    monkeypatch.setattr(updater.subprocess, "run", Runs())
    code, _, _ = _run(updater.UpdateOptions(json=True))
    assert code == 0
    assert started[0]["uv_env"] == {
        "UV_SYSTEM_CERTS": "1",
        "UV_HTTP_TIMEOUT": "300",
        "UV_DEFAULT_INDEX": CN_SIMPLE,
        "UV_PYTHON_INSTALL_MIRROR": uv_tool.CN_PYTHON_MIRROR,
    }


def test_terminal_handover_without_a_desktop_stays_on_this_console(
    uv_install, monkeypatch
):
    """In an SSH session nobody would see a new window: the helper writes to
    this console and does not wait."""
    from EvoScientist import update_helper

    started: list[dict] = []
    monkeypatch.setattr(updater.sys, "platform", "win32")
    monkeypatch.setattr(update_helper, "has_visible_desktop", lambda: False)
    monkeypatch.setattr(
        update_helper,
        "start_helper",
        lambda tmp, config, json_mode, own_window: started.append(
            {"config": config, "own": own_window}
        ),
    )
    _code, lines, _ = _run(updater.UpdateOptions())
    assert "continues here" in lines[-1]["message"]
    assert started[0]["own"] is False
    assert started[0]["config"]["unattended"] is True


def test_start_helper_json_keeps_the_callers_pipes(tmp_path, monkeypatch):
    from EvoScientist import update_helper

    calls: list[dict] = []
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x8000000, raising=False)
    monkeypatch.setattr(
        update_helper.subprocess,
        "Popen",
        lambda cmd, **kwargs: calls.append({"cmd": cmd, **kwargs}),
    )
    update_helper.start_helper(
        tmp_path, {"env_dir": "C:\\env"}, json_mode=True, own_window=False
    )
    (call,) = calls
    assert call["creationflags"] == 0x200 | 0x8000000
    assert call["stdin"] is sys.stdin
    assert call["stdout"] is sys.stdout
    # A script block, not -File (an AllSigned policy refuses the file); the
    # paths come through the environment.
    assert call["cmd"][1:3] == ["-NoProfile", "-Command"]
    assert call["cmd"][3].startswith(". ([scriptblock]::Create(")
    assert call["cmd"][3].endswith(" -Json")
    assert call["env"]["EVOSCI_UPDATE_HELPER"] == str(
        tmp_path / update_helper.HELPER_NAME
    )
    assert call["env"]["EVOSCI_UPDATE_CONFIG"] == str(tmp_path / "config.json")
    assert call["env"]["EVOSCI_UPDATE_TEMP"] == str(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config == {"env_dir": "C:\\env"}


def test_start_helper_terminal_opens_its_own_window(tmp_path, monkeypatch):
    """A prompt reads Ctrl+C as a key, so the helper gets a console of its own,
    where Ctrl+C reaches it."""
    from EvoScientist import update_helper

    calls: list[dict] = []
    monkeypatch.setattr(subprocess, "CREATE_NEW_CONSOLE", 0x10, raising=False)
    monkeypatch.setattr(
        update_helper.subprocess,
        "Popen",
        lambda cmd, **kwargs: calls.append({"cmd": cmd, **kwargs}),
    )
    update_helper.start_helper(tmp_path, {}, json_mode=False, own_window=True)
    (call,) = calls
    assert call["creationflags"] == 0x10
    assert "stdout" not in call
    assert not call["cmd"][3].endswith(" -Json")


def test_terminal_handover_says_new_window(uv_install, monkeypatch):
    from EvoScientist import update_helper

    monkeypatch.setattr(updater.sys, "platform", "win32")
    monkeypatch.setattr(update_helper, "start_helper", lambda *a, **kw: None)
    _code, lines, _ = _run(updater.UpdateOptions())
    assert "new window" in lines[-1]["message"]


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="needs pwsh")
def test_helper_script_parses(tmp_path):
    from EvoScientist import update_helper

    script = tmp_path / "helper.ps1"
    script.write_text(update_helper.HELPER, encoding="utf-8")
    check = (
        "$e = $null; [System.Management.Automation.Language.Parser]::ParseFile("
        f"'{script}', [ref]$null, [ref]$e) | Out-Null; $e.Count"
    )
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-Command", check], capture_output=True, text=True
    )
    assert result.stdout.strip() == "0", result.stdout + result.stderr


def test_tool_bin_dir_reads_utf8(monkeypatch):
    """uv prints UTF-8; the Windows locale code page would garble the path."""
    seen: list[dict] = []
    path = "C:\\Users\\Public\\Evo Upd \u0105\u0119\\bin"

    def fake_run(cmd, **kwargs):
        seen.append(kwargs)
        return subprocess.CompletedProcess(cmd, 0, path + "\n", "")

    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    assert updater.tool_bin_dir("uv") == Path(path)
    assert seen[0]["encoding"] == "utf-8"


def test_check_works_for_installs_it_does_not_update(
    tmp_path, monkeypatch, no_subprocess
):
    """--check only prints the versions, also for an editable install."""
    _plain_env(tmp_path, monkeypatch, editable=True)
    code, lines, _ = _run(updater.UpdateOptions(check=True, json=True))
    assert code == 0
    assert _terminal(lines)["detail"] == {"version": "0.3.7", "latest": "0.3.10"}


def test_truncated_constraints_download_is_download_failed(tmp_path, monkeypatch):
    import http.client

    def truncated(url, timeout):
        raise http.client.IncompleteRead(b"six==", 10)

    monkeypatch.setattr(updater.urllib.request, "urlopen", truncated)
    with pytest.raises(updater.UpdateError) as exc:
        updater.download_constraints("0.3.7", "default", tmp_path / "c.txt")
    assert exc.value.code == "download_failed"


def test_ctrl_c_while_uv_runs_still_ends_with_a_terminal_line(uv_install, monkeypatch):
    def interrupted(cmd, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(updater.subprocess, "run", interrupted)
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.8", json=True))
    end = _terminal(lines)
    assert code == 130
    assert end["code"] == "install_failed"
    assert "install script" in end["message"]


def test_newest_release_answer_for_installs_it_does_not_update(
    tmp_path, monkeypatch, no_subprocess
):
    """An editable install on the newest release is told so, not manual_update;
    with --version it still gets manual_update."""
    _plain_env(tmp_path, monkeypatch, editable=True)
    monkeypatch.setattr(updater, "installed_version", lambda: "0.3.10")
    code, lines, _ = _run(updater.UpdateOptions(json=True))
    end = _terminal(lines)
    assert code == 0
    assert end["status"] == "done"
    assert end["detail"] == {"version": "0.3.10"}
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.10", json=True))
    assert _terminal(lines)["code"] == "manual_update"


def test_exiting_pids_include_a_parent_in_the_environment(tmp_path, monkeypatch):
    import psutil

    env = tmp_path / "env"
    (env / "Scripts").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(env))

    class Parent:
        def __init__(self, exe):
            self.pid = 4242
            self._exe = exe

        def exe(self):
            return self._exe

    monkeypatch.setattr(
        psutil, "Process", lambda pid: Parent(str(env / "Scripts" / "python.exe"))
    )
    assert updater._exiting_pids() == [os.getpid(), 4242]
    # A parent outside the environment (a shell, the desktop app) is not waited for.
    monkeypatch.setattr(psutil, "Process", lambda pid: Parent(str(tmp_path / "node")))
    assert updater._exiting_pids() == [os.getpid()]


def test_receipt_entry_that_cannot_be_reinstalled(tmp_path, monkeypatch, no_subprocess):
    """A --with entry the update cannot write back ends with manual_update,
    not with install_failed after the lookup."""
    _plain_env(tmp_path, monkeypatch)
    (tmp_path / "uv-receipt.toml").write_text(
        "[tool]\nrequirements = [\n"
        '  { name = "evoscientist", specifier = "==0.3.7" },\n'
        '  { name = "wheelpkg", git = "https://example.org/r?path=dist/w.whl&rev=v1" },\n'
        "]\n"
    )
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.8", json=True))
    end = _terminal(lines)
    assert code == 1
    assert end["code"] == "manual_update"
    assert "install script" in end["message"]


# --------------------------------------------------------------------------- #
# The Windows helper itself, under pwsh with POSIX stand-ins: a copy of
# `sleep` started from the environment blocks it; shell scripts play uv and
# EvoSci. macOS kills a copied system binary, so those tests skip there.
# --------------------------------------------------------------------------- #
needs_pwsh = pytest.mark.skipif(
    shutil.which("pwsh") is None or sys.platform == "win32",
    reason="runs the helper under pwsh with POSIX stand-ins",
)
needs_blocker = pytest.mark.skipif(
    shutil.which("pwsh") is None
    or sys.platform in ("win32", "darwin")
    or not shutil.which("sleep"),
    reason="runs the helper under pwsh with a copy of sleep (macOS kills it)",
)


@contextlib.contextmanager
def _bounded(*procs, seconds=60):
    """Kills *procs* after *seconds*, so a stalled helper cannot block a read
    of its stdout, and fails the test when that happened; kills and reaps
    them on the way out."""
    import threading

    expired = threading.Event()

    def expire():
        expired.set()
        for p in procs:
            p.kill()

    timer = threading.Timer(seconds, expire)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        for p in procs:
            p.kill()
            p.wait()
    assert not expired.is_set(), f"still running after {seconds} s"


def _script(path: Path, body: str) -> Path:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def _start_blocker(folder: Path, seconds: int = 30) -> int:
    """A copy of sleep at *folder*/sleep, started through a shell that exits
    at once, so init reaps it: a zombie child of the test would still count.

    The copy keeps the name ``sleep``: on ubuntu-26.04 ``sleep`` is the uutils
    multi-call binary, which picks the program from its file name and exits
    under any other (``coreutils: unknown program 'blocker'``). A copy that is
    not running fails the test here, not later for another reason, or not at
    all."""
    import time

    import psutil

    folder.mkdir(parents=True, exist_ok=True)
    path, err = folder / "sleep", folder / "sleep.err"
    shutil.copy(shutil.which("sleep"), path)
    started = subprocess.run(
        ["sh", "-c", f"'{path}' {seconds} >/dev/null 2>'{err}' & echo $!"],
        capture_output=True,
        text=True,
        check=True,
    )
    pid = int(started.stdout.strip())
    time.sleep(0.3)
    try:
        running = psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        running = False
    if not running:
        pytest.fail(f"{path} did not keep running: {err.read_text().strip()}")
    return pid


def _helper(root: Path, *, json_mode: bool = True, **config) -> list[str]:
    """Writes the helper and its config under *root* (environment
    ``root/env``, uv and EvoSci stand-ins that succeed) and returns the
    command that runs it."""
    from EvoScientist import update_helper

    (root / "env" / "bin").mkdir(parents=True, exist_ok=True)
    work = root / "work"
    work.mkdir(exist_ok=True)
    if "uv" not in config:
        config["uv"] = str(_script(root / "uv", "exit 0"))
    if "evosci" not in config:
        config["evosci"] = str(
            _script(
                root / "EvoSci",
                'echo \'{"protocol": 1, "stage": "node", "status": "done"}\'',
            )
        )
    full = {
        "env_dir": str(root / "env"),
        "wait_pids": [],
        "uv_args": [],
        "uv_env": {},
        "setup_command": "EvoSci setup",
        "skip": [],
        "installed": "0.3.6",
        "target": "0.3.7",
        "install_command": "install-script",
        "install_hint": "",
        "lock": str(root / "update.lock"),
        "unattended": False,
        **config,
    }
    (root / "config.json").write_text(json.dumps(full))
    script = root / "helper.ps1"
    script.write_text(update_helper.HELPER, encoding="utf-8")
    cmd = ["pwsh", "-NoProfile", "-File", str(script)]
    cmd += ["-ConfigPath", str(root / "config.json"), "-TempDir", str(work)]
    return cmd + (["-Json"] if json_mode else [])


@needs_blocker
def test_start_blocker_fails_when_the_copy_does_not_keep_running(tmp_path, monkeypatch):
    """A blocker that exits at once, as ubuntu-26.04's multi-call coreutils
    does under another name, fails the test where it starts, with its error."""
    exits = _script(tmp_path / "exits", "echo 'unknown program' >&2; exit 1")
    monkeypatch.setattr(shutil, "which", lambda _name: str(exits))
    with pytest.raises(
        pytest.fail.Exception, match="did not keep running: unknown program"
    ):
        _start_blocker(tmp_path / "env" / "bin")


def _kill(*pids: int) -> None:
    for pid in pids:
        subprocess.run(["kill", str(pid)], check=False, capture_output=True)


@needs_pwsh
def test_helper_runs_through_the_start_command(tmp_path):
    """The script-block start command, with paths that hold a space and
    non-ASCII characters, runs the helper to its terminal line."""
    from EvoScientist import update_helper

    root = tmp_path / "Evo Upd ąę"
    cmd = _helper(root)
    env = {
        **os.environ,
        "EVOSCI_UPDATE_HELPER": cmd[3],
        "EVOSCI_UPDATE_CONFIG": cmd[5],
        "EVOSCI_UPDATE_TEMP": cmd[7],
    }
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-Command", update_helper._START + " -Json"],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]["status"] == "done", result.stdout + result.stderr
    assert not (root / "work").exists()  # the helper removes its temp dir


@needs_pwsh
def test_helper_unreadable_config_ends_with_a_terminal_line(tmp_path):
    cmd = _helper(tmp_path)
    (tmp_path / "config.json").write_text("{ not json")
    result = subprocess.run(
        cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60
    )
    (end,) = [json.loads(line) for line in result.stdout.splitlines()]
    assert end["status"] == "error"
    assert end["code"] == "install_failed"
    assert "Nothing was changed" in end["message"]
    assert not (tmp_path / "work").exists()


@needs_pwsh
def test_helper_keeps_setup_output_that_is_not_an_event_off_stdout(tmp_path):
    cmd = _helper(
        tmp_path,
        evosci=str(
            _script(
                tmp_path / "EvoSci",
                "echo 'stray print'\n"
                'echo \'{"protocol": 1, "stage": "node", "status": "done"}\'',
            )
        ),
    )
    result = subprocess.run(
        cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60
    )
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]["status"] == "done"
    assert any(e.get("stage") == "node" for e in events)
    assert "stray print" in result.stderr


@needs_pwsh
def test_helper_install_failure_names_exclude_newer(tmp_path):
    hint = " ~/.config/uv/uv.toml sets exclude-newer: uv ignores releases."
    cmd = _helper(
        tmp_path, uv=str(_script(tmp_path / "uv", "exit 1")), install_hint=hint
    )
    result = subprocess.run(
        cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60
    )
    end = json.loads(result.stdout.splitlines()[-1])
    assert end["code"] == "install_failed"
    assert "sets exclude-newer" in end["message"]
    assert "install-script" in end["message"]


@needs_blocker
def test_helper_cancel_before_retry_is_install_failed(tmp_path):
    """A failed uv run may have changed the environment: a wait cancelled
    before the retry must not claim that nothing was changed."""
    # Named sleep, as in _start_blocker: uutils runs under no other name.
    blocker = tmp_path / "env" / "bin" / "sleep"
    blocker.parent.mkdir(parents=True)
    shutil.copy(shutil.which("sleep"), blocker)
    # Starts a process from the environment, then fails, as a uv run that
    # met a locked file would leave things.
    uv = _script(tmp_path / "uv", f"'{blocker}' 30 >/dev/null 2>&1 &\nexit 1")
    proc = subprocess.Popen(
        _helper(tmp_path, uv=str(uv)),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    events = []
    try:
        with _bounded(proc):
            for line in proc.stdout:
                event = json.loads(line)
                events.append(event)
                detail = event.get("detail") or {}
                if "cancel_file" in detail:
                    assert (
                        Path(detail["waiting_for"][0]["path"]).resolve()
                        == blocker.resolve()
                    )
                    Path(detail["cancel_file"]).write_text("")
    finally:
        subprocess.run(["pkill", "-f", str(blocker)], check=False)
    end = events[-1]
    assert end["status"] == "error"
    assert end["code"] == "install_failed"
    # The cancel before the retry, not a second failed uv run, which ends on
    # install_failed with the install command too.
    assert end["message"].startswith("Update cancelled after a failed install")
    assert "install-script" in end["message"]
    assert any("trying once more" in e.get("message", "") for e in events)


@needs_blocker
def test_helper_cancelled_wait_is_environment_busy(tmp_path):
    blocker_pid = _start_blocker(tmp_path / "env" / "bin")
    ran = tmp_path / "uv-ran"
    cmd = _helper(tmp_path, uv=str(_script(tmp_path / "uv", f"touch '{ran}'")))
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    events = []
    try:
        with _bounded(proc):
            for line in proc.stdout:
                event = json.loads(line)
                events.append(event)
                detail = event.get("detail") or {}
                if "cancel_file" in detail:
                    assert detail["waiting_for"][0]["pid"] == blocker_pid
                    Path(detail["cancel_file"]).write_text("")
    finally:
        _kill(blocker_pid)
    end = events[-1]
    assert end["status"] == "error"
    assert end["code"] == "environment_busy"
    assert end["detail"]["waiting_for"][0]["pid"] == blocker_pid
    assert not ran.exists()


@needs_blocker
def test_second_helper_ends_while_one_waits(tmp_path):
    lock = str(tmp_path / "update.lock")
    first_blocker = _start_blocker(tmp_path / "first" / "env" / "bin")
    second_ran = tmp_path / "second-uv-ran"
    first = subprocess.Popen(
        _helper(tmp_path / "first", lock=lock),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        with _bounded(first):
            assert "pending" in json.loads(first.stdout.readline())["message"]
            second = subprocess.run(
                _helper(
                    tmp_path / "second",
                    lock=lock,
                    uv=str(_script(tmp_path / "uv2", f"touch '{second_ran}'")),
                ),
                stdin=subprocess.PIPE,
                capture_output=True,
                text=True,
                timeout=60,
            )
    finally:
        _kill(first_blocker)
    end = json.loads(second.stdout.splitlines()[-1])
    assert end["code"] == "environment_busy"
    assert "Another EvoSci update" in end["message"]
    assert end["detail"] == {"waiting_for": []}
    assert not second_ran.exists()


@needs_blocker
def test_helper_waits_for_its_exiting_caller(tmp_path):
    """The caller still exiting when the helper starts is waited for, not
    reported as a process that blocks the environment."""
    exiting_pid = _start_blocker(tmp_path / "env" / "bin" / "caller", seconds=10)
    try:
        result = subprocess.run(
            _helper(tmp_path, wait_pids=[exiting_pid]),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        _kill(exiting_pid)
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert not any("pending" in e.get("message", "") for e in events), events
    assert any("Installing EvoScientist 0.3.7" in e.get("message", "") for e in events)


@needs_blocker
def test_helper_without_a_desktop_does_not_wait(tmp_path):
    """Unattended text mode (an SSH session): plain lines on stdout, and a
    blocked environment ends the update at once."""
    blocker_pid = _start_blocker(tmp_path / "env" / "bin")
    try:
        result = subprocess.run(
            _helper(tmp_path, json_mode=False, unattended=True),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        _kill(blocker_pid)
    assert "Update pending until you close: sleep" in result.stdout
    assert "Ctrl+C" not in result.stdout
    assert result.stdout.rstrip().endswith("(environment_busy)")


# --------------------------------------------------------------------------- #
# macOS and Linux: waiting for other processes that run from the environment
# --------------------------------------------------------------------------- #
TUI = {"pid": 4321, "name": "EvoSci", "path": "/env/bin/python"}


@pytest.fixture
def waits(uv_install, monkeypatch):
    """Blockers for the first *n* checks (``state["n"]``), a piped stdin and
    no sleeping."""
    state = {"n": 1, "checks": 0}

    def blockers():
        state["checks"] += 1
        return [TUI] if state["checks"] <= state["n"] else []

    monkeypatch.setattr(updater, "running_from_env", blockers)
    monkeypatch.setattr(updater, "stdin_is_pipe", lambda: True)
    monkeypatch.setattr(updater, "_reader_gone", lambda: False)
    monkeypatch.setattr(updater.time, "sleep", lambda s: None)
    return state


def test_update_waits_until_other_processes_close(waits, monkeypatch):
    runs = Runs()
    monkeypatch.setattr(updater.subprocess, "run", runs)
    monkeypatch.setattr(updater.subprocess, "Popen", FakePopen([]))
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.8", json=True))
    assert code == 0
    pending = [e for e in lines if "pending" in e.get("message", "")]
    assert len(pending) == 1
    assert pending[0]["detail"]["waiting_for"] == [TUI]
    assert pending[0]["detail"]["cancel_file"].endswith("cancel")
    assert runs.calls[0]["cmd"][0] == "/opt/uv"
    assert _terminal(lines)["status"] == "done"


def test_cancel_file_ends_the_wait_and_changes_nothing(waits, monkeypatch):
    waits["n"] = 100
    runs = Runs()
    monkeypatch.setattr(updater.subprocess, "run", runs)

    def cancel_on_first_check(s):
        raise AssertionError("cancelled before sleeping")

    out = io.StringIO()

    def emit_line(line):
        out.write(line + "\n")

    def emit(event):
        out.write(json.dumps(event) + "\n")
        cancel = (event.get("detail") or {}).get("cancel_file")
        if cancel:
            Path(cancel).write_text("")

    monkeypatch.setattr(updater.time, "sleep", cancel_on_first_check)
    code = updater.run_update(
        updater.UpdateOptions(version="0.3.8", json=True),
        emit,
        confirm=None,
        emit_line=emit_line,
    )
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    end = _terminal(lines)
    assert code == 1
    assert end["code"] == "environment_busy"
    assert end["detail"] == {"waiting_for": [TUI]}
    assert "Nothing was changed" in end["message"]
    assert runs.calls == []


def test_unattended_json_update_does_not_wait(waits, monkeypatch):
    waits["n"] = 100
    monkeypatch.setattr(updater, "stdin_is_pipe", lambda: False)
    runs = Runs()
    monkeypatch.setattr(updater.subprocess, "run", runs)
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.8", json=True))
    assert code == 1
    assert _terminal(lines)["code"] == "environment_busy"
    assert runs.calls == []


def test_gone_reader_ends_the_wait(waits, monkeypatch):
    waits["n"] = 100
    monkeypatch.setattr(updater, "_reader_gone", lambda: True)
    monkeypatch.setattr(updater.subprocess, "run", Runs())
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.8", json=True))
    end = _terminal(lines)
    assert code == 1
    assert end["code"] == "environment_busy"
    assert "has exited" in end["message"]


def test_ctrl_c_while_waiting_is_environment_busy(waits, monkeypatch):
    waits["n"] = 100
    monkeypatch.setattr(updater.sys.stdin, "isatty", lambda: True, raising=False)

    def interrupted(s):
        raise KeyboardInterrupt

    monkeypatch.setattr(updater.time, "sleep", interrupted)
    runs = Runs()
    monkeypatch.setattr(updater.subprocess, "run", runs)
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.8"))
    end = _terminal(lines)
    assert code == 1
    assert end["code"] == "environment_busy"
    assert "Ctrl+C cancels" in lines[-2]["message"]
    assert runs.calls == []


def test_running_from_env_matches_the_interpreter_on_the_command_line(
    tmp_path, monkeypatch
):
    import psutil

    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(env))

    class Proc:
        def __init__(self, pid, cmdline, exe):
            self.info = {"pid": pid, "name": f"p{pid}", "cmdline": cmdline, "exe": exe}

    base = "/usr/lib/uv-python/cpython-3.12/bin/python3.12"
    procs = [
        Proc(os.getpid(), [str(env / "bin" / "python")], base),  # this process
        Proc(11, [str(env / "bin" / "python"), str(env / "bin" / "EvoSci")], base),
        Proc(12, ["node", "server.js"], "/usr/bin/node"),
        Proc(13, ["tool"], str(env / "bin" / "tool")),  # a binary in the env
        Proc(14, [], None),  # access denied
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: procs)
    assert [p["pid"] for p in updater.running_from_env()] == [11, 13]


@pytest.mark.skipif(sys.platform == "win32", reason="poll on a pipe")
def test_reader_gone_sees_a_closed_pipe(monkeypatch):
    read_fd, write_fd = os.pipe()
    out = os.fdopen(write_fd, "w")
    monkeypatch.setattr(sys, "stdout", out)
    try:
        assert updater._reader_gone() is False
        os.close(read_fd)
        assert updater._reader_gone() is True
    finally:
        monkeypatch.undo()
        out.close()


# --------------------------------------------------------------------------- #
# Behaviours the command promises: each reached by a test
# --------------------------------------------------------------------------- #
def test_json_command_never_asks_and_keeps_stdout_for_events(monkeypatch):
    from typer.testing import CliRunner

    from EvoScientist.cli import app

    seen: dict = {}

    def fake_run(opts, emit, *, confirm):
        seen.update(opts=opts, emit=emit, confirm=confirm)
        return 0

    monkeypatch.setattr(updater, "run_update", fake_run)
    monkeypatch.setattr(
        "EvoScientist.stream.json_sink.redirect_console_to_stderr",
        lambda: seen.setdefault("redirected", True),
    )
    result = CliRunner().invoke(app, ["update", "--json", "--skip", "webui"])
    assert result.exit_code == 0
    assert seen["confirm"] is None
    assert isinstance(seen["emit"], JsonEmitter)
    assert seen["redirected"] is True
    assert seen["opts"].skip == ["webui"]


def test_setup_output_that_is_not_an_event_stays_off_stdout(
    uv_install, monkeypatch, capsys
):
    monkeypatch.setattr(updater.subprocess, "run", Runs())
    monkeypatch.setattr(updater.subprocess, "Popen", FakePopen(["stray print"], rc=2))
    code, lines, _ = _run(updater.UpdateOptions(json=True))
    end = _terminal(lines)
    assert code == 1
    assert end["code"] == "setup_failed"
    assert end["detail"] == {"stage": None, "code": None, "exit_code": 2}
    assert "stray print" in capsys.readouterr().err


def test_terminal_setup_failure_is_setup_failed(uv_install, monkeypatch):
    monkeypatch.setattr(updater.subprocess, "run", Runs(setup_rc=1))
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.8"))
    end = _terminal(lines)
    assert code == 1
    assert end["code"] == "setup_failed"
    # Ready to paste: the path quoted for a shell.
    assert "Run it again: " in end["message"]
    assert end["message"].endswith(" setup")


def test_editable_direct_url_says_git(tmp_path, monkeypatch, no_subprocess):
    """The real direct_url.json check, not a stub of it."""

    class Dist:
        metadata: ClassVar[dict] = {"Name": "EvoScientist"}

        def read_text(self, name):
            return '{"url": "file:///src/evo", "dir_info": {"editable": true}}'

    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(updater, "installed_version", lambda: "0.3.7")
    monkeypatch.setattr("EvoScientist.config.load_config", lambda: FakeConfig())
    monkeypatch.setattr(
        "EvoScientist.update_check.fetch_versions", lambda index, timeout: VERSIONS
    )
    monkeypatch.setattr("importlib.metadata.distributions", lambda: [Dist()])
    _code, lines, _ = _run(updater.UpdateOptions(json=True))
    end = _terminal(lines)
    assert end["code"] == "manual_update"
    assert "source checkout" in end["message"]


def test_editable_receipt_says_git(tmp_path, monkeypatch, no_subprocess):
    _plain_env(tmp_path, monkeypatch)
    (tmp_path / "uv-receipt.toml").write_text(
        '[tool]\nrequirements = [{ name = "evoscientist", editable = "/opt/src/evo" }]\n'
    )
    _code, lines, _ = _run(updater.UpdateOptions(json=True))
    end = _terminal(lines)
    assert end["code"] == "manual_update"
    assert "editable mode" in end["message"]


# The real one: uv_install replaces it.
_CHECK_SERVER = updater.check_server


def test_server_refusal_comes_before_uv(uv_install, server, monkeypatch):
    monkeypatch.setattr(updater, "check_server", _CHECK_SERVER)
    runs = Runs()
    monkeypatch.setattr(updater.subprocess, "run", runs)
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.8", json=True))
    end = _terminal(lines)
    assert code == 1
    assert end["code"] == "server_running"
    assert "--yes" in end["message"]
    assert runs.calls == []
    assert server["stopped"] == 0


@pytest.mark.skipif(sys.platform == "win32", reason="poll on a socket")
def test_reader_gone_sees_a_closed_socket(monkeypatch):
    """Node (the desktop app) gives a child UNIX socketpairs, not pipes."""
    import socket

    stdin_ours, stdin_theirs = socket.socketpair()
    stdout_ours, stdout_theirs = socket.socketpair()
    stdin = os.fdopen(stdin_ours.detach(), "r")
    out = os.fdopen(stdout_ours.detach(), "w")
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", out)
    try:
        assert updater.stdin_is_pipe() is True
        assert updater._reader_gone() is False
        stdout_theirs.close()
        assert updater._reader_gone() is True
    finally:
        monkeypatch.undo()
        stdin.close()
        out.close()
        stdin_theirs.close()


SERVER = {"pid": 42, "name": "python3.12", "path": "/env/bin/python"}


def test_unattended_yes_leaves_the_server_running_when_something_blocks(
    uv_install, server, monkeypatch
):
    monkeypatch.setattr(updater, "check_server", _CHECK_SERVER)
    monkeypatch.setattr(updater, "_server_tree", lambda: {42})
    monkeypatch.setattr(updater, "running_from_env", lambda: [SERVER, TUI])
    monkeypatch.setattr(updater, "stdin_is_pipe", lambda: False)
    runs = Runs()
    monkeypatch.setattr(updater.subprocess, "run", runs)
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.8", json=True, yes=True))
    end = _terminal(lines)
    assert code == 1
    assert end["code"] == "environment_busy"
    assert end["detail"] == {"waiting_for": [TUI]}
    assert server["stopped"] == 0
    assert runs.calls == []


def test_the_server_is_not_waited_for_before_it_is_stopped(
    uv_install, server, monkeypatch
):
    monkeypatch.setattr(updater, "check_server", _CHECK_SERVER)
    monkeypatch.setattr(updater, "_server_tree", lambda: {42})
    monkeypatch.setattr(
        updater, "running_from_env", lambda: [] if server["stopped"] else [SERVER]
    )
    monkeypatch.setattr(updater, "stdin_is_pipe", lambda: False)
    runs = Runs()
    monkeypatch.setattr(updater.subprocess, "run", runs)
    monkeypatch.setattr(updater.subprocess, "Popen", FakePopen([]))
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.8", json=True, yes=True))
    assert code == 0
    assert _terminal(lines)["status"] == "done"
    assert server["stopped"] == 1
    assert runs.calls[0]["cmd"][0] == "/opt/uv"


def test_a_server_child_started_during_the_wait_is_not_waited_for(
    uv_install, server, monkeypatch
):
    child = {"pid": 43, "name": "python3.12", "path": "/env/bin/python"}
    checks = {"n": 0}

    def blockers():
        checks["n"] += 1
        if server["stopped"]:
            return []
        return [SERVER, TUI] if checks["n"] == 1 else [SERVER, child]

    def tree():
        return {42} if checks["n"] < 2 else {42, 43}

    def sleep(s):
        if checks["n"] > 5:
            raise AssertionError("still waiting")

    monkeypatch.setattr(updater, "check_server", _CHECK_SERVER)
    monkeypatch.setattr(updater, "_server_tree", tree)
    monkeypatch.setattr(updater, "running_from_env", blockers)
    monkeypatch.setattr(updater, "stdin_is_pipe", lambda: True)
    monkeypatch.setattr(updater, "_reader_gone", lambda: False)
    monkeypatch.setattr(updater.time, "sleep", sleep)
    monkeypatch.setattr(updater.subprocess, "run", Runs())
    monkeypatch.setattr(updater.subprocess, "Popen", FakePopen([]))
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.8", json=True, yes=True))
    assert code == 0
    assert _terminal(lines)["status"] == "done"
    assert server["stopped"] == 1


def test_server_tree_reads_the_recorded_server(monkeypatch):
    import EvoScientist.langgraph_dev.manager as manager

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        monkeypatch.setattr(
            manager, "_read_workspace_sidecar", lambda: {"pid": os.getpid()}
        )
        assert {os.getpid(), child.pid} <= updater._server_tree()
        monkeypatch.setattr(manager, "_read_workspace_sidecar", lambda: None)
        assert updater._server_tree() == set()
        monkeypatch.setattr(manager, "_read_workspace_sidecar", lambda: {"pid": "x"})
        assert updater._server_tree() == set()
        monkeypatch.setattr(
            manager, "_read_workspace_sidecar", lambda: {"pid": float("inf")}
        )
        assert updater._server_tree() == set()
    finally:
        child.kill()
        child.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="the Windows helper holds it")
def test_second_update_ends_while_one_holds_the_lock(uv_install, monkeypatch):
    """macOS and Linux: one update at a time, as on Windows."""
    runs = Runs()
    monkeypatch.setattr(updater.subprocess, "run", runs)
    held = updater._hold_update_lock()
    try:
        code, lines, _ = _run(updater.UpdateOptions(version="0.3.8", json=True))
    finally:
        held.close()
    end = _terminal(lines)
    assert code == 1
    assert end["code"] == "environment_busy"
    assert "Another EvoSci update" in end["message"]
    assert end["detail"] == {"waiting_for": []}
    assert runs.calls == []
    # Released at the end: the next update runs.
    monkeypatch.setattr(updater.subprocess, "Popen", FakePopen([]))
    code, _, _ = _run(updater.UpdateOptions(version="0.3.8", json=True))
    assert code == 0


def test_ctrl_c_before_the_first_look_has_a_plain_message(uv_install, monkeypatch):
    def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr(updater, "running_from_env", interrupted)
    monkeypatch.setattr(updater.subprocess, "run", Runs())
    code, lines, _ = _run(updater.UpdateOptions(version="0.3.8", json=True))
    end = _terminal(lines)
    assert code == 1
    assert end["code"] == "environment_busy"
    assert end["message"] == "Update cancelled. Nothing was changed."


def test_deploy_owned_server_is_not_restarted(server, monkeypatch):
    """``EvoSci deploy`` or the desktop app owned it; nobody asked for a
    lasting one."""
    import EvoScientist.langgraph_dev.manager as manager
    from EvoScientist.setup import server as server_stage

    monkeypatch.setattr(
        manager,
        "_read_workspace_sidecar",
        lambda: {"pid": 42, "workspace": "/proj", "lasting": False},
    )
    recorded: list = []
    monkeypatch.setattr(server_stage, "remember_for_restart", recorded.append)

    updater.check_server(updater.UpdateOptions(yes=True), _emits(), None)

    assert recorded == []


@pytest.mark.parametrize("running", [True, False])
def test_update_stops_service_ccproxy(server, monkeypatch, running):
    """It runs from the install the update replaces."""
    import EvoScientist.ccproxy_manager as ccproxy_mod

    server["running"] = running
    stopped = []
    monkeypatch.setattr(
        ccproxy_mod, "stop_recorded_ccproxy", lambda: stopped.append(1) or 7
    )

    updater.check_server(updater.UpdateOptions(yes=True), _emits(), None)

    assert stopped == [1]
