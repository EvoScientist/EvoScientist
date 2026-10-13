"""Tests for EvoScientist.updater (``EvoSci update``). The network and
subprocesses are mocked."""

from __future__ import annotations

import io
import json
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
    text = " ".join(result.output.split())
    assert "config.yaml.bak-<current version>" in text
    assert "rolls back" in text


def test_no_code_path_runs_uv_tool_upgrade():
    root = Path(updater.__file__).parent
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert '"upgrade"' not in text, path
        assert "'upgrade'" not in text, path


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


def test_cn_mirror_environment(monkeypatch):
    monkeypatch.delenv("UV_SYSTEM_CERTS", raising=False)
    env = updater.uv_environment("cn")
    assert env["UV_DEFAULT_INDEX"] == "https://pypi.tuna.tsinghua.edu.cn/simple"
    assert env["UV_PYTHON_INSTALL_MIRROR"] == (
        "https://registry.npmmirror.com/-/binary/python-build-standalone/"
    )
    assert env["UV_SYSTEM_CERTS"] == "1"


def test_default_mirror_keeps_user_settings(monkeypatch):
    monkeypatch.setenv("UV_SYSTEM_CERTS", "0")
    monkeypatch.delenv("UV_DEFAULT_INDEX", raising=False)
    env = updater.uv_environment("default")
    assert env["UV_SYSTEM_CERTS"] == "0"
    assert "UV_DEFAULT_INDEX" not in env


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
        lambda tmp, config, json_mode: started.append(
            {"tmp": tmp, "config": config, "json": json_mode}
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
    assert call["tmp"].is_dir()  # the helper deletes it
    assert (call["tmp"] / "constraints.txt").read_text() == "six==1.17.0\n"
    assert config["env_dir"] == str(Path(sys.prefix).resolve())
    assert config["uv"] == "/opt/uv"
    assert config["uv_args"][:3] == ["tool", "install", "--force"]
    assert config["evosci"] == str(uv_install["tmp"] / "bin" / "EvoSci.exe")
    assert config["skip"] == ["webui"]
    assert (config["installed"], config["target"]) == ("0.3.7", "0.3.10")


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
    update_helper.start_helper(tmp_path, {"env_dir": "C:\\env"}, json_mode=True)
    (call,) = calls
    assert call["creationflags"] == 0x200 | 0x8000000
    assert call["stdin"] is sys.stdin
    assert call["stdout"] is sys.stdout
    assert call["cmd"][-3:] == [
        str(tmp_path / update_helper.HELPER_NAME),
        "-ConfigPath",
        str(tmp_path / "config.json"),
    ]
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config == {"env_dir": "C:\\env", "temp_dir": str(tmp_path), "json": True}


def test_start_helper_terminal_opens_its_own_window(tmp_path, monkeypatch):
    """A prompt reads Ctrl+C as a key, so the helper gets a console of its own,
    where Ctrl+C reaches it."""
    from EvoScientist import update_helper

    calls: list[dict] = []
    monkeypatch.setattr(subprocess, "CREATE_NEW_CONSOLE", 0x10, raising=False)
    monkeypatch.setattr(
        update_helper.subprocess,
        "Popen",
        lambda cmd, **kwargs: calls.append(kwargs),
    )
    update_helper.start_helper(tmp_path, {}, json_mode=False)
    assert calls == [{"creationflags": 0x10}]


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


@pytest.mark.skipif(
    shutil.which("pwsh") is None
    or sys.platform == "win32"
    or not shutil.which("sleep"),
    reason="runs the helper under pwsh with POSIX stand-ins",
)
def test_helper_cancel_before_retry_is_install_failed(tmp_path):
    """A failed uv run may have changed the environment: a wait cancelled
    before the retry must not claim that nothing was changed."""
    from EvoScientist import update_helper

    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    blocker = env / "bin" / "blocker"
    shutil.copy(shutil.which("sleep"), blocker)
    uv = tmp_path / "uv"
    # Starts a process from the environment, then fails, as a uv run that
    # met a locked file would leave things.
    uv.write_text(f"#!/bin/sh\n'{blocker}' 30 >/dev/null 2>&1 &\nexit 1\n")
    uv.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    config = {
        "env_dir": str(env),
        "uv": str(uv),
        "uv_args": ["tool", "install"],
        "uv_env": {},
        "evosci": str(tmp_path / "EvoSci"),
        "skip": [],
        "installed": "0.3.6",
        "target": "0.3.7",
        "install_command": "install-script",
        "temp_dir": str(work),
        "json": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    script = tmp_path / "helper.ps1"
    script.write_text(update_helper.HELPER, encoding="utf-8")
    proc = subprocess.Popen(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(script),
            "-ConfigPath",
            str(tmp_path / "config.json"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    events = []
    try:
        for line in proc.stdout:
            event = json.loads(line)
            events.append(event)
            detail = event.get("detail") or {}
            if "cancel_file" in detail:
                Path(detail["cancel_file"]).write_text("")
        proc.wait(timeout=60)
    finally:
        proc.kill()
        subprocess.run(["pkill", "-f", str(blocker)], check=False)
    end = events[-1]
    assert end["status"] == "error"
    assert end["code"] == "install_failed"
    assert "install-script" in end["message"]
    assert any("trying once more" in e.get("message", "") for e in events)
