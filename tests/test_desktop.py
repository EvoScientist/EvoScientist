"""Tests for the desktop shell (``EvoScientist.desktop``).

Cover the GUI-free pieces: the boot/shutdown controller (via a recording fake
window + fake launcher), app-path resolution, and status/error HTML escaping.
The pywebview glue in ``run_desktop`` needs a real display and is not exercised
here.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from EvoScientist.config.settings import EvoScientistConfig
from EvoScientist.deploy.launcher import LauncherError
from EvoScientist.desktop import app_paths, shell
from EvoScientist.desktop import setup as dsetup
from EvoScientist.desktop.controller import DesktopController


class _FakeWindow:
    def __init__(self):
        self.status: list[str] = []
        self.loaded_url: str | None = None
        self.error: tuple | None = None

    def show_status(self, message):
        self.status.append(message)

    def load_url(self, url):
        self.loaded_url = url

    def show_error(self, code, message, detail):
        self.error = (code, message, detail)


class _FakeLauncher:
    def __init__(
        self, *, start_exc=None, ready_exc=None, webui_url="http://127.0.0.1:4716"
    ):
        self._start_exc = start_exc
        self._ready_exc = ready_exc
        self._webui_url = webui_url
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        if self._start_exc:
            raise self._start_exc

    def wait_ready(self, timeout):
        if self._ready_exc:
            raise self._ready_exc
        return SimpleNamespace(webui_url=self._webui_url)

    def stop(self):
        self.stopped = True


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
def test_boot_success_loads_webui_url():
    win = _FakeWindow()
    launcher = _FakeLauncher(webui_url="http://127.0.0.1:4800")
    ok = DesktopController(launcher, win).boot()
    assert ok is True
    assert launcher.started
    assert win.loaded_url == "http://127.0.0.1:4800"
    assert win.error is None
    assert win.status  # showed at least one progress message


def test_boot_launcher_error_on_start_shows_error():
    win = _FakeWindow()
    launcher = _FakeLauncher(
        start_exc=LauncherError("stripped_backend", "needs deploy mode", "stop it")
    )
    ok = DesktopController(launcher, win).boot()
    assert ok is False
    assert win.loaded_url is None
    assert win.error == ("stripped_backend", "needs deploy mode", "stop it")


def test_boot_launcher_error_on_ready_shows_error():
    win = _FakeWindow()
    launcher = _FakeLauncher(ready_exc=LauncherError("not_ready", "timed out", None))
    ok = DesktopController(launcher, win).boot()
    assert ok is False
    assert win.error[0] == "not_ready"
    assert win.loaded_url is None


def test_boot_unexpected_exception_is_shown_not_raised():
    win = _FakeWindow()
    launcher = _FakeLauncher(start_exc=RuntimeError("boom"))
    ok = DesktopController(launcher, win).boot()
    assert ok is False
    assert win.error[0] == "unexpected"
    assert "boom" in win.error[1]


def test_shutdown_stops_launcher():
    win = _FakeWindow()
    launcher = _FakeLauncher()
    DesktopController(launcher, win).shutdown()
    assert launcher.stopped


# --------------------------------------------------------------------------- #
# app_paths
# --------------------------------------------------------------------------- #
def test_app_paths_env_overrides_win(monkeypatch, tmp_path):
    monkeypatch.setenv(app_paths.ENV_WEBUI_DIR, str(tmp_path / "wu"))
    monkeypatch.setenv(app_paths.ENV_NODE_EXE, str(tmp_path / "n" / "node"))
    assert app_paths.webui_dir() == tmp_path / "wu"
    assert app_paths.node_exe() == tmp_path / "n" / "node"


def test_app_paths_defaults_from_app_root(monkeypatch, tmp_path):
    monkeypatch.delenv(app_paths.ENV_WEBUI_DIR, raising=False)
    monkeypatch.delenv(app_paths.ENV_NODE_EXE, raising=False)
    monkeypatch.setenv(app_paths.ENV_APP_ROOT, str(tmp_path))
    assert app_paths.webui_dir() == tmp_path / "webui"
    node_name = "node.exe" if os.name == "nt" else "node"
    assert app_paths.node_exe() == tmp_path / "runtime" / "node" / node_name


def test_app_root_prefers_env_over_cwd(monkeypatch, tmp_path):
    monkeypatch.setenv(app_paths.ENV_APP_ROOT, str(tmp_path))
    assert app_paths.app_root() == Path(tmp_path)


def test_python_exe_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(app_paths.ENV_PYTHON_EXE, str(tmp_path / "py" / "python.exe"))
    assert app_paths.python_exe() == tmp_path / "py" / "python.exe"


def test_python_exe_default_from_app_root(monkeypatch, tmp_path):
    monkeypatch.delenv(app_paths.ENV_PYTHON_EXE, raising=False)
    monkeypatch.setenv(app_paths.ENV_APP_ROOT, str(tmp_path))
    name = "python.exe" if os.name == "nt" else "python3"
    assert app_paths.python_exe() == tmp_path / "runtime" / "python" / name


def test_user_pypackages_dir_under_data_dir(monkeypatch, tmp_path):
    from EvoScientist import paths as epaths

    monkeypatch.setattr(epaths, "DATA_DIR", tmp_path / ".evoscientist")
    assert app_paths.user_pypackages_dir() == tmp_path / ".evoscientist" / "pypackages"


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #
def test_status_html_escapes():
    out = shell._status_html("<b>hi & bye</b>")
    assert "&lt;b&gt;" in out
    assert "hi &amp; bye" in out


def test_error_html_includes_code_and_escapes():
    out = shell._error_html("port_conflict", "<x>", "do <y>")
    assert "port_conflict" in out
    assert "&lt;x&gt;" in out
    assert "do &lt;y&gt;" in out


def test_error_html_omits_detail_block_when_none():
    out = shell._error_html("node_missing", "no node", None)
    assert "class=detail" not in out


class _RecordingLoaded:
    """Stand-in for pywebview's ``events.loaded`` that logs every ``wait``."""

    def __init__(self, log):
        self._log = log

    def wait(self, timeout=None):
        self._log.append("wait")
        return True


class _RecordingPywebviewWindow:
    """Fake pywebview window recording the order of navigations and loaded-waits."""

    def __init__(self):
        self.calls: list[str] = []
        self.events = SimpleNamespace(loaded=_RecordingLoaded(self.calls))

    def load_html(self, markup, *args, **kwargs):
        self.calls.append("load_html")

    def load_url(self, url):
        self.calls.append("load_url")


def test_webview_window_waits_for_loaded_around_html_swap():
    # A swap must wait for the initial content to load, then wait again after
    # issuing it — so it never races the previous (async) navigation.
    win = _RecordingPywebviewWindow()
    shell._WebviewWindow(win).show_status("starting")
    assert win.calls == ["wait", "load_html", "wait"]


def test_webview_window_awaits_initial_load_only_once():
    # The initial-load gate fires once; later swaps only post-wait.
    win = _RecordingPywebviewWindow()
    w = shell._WebviewWindow(win)
    w.show_status("starting")
    w.show_error("node_missing", "no node", None)
    assert win.calls == ["wait", "load_html", "wait", "load_html", "wait"]


def test_webview_window_load_url_awaits_initial_load():
    win = _RecordingPywebviewWindow()
    shell._WebviewWindow(win).load_url("http://127.0.0.1:4716")
    assert win.calls == ["wait", "load_url"]


# --------------------------------------------------------------------------- #
# First-run setup (EvoScientist.desktop.setup)
# --------------------------------------------------------------------------- #
def test_setup_needed_true_without_key():
    # Default config: provider=anthropic, no key → setup required.
    assert dsetup.setup_needed(EvoScientistConfig()) is True


def test_setup_needed_false_with_key():
    # A key set in the config (env keys reach here the same way, folded into
    # the config fields by get_effective_config) → setup skipped.
    cfg = EvoScientistConfig()
    cfg.provider = "anthropic"
    cfg.anthropic_api_key = "sk-ant-x"
    assert dsetup.setup_needed(cfg) is False


def test_apply_setup_writes_openrouter_fields(monkeypatch):
    saved = {}
    monkeypatch.setattr(
        "EvoScientist.config.save_config", lambda c: saved.setdefault("cfg", c)
    )
    cfg = EvoScientistConfig()
    dsetup.apply_setup(
        "openrouter", " deepseek/deepseek-chat ", " sk-or-1 ", "/tmp/ws", config=cfg
    )
    assert cfg.provider == "openrouter"
    assert cfg.model == "deepseek/deepseek-chat"  # trimmed
    assert cfg.openrouter_api_key == "sk-or-1"  # trimmed, correct field
    assert cfg.default_workdir == "/tmp/ws"
    assert saved["cfg"] is cfg


def test_apply_setup_maps_anthropic_key_field(monkeypatch):
    monkeypatch.setattr("EvoScientist.config.save_config", lambda c: None)
    cfg = EvoScientistConfig()
    dsetup.apply_setup("anthropic", "claude-sonnet-4-6", "sk-ant", "", config=cfg)
    assert cfg.anthropic_api_key == "sk-ant"


def test_validate_setup_flags_missing_and_unknown():
    assert dsetup.validate_setup("anthropic", "m", "") is not None  # no key
    assert dsetup.validate_setup("anthropic", "", "k") is not None  # no model
    assert dsetup.validate_setup("mystery", "m", "k") is not None  # unknown provider
    assert dsetup.validate_setup("anthropic", "m", "k") is None


def test_render_setup_html_prefills_escapes_and_hides_key():
    out = dsetup.render_setup_html(provider="openrouter", model="<m>", workspace="/w")
    assert 'value="openrouter" selected' in out  # provider preselected
    assert 'value="&lt;m&gt;"' in out  # model prefilled + escaped
    assert "/w" in out  # workspace prefilled
    assert "type=password" in out  # key field present
    assert "sk-" not in out  # key never prefilled
    assert "pywebview.api.submit" in out  # wired to the js_api


# --------------------------------------------------------------------------- #
# Bundled-python env for the agent's shell (EvoScientist._agent_shell_env)
# --------------------------------------------------------------------------- #
def test_agent_shell_env_none_without_bundled_python(monkeypatch, tmp_path):
    from EvoScientist import EvoScientist as ev
    from EvoScientist.desktop import app_paths as ap

    monkeypatch.setattr(ap, "python_exe", lambda: tmp_path / "absent" / "python.exe")
    assert ev._agent_shell_env() is None


def test_agent_shell_env_injects_bundled_python(monkeypatch, tmp_path):
    from EvoScientist import EvoScientist as ev
    from EvoScientist.desktop import app_paths as ap

    py = tmp_path / "py" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_text("x")
    monkeypatch.setattr(ap, "python_exe", lambda: py)
    monkeypatch.setattr(ap, "user_pypackages_dir", lambda: tmp_path / "pp")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = ev._agent_shell_env()
    assert env["PIP_USER"] == "1"
    assert env["PYTHONUSERBASE"] == str(tmp_path / "pp")
    assert env["PATH"].startswith(str(py.parent) + os.pathsep)
    assert (tmp_path / "pp").is_dir()  # created for the pip target
