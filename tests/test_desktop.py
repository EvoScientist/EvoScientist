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

from EvoScientist.deploy.launcher import LauncherError
from EvoScientist.desktop import app_paths, shell
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
