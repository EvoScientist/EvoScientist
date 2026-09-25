"""Tests for the desktop shell (``EvoScientist.desktop``).

Cover the GUI-free pieces: the boot/shutdown controller (via a recording fake
window + fake launcher), app-path resolution, and status/error HTML escaping.
The pywebview glue in ``run_desktop`` needs a real display and is not exercised
here.
"""

from __future__ import annotations

import json
import logging
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
        self.pending: list[str] = []
        self.pending_cleared = 0

    def show_status(self, message):
        self.status.append(message)

    def load_url(self, url):
        self.loaded_url = url

    def show_error(self, code, message, detail):
        self.error = (code, message, detail)

    def show_pending(self, message):
        self.pending.append(message)

    def clear_pending(self):
        self.pending_cleared += 1


class _FakeLauncher:
    def __init__(
        self,
        *,
        start_exc=None,
        ready_exc=None,
        webui_url="http://127.0.0.1:4716",
        backend_url="http://127.0.0.1:2024",
        workspace_dir="/ws/old",
        backend_started=True,
    ):
        self._start_exc = start_exc
        self._ready_exc = ready_exc
        self._webui_url = webui_url
        self.backend_url = backend_url
        self.workspace_dir = workspace_dir
        self.backend_started = backend_started
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


def test_switch_workspace_no_factory_is_noop(caplog):
    win = _FakeWindow()
    launcher = _FakeLauncher()
    with caplog.at_level(logging.INFO, logger="EvoScientist.desktop"):
        DesktopController(launcher, win).switch_workspace("/ws/new")
    assert not launcher.stopped
    assert any("no launcher factory" in r.message for r in caplog.records)


def test_switch_workspace_restarts_with_new_launcher():
    win = _FakeWindow()
    old = _FakeLauncher(workspace_dir="/ws/old")
    new = _FakeLauncher(webui_url="http://127.0.0.1:4900")
    built: dict = {}

    def factory(ws):
        built["ws"] = ws
        return new

    ctl = DesktopController(
        old,
        win,
        launcher_factory=factory,
        active_probe=lambda url: "idle",  # nothing running -> switch immediately
    )
    ctl.switch_workspace("/ws/new")

    assert old.stopped  # old backend torn down
    assert new.started  # new one booted
    assert built["ws"] == "/ws/new"
    assert ctl.launcher is new  # live launcher swapped
    assert win.loaded_url == "http://127.0.0.1:4900"
    assert win.pending == []  # idle from the start -> no waiting banner


def test_switch_workspace_reused_backend_skips_wait(monkeypatch):
    """A reused (not app-owned) backend's runs survive the switch, so the switch
    must not wait on it — mirrors the close path's backend_started guard."""
    from EvoScientist.desktop import shutdown as dshutdown

    monkeypatch.setattr(dshutdown, "running_bg_process_names", lambda url, **k: [])
    win = _FakeWindow()
    old = _FakeLauncher(backend_started=False)  # reused, not ours
    new = _FakeLauncher(webui_url="http://127.0.0.1:4903")
    probed = {"n": 0}

    def probe(url):
        probed["n"] += 1
        return "active"  # would block forever if it were consulted

    ctl = DesktopController(
        old,
        win,
        launcher_factory=lambda ws: new,
        active_probe=probe,
        sleep=lambda s: None,
    )
    ctl.switch_workspace("/ws/new")

    assert probed["n"] == 0  # never probed/waited on a backend we don't own
    assert win.pending == []  # no waiting banner
    assert old.stopped  # went straight to the rebuild
    assert new.started
    assert ctl.launcher is new


def test_switch_workspace_waits_until_idle(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    monkeypatch.setattr(dshutdown, "running_bg_process_names", lambda url, **k: [])
    win = _FakeWindow()
    old = _FakeLauncher()
    new = _FakeLauncher(webui_url="http://127.0.0.1:4901")
    calls = {"n": 0}
    slept: list = []

    def probe(url):
        calls["n"] += 1
        return "idle" if calls["n"] >= 3 else "active"

    ctl = DesktopController(
        old,
        win,
        launcher_factory=lambda ws: new,
        active_probe=probe,
        sleep=slept.append,
        poll_interval=0.01,
    )
    ctl.switch_workspace("/ws/new")

    assert slept  # did not restart until the backend went idle
    assert win.pending  # showed the non-blocking waiting banner (not a full page)
    assert win.pending_cleared  # and cleared it before restarting
    assert old.stopped
    assert new.started
    assert ctl.launcher is new


def test_switch_workspace_aborts_when_cancelled(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    monkeypatch.setattr(dshutdown, "running_bg_process_names", lambda url, **k: [])
    win = _FakeWindow()
    old = _FakeLauncher()

    ctl = DesktopController(
        old,
        win,
        launcher_factory=lambda ws: _FakeLauncher(),
        active_probe=lambda url: "active",  # would wait forever...
        should_cancel=lambda: True,  # ...but the app is shutting down
        sleep=lambda s: None,
    )
    ctl.switch_workspace("/ws/new")

    assert not old.stopped  # no teardown, no relaunch on a closing app
    assert ctl.launcher is old
    assert win.pending_cleared  # banner removed on abort


def test_switch_workspace_stop_now_kills_and_switches(monkeypatch):
    """The banner's "Stop tasks and switch now": end the wait even though work is
    still active, and proceed with the restart (which kills the running tasks)."""
    from EvoScientist.desktop import shutdown as dshutdown

    monkeypatch.setattr(
        dshutdown, "running_bg_process_names", lambda url, **k: ["train"]
    )
    win = _FakeWindow()
    old = _FakeLauncher()
    new = _FakeLauncher(webui_url="http://127.0.0.1:4902")

    ctl = DesktopController(
        old,
        win,
        launcher_factory=lambda ws: new,
        active_probe=lambda url: "active",  # never goes idle on its own
        should_proceed_now=lambda: True,  # ...but the user clicked "stop and switch"
        sleep=lambda s: None,
    )
    ctl.switch_workspace("/ws/new")

    assert old.stopped  # restart happened -> running tasks torn down
    assert new.started
    assert ctl.launcher is new
    assert win.pending  # banner shown while active
    assert win.pending_cleared  # then cleared before restart


def test_switch_workspace_passes_watched_thread_to_default_probe(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    seen: dict = {}

    def fake_probe(url, *, watched_thread_id=None, timeout=3.0):
        seen["watched"] = watched_thread_id
        return "idle"

    monkeypatch.setattr(dshutdown, "_probe_active_state", fake_probe)
    win = _FakeWindow()
    ctl = DesktopController(  # no active_probe -> uses the default via a partial
        _FakeLauncher(), win, launcher_factory=lambda ws: _FakeLauncher()
    )
    ctl.switch_workspace("/ws/new", watched_thread_id="abc-123")
    assert seen["watched"] == "abc-123"


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


def test_error_html_shows_log_location():
    out = shell._error_html("node_missing", "no node", None)
    assert "Logs:" in out


# --------------------------------------------------------------------------- #
# Diagnostic-log paths
# --------------------------------------------------------------------------- #
def test_log_paths_under_config_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "EvoScientist.config.settings.get_config_dir", lambda: tmp_path / "cfg"
    )
    assert app_paths.desktop_log_path() == tmp_path / "cfg" / "desktop.log"
    assert app_paths.webui_log_path() == tmp_path / "cfg" / "webui.log"


# --------------------------------------------------------------------------- #
# Safe-shutdown decision + active-run probe (EvoScientist.desktop.shutdown)
# --------------------------------------------------------------------------- #
def test_should_confirm_only_when_owned_and_active():
    from EvoScientist.desktop import shutdown as dshutdown

    assert dshutdown.should_confirm_close(True, True) is True
    assert dshutdown.should_confirm_close(True, False) is False
    assert dshutdown.should_confirm_close(False, True) is False  # reused backend
    assert dshutdown.should_confirm_close(False, False) is False


def _patch_probe(monkeypatch, *, reachable, result=None, exc=None):
    """Patch the reachability guard + the raw httpx probe used by
    ``backend_has_active_runs`` (which bypasses the proxy via trust_env=False)."""
    import httpx

    from EvoScientist.desktop import shutdown as dshutdown
    from EvoScientist.langgraph_dev import manager as lgm

    monkeypatch.setattr(lgm, "is_langgraph_dev_running", lambda **k: reachable)
    # Default: no background jobs running (thread-based tests own the result).
    monkeypatch.setattr(dshutdown, "_running_bg_processes", lambda url, **k: [])
    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return result if result is not None else []

    def _fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        if exc is not None:
            raise exc
        return _Resp()

    monkeypatch.setattr(httpx, "post", _fake_post)
    return captured


def test_active_runs_true_when_busy_thread(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    captured = _patch_probe(monkeypatch, reachable=True, result=[{"thread_id": "t"}])
    assert dshutdown.backend_has_active_runs("http://127.0.0.1:6174") is True
    assert captured["url"].endswith("/threads/search")
    assert captured["kwargs"]["json"] == {"status": "busy", "limit": 1}
    assert captured["kwargs"]["trust_env"] is False  # must bypass the proxy


def test_active_runs_false_when_idle(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    _patch_probe(monkeypatch, reachable=True, result=[])
    assert dshutdown.backend_has_active_runs("http://127.0.0.1:6174") is False


def test_active_runs_true_when_bg_running(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    # Threads idle, but a background job runs -> the close must confirm (quitting
    # would tree-kill it). Same bg-awareness the switch wait uses.
    _patch_probe(monkeypatch, reachable=True, result=[])
    monkeypatch.setattr(
        dshutdown, "_running_bg_processes", lambda url, **k: [{"name": "train"}]
    )
    assert dshutdown.backend_has_active_runs("http://127.0.0.1:6174") is True


def test_active_runs_false_when_unreachable(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    captured = _patch_probe(monkeypatch, reachable=False, result=[{"thread_id": "t"}])
    # Unreachable backend: no reachable runs to protect → fail-open (False),
    # and the probe endpoint is never hit.
    assert dshutdown.backend_has_active_runs("http://127.0.0.1:6174") is False
    assert captured == {}


def test_active_runs_false_on_probe_error(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    _patch_probe(monkeypatch, reachable=True, exc=RuntimeError("boom"))
    assert dshutdown.backend_has_active_runs("http://127.0.0.1:6174") is False


_WATCHED = "01a07c30-3858-7bb2-a0e4-288cdc91f264"


def _patch_probe_by_status(monkeypatch, mapping):
    """Patch the probe so each ``/threads/search`` call returns the result list
    ``mapping`` gives for its requested status (default empty). Lets a test make
    ``busy`` empty but ``interrupted`` non-empty, and vice versa. Returns a list
    that captures each call's JSON payload, so a test can assert the ``ids``
    filter carried the watched thread."""
    import httpx

    from EvoScientist.desktop import shutdown as dshutdown
    from EvoScientist.langgraph_dev import manager as lgm

    monkeypatch.setattr(lgm, "is_langgraph_dev_running", lambda **k: True)
    # Default: no background jobs running (these tests exercise thread status).
    monkeypatch.setattr(dshutdown, "_running_bg_processes", lambda url, **k: [])
    calls: list[dict] = []

    class _Resp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    def _fake_post(url, **kwargs):
        payload = kwargs["json"]
        calls.append(payload)
        return _Resp(mapping.get(payload["status"], []))

    monkeypatch.setattr(httpx, "post", _fake_post)
    return calls


# --------------------------------------------------------------------------- #
# Tri-state probe + wait-for-idle (switch path)
# --------------------------------------------------------------------------- #
def test_probe_active_state_busy(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    _patch_probe(monkeypatch, reachable=True, result=[{"thread_id": "t"}])
    assert dshutdown._probe_active_state("http://127.0.0.1:6174") == "active"


def test_probe_active_state_watched_interrupted_is_active(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    # The WATCHED thread awaiting HITL input is active — the switch must wait.
    calls = _patch_probe_by_status(
        monkeypatch, {"busy": [], "interrupted": [{"id": _WATCHED}]}
    )
    assert (
        dshutdown._probe_active_state(
            "http://127.0.0.1:6174", watched_thread_id=_WATCHED
        )
        == "active"
    )
    # the interrupted probe scoped to the watched thread via the ids filter
    interrupted = [c for c in calls if c["status"] == "interrupted"]
    assert interrupted
    assert interrupted[0]["ids"] == [_WATCHED]


def test_probe_active_state_unwatched_interrupted_is_idle(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    # An interrupted thread the user is NOT watching (or with no watched thread)
    # must not block: interrupted turns are saved/resumable and accumulate.
    _patch_probe_by_status(monkeypatch, {"busy": [], "interrupted": [{"id": "other"}]})
    assert dshutdown._probe_active_state("http://127.0.0.1:6174") == "idle"


def test_probe_active_state_idle(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    _patch_probe(monkeypatch, reachable=True, result=[])
    assert (
        dshutdown._probe_active_state(
            "http://127.0.0.1:6174", watched_thread_id=_WATCHED
        )
        == "idle"
    )


def test_probe_active_state_unreachable_is_idle(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    # Backend gone -> its runs are gone -> a restart is safe -> "idle".
    _patch_probe(monkeypatch, reachable=False)
    assert dshutdown._probe_active_state("http://127.0.0.1:6174") == "idle"


def test_probe_active_state_error_is_unknown(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    # Backend up but the probe threw -> we cannot tell -> "unknown".
    _patch_probe(monkeypatch, reachable=True, exc=RuntimeError("boom"))
    assert dshutdown._probe_active_state("http://127.0.0.1:6174") == "unknown"


def test_probe_active_state_bg_running_is_active(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    # No busy/interrupted threads, but a background job is still running -> active
    # (a restart would tree-kill it, and it is not resumable).
    _patch_probe(monkeypatch, reachable=True, result=[])
    monkeypatch.setattr(
        dshutdown,
        "_running_bg_processes",
        lambda url, **k: [{"process_id": "p1", "name": "train"}],
    )
    assert dshutdown._probe_active_state("http://127.0.0.1:6174") == "active"


def test_probe_active_state_bg_probe_error_is_unknown(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    # Threads idle, but the bg-process probe threw -> cannot tell -> "unknown"
    # (the switch keeps waiting; the close fails open).
    _patch_probe(monkeypatch, reachable=True, result=[])

    def _boom(url, **k):
        raise RuntimeError("bg probe down")

    monkeypatch.setattr(dshutdown, "_running_bg_processes", _boom)
    assert dshutdown._probe_active_state("http://127.0.0.1:6174") == "unknown"


def test_running_bg_process_names_lists_names_and_empty_on_error(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    monkeypatch.setattr(
        dshutdown,
        "_running_bg_processes",
        lambda url, **k: [{"name": "train"}, {"name": "eval"}],
    )
    assert dshutdown.running_bg_process_names("http://127.0.0.1:6174") == [
        "train",
        "eval",
    ]

    def _boom(url, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(dshutdown, "_running_bg_processes", _boom)
    assert dshutdown.running_bg_process_names("http://127.0.0.1:6174") == []


def test_active_runs_false_when_interrupted_not_watched(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    # With no watched thread, an interrupted turn does not count — a stale
    # interrupt the user is not looking at must not make close prompt.
    _patch_probe_by_status(monkeypatch, {"busy": [], "interrupted": [{"id": "t"}]})
    assert dshutdown.backend_has_active_runs("http://127.0.0.1:6174") is False


def test_active_runs_true_when_watched_interrupted(monkeypatch):
    from EvoScientist.desktop import shutdown as dshutdown

    # Close prompts on the WATCHED interrupt too (same rule as the switch):
    # global busy + watched interrupt.
    _patch_probe_by_status(monkeypatch, {"busy": [], "interrupted": [{"id": _WATCHED}]})
    assert (
        dshutdown.backend_has_active_runs(
            "http://127.0.0.1:6174", watched_thread_id=_WATCHED
        )
        is True
    )


def test_wait_for_backend_idle_returns_immediately_when_idle():
    from EvoScientist.desktop import shutdown as dshutdown

    slept: list = []
    ok = dshutdown.wait_for_backend_idle(
        "http://x", probe=lambda url: "idle", sleep=slept.append
    )
    assert ok is True
    assert slept == []  # never waited


def test_wait_for_backend_idle_treats_unknown_as_busy():
    from EvoScientist.desktop import shutdown as dshutdown

    states = iter(["busy", "unknown", "idle"])
    slept: list = []
    ok = dshutdown.wait_for_backend_idle(
        "http://x",
        probe=lambda url: next(states),
        sleep=slept.append,
        poll_interval=0.5,
    )
    assert ok is True
    # "unknown" must NOT end the wait: two sleeps (after busy, after unknown).
    assert slept == [0.5, 0.5]


def test_wait_for_backend_idle_cancel_returns_false():
    from EvoScientist.desktop import shutdown as dshutdown

    probed: list = []

    def _probe(url):
        probed.append(url)
        return "busy"

    ok = dshutdown.wait_for_backend_idle(
        "http://x", probe=_probe, sleep=lambda s: None, should_cancel=lambda: True
    )
    assert ok is False
    assert probed == []  # cancel is checked before probing


# --------------------------------------------------------------------------- #
# Shell folder-picker helpers
# --------------------------------------------------------------------------- #
def test_first_path_folds_dialog_results():
    assert shell._first_path(None) is None
    assert shell._first_path(()) is None
    assert shell._first_path(("/a", "/b")) == "/a"
    assert shell._first_path("/a") == "/a"


def test_same_dir_normalises(tmp_path):
    d = tmp_path / "ws"
    d.mkdir()
    assert shell._same_dir(str(d), str(d)) is True
    assert shell._same_dir(str(d), str(tmp_path / "other")) is False
    assert shell._same_dir(None, str(d)) is False


class _FakeRawWindow:
    """Stand-in for a pywebview window: records ``evaluate_js`` scripts and
    returns a canned ``get_current_url`` (a URL string, or an Exception to
    raise)."""

    def __init__(self, result=None, url=None):
        self._result = result
        self._url = url
        self.scripts: list[str] = []

    def evaluate_js(self, script):
        self.scripts.append(script)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def get_current_url(self):
        if isinstance(self._url, Exception):
            raise self._url
        return self._url


def test_current_thread_id_reads_valid_uuid():
    url = f"http://127.0.0.1:5150/?assistantId=E&sidebar=1&threadId={_WATCHED}"
    win = shell._WebviewWindow(_FakeRawWindow(url=url))
    assert win.current_thread_id() == _WATCHED


def test_current_thread_id_rejects_non_uuid_and_missing():
    # No threadId in the URL, a malformed value, or no URL -> None, so the
    # switch/close gating falls back to busy-only instead of a bad backend query.
    no_tid = "http://127.0.0.1:5150/?assistantId=E&sidebar=1"
    bad = "http://127.0.0.1:5150/?threadId=not-a-uuid"
    assert shell._WebviewWindow(_FakeRawWindow(url=no_tid)).current_thread_id() is None
    assert shell._WebviewWindow(_FakeRawWindow(url=bad)).current_thread_id() is None
    assert shell._WebviewWindow(_FakeRawWindow(url=None)).current_thread_id() is None


def test_current_thread_id_swallows_url_errors():
    # get_current_url is a cached attribute read (no evaluate_js -> no close
    # deadlock); any failure still degrades to None.
    win = shell._WebviewWindow(_FakeRawWindow(url=RuntimeError("no page")))
    assert win.current_thread_id() is None


def test_show_pending_banner_has_cancel_bridge():
    raw = _FakeRawWindow()
    shell._WebviewWindow(raw).show_pending("Waiting for tasks…")
    js = raw.scripts[-1]
    # the banner's Cancel button calls the exposed JS->Python bridge
    assert "window.pywebview.api.cancel_workspace_switch()" in js
    # and the message is injected as a safe JS string literal
    assert json.dumps("Waiting for tasks…") in js


def test_clear_pending_removes_banner():
    raw = _FakeRawWindow()
    shell._WebviewWindow(raw).clear_pending()
    assert "__evosci_pending__" in raw.scripts[-1]
    assert "remove()" in raw.scripts[-1]


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


def test_apply_setup_blank_workspace_defaults_to_documents_subdir(monkeypatch):
    """A blank workspace field must not become "" (which resolves to the cwd —
    the whole Documents folder under the installer's WorkingDir); it defaults to
    the dedicated Documents\\EvoScientist folder instead."""
    monkeypatch.setattr("EvoScientist.config.save_config", lambda c: None)
    cfg = EvoScientistConfig()
    dsetup.apply_setup("anthropic", "claude-sonnet-4-6", "sk-ant", "  ", config=cfg)
    assert cfg.default_workdir == str(app_paths.default_workspace())
    assert cfg.default_workdir  # never blank


def test_default_workspace_is_a_named_documents_subfolder():
    ws = app_paths.default_workspace()
    assert ws.name == "EvoScientist"
    assert ws.parent.name == "Documents"


def test_validate_setup_flags_missing_and_unknown():
    assert dsetup.validate_setup("anthropic", "m", "") is not None  # no key
    assert dsetup.validate_setup("anthropic", "", "k") is not None  # no model
    assert dsetup.validate_setup("mystery", "m", "k") is not None  # unknown provider
    assert dsetup.validate_setup("anthropic", "m", "k") is None


def test_setup_api_submit_is_one_shot(monkeypatch):
    """A second submit is rejected once setup completed — a later WebUI page
    must not be able to overwrite the config through the still-attached bridge."""
    import threading

    calls = []
    monkeypatch.setattr(shell, "validate_setup", lambda *a, **k: None)
    monkeypatch.setattr(shell, "apply_setup", lambda *a, **k: calls.append(a))
    api = shell._SetupApi(EvoScientistConfig(), threading.Event())

    payload = {"provider": "anthropic", "model": "m", "api_key": "k", "workspace": ""}
    assert api.submit(payload) == {"ok": True}
    second = api.submit(payload)
    assert second["ok"] is False
    assert len(calls) == 1  # config written exactly once


def test_setup_api_submit_rejected_when_already_done(monkeypatch):
    """With setup pre-completed (existing config), submit never writes."""
    import threading

    writes = []
    monkeypatch.setattr(shell, "validate_setup", lambda *a, **k: None)
    monkeypatch.setattr(shell, "apply_setup", lambda *a, **k: writes.append(a))
    done = threading.Event()
    done.set()
    api = shell._SetupApi(EvoScientistConfig(), done)
    result = api.submit({"provider": "anthropic", "model": "m", "api_key": "k"})
    assert result["ok"] is False
    assert writes == []  # nothing written


def test_render_setup_html_prefills_escapes_and_hides_key():
    out = dsetup.render_setup_html(provider="openrouter", model="<m>", workspace="/w")
    assert 'value="openrouter" selected' in out  # provider preselected
    assert 'value="&lt;m&gt;"' in out  # model prefilled + escaped
    assert "/w" in out  # workspace prefilled
    assert "type=password" in out  # key field present
    assert "sk-" not in out  # key never prefilled
    assert "pywebview.api.submit" in out  # wired to the js_api


def test_render_setup_html_unknown_provider_stays_selected(monkeypatch):
    # A configured provider outside the curated six must be shown+selected, not
    # silently reverted to Anthropic (the first <option>).
    out = dsetup.render_setup_html(provider="siliconflow", model="glm-4")
    assert 'value="siliconflow" selected' in out
    # None of the curated options is selected in its place.
    assert 'value="anthropic" selected' not in out


def test_render_setup_html_provider_change_cascades_model(monkeypatch):
    import json
    import re

    from EvoScientist.llm import get_models_for_provider

    out = dsetup.render_setup_html(provider="anthropic", model="claude-custom")
    assert "onProviderChange()" in out  # select wired to the cascade
    m = re.search(r"var MODEL_DEFAULTS=(\{.*?\});", out)
    assert m, "MODEL_DEFAULTS map must be embedded"
    defaults = json.loads(m.group(1))
    # Current provider keeps the prefilled model as its default...
    assert defaults["anthropic"] == "claude-custom"
    # ...and another provider maps to its own registry default (first entry).
    entries = get_models_for_provider("openai")
    if entries:
        assert defaults["openai"] == entries[0][0]


# --------------------------------------------------------------------------- #
# Bundled-python env for the agent's shell (EvoScientist._agent_shell_env)
# --------------------------------------------------------------------------- #
def test_agent_shell_env_none_without_bundled_python(monkeypatch, tmp_path):
    from EvoScientist import EvoScientist as ev
    from EvoScientist.desktop import app_paths as ap

    monkeypatch.setattr(ap, "python_exe", lambda: tmp_path / "absent" / "python.exe")
    assert ev._agent_shell_env() is None


def test_agent_shell_env_none_in_dev_checkout_without_override(monkeypatch, tmp_path):
    """Not frozen and no explicit override: leave PATH untouched even if a
    runtime/python/ happens to exist under the cwd (untrusted workspace tree)."""
    from EvoScientist import EvoScientist as ev
    from EvoScientist.desktop import app_paths as ap

    py = tmp_path / "py" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_text("x")
    monkeypatch.setattr(ap, "is_frozen", lambda: False)
    monkeypatch.delenv(ap.ENV_PYTHON_EXE, raising=False)
    monkeypatch.setattr(ap, "python_exe", lambda: py)
    assert ev._agent_shell_env() is None


def test_agent_shell_env_injects_bundled_python(monkeypatch, tmp_path):
    from EvoScientist import EvoScientist as ev
    from EvoScientist.desktop import app_paths as ap

    py = tmp_path / "py" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_text("x")
    # An explicit override marks the interpreter as trusted (the frozen bundle is
    # the other trusted source).
    monkeypatch.setenv(ap.ENV_PYTHON_EXE, str(py))
    monkeypatch.setattr(ap, "python_exe", lambda: py)
    monkeypatch.setattr(ap, "user_pypackages_dir", lambda: tmp_path / "pp")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = ev._agent_shell_env()
    assert env["PIP_USER"] == "1"
    assert env["PYTHONUSERBASE"] == str(tmp_path / "pp")
    assert env["PATH"].startswith(str(py.parent) + os.pathsep)
    assert (tmp_path / "pp").is_dir()  # created for the pip target
