"""Tests for the backend busy check (``EvoScientist.deploy.busy``)."""

from __future__ import annotations

from EvoScientist.deploy import busy


def _patch_probe(monkeypatch, *, reachable, result=None, exc=None):
    """Patch the reachability guard + the raw httpx probe used by
    ``backend_has_active_runs`` (which bypasses the proxy via trust_env=False)."""
    import httpx

    def _fake_get(url, **kwargs):
        if not reachable:
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _fake_get)
    # Default: no background jobs running (thread-based tests own the result).
    monkeypatch.setattr(busy, "_running_bg_processes", lambda url, **k: [])
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
    captured = _patch_probe(monkeypatch, reachable=True, result=[{"thread_id": "t"}])
    assert busy.backend_has_active_runs("http://127.0.0.1:6174") is True
    assert captured["url"].endswith("/threads/search")
    assert captured["kwargs"]["json"] == {"status": "busy", "limit": 1}
    assert captured["kwargs"]["trust_env"] is False  # must bypass the proxy


def test_active_runs_false_when_idle(monkeypatch):
    _patch_probe(monkeypatch, reachable=True, result=[])
    assert busy.backend_has_active_runs("http://127.0.0.1:6174") is False


def test_active_runs_true_when_bg_running(monkeypatch):
    # Threads idle, but a background job runs -> active (stopping the backend
    # would tree-kill it).
    _patch_probe(monkeypatch, reachable=True, result=[])
    monkeypatch.setattr(
        busy, "_running_bg_processes", lambda url, **k: [{"name": "train"}]
    )
    assert busy.backend_has_active_runs("http://127.0.0.1:6174") is True


def test_active_runs_false_when_unreachable(monkeypatch):
    captured = _patch_probe(monkeypatch, reachable=False, result=[{"thread_id": "t"}])
    # Unreachable backend: no reachable runs to protect → fail-open (False),
    # and the probe endpoint is never hit.
    assert busy.backend_has_active_runs("http://127.0.0.1:6174") is False
    assert captured == {}


def test_active_runs_false_on_probe_error(monkeypatch):
    _patch_probe(monkeypatch, reachable=True, exc=RuntimeError("boom"))
    assert busy.backend_has_active_runs("http://127.0.0.1:6174") is False


_WATCHED = "01a07c30-3858-7bb2-a0e4-288cdc91f264"


def _patch_probe_by_status(monkeypatch, mapping):
    """Patch the probe so each ``/threads/search`` call returns the result list
    ``mapping`` gives for its requested status (default empty). Lets a test make
    ``busy`` empty but ``interrupted`` non-empty, and vice versa. Returns a list
    that captures each call's JSON payload, so a test can assert the ``ids``
    filter carried the watched thread."""
    import httpx

    monkeypatch.setattr(httpx, "get", lambda url, **kwargs: None)
    # Default: no background jobs running (these tests exercise thread status).
    monkeypatch.setattr(busy, "_running_bg_processes", lambda url, **k: [])
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
# Tri-state probe + wait-for-idle
# --------------------------------------------------------------------------- #
def test_probe_active_state_busy(monkeypatch):
    _patch_probe(monkeypatch, reachable=True, result=[{"thread_id": "t"}])
    assert busy._probe_active_state("http://127.0.0.1:6174") == "active"


def test_probe_active_state_watched_interrupted_is_active(monkeypatch):
    # The WATCHED thread awaiting HITL input is active.
    calls = _patch_probe_by_status(
        monkeypatch, {"busy": [], "interrupted": [{"id": _WATCHED}]}
    )
    assert (
        busy._probe_active_state("http://127.0.0.1:6174", watched_thread_id=_WATCHED)
        == "active"
    )
    # the interrupted probe scoped to the watched thread via the ids filter
    interrupted = [c for c in calls if c["status"] == "interrupted"]
    assert interrupted
    assert interrupted[0]["ids"] == [_WATCHED]


def test_probe_active_state_unwatched_interrupted_is_idle(monkeypatch):
    # An interrupted thread the user is NOT watching (or with no watched thread)
    # must not count: interrupted turns are saved/resumable and accumulate.
    _patch_probe_by_status(monkeypatch, {"busy": [], "interrupted": [{"id": "other"}]})
    assert busy._probe_active_state("http://127.0.0.1:6174") == "idle"


def test_probe_active_state_idle(monkeypatch):
    _patch_probe(monkeypatch, reachable=True, result=[])
    assert (
        busy._probe_active_state("http://127.0.0.1:6174", watched_thread_id=_WATCHED)
        == "idle"
    )


def test_probe_active_state_unreachable_is_idle(monkeypatch):
    # Backend gone -> its runs are gone -> a restart is safe -> "idle".
    _patch_probe(monkeypatch, reachable=False)
    assert busy._probe_active_state("http://127.0.0.1:6174") == "idle"


def test_probe_active_state_slow_ok_still_checks_runs(monkeypatch):
    import httpx

    _patch_probe(monkeypatch, reachable=True, result=[{"thread_id": "t1"}])

    def _slow_get(url, **kwargs):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(httpx, "get", _slow_get)
    assert busy._probe_active_state("http://127.0.0.1:6174") == "active"


def test_probe_active_state_error_is_unknown(monkeypatch):
    # Backend up but the probe threw -> we cannot tell -> "unknown".
    _patch_probe(monkeypatch, reachable=True, exc=RuntimeError("boom"))
    assert busy._probe_active_state("http://127.0.0.1:6174") == "unknown"


def test_probe_active_state_bg_running_is_active(monkeypatch):
    # No busy/interrupted threads, but a background job is still running -> active
    # (a restart would tree-kill it, and it is not resumable).
    _patch_probe(monkeypatch, reachable=True, result=[])
    monkeypatch.setattr(
        busy,
        "_running_bg_processes",
        lambda url, **k: [{"process_id": "p1", "name": "train"}],
    )
    assert busy._probe_active_state("http://127.0.0.1:6174") == "active"


def test_probe_active_state_bg_probe_error_is_unknown(monkeypatch):
    # Threads idle, but the bg-process probe threw -> cannot tell -> "unknown"
    # (a wait keeps waiting; a stop prompt fails open).
    _patch_probe(monkeypatch, reachable=True, result=[])

    def _boom(url, **k):
        raise RuntimeError("bg probe down")

    monkeypatch.setattr(busy, "_running_bg_processes", _boom)
    assert busy._probe_active_state("http://127.0.0.1:6174") == "unknown"


def test_running_bg_process_names_lists_names_and_empty_on_error(monkeypatch):
    monkeypatch.setattr(
        busy,
        "_running_bg_processes",
        lambda url, **k: [{"name": "train"}, {"name": "eval"}],
    )
    assert busy.running_bg_process_names("http://127.0.0.1:6174") == [
        "train",
        "eval",
    ]

    def _boom(url, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(busy, "_running_bg_processes", _boom)
    assert busy.running_bg_process_names("http://127.0.0.1:6174") == []


def test_active_runs_false_when_interrupted_not_watched(monkeypatch):
    # With no watched thread, an interrupted turn does not count — a stale
    # interrupt the user is not looking at must not make a stop prompt.
    _patch_probe_by_status(monkeypatch, {"busy": [], "interrupted": [{"id": "t"}]})
    assert busy.backend_has_active_runs("http://127.0.0.1:6174") is False


def test_active_runs_true_when_watched_interrupted(monkeypatch):
    # A stop prompts on the WATCHED interrupt too: global busy + watched interrupt.
    _patch_probe_by_status(monkeypatch, {"busy": [], "interrupted": [{"id": _WATCHED}]})
    assert (
        busy.backend_has_active_runs(
            "http://127.0.0.1:6174", watched_thread_id=_WATCHED
        )
        is True
    )


def test_wait_for_backend_idle_returns_immediately_when_idle():
    slept: list = []
    ok = busy.wait_for_backend_idle(
        "http://x", probe=lambda url: "idle", sleep=slept.append
    )
    assert ok is True
    assert slept == []  # never waited


def test_wait_for_backend_idle_treats_unknown_as_busy():
    states = iter(["busy", "unknown", "idle"])
    slept: list = []
    ok = busy.wait_for_backend_idle(
        "http://x",
        probe=lambda url: next(states),
        sleep=slept.append,
        poll_interval=0.5,
    )
    assert ok is True
    # "unknown" must NOT end the wait: two sleeps (after busy, after unknown).
    assert slept == [0.5, 0.5]


def test_wait_for_backend_idle_cancel_returns_false():
    probed: list = []

    def _probe(url):
        probed.append(url)
        return "busy"

    ok = busy.wait_for_backend_idle(
        "http://x", probe=_probe, sleep=lambda s: None, should_cancel=lambda: True
    )
    assert ok is False
    assert probed == []  # cancel is checked before probing
