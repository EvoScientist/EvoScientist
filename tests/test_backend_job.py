"""Tests for the Windows job object that ends a ``deploy --json`` backend.

The kernel32 calls are faked, so these run on every platform.
"""

from __future__ import annotations

import pytest

from EvoScientist import agent_shell
from EvoScientist.deploy import backend_job


class FakeProcess:
    def __init__(self, handle: int) -> None:
        self._handle = handle


class FakeJobApi:
    def __init__(
        self, *, create_ok: bool = True, limit_ok: bool = True, assign_ok: bool = True
    ) -> None:
        self.create_ok = create_ok
        self.limit_ok = limit_ok
        self.assign_ok = assign_ok
        self.calls: list[tuple] = []

    def create(self):
        self.calls.append(("create",))
        return 77 if self.create_ok else None

    def kill_on_close(self, job):
        self.calls.append(("kill_on_close", job))
        return self.limit_ok

    def assign(self, job, process):
        self.calls.append(("assign", job, process))
        return self.assign_ok

    def last_error(self):
        return 5

    def resume(self, process):
        self.calls.append(("resume", process))

    def close(self, job):
        self.calls.append(("close", job))


@pytest.fixture
def api(monkeypatch):
    fake = FakeJobApi()
    monkeypatch.setattr(backend_job, "_job_api", lambda: fake)
    return fake


def test_backend_joins_a_kill_on_close_job_before_it_runs(api):
    job = backend_job.BackendJob()
    assert job.creationflags == agent_shell._CREATE_SUSPENDED
    job.started(FakeProcess(1234))
    assert api.calls == [
        ("create",),
        ("kill_on_close", 77),
        ("assign", 77, 1234),
        ("resume", 1234),
    ]
    assert job.warning is None


def test_close_closes_the_handle_once(api):
    job = backend_job.BackendJob()
    job.close()
    job.close()
    assert api.calls.count(("close", 77)) == 1


def test_a_failed_assign_resumes_and_warns(api):
    api.assign_ok = False
    job = backend_job.BackendJob()
    job.started(FakeProcess(1234))
    assert api.calls[-1] == ("resume", 1234)
    assert "error 5" in job.warning
    assert "keeps running" in job.warning


@pytest.mark.parametrize("failing", ["create_ok", "limit_ok"])
def test_no_job_starts_the_backend_normally_and_warns(monkeypatch, failing):
    fake = FakeJobApi(**{failing: False})
    monkeypatch.setattr(backend_job, "_job_api", lambda: fake)
    job = backend_job.BackendJob()
    assert job.creationflags == 0
    job.started(FakeProcess(1234))
    assert ("assign", 77, 1234) not in fake.calls
    assert ("resume", 1234) not in fake.calls
    assert job.warning is not None
    if failing == "limit_ok":
        # A job that would not end the backend is not kept.
        assert fake.calls[-1] == ("close", 77)


def test_no_job_api_warns(monkeypatch):
    def _no_api():
        raise OSError("no kernel32")

    monkeypatch.setattr(backend_job, "_job_api", _no_api)
    job = backend_job.BackendJob()
    assert job.creationflags == 0
    assert job.warning is not None


def test_only_windows_gets_a_job(monkeypatch, api):
    monkeypatch.setattr(backend_job.sys, "platform", "linux")
    assert backend_job.backend_job() is None
    monkeypatch.setattr(backend_job.sys, "platform", "win32")
    assert isinstance(backend_job.backend_job(), backend_job.BackendJob)
