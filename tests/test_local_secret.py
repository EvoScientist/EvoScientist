"""Tests for the local secret: its file and the server's verify route."""

from __future__ import annotations

import errno
import json
import logging
import os
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from EvoScientist import local_secret, paths
from EvoScientist.langgraph_dev.http import app

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")


def _secret_file():
    return paths.SERVER_HOME / local_secret.FILE_NAME


@posix_only
def test_ensure_local_secret_creates_a_private_file():
    value = local_secret.ensure_local_secret()
    assert len(value) >= 40
    assert stat.S_IMODE(_secret_file().stat().st_mode) == 0o600
    assert stat.S_IMODE(paths.SERVER_HOME.stat().st_mode) == 0o700


def test_ensure_local_secret_reuses_the_file():
    """Never rotated on start: another running server would lock its clients out."""
    assert local_secret.ensure_local_secret() == local_secret.ensure_local_secret()


def test_ensure_local_secret_gives_servers_starting_at_once_one_value(monkeypatch):
    """Every starter writes its own value, then they all link at once."""
    starters = 4
    linking = threading.Barrier(starters)
    real_link = os.link

    def link_together(src, dst):
        linking.wait(timeout=10)
        real_link(src, dst)

    monkeypatch.setattr(local_secret.os, "link", link_together)
    with ThreadPoolExecutor(starters) as pool:
        values = set(pool.map(lambda _: local_secret.ensure_local_secret(), range(4)))
    assert values == {_secret_file().read_text()}
    assert [p.name for p in paths.SERVER_HOME.iterdir()] == ["local-secret"]


def test_ensure_local_secret_works_without_hard_links(monkeypatch):
    def no_links(*_args):
        raise OSError(errno.EPERM, "hard links not supported")

    monkeypatch.setattr(local_secret.os, "link", no_links)
    value = local_secret.ensure_local_secret()
    assert _secret_file().read_text() == value
    assert local_secret.ensure_local_secret() == value
    assert [p.name for p in paths.SERVER_HOME.iterdir()] == ["local-secret"]


@posix_only
def test_ensure_local_secret_tightens_a_file_others_can_read(caplog):
    value = local_secret.ensure_local_secret()
    _secret_file().chmod(0o644)
    with caplog.at_level(logging.WARNING, logger="EvoScientist.local_secret"):
        assert local_secret.ensure_local_secret() == value
    assert stat.S_IMODE(_secret_file().stat().st_mode) == 0o600
    assert "readable by other users" in caplog.text
    assert value not in caplog.text


@posix_only
def test_ensure_local_secret_refuses_a_file_system_without_modes(monkeypatch):
    """E.g. a data folder on a FAT drive or WSL's /mnt/c: chmod changes nothing."""
    local_secret.ensure_local_secret()
    _secret_file().chmod(0o644)
    monkeypatch.setattr(local_secret.os, "chmod", lambda *_a: None)
    with pytest.raises(PermissionError, match="ignores permissions"):
        local_secret.ensure_local_secret()


@posix_only
def test_ensure_local_secret_refuses_a_file_of_another_user(monkeypatch):
    local_secret.ensure_local_secret()
    real_stat = os.stat

    def file_owned_by_someone_else(path, *args, **kwargs):
        st = real_stat(path, *args, **kwargs)
        if os.fspath(path) != os.fspath(_secret_file()):
            return st
        return SimpleNamespace(st_uid=st.st_uid + 1, st_mode=st.st_mode)

    monkeypatch.setattr(local_secret.os, "stat", file_owned_by_someone_else)
    with pytest.raises(PermissionError, match="belongs to another user"):
        local_secret.ensure_local_secret()


def _write_secret_file(content: bytes) -> None:
    paths.SERVER_HOME.mkdir(mode=0o700, parents=True)
    _secret_file().write_bytes(content)
    if os.name == "posix":
        _secret_file().chmod(0o600)


@pytest.mark.parametrize("content", [b"", b"  \n", "café".encode()])
def test_ensure_local_secret_refuses_a_file_without_a_secret(content):
    _write_secret_file(content)
    with pytest.raises(OSError, match="does not hold a local secret"):
        local_secret.ensure_local_secret()


def test_ensure_local_secret_removes_the_temp_file_when_writing_fails(monkeypatch):
    class FullDisk:
        def __init__(self, fd, *_a, **_kw):
            os.close(fd)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def write(self, _text):
            raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(local_secret.os, "fdopen", FullDisk)
    with pytest.raises(OSError, match="No space"):
        local_secret.ensure_local_secret()
    assert list(paths.SERVER_HOME.iterdir()) == []


def test_ensure_local_secret_restricts_the_empty_temp_file_on_windows(monkeypatch):
    """The access list is set before the value is written, and set again on an
    existing file at every start."""
    restricted = []

    def record(path, *, existing=False):
        restricted.append((path.name, path.stat().st_size, existing))

    monkeypatch.setattr(local_secret.sys, "platform", "win32")
    monkeypatch.setattr(local_secret, "_restrict_to_owner", record)
    value = local_secret.ensure_local_secret()
    local_secret.ensure_local_secret()
    (temp_name, temp_size, temp_existing), existing = restricted
    assert temp_name.startswith(".local-secret.")
    assert (temp_size, temp_existing) == (0, False)
    assert existing == ("local-secret", len(value), True)


def test_ensure_local_secret_leaves_no_file_when_restricting_fails_on_windows(
    monkeypatch,
):
    def fail(_path, **_kw):
        raise OSError("icacls failed: Access is denied.")

    monkeypatch.setattr(local_secret.sys, "platform", "win32")
    monkeypatch.setattr(local_secret, "_restrict_to_owner", fail)
    with pytest.raises(OSError, match="Access is denied"):
        local_secret.ensure_local_secret()
    assert list(paths.SERVER_HOME.iterdir()) == []


def test_open_new_retries_a_briefly_locked_file_on_windows(tmp_path, monkeypatch):
    real_open = os.open
    calls = []

    def locked_twice(path, flags, mode=0o777):
        calls.append(path)
        if len(calls) <= 2:
            raise PermissionError(13, "in use by another process")
        return real_open(path, flags, mode)

    monkeypatch.setattr(local_secret.sys, "platform", "win32")
    monkeypatch.setattr(local_secret.os, "open", locked_twice)
    monkeypatch.setattr(local_secret.time, "sleep", lambda _s: None)
    os.close(local_secret._open_new(tmp_path / "f"))
    assert len(calls) == 3


@pytest.fixture
def server(monkeypatch, caplog):
    """The custom app with its lifespan run, as langgraph-api runs it."""
    monkeypatch.setattr(local_secret, "_value", None)
    caplog.set_level(logging.DEBUG)
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as client:
        yield client


def test_verify_route_accepts_the_secret_from_the_file(server):
    value = _secret_file().read_text()
    response = server.get(
        "/api/local-secret/verify", headers={local_secret.HEADER: value}
    )
    assert response.status_code == 204


@pytest.mark.parametrize("headers", [{}, {local_secret.HEADER: "wrong"}])
def test_verify_route_refuses_without_the_secret(server, headers):
    """Also from 127.0.0.1: the address is no shortcut."""
    response = server.get("/api/local-secret/verify", headers=headers)
    assert response.status_code == 403
    assert response.json() == {"error": "local_secret_required"}


@pytest.mark.parametrize("content", [b"", "café".encode()])
def test_verify_route_refuses_when_the_file_is_unusable(content, monkeypatch, caplog):
    """The server still starts; only routes that need the secret refuse."""
    _write_secret_file(content)
    monkeypatch.setattr(local_secret, "_value", None)
    with TestClient(app, base_url="http://localhost") as client:
        response = client.get(
            "/api/local-secret/verify", headers={local_secret.HEADER: ""}
        )
    assert response.status_code == 403
    assert "Local secret unavailable" in caplog.text


def test_local_secret_never_reaches_logs_or_responses(server, caplog):
    value = _secret_file().read_text()
    responses = [
        server.get("/api/local-secret/verify", headers={local_secret.HEADER: value}),
        server.get("/api/local-secret/verify"),
    ]
    assert value not in caplog.text
    assert all(value not in r.text for r in responses)
    assert all(value not in json.dumps(dict(r.headers)) for r in responses)


def test_restrict_to_owner_grants_this_user_not_owner_rights(tmp_path, monkeypatch):
    """A file created from an elevated prompt is owned by Administrators;
    OWNER RIGHTS would then shut this user's own programs out."""
    calls = []
    monkeypatch.setattr(
        local_secret, "_current_user_sid", lambda: "S-1-5-21-1-2-3-1001"
    )
    monkeypatch.setattr(local_secret, "_run_system_tool", lambda *a: calls.append(a))
    local_secret._restrict_to_owner(tmp_path / "f", existing=True)
    reset, args = calls
    assert reset == ("icacls", str(tmp_path / "f"), "/reset")
    assert "/inheritance:r" in args
    assert "*S-1-5-21-1-2-3-1001:F" in args
    assert not any("S-1-3-4" in a for a in args)


def test_current_user_sid_reads_whoami(monkeypatch):
    monkeypatch.setattr(
        local_secret,
        "_run_system_tool",
        lambda *_a: '"host\\Müller","S-1-5-21-1-2-3-1001"\r\n',
    )
    local_secret._current_user_sid.cache_clear()
    try:
        assert local_secret._current_user_sid() == "S-1-5-21-1-2-3-1001"
    finally:
        local_secret._current_user_sid.cache_clear()


def test_run_system_tool_failure_is_an_os_error(monkeypatch):
    """So the server refuses the secret instead of failing to start."""

    def fail(cmd, **_kw):
        raise subprocess.CalledProcessError(
            5, cmd, output="", stderr="Access is denied."
        )

    monkeypatch.setattr(local_secret.subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    with pytest.raises(OSError, match="Access is denied"):
        local_secret._run_system_tool("icacls", "x")


windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows access lists"
)


def _sddl(path):
    """The file's access list as SDDL (SIDs, not localised names)."""
    saved = path.with_name(path.name + ".acl")
    subprocess.run(["icacls", str(path), "/save", str(saved)], check=True)
    try:
        # UTF-16 LE, without a byte-order mark (drop one if a version adds it).
        text = saved.read_bytes().decode("utf-16-le").lstrip("\ufeff")
        return text.splitlines()[1]
    finally:
        saved.unlink()


def _ace_sids(sddl):
    return {
        ace.split(";")[-1] for ace in sddl.split("D:", 1)[1].strip("()").split(")(")
    }


@windows_only
def test_ensure_local_secret_restricts_the_file_on_windows():
    local_secret.ensure_local_secret()
    sddl = _sddl(_secret_file())
    assert sddl.split("D:", 1)[1].startswith("P")  # nothing inherited
    assert ";ID;" not in sddl
    assert _ace_sids(sddl) == {local_secret._current_user_sid(), "SY", "BA"}


@windows_only
def test_ensure_local_secret_tightens_a_file_others_can_read_on_windows():
    value = local_secret.ensure_local_secret()
    subprocess.run(
        ["icacls", str(_secret_file()), "/grant", "*S-1-5-32-545:R"], check=True
    )
    assert "BU" in _ace_sids(_sddl(_secret_file()))
    assert local_secret.ensure_local_secret() == value
    assert "BU" not in _ace_sids(_sddl(_secret_file()))
