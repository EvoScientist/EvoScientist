"""The POSIX lock tests for #785: the reviewer's third finding.

The lock bound a Linux abstract-namespace address on every POSIX platform
and retried every error forever. On a platform without that namespace the
bind fails permanently, so every submit, approve, reject, revert, move and
upgrade hung instead of taking the lock.

The tests drive the retry logic through an injected socket, so they run
wherever the suite runs; the real socket is only exercised where the
platform has one.
"""

from __future__ import annotations

import errno
import socket
from pathlib import Path

import pytest

from EvoScientist.memory.autoskills import proposals


class _FakeSocket:
    """A stand-in for a POSIX socket that records what it was asked to do."""

    def __init__(self, state: dict, family: int, kind: int) -> None:
        self._state = state
        self._file = None
        self.address = None
        self.closed = False

    def bind(self, address):
        self._state["calls"].append(address)
        if self._state["fail"]:
            raise OSError(self._state["fail"].pop(0), "no")
        if isinstance(address, str):
            # A real file at the path, so the inode check has something
            # truthful to compare.
            self._file = open(address, "wb")
        self.address = address

    def connect(self, address):
        raise OSError(errno.ECONNREFUSED, "no")

    def fileno(self):
        assert self._file is not None, "the socket was not bound to a path"
        return self._file.fileno()

    def settimeout(self, value):
        pass

    def close(self):
        self.closed = True
        if self._file is not None and not self._file.closed:
            self._file.close()
        self._file = None


@pytest.fixture
def fake_sockets(monkeypatch, tmp_path):
    """Replace the POSIX socket with one the test can steer."""
    state = {"calls": [], "fail": []}
    locks = tmp_path / "locks"
    locks.mkdir()
    monkeypatch.setattr(socket, "socket", lambda *a, **k: _FakeSocket(state, *a))
    monkeypatch.setattr(socket, "AF_UNIX", 1, raising=False)
    monkeypatch.setattr(proposals, "_lock_directory", lambda: locks)
    return state


def _proposal_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "draft"
    folder.mkdir(exist_ok=True)
    return folder


def test_a_setup_failure_is_not_retried_forever(fake_sockets, tmp_path):
    """A permanent bind failure raises instead of hanging the caller."""
    fake_sockets["fail"] = [errno.EINVAL] * 50

    with pytest.raises(OSError, match=r"\[Errno 22\]") as raised:
        proposals._ProposalLock(_proposal_dir(tmp_path))._acquire_posix()

    assert raised.value.errno == errno.EINVAL
    # The permanent failure was tried once on the abstract address and once
    # on the fallback path. It was not retried.
    assert len(fake_sockets["calls"]) <= 3, fake_sockets["calls"]


def test_contention_is_retried(fake_sockets, tmp_path):
    """Another holder of the same draft is waited for, not refused."""
    fake_sockets["fail"] = [errno.EADDRINUSE, errno.EADDRINUSE]

    held = proposals._ProposalLock(_proposal_dir(tmp_path))._acquire_posix()

    assert len(fake_sockets["calls"]) == 3, fake_sockets["calls"]
    held.close()


def test_the_abstract_namespace_is_used_when_it_binds(fake_sockets, tmp_path):
    """Where the platform has the namespace, the lock has no pathname."""
    proposals._ProposalLock(_proposal_dir(tmp_path))._acquire_posix()

    assert fake_sockets["calls"][0].startswith(b"\0")


def test_a_platform_without_the_abstract_namespace_falls_back(fake_sockets, tmp_path):
    """Where the namespace is missing, the lock takes a path instead."""
    fake_sockets["fail"] = [errno.EAFNOSUPPORT]

    proposals._ProposalLock(_proposal_dir(tmp_path))._acquire_posix()

    assert len(fake_sockets["calls"]) == 2, fake_sockets["calls"]
    assert isinstance(fake_sockets["calls"][1], str)
    assert fake_sockets["calls"][1].endswith(".sock")


def test_a_replaced_fallback_name_is_refused(fake_sockets, tmp_path, monkeypatch):
    """A name taken over after the bind is not trusted as this lock."""
    lock = proposals._ProposalLock(_proposal_dir(tmp_path))
    fake_sockets["fail"] = [errno.EAFNOSUPPORT]
    monkeypatch.setattr(proposals, "_socket_is_ours", lambda path, held: False)

    with pytest.raises(OSError, match="replaced"):
        lock._acquire_posix()

    # The replaced name was retried a bounded number of times, not forever.
    assert len(fake_sockets["calls"]) <= 8, fake_sockets["calls"]


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="this platform has no POSIX socket"
)
def test_the_real_socket_is_taken_and_released(tmp_path):
    """The lock this platform uses is really taken, and really released."""
    folder = _proposal_dir(tmp_path)

    held = proposals._ProposalLock(folder)._acquire_posix()
    try:
        assert isinstance(held, socket.socket)
    finally:
        held.close()


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="this platform has no POSIX socket"
)
def test_a_dead_holder_leaves_a_socket_the_next_acquirer_clears(tmp_path, monkeypatch):
    """A holder that died leaves a file; the next acquirer clears it.

    The file exists only where the platform has no abstract namespace, so the
    abstract bind is refused here and the fallback is what runs.
    """
    directory = tmp_path / "locks"
    directory.mkdir()
    monkeypatch.setattr(proposals, "_lock_directory", lambda: directory)

    real_bind = socket.socket.bind

    def refuse_abstract(sock, address):
        if isinstance(address, bytes) and address.startswith(b"\0"):
            raise OSError(errno.EAFNOSUPPORT, "no abstract namespace")
        return real_bind(sock, address)

    monkeypatch.setattr(socket.socket, "bind", refuse_abstract)

    folder = _proposal_dir(tmp_path)
    lock = proposals._ProposalLock(folder)
    held = lock._acquire_posix()
    path = directory / f"evoskill-{lock._name}.sock"
    assert path.exists()
    held.close()

    again = proposals._ProposalLock(folder)
    second = again._acquire_posix()
    try:
        assert isinstance(second, socket.socket)
    finally:
        second.close()


def test_a_long_home_directory_still_yields_a_bindable_path(tmp_path, monkeypatch):
    """A home directory too long for a socket path still gets a lock.

    The socket's path has to fit in sun_path: 107 bytes on Linux, 103 on
    macOS. A home directory long enough to push it past that made every
    bind fail with "AF_UNIX path too long" -- on the CI runner, and on
    every macOS run, because macOS has no abstract namespace at all.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / ("x" * 200)))

    path = proposals._socket_path("0123456789abcdef0123456789abcdef")

    assert len(str(path).encode("utf-8")) <= proposals._SUN_PATH_MAX


def test_the_private_directory_is_used_when_the_path_fits(tmp_path, monkeypatch):
    """The private directory is preferred over the temporary one.

    The base is a relative path, so the test does not depend on how long
    this machine's temporary directory happens to be.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(proposals, "_lock_base", lambda: Path("evtest"))

    path = proposals._socket_path("0123456789abcdef0123456789abcdef")

    assert path.parent == Path("evtest")
    assert path.name == "evoskill-0123456789abcdef0123456789abcdef.sock"


def test_the_fallback_directory_is_private(tmp_path, monkeypatch):
    """The fallback lives in a directory under this user's own home."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    directory = proposals._lock_directory()

    assert directory.is_dir()
    assert directory == tmp_path / ".cache" / "evoscientist" / "locks"
    assert directory.parent.name == "evoscientist"
