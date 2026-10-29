"""Download helpers shared by the setup stages.

Downloads use ``urllib``'s proxy handling: it honours ``HTTPS_PROXY`` (and, on
Windows and macOS, the system proxy settings) and verifies TLS with the default
SSL context, which honours ``SSL_CERT_FILE`` and on Windows also loads the
system certificate store. That is deliberate and differs from the loopback
probes elsewhere, which bypass proxies.

Two time limits apply to every request, and none to its total time:

- the connect budget covers connecting to the host's addresses in turn (each
  gets a share of it, so one that drops packets does not use it all up), the
  proxy's ``CONNECT`` tunnel and the TLS handshake, all together, from when
  the name is resolved (name resolution is not bounded; a read in the tunnel
  or handshake waits at most what is left of the budget when it starts);
- the no-data timeout covers each wait for the next bytes afterwards.

A failure before the connection is up (no route, refused, the budget spent, a
proxy that cannot reach the host or answers its tunnel request with garbage,
a TLS error) is the same on every try, so it fails at once. One after it is up
(an HTTP 5xx, a reset, a stall, a truncated body) is retried a few times. A
retry of a download that had received data asks only for the rest (an HTTP
``Range`` request): a slow proxy may cut a long download near its end, and
each full restart would load the link for as long again. Such a retry is
retried also when it cannot connect, since the cut that preceded it is the
likely cause (a VPN or Wi-Fi reconnecting). The callers check the whole file
against a pinned or published hash afterwards.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import itertools
import logging
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from .protocol import CN_MIRROR_HINT, StageError

logger = logging.getLogger(__name__)

T = TypeVar("T")

_CHUNK = 256 * 1024
# A truncated chunked body (IncompleteRead) or a garbled proxy reply
# (BadStatusLine) raises http.client errors, which are neither URLError nor
# OSError.
_NETWORK_ERRORS = (
    urllib.error.URLError,
    http.client.HTTPException,
    OSError,
    ValueError,
)
# Seconds for the whole connect phase: a working tunnel and TLS handshake take
# well under a second, and a proxy's own error page arrives within about 10 s.
CONNECT_BUDGET = 15.0
# Seconds each resolved address gets at least, out of what is left of the
# budget.
_MIN_ADDRESS_SLICE = 3.0
# Seconds an archive download may receive nothing (no limit on its total time).
_ARCHIVE_TIMEOUT = 120
# Pauses before the 2nd, 3rd and 4th attempt.
_RETRY_PAUSES = (1.0, 2.0, 4.0)

ProgressFn = Callable[[float], None]


class NotConnected(Exception):
    """No connection to the host within the connect budget; ``__cause__`` says
    why. ``proxy`` is set when the proxy itself could not be reached."""

    def __init__(
        self, host: str, cause: BaseException, *, proxy: str | None = None
    ) -> None:
        if isinstance(cause, TimeoutError):
            reason = f"no answer within {CONNECT_BUDGET:g} s"
        else:
            reason = str(cause) or type(cause).__name__
        self.proxy = proxy
        target = f"the proxy {proxy} (for {host})" if proxy else host
        super().__init__(f"Could not connect to {target}: {reason}")


class _Truncated(Exception):
    """The body ended before its length, or a resume came back at another
    offset."""


class _ResumeNotConnected(Exception):
    """A retry that would resume a download could not connect; retried, unlike
    :class:`NotConnected` on the first attempt."""


def _connect_within(
    resolved: list[tuple[Any, ...]],
    source_address: tuple[str, int] | None = None,
    *,
    deadline: float,
) -> socket.socket:
    """``socket.create_connection`` over the ``getaddrinfo`` result
    ``resolved``, with one deadline for all addresses.

    The stdlib version gives every resolved address the full timeout, so a
    host with many addresses on a network that drops packets takes that many
    timeouts. Here each address gets an equal share of what is left, at least
    :data:`_MIN_ADDRESS_SLICE`, so a first address that drops packets (a broken
    IPv6 route) leaves time for the others. The address families take turns
    (RFC 8305), so a family with many such addresses cannot use up the budget
    before the other one is tried.
    """
    error: OSError | None = None
    by_family: dict[int, list] = {}
    for entry in resolved:
        by_family.setdefault(entry[0], []).append(entry)
    addresses = [
        entry
        for turn in itertools.zip_longest(*by_family.values())
        for entry in turn
        if entry is not None
    ]
    for index, (family, kind, proto, _name, sockaddr) in enumerate(addresses):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        share = max(remaining / (len(addresses) - index), _MIN_ADDRESS_SLICE)
        sock = None
        try:
            sock = socket.socket(family, kind, proto)
            sock.settimeout(min(remaining, share))
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
        except OSError as exc:
            error = exc
            if sock is not None:
                sock.close()
            continue
        # The tunnel and the TLS handshake get what is left of the budget.
        sock.settimeout(max(deadline - time.monotonic(), 1.0))
        return sock
    raise error if error is not None else TimeoutError("timed out")


class _BudgetedConnection:
    """Connect within :data:`CONNECT_BUDGET`, counted from when the name is
    resolved (the lookup itself is not bounded), then read with ``timeout``.

    The proxy's ``CONNECT`` reply and the TLS handshake each start with what
    is left of the budget; a single read in them may still wait that long.
    """

    host: str
    port: int
    timeout: float | None
    sock: socket.socket | None
    _create_connection: Callable[..., socket.socket]
    _tunnel_host: str | None
    _deadline: float

    def _left(self) -> float:
        """Seconds left of the connect budget; TimeoutError once it is spent."""
        left = self._deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("timed out")
        return left

    def connect(self) -> None:
        read_timeout = self.timeout

        def create(address, timeout, source_address=None):
            try:
                resolved = socket.getaddrinfo(*address, 0, socket.SOCK_STREAM)
                self._deadline = time.monotonic() + CONNECT_BUDGET
                return _connect_within(
                    resolved, source_address, deadline=self._deadline
                )
            except OSError as exc:
                if not self._tunnel_host:
                    raise
                proxy = f"{self.host}:{self.port}"
                raise NotConnected(self._tunnel_host, exc, proxy=proxy) from exc

        self._create_connection = create
        try:
            super().connect()  # type: ignore[misc]
        except (OSError, http.client.HTTPException) as exc:
            # Includes a refused proxy tunnel ("Tunnel connection failed: 502"),
            # a garbled tunnel reply (BadStatusLine) and TLS errors.
            target = self.host
            if self._tunnel_host:
                target = (
                    f"{self._tunnel_host} through the proxy {self.host}:{self.port}"
                )
            raise NotConnected(target, exc) from exc
        assert self.sock is not None
        self.sock.settimeout(read_timeout)

    def _tunnel(self) -> None:
        assert self.sock is not None
        self.sock.settimeout(self._left())
        super()._tunnel()  # type: ignore[misc]
        # The TLS handshake that follows gets what is left after the tunnel.
        self.sock.settimeout(self._left())


class _HTTPConnection(_BudgetedConnection, http.client.HTTPConnection):
    pass


class _HTTPSConnection(_BudgetedConnection, http.client.HTTPSConnection):
    pass


class _HTTPHandler(urllib.request.HTTPHandler):
    def do_open(self, http_class, req, **kwargs):
        return super().do_open(_HTTPConnection, req, **kwargs)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def do_open(self, http_class, req, **kwargs):
        return super().do_open(_HTTPSConnection, req, **kwargs)


def open_url(request: str | urllib.request.Request, timeout: float):
    """``urlopen`` with the connect budget; ``timeout`` is the no-data timeout.

    Raises :class:`NotConnected` before the connection is up, ``urllib`` and
    ``http.client`` errors after it.
    """
    opener = urllib.request.build_opener(_HTTPHandler, _HTTPSHandler)
    return opener.open(request, timeout=timeout)


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500
    if isinstance(exc, NotConnected | ValueError):
        return False
    return isinstance(
        exc,
        _Truncated
        | _ResumeNotConnected
        | urllib.error.URLError
        | http.client.HTTPException
        | OSError,
    )


def with_retries(attempt: Callable[[], T], what: str) -> T:
    """Run ``attempt``; retry a failure after the connection was up.

    At most ``len(_RETRY_PAUSES)`` retries. Anything else, and the last
    failure, is raised as it is; the caller turns it into a stage error.
    """
    for pause in (*_RETRY_PAUSES, None):
        try:
            return attempt()
        except Exception as exc:
            if pause is None or not _retryable(exc):
                raise
            reason = str(exc).strip() or type(exc).__name__
            logger.warning(f"{what} failed ({reason}); retrying in {pause:g} s.")
            time.sleep(pause)
    raise AssertionError("unreachable")


def with_hint(message: str, cn_hint: bool) -> str:
    """``message``, plus the ``--cn`` hint when the caller asks for it."""
    return f"{message} {CN_MIRROR_HINT}" if cn_hint else message


@contextlib.contextmanager
def _local_io(dest: Path) -> Iterator[None]:
    """Report a failed write to ``dest`` as ``install_failed``, not as a network error."""
    try:
        yield
    except OSError as exc:
        raise StageError("install_failed", f"Could not write {dest}: {exc}") from exc


@dataclass
class _Partial:
    """What earlier attempts of one download left in the file."""

    digest: Any = field(default_factory=hashlib.sha256)
    done: int = 0
    total: int = 0
    # The ETag or Last-Modified of the first reply, sent back as If-Range so a
    # file changed in between comes back whole instead of being spliced.
    validator: str | None = None

    def restart(self, resp) -> None:
        """Forget earlier attempts; ``resp`` is a reply with the whole file."""
        self.digest = hashlib.sha256()
        self.done = 0
        self.total = int(resp.headers.get("Content-Length") or 0)
        etag = resp.headers.get("ETag")
        # A weak ETag may not be sent in If-Range.
        if etag and not etag.startswith("W/"):
            self.validator = etag
        else:
            self.validator = resp.headers.get("Last-Modified")


def _resume_offset(resp, partial: _Partial) -> int:
    """Where ``resp``'s body starts: ``partial.done`` for a matching ``206``,
    else 0 (the server sent the whole file). When the first reply had no
    ``Content-Length``, a ``206`` with a numeric size sets ``partial.total``,
    so a cut resume is caught."""
    if not partial.done or resp.status != 206:
        return 0
    unit, _, spec = resp.headers.get("Content-Range", "").partition(" ")
    span, _, size = spec.partition("/")
    start, _, _end = span.partition("-")
    if (
        unit != "bytes"
        or start != str(partial.done)
        # Without a Content-Length on the first reply, any size will do.
        or (partial.total and size not in (str(partial.total), "*"))
    ):
        raise _Truncated(f"the server resumed with Content-Range {spec!r}")
    # A "*" size gives no total: the range may end before the file does.
    if not partial.total and size.isdecimal():
        partial.total = int(size)
    return partial.done


def _download_once(
    url: str, dest: Path, progress: ProgressFn | None, partial: _Partial
) -> str:
    request = urllib.request.Request(url)
    if partial.done:
        request.add_header("Range", f"bytes={partial.done}-")
        if partial.validator:
            request.add_header("If-Range", partial.validator)
    try:
        opened = open_url(request, _ARCHIVE_TIMEOUT)
    except NotConnected as exc:
        if not partial.done:
            raise
        raise _ResumeNotConnected(str(exc)) from exc
    with opened as resp:
        try:
            offset = _resume_offset(resp, partial)
        except _Truncated:
            # The next attempt asks for the whole file.
            partial.done = 0
            raise
        if offset == 0:
            partial.restart(resp)
        with _local_io(dest):
            fh = open(dest, "r+b" if offset else "wb")
            # Drop whatever an earlier attempt wrote past what was hashed.
            fh.seek(offset)
            fh.truncate()
        length = resp.headers.get("Content-Length") or ""
        start = partial.done
        try:
            while chunk := resp.read(_CHUNK):
                with _local_io(dest):
                    fh.write(chunk)
                partial.digest.update(chunk)
                partial.done += len(chunk)
                if progress is not None and partial.total:
                    progress(partial.done / partial.total)
        except BaseException:
            with contextlib.suppress(OSError):
                fh.close()
            raise
        # Closing writes out what is still buffered, so it can fail like a write.
        with _local_io(dest):
            fh.close()
        # http.client's read(amt) returns b"" at an early end of stream
        # instead of raising IncompleteRead. Each reply is held to its own
        # length too: a 206 of unknown size ("*") sets no total.
        if length.isdecimal() and partial.done - start != int(length):
            raise _Truncated(
                f"the reply ended after {partial.done - start} of {length} bytes"
            )
        if partial.total and partial.done != partial.total:
            raise _Truncated(
                f"the body ended after {partial.done} of {partial.total} bytes"
            )
    return partial.digest.hexdigest()


def download(
    url: str, dest: Path, progress: ProgressFn | None = None, *, cn_hint: bool = False
) -> str:
    """Stream ``url`` into ``dest`` and return the file's SHA-256 hex digest.

    ``progress`` receives the downloaded fraction when the server sends a
    ``Content-Length``. Network and HTTP errors raise
    ``StageError("download_failed")``, with the ``--cn`` hint when
    ``cn_hint``; a failed write to ``dest`` raises
    ``StageError("install_failed")``.
    """
    host = urllib.parse.urlsplit(url).hostname or url
    partial = _Partial()
    try:
        return with_retries(
            lambda: _download_once(url, dest, progress, partial),
            f"Download from {host}",
        )
    except NotConnected as exc:
        raise StageError(
            "download_failed", with_hint(f"{exc}.", cn_hint and not exc.proxy)
        ) from exc
    except (_Truncated, _ResumeNotConnected, *_NETWORK_ERRORS) as exc:
        # No --cn hint when the proxy is down: the mirror goes through it too.
        proxy_down = getattr(exc.__cause__, "proxy", None)
        raise StageError(
            "download_failed",
            with_hint(f"Download of {url} failed: {exc}.", cn_hint and not proxy_down),
        ) from exc
