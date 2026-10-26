"""Tests for EvoScientist.setup.download: time limits, retries, error mapping.

Requests go to a real HTTP server on 127.0.0.1, so the connection class and
urllib's handlers run as they do for a real download.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import socket
import threading
import time
from collections.abc import Callable

import pytest

from EvoScientist.setup import download as dl
from EvoScientist.setup.protocol import CN_MIRROR_HINT, StageError

BODY = b"0123456789" * 1000

Reply = Callable[[http.server.BaseHTTPRequestHandler], None]


def ok(handler) -> None:
    handler.send_response(200)
    handler.send_header("Content-Length", str(len(BODY)))
    handler.end_headers()
    handler.wfile.write(BODY)


def status(code: int) -> Reply:
    def reply(handler) -> None:
        handler.send_response(code)
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    return reply


def truncated(handler) -> None:
    handler.send_response(200)
    handler.send_header("Content-Length", str(len(BODY)))
    handler.send_header("ETag", '"v1"')
    handler.end_headers()
    handler.wfile.write(BODY[:100])
    handler.close_connection = True


def ranged(handler) -> None:
    """Honour a ``Range: bytes=<start>-`` request, else send the whole body."""
    asked = handler.headers.get("Range", "")
    if not asked.startswith("bytes="):
        ok(handler)
        return
    start = int(asked.removeprefix("bytes=").rstrip("-"))
    handler.send_response(206)
    handler.send_header("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")
    handler.send_header("Content-Length", str(len(BODY) - start))
    handler.end_headers()
    handler.wfile.write(BODY[start:])


def wrong_range(handler) -> None:
    """A ``206`` that does not start where it was asked to."""
    handler.send_response(206)
    handler.send_header("Content-Range", f"bytes 0-{len(BODY) - 1}/{len(BODY)}")
    handler.send_header("Content-Length", str(len(BODY)))
    handler.end_headers()
    handler.wfile.write(BODY)


def stall(seconds: float) -> Reply:
    def reply(handler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Length", str(len(BODY)))
        handler.end_headers()
        handler.wfile.write(BODY[:100])
        handler.wfile.flush()
        time.sleep(seconds)
        handler.close_connection = True

    return reply


def trickle(gap: float, pieces: int = 3) -> Reply:
    """The whole body, in ``pieces`` parts ``gap`` seconds apart."""

    def reply(handler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Length", str(len(BODY)))
        handler.end_headers()
        step = len(BODY) // pieces + 1
        for start in range(0, len(BODY), step):
            handler.wfile.write(BODY[start : start + step])
            handler.wfile.flush()
            time.sleep(gap)

    return reply


class Server:
    """Answers each GET with the next scripted reply (the last one repeats)."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.requests = 0
        self.headers: list[dict[str, str]] = []
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                index = min(server.requests, len(server.replies) - 1)
                server.requests += 1
                server.headers.append(dict(self.headers))
                server.replies[index](self)

            def log_message(self, *_args) -> None:
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/node.tar.xz"
        threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(autouse=True)
def direct(monkeypatch):
    """No proxy from the environment (or, on Windows, the registry), no pauses."""
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    # A non-empty proxy environment keeps urllib from reading the registry.
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setattr(dl, "_RETRY_PAUSES", (0, 0, 0))


@pytest.fixture
def serve():
    servers: list[Server] = []

    def start(*replies: Reply) -> Server:
        servers.append(Server(*replies))
        return servers[-1]

    yield start
    for server in servers:
        server.close()


def test_download_returns_the_digest_and_reports_progress(serve, tmp_path):
    server = serve(ok)
    seen: list[float] = []
    digest = dl.download(server.url, tmp_path / "a", seen.append)
    assert digest == hashlib.sha256(BODY).hexdigest()
    assert (tmp_path / "a").read_bytes() == BODY
    assert seen[-1] == 1.0


def test_server_error_is_retried_until_it_succeeds(serve, tmp_path):
    server = serve(status(503), status(502), ok)
    dl.download(server.url, tmp_path / "a")
    assert server.requests == 3
    assert (tmp_path / "a").read_bytes() == BODY


def test_server_error_gives_up_after_three_retries(serve, tmp_path):
    server = serve(status(500))
    with pytest.raises(StageError) as ei:
        dl.download(server.url, tmp_path / "a", cn_hint=True)
    assert server.requests == 4
    assert ei.value.code == "download_failed"
    assert "500" in ei.value.message
    assert ei.value.message.count(CN_MIRROR_HINT) == 1


@pytest.mark.parametrize("code", [403, 404, 407])
def test_client_error_is_not_retried(serve, tmp_path, code):
    server = serve(status(code))
    with pytest.raises(StageError) as ei:
        dl.download(server.url, tmp_path / "a")
    assert server.requests == 1
    assert ei.value.code == "download_failed"


def test_truncated_body_is_resumed_from_where_it_ended(serve, tmp_path):
    server = serve(truncated, ranged)
    digest = dl.download(server.url, tmp_path / "a")
    assert server.requests == 2
    assert server.headers[1]["Range"] == "bytes=100-"
    assert server.headers[1]["If-Range"] == '"v1"'
    assert (tmp_path / "a").read_bytes() == BODY
    assert digest == hashlib.sha256(BODY).hexdigest()


def test_server_that_ignores_range_restarts_from_an_empty_file(serve, tmp_path):
    server = serve(truncated, ok)
    digest = dl.download(server.url, tmp_path / "a")
    assert server.requests == 2
    assert (tmp_path / "a").read_bytes() == BODY
    assert digest == hashlib.sha256(BODY).hexdigest()


def chunked_then_cut(handler) -> None:
    """A chunked body (no Content-Length) that ends after its first chunk."""
    handler.send_response(200)
    handler.send_header("Transfer-Encoding", "chunked")
    handler.send_header("ETag", '"v1"')
    handler.end_headers()
    handler.wfile.write(b"%x\r\n%s\r\n" % (100, BODY[:100]))
    handler.close_connection = True


def test_resume_without_a_content_length_accepts_the_size(serve, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_CHUNK", 100)
    server = serve(chunked_then_cut, ranged)
    digest = dl.download(server.url, tmp_path / "a")
    assert server.requests == 2
    assert server.headers[1]["Range"] == "bytes=100-"
    assert digest == hashlib.sha256(BODY).hexdigest()


def ranged_then_cut(size: str, *, length: bool) -> Reply:
    """A ``206`` for the asked range, of a file of ``size`` bytes (``*``:
    unknown), that ends after 100 bytes; with or without a Content-Length."""

    def reply(handler) -> None:
        start = int(handler.headers["Range"].removeprefix("bytes=").rstrip("-"))
        handler.send_response(206)
        handler.send_header("Content-Range", f"bytes {start}-{len(BODY) - 1}/{size}")
        if length:
            handler.send_header("Content-Length", str(len(BODY) - start))
        handler.end_headers()
        handler.wfile.write(BODY[start : start + 100])
        handler.close_connection = True

    return reply


@pytest.mark.parametrize(
    ("size", "length"),
    [(str(len(BODY)), True), (str(len(BODY)), False), ("*", True)],
)
def test_cut_resume_without_a_content_length_is_retried(
    serve, tmp_path, monkeypatch, size, length
):
    monkeypatch.setattr(dl, "_CHUNK", 100)
    server = serve(chunked_then_cut, ranged_then_cut(size, length=length), ranged)
    digest = dl.download(server.url, tmp_path / "a")
    assert server.requests == 3
    assert server.headers[2]["Range"] == "bytes=200-"
    assert (tmp_path / "a").read_bytes() == BODY
    assert digest == hashlib.sha256(BODY).hexdigest()


def test_cut_resume_of_unknown_size_and_length_cannot_be_told(
    serve, tmp_path, monkeypatch
):
    """Known limit: a ``206`` with a ``*`` size and no Content-Length gives no
    length to hold it to, so its cut ends the download; the caller's hash
    check then refuses the file."""
    monkeypatch.setattr(dl, "_CHUNK", 100)
    server = serve(chunked_then_cut, ranged_then_cut("*", length=False), ranged)
    digest = dl.download(server.url, tmp_path / "a")
    assert server.requests == 2
    assert digest != hashlib.sha256(BODY).hexdigest()


def ranged_part_of_unknown_size(handler) -> None:
    """Like :func:`ranged_part`, with a ``*`` size."""
    start = int(handler.headers["Range"].removeprefix("bytes=").rstrip("-"))
    handler.send_response(206)
    handler.send_header("Content-Range", f"bytes {start}-{start + 99}/*")
    handler.send_header("Content-Length", "100")
    handler.end_headers()
    handler.wfile.write(BODY[start : start + 100])


def test_a_resume_of_unknown_size_sets_no_total(serve, tmp_path, monkeypatch):
    """``bytes 100-199/*`` says nothing of the file's end: no total is taken
    from it, so its progress is not reported as complete."""
    monkeypatch.setattr(dl, "_CHUNK", 100)
    fractions: list[float] = []
    server = serve(chunked_then_cut, ranged_part_of_unknown_size)
    dl.download(server.url, tmp_path / "a", progress=fractions.append)
    assert server.requests == 2
    assert fractions == []


def ranged_part(handler) -> None:
    """A ``206`` with only the next 100 bytes of the asked range."""
    start = int(handler.headers["Range"].removeprefix("bytes=").rstrip("-"))
    handler.send_response(206)
    handler.send_header("Content-Range", f"bytes {start}-{start + 99}/{len(BODY)}")
    handler.send_header("Content-Length", "100")
    handler.end_headers()
    handler.wfile.write(BODY[start : start + 100])


def test_a_resume_with_part_of_the_rest_goes_on(serve, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_CHUNK", 100)
    server = serve(chunked_then_cut, ranged_part, ranged)
    digest = dl.download(server.url, tmp_path / "a")
    assert server.requests == 3
    assert server.headers[2]["Range"] == "bytes=200-"
    assert digest == hashlib.sha256(BODY).hexdigest()


def test_a_resume_that_cannot_connect_is_retried(serve, tmp_path, monkeypatch):
    """After a cut, a failed reconnect is likely the same blip (a VPN
    reconnecting), not a blocked network."""
    server = serve(truncated, ranged)
    real = dl.open_url
    calls: list[int] = []

    def flaky(request, timeout):
        calls.append(1)
        if len(calls) == 2:
            raise dl.NotConnected("127.0.0.1", ConnectionRefusedError("refused"))
        return real(request, timeout)

    monkeypatch.setattr(dl, "open_url", flaky)
    digest = dl.download(server.url, tmp_path / "a")
    assert len(calls) == 3
    assert server.headers[1]["Range"] == "bytes=100-"
    assert digest == hashlib.sha256(BODY).hexdigest()


def test_garbled_tunnel_reply_fails_at_once(tmp_path, monkeypatch):
    """A proxy that answers CONNECT with garbage never connected."""
    tunnels: list[bytes] = []
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()

    def proxy() -> None:
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            with conn:
                tunnels.append(conn.recv(4096).split(b"\r\n")[0])
                conn.sendall(b"garbage\r\n\r\n")

    threading.Thread(target=proxy, daemon=True).start()
    monkeypatch.delenv("no_proxy")
    monkeypatch.setenv("https_proxy", f"http://127.0.0.1:{listener.getsockname()[1]}")
    try:
        with pytest.raises(StageError) as ei:
            dl.download("https://nodejs.example/node.tar.xz", tmp_path / "a")
    finally:
        listener.close()
    assert len(tunnels) == 1
    assert "Could not connect to nodejs.example" in ei.value.message


def test_resume_at_the_wrong_offset_restarts_the_download(serve, tmp_path):
    server = serve(truncated, wrong_range, ranged)
    digest = dl.download(server.url, tmp_path / "a")
    assert server.requests == 3
    assert "Range" not in server.headers[2]
    assert (tmp_path / "a").read_bytes() == BODY
    assert digest == hashlib.sha256(BODY).hexdigest()


def wrong_total(handler) -> None:
    """A ``206`` at the asked offset, of a file with another size."""
    start = int(handler.headers["Range"].removeprefix("bytes=").rstrip("-"))
    handler.send_response(206)
    handler.send_header(
        "Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY) + 1}"
    )
    handler.send_header("Content-Length", str(len(BODY) - start))
    handler.end_headers()
    handler.wfile.write(BODY[start:])


def test_resume_of_a_file_with_another_size_restarts_the_download(serve, tmp_path):
    server = serve(truncated, wrong_total, ranged)
    digest = dl.download(server.url, tmp_path / "a")
    assert server.requests == 3
    assert "Range" not in server.headers[2]
    assert (tmp_path / "a").read_bytes() == BODY
    assert digest == hashlib.sha256(BODY).hexdigest()


def truncated_with_a_weak_etag(handler) -> None:
    """Like :func:`truncated`, with a weak ETag and a Last-Modified date."""
    handler.send_response(200)
    handler.send_header("Content-Length", str(len(BODY)))
    handler.send_header("ETag", 'W/"v1"')
    handler.send_header("Last-Modified", "Wed, 21 Oct 2015 07:28:00 GMT")
    handler.end_headers()
    handler.wfile.write(BODY[:100])
    handler.close_connection = True


def test_resume_after_a_weak_etag_sends_the_date_in_if_range(serve, tmp_path):
    server = serve(truncated_with_a_weak_etag, ranged)
    dl.download(server.url, tmp_path / "a")
    assert server.headers[1]["If-Range"] == "Wed, 21 Oct 2015 07:28:00 GMT"
    assert (tmp_path / "a").read_bytes() == BODY


def test_stall_past_the_no_data_timeout_is_retried(serve, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_ARCHIVE_TIMEOUT", 0.3)
    server = serve(stall(1.5), ranged)
    dl.download(server.url, tmp_path / "a")
    assert server.requests == 2
    # The 100 bytes never completed a read, so nothing was kept to resume.
    assert "Range" not in server.headers[1]
    assert (tmp_path / "a").read_bytes() == BODY


def test_slow_but_steady_body_outlasts_the_connect_budget(serve, tmp_path, monkeypatch):
    """The connect budget limits connecting only; a slow body is fine as long
    as some data arrives within the no-data timeout."""
    monkeypatch.setattr(dl, "CONNECT_BUDGET", 0.2)
    monkeypatch.setattr(dl, "_ARCHIVE_TIMEOUT", 2)
    server = serve(trickle(0.5))
    dl.download(server.url, tmp_path / "a")
    assert server.requests == 1
    assert (tmp_path / "a").read_bytes() == BODY


def test_name_lookup_is_outside_the_connect_budget(serve, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CONNECT_BUDGET", 0.2)
    server = serve(ok)
    real = socket.getaddrinfo

    def slow_lookup(host, port, *args):
        time.sleep(0.5)
        return real("127.0.0.1", port, *args)

    monkeypatch.setattr(dl.socket, "getaddrinfo", slow_lookup)
    dl.download(server.url.replace("127.0.0.1", "nodejs.example"), tmp_path / "a")
    assert server.requests == 1
    assert (tmp_path / "a").read_bytes() == BODY


def test_refused_connection_fails_at_once(tmp_path, monkeypatch):
    calls: list[tuple] = []
    real = dl._connect_within

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(dl, "_connect_within", counting)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(StageError) as ei:
        dl.download(f"http://127.0.0.1:{port}/x", tmp_path / "a", cn_hint=True)
    assert len(calls) == 1
    assert ei.value.code == "download_failed"
    assert ei.value.message.startswith("Could not connect to 127.0.0.1")
    assert CN_MIRROR_HINT in ei.value.message


def test_proxy_that_cannot_reach_the_host_fails_at_once(tmp_path, monkeypatch):
    """A proxy answers CONNECT with 502 when it cannot reach the host; urllib
    raises an OSError for it, not an HTTPError."""
    tunnels: list[bytes] = []
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()

    def proxy() -> None:
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            with conn:
                tunnels.append(conn.recv(4096).split(b"\r\n")[0])
                conn.sendall(b"HTTP/1.1 502 cannotconnect\r\nContent-Length: 0\r\n\r\n")

    threading.Thread(target=proxy, daemon=True).start()
    monkeypatch.delenv("no_proxy")
    monkeypatch.setenv("https_proxy", f"http://127.0.0.1:{listener.getsockname()[1]}")
    try:
        with pytest.raises(StageError) as ei:
            dl.download("https://nodejs.example/node.tar.xz", tmp_path / "a")
    finally:
        listener.close()
    # Python 3.11 sends HTTP/1.0 here, 3.12 and later HTTP/1.1.
    assert len(tunnels) == 1
    assert tunnels[0].startswith(b"CONNECT nodejs.example:443 HTTP/1.")
    assert ei.value.code == "download_failed"
    assert "Could not connect to nodejs.example" in ei.value.message
    assert "502" in ei.value.message


def test_proxy_that_cannot_be_reached_is_named(tmp_path, monkeypatch):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    monkeypatch.delenv("no_proxy")
    monkeypatch.setenv("https_proxy", f"http://127.0.0.1:{port}")
    with pytest.raises(StageError) as ei:
        dl.download("https://nodejs.example/node.tar.xz", tmp_path / "a", cn_hint=True)
    assert ei.value.message.startswith(
        f"Could not connect to the proxy 127.0.0.1:{port} (for nodejs.example): "
    )
    assert CN_MIRROR_HINT not in ei.value.message


def test_a_resume_through_a_proxy_that_went_down_names_no_mirror(
    serve, tmp_path, monkeypatch
):
    server = serve(truncated)
    opened = dl.open_url

    def open_url(request, timeout):
        if server.requests:
            raise dl.NotConnected(
                "nodejs.example", ConnectionRefusedError(), proxy="p:1"
            )
        return opened(request, timeout)

    monkeypatch.setattr(dl, "open_url", open_url)
    with pytest.raises(StageError) as ei:
        dl.download(server.url, tmp_path / "a", cn_hint=True)
    assert "the proxy p:1" in ei.value.message
    assert CN_MIRROR_HINT not in ei.value.message


def test_tls_after_a_slow_tunnel_gets_only_what_is_left(tmp_path, monkeypatch):
    """A proxy that answers CONNECT late and then stalls the TLS handshake
    fails within the budget, not the budget again on top."""
    monkeypatch.setattr(dl, "CONNECT_BUDGET", 2.0)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    held: list[socket.socket] = []

    def proxy() -> None:
        try:
            conn, _ = listener.accept()
        except OSError:
            return
        held.append(conn)
        conn.recv(4096)
        time.sleep(1.5)
        conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")

    threading.Thread(target=proxy, daemon=True).start()
    monkeypatch.delenv("no_proxy")
    monkeypatch.setenv("https_proxy", f"http://127.0.0.1:{listener.getsockname()[1]}")
    started = time.monotonic()
    try:
        with pytest.raises(StageError) as ei:
            dl.download("https://nodejs.example/node.tar.xz", tmp_path / "a")
    finally:
        listener.close()
        for conn in held:
            conn.close()
    # Twice the budget would be 4 s; the rest is slack for a slow CI runner.
    assert time.monotonic() - started < 3.5
    assert "no answer within 2 s" in ei.value.message


def test_connect_budget_is_shared_by_all_addresses(monkeypatch):
    """socket.create_connection gives each address the full timeout; the
    budget here covers them all, in shares."""
    clock = {"now": 0.0}
    timeouts: list[float] = []

    class DroppingSocket:
        def __init__(self, *_args) -> None:
            self.timeout = None

        def settimeout(self, value) -> None:
            self.timeout = value

        def connect(self, addr) -> None:
            timeouts.append(self.timeout)
            if addr[0] in reachable:
                return
            clock["now"] += self.timeout
            raise TimeoutError("timed out")

        def close(self) -> None:
            pass

    addresses = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (f"192.0.2.{i}", 443))
        for i in range(12)
    ]
    monkeypatch.setattr(dl.socket, "getaddrinfo", lambda *_a: addresses)
    monkeypatch.setattr(dl.socket, "socket", DroppingSocket)
    monkeypatch.setattr(dl.time, "monotonic", lambda: clock["now"])
    reachable: set[str] = set()
    with pytest.raises(TimeoutError):
        dl._connect_within(
            dl.socket.getaddrinfo("registry.example", 443), deadline=15.0
        )
    # Each address gets an equal share, at least 3 s, until the budget is spent.
    assert timeouts == [3.0] * 5
    assert clock["now"] == 15.0

    # A first address that drops packets leaves time for the next one.
    timeouts.clear()
    clock["now"] = 0.0
    reachable.add("192.0.2.1")
    sock = dl._connect_within(
        dl.socket.getaddrinfo("registry.example", 443), deadline=15.0
    )
    assert isinstance(sock, DroppingSocket)
    assert timeouts[:2] == [3.0, 3.0]


@pytest.mark.parametrize("ipv6", ["drops", "unsupported"])
def test_ipv4_gets_a_turn_while_ipv6_fails(monkeypatch, ipv6):
    """getaddrinfo lists every IPv6 address before the IPv4 ones."""
    clock = {"now": 0.0}
    tried: list[str] = []

    class FakeSocket:
        def __init__(self, family, *_args) -> None:
            if family == socket.AF_INET6 and ipv6 == "unsupported":
                raise OSError("Address family not supported by protocol")
            self.family = family
            self.timeout = None

        def settimeout(self, value) -> None:
            self.timeout = value

        def connect(self, addr) -> None:
            tried.append(addr[0])
            if self.family == socket.AF_INET6:
                clock["now"] += self.timeout
                raise TimeoutError("timed out")

        def close(self) -> None:
            pass

    v6 = [
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (f"2001:db8::{i}", 443, 0, 0))
        for i in range(12)
    ]
    v4 = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (f"192.0.2.{i}", 443))
        for i in range(12)
    ]
    monkeypatch.setattr(dl.socket, "getaddrinfo", lambda *_a: v6 + v4)
    monkeypatch.setattr(dl.socket, "socket", FakeSocket)
    monkeypatch.setattr(dl.time, "monotonic", lambda: clock["now"])
    dl._connect_within(dl.socket.getaddrinfo("registry.example", 443), deadline=15.0)
    assert tried[-1] == "192.0.2.0"
    assert clock["now"] <= 3.0


def test_notconnected_names_the_budget_for_a_timeout():
    exc = dl.NotConnected("nodejs.org", TimeoutError("timed out"))
    assert (
        str(exc)
        == f"Could not connect to nodejs.org: no answer within {dl.CONNECT_BUDGET:g} s"
    )


def test_disk_error_while_downloading_is_install_failed(monkeypatch, tmp_path):
    def open_url(url, timeout):
        response = io.BytesIO(b"data")
        response.headers = {}
        return response

    monkeypatch.setattr(dl, "open_url", open_url)
    with pytest.raises(StageError) as ei:
        dl.download("https://example.invalid/x", tmp_path / "missing-dir" / "x")
    assert ei.value.code == "install_failed"


def test_disk_error_when_closing_the_file_is_install_failed(
    serve, monkeypatch, tmp_path
):
    """Buffered bytes are written when the file is closed; that failure is the
    disk's, not the network's, so it is not retried."""
    server = serve(ok)
    real_open = open

    class FullDisk:
        def __init__(self, *args) -> None:
            self.fh = real_open(*args)

        def __getattr__(self, name):
            return getattr(self.fh, name)

        def close(self) -> None:
            self.fh.close()
            raise OSError(27, "File too large")

    monkeypatch.setattr(dl, "open", FullDisk, raising=False)
    with pytest.raises(StageError) as ei:
        dl.download(server.url, tmp_path / "a")
    assert ei.value.code == "install_failed"
    assert server.requests == 1


def test_retries_log_a_warning(serve, tmp_path, caplog):
    server = serve(status(503), ok)
    with caplog.at_level("WARNING", logger=dl.__name__):
        dl.download(server.url, tmp_path / "a")
    assert "retrying" in caplog.text
    assert "127.0.0.1" in caplog.text
