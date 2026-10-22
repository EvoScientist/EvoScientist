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
        dl._connect_within(("registry.example", 443), None, deadline=15.0)
    # Each address gets an equal share, at least 3 s, until the budget is spent.
    assert timeouts == [3.0] * 5
    assert clock["now"] == 15.0

    # A first address that drops packets leaves time for the next one.
    timeouts.clear()
    clock["now"] = 0.0
    reachable.add("192.0.2.1")
    sock = dl._connect_within(("registry.example", 443), None, deadline=15.0)
    assert isinstance(sock, DroppingSocket)
    assert timeouts[:2] == [3.0, 3.0]


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


def test_retries_log_a_warning(serve, tmp_path, caplog):
    server = serve(status(503), ok)
    with caplog.at_level("WARNING", logger=dl.__name__):
        dl.download(server.url, tmp_path / "a")
    assert "retrying" in caplog.text
    assert "127.0.0.1" in caplog.text
