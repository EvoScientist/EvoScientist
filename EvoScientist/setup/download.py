"""Download and checksum helpers shared by the setup stages.

Downloads use ``urllib``'s default opener: it honours ``HTTPS_PROXY`` (and, on
Windows and macOS, the system proxy settings) and verifies TLS with the default
SSL context, which honours ``SSL_CERT_FILE`` and on Windows also loads the
system certificate store. That is deliberate and differs from the loopback
probes elsewhere, which bypass proxies.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .protocol import StageError

_CHUNK = 256 * 1024
_TIMEOUT = 120

ProgressFn = Callable[[float], None]


def download(url: str, dest: Path, progress: ProgressFn | None = None) -> str:
    """Stream ``url`` into ``dest`` and return the file's SHA-256 hex digest.

    ``progress`` receives the downloaded fraction when the server sends a
    ``Content-Length``. Network and HTTP errors raise
    ``StageError("download_failed")``.
    """
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with open(dest, "wb") as fh:
                while chunk := resp.read(_CHUNK):
                    fh.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if progress is not None and total:
                        progress(done / total)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise StageError("download_failed", f"Download of {url} failed: {exc}") from exc
    return digest.hexdigest()


def fetch_text(url: str) -> str:
    """GET a small text resource, e.g. ``SHASUMS256.txt``."""
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise StageError("download_failed", f"Download of {url} failed: {exc}") from exc


def verify_sha256_from_sums(actual: str, sums_text: str, filename: str) -> None:
    """Check ``actual`` against ``filename``'s entry in a ``SHASUMS256.txt``.

    Parses ``<sha256>  <filename>`` lines (``*name`` marks binary mode in
    ``sha256sum`` output). A missing entry fails too: the publisher lists every
    asset, so an absent line means the wrong file or a tampered list.
    """
    expected = None
    for line in sums_text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == filename:
            expected = parts[0].lower()
            break
    if expected is None:
        raise StageError(
            "checksum_mismatch", f"{filename} is not listed in the published checksums."
        )
    if actual.lower() != expected:
        raise StageError(
            "checksum_mismatch",
            f"{filename} sha256 {actual} does not match the published {expected}.",
        )
