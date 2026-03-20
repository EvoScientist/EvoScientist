"""Background update check for EvoScientist.

Compares the installed version against PyPI and caches the result
(see ``CACHE_TTL``).  All errors are silently swallowed so startup
is never blocked or degraded.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PYPI_URL = "https://pypi.org/pypi/EvoScientist/json"
CACHE_DIR = Path("~/.config/evoscientist").expanduser()
CACHE_FILE = CACHE_DIR / "latest_version.json"
CACHE_TTL = 86_400  # 24 hours


def _installed_version() -> str:
    """Return the currently installed EvoScientist version."""
    from importlib.metadata import version as _pkg_version

    return _pkg_version("EvoScientist")


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a dotted version string into a comparable integer tuple."""
    return tuple(int(x) for x in v.strip().split("."))


def get_latest_version() -> str | None:
    """Fetch the latest EvoScientist version from PyPI, with 24h caching."""
    # Try cache first
    try:
        if CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if time.time() - data.get("checked_at", 0) < CACHE_TTL:
                return data["version"]
    except Exception:
        logger.debug("Failed to read update-check cache", exc_info=True)

    # Fetch from PyPI (use urllib to avoid extra deps)
    try:
        import urllib.request

        req = urllib.request.Request(
            PYPI_URL,
            headers={"User-Agent": "EvoScientist update-check"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            latest: str = json.loads(resp.read())["info"]["version"]
    except Exception:
        logger.debug("Failed to fetch latest version from PyPI", exc_info=True)
        return None

    # Cache result
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(
            json.dumps({"version": latest, "checked_at": time.time()}),
            encoding="utf-8",
        )
    except Exception:
        logger.debug("Failed to write update-check cache", exc_info=True)

    return latest


def is_update_available() -> tuple[bool, str | None]:
    """Check whether a newer version of EvoScientist is available on PyPI.

    Returns:
        ``(available, latest)`` tuple.  *available* is ``True`` when
        the PyPI version is strictly newer; *latest* is the version
        string (or ``None`` when the check fails).
    """
    latest = get_latest_version()
    if latest is None:
        return False, None

    try:
        current = _installed_version()
        if _parse_version(latest) > _parse_version(current):
            return True, latest
    except (ValueError, TypeError):
        logger.debug("Failed to compare versions", exc_info=True)

    return False, None
