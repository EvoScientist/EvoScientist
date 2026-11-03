"""Background update check for EvoScientist.

Compares the installed version against PyPI and caches the result
(see ``CACHE_TTL``).  All errors are silently swallowed so startup
is never blocked or degraded.
"""

from __future__ import annotations

import json
import logging
import time

from .config.settings import get_config_dir

logger = logging.getLogger(__name__)

PYPI_URL = "https://pypi.org/pypi/EvoScientist/json"
PYPI_SIMPLE = "https://pypi.org/simple"
# The Tsinghua PyPI mirror, used with ``mirror: cn`` (the index the install
# scripts and ``EvoSci update`` install from there).
CN_SIMPLE = "https://pypi.tuna.tsinghua.edu.cn/simple"
_SIMPLE_JSON = "application/vnd.pypi.simple.v1+json"
CACHE_DIR = get_config_dir()
CACHE_FILE = CACHE_DIR / "latest_version.json"
CACHE_TTL = 86_400  # 24 hours


def _installed_version() -> str:
    """Return the currently installed EvoScientist version."""
    from importlib.metadata import version as _pkg_version

    return _pkg_version("EvoScientist")


def index_url(mirror: str) -> str:
    """The package index for a ``mirror`` setting."""
    return CN_SIMPLE if mirror == "cn" else PYPI_SIMPLE


def fetch_versions(index: str, timeout: float) -> list[str]:
    """Every EvoScientist version *index* lists, from its JSON simple API
    (PEP 691). The list is not sorted (the Tsinghua mirror sorts it as
    strings). Raises ``OSError`` / ``ValueError`` / ``KeyError`` on failure."""
    import urllib.request

    req = urllib.request.Request(
        f"{index.rstrip('/')}/evoscientist/",
        headers={"Accept": _SIMPLE_JSON, "User-Agent": "EvoScientist update-check"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        versions = json.loads(resp.read())["versions"]
    return [v for v in versions if isinstance(v, str)]


def newest_release(versions: list[str]) -> str | None:
    """The newest final release among *versions* (PEP 440); pre-releases and
    versions that do not parse are skipped."""
    from packaging.version import InvalidVersion, Version

    releases = []
    for v in versions:
        try:
            parsed = Version(v)
        except InvalidVersion:
            continue
        if not (parsed.is_prerelease or parsed.is_devrelease):
            releases.append((parsed, v))
    return max(releases)[1] if releases else None


def _fetch_latest(mirror: str, timeout: float) -> str | None:
    if mirror == "cn":
        return newest_release(fetch_versions(CN_SIMPLE, timeout))
    import urllib.request

    req = urllib.request.Request(
        PYPI_URL, headers={"User-Agent": "EvoScientist update-check"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["info"]["version"]


def get_latest_version() -> str | None:
    """Fetch the latest EvoScientist version, with 24h caching.

    With ``mirror: cn`` in the config the Tsinghua mirror is asked instead of
    PyPI. (``EvoSci update`` reads the index itself, without the cache.)
    """
    try:
        if CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if time.time() - data.get("checked_at", 0) < CACHE_TTL:
                return data["version"]
    except Exception:
        logger.debug("Failed to read update-check cache", exc_info=True)

    try:
        from .config import load_config

        latest = _fetch_latest(load_config().mirror, timeout=3)
    except Exception:
        logger.debug("Failed to fetch the latest version", exc_info=True)
        return None
    if latest is None:
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

    from packaging.version import InvalidVersion, Version

    try:
        current = _installed_version()
        if Version(latest) > Version(current):
            return True, latest
    except (InvalidVersion, TypeError):
        logger.debug("Failed to compare versions", exc_info=True)

    return False, None
