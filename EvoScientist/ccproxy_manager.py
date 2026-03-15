"""ccproxy lifecycle management for OAuth-based Anthropic access.

Provides functions to start/stop/health-check ccproxy, which allows
users with a Claude Pro/Max subscription to use EvoScientist without
a separate API key by reusing Claude Code's OAuth tokens.

ccproxy is invoked via subprocess (not Python imports) so the
``ccproxy-api`` package is truly optional at runtime.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8000


# =============================================================================
# Availability & auth checks
# =============================================================================


def is_ccproxy_available() -> bool:
    """Check whether the ``ccproxy`` CLI binary is on PATH."""
    return shutil.which("ccproxy") is not None


def check_ccproxy_auth() -> tuple[bool, str]:
    """Check if ccproxy has valid OAuth credentials.

    Returns:
        (is_valid, message) tuple.
    """
    try:
        result = subprocess.run(
            ["ccproxy", "auth", "status", "claude_api"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = (result.stdout + result.stderr).strip()
        # ccproxy auth status exits 0 when authed
        if result.returncode == 0:
            return True, output or "Authenticated"
        return False, output or "Not authenticated"
    except FileNotFoundError:
        return False, "ccproxy not found"
    except subprocess.TimeoutExpired:
        return False, "Auth check timed out"
    except Exception as exc:
        return False, f"Auth check failed: {exc}"


# =============================================================================
# Process management
# =============================================================================


def is_ccproxy_running(port: int = _DEFAULT_PORT) -> bool:
    """Check if ccproxy is already serving on the given port."""
    import httpx

    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/", timeout=2.0)
        return resp.status_code < 500
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        return False


def start_ccproxy(port: int = _DEFAULT_PORT) -> subprocess.Popen:
    """Start ccproxy serve as a background process.

    Args:
        port: Port number for the proxy server.

    Returns:
        The Popen handle for the ccproxy process.

    Raises:
        RuntimeError: If ccproxy fails to become healthy within 10 seconds.
        FileNotFoundError: If ccproxy binary is not found.
    """
    proc = subprocess.Popen(
        ["ccproxy", "serve", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for health
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"ccproxy exited immediately with code {proc.returncode}"
            )
        if is_ccproxy_running(port):
            return proc
        time.sleep(0.3)

    # Timed out — clean up
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
    raise RuntimeError("ccproxy did not become healthy within 10 seconds")


def stop_ccproxy(proc: subprocess.Popen | None) -> None:
    """Gracefully stop a ccproxy process.

    Safe to call with None (no-op).
    """
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    except Exception:
        pass


def ensure_ccproxy(port: int = _DEFAULT_PORT) -> subprocess.Popen | None:
    """Ensure ccproxy is running — reuse existing or start new.

    Returns:
        Popen handle if we started a new process, None if already running.
    """
    if is_ccproxy_running(port):
        logger.debug("ccproxy already running on port %d", port)
        return None
    return start_ccproxy(port)


# =============================================================================
# Environment setup
# =============================================================================


def setup_ccproxy_env(port: int = _DEFAULT_PORT) -> None:
    """Set environment variables for ccproxy routing.

    Force-sets ``ANTHROPIC_BASE_URL`` and ``ANTHROPIC_API_KEY`` so that
    downstream LangChain/Anthropic clients route through ccproxy.

    Always overrides existing values — when this function is called,
    we've decided to use ccproxy, so env must point to it.
    """
    os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}/claude"
    os.environ["ANTHROPIC_API_KEY"] = "ccproxy-oauth"


# =============================================================================
# High-level orchestration
# =============================================================================


def maybe_start_ccproxy(config: object) -> subprocess.Popen | None:
    """High-level: conditionally start ccproxy based on config.

    Checks ``config.anthropic_auth_mode``:
    - ``oauth``: ccproxy must work — raises on failure.
    - ``api_key``: no-op.

    Args:
        config: An ``EvoScientistConfig`` instance.

    Returns:
        Popen handle if we started ccproxy, None otherwise.
    """
    auth_mode = getattr(config, "anthropic_auth_mode", "api_key")
    if auth_mode != "oauth":
        return None

    if not is_ccproxy_available():
        raise RuntimeError(
            "ccproxy is required for OAuth mode but not found. "
            "Install it with: pip install 'evoscientist[oauth]'"
        )

    authed, msg = check_ccproxy_auth()
    if not authed:
        raise RuntimeError(
            f"ccproxy OAuth not authenticated: {msg}\n"
            "Run: ccproxy auth login claude_api"
        )

    proc = ensure_ccproxy()
    setup_ccproxy_env()
    if proc:
        logger.info("Started ccproxy on port %d", _DEFAULT_PORT)
    else:
        logger.info("Reusing existing ccproxy on port %d", _DEFAULT_PORT)
    return proc
