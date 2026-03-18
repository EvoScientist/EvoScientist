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

from EvoScientist.config import EvoScientistConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Availability & auth checks
# =============================================================================


def _ccproxy_exe() -> str | None:
    """Return the path to the ccproxy binary, or None if not found.

    Checks PATH first, then the current Python environment's bin directory
    (handles conda envs where newly installed binaries may not be visible
    to shutil.which immediately after pip install).
    """
    found = shutil.which("ccproxy")
    if found:
        return found
    import sys as _sys

    candidate = os.path.join(os.path.dirname(_sys.executable), "ccproxy")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def is_ccproxy_available() -> bool:
    """Check whether the ``ccproxy`` CLI binary is available."""
    return _ccproxy_exe() is not None


def _summarize_auth_output(raw: str) -> str:
    """Extract key fields from ccproxy auth status output into a one-line summary.

    Parses the Rich table output for Email, Subscription, and Status fields.
    Returns e.g. ``"user@example.com (plus, active)"``.
    Falls back to ``"Authenticated"`` if parsing fails.
    """
    import re as _re

    # Strip ANSI escape sequences
    clean = _re.sub(r"\x1b\[[0-9;]*m", "", raw)

    # Parse "Key<2+ spaces>Value" table rows, match exact key names
    fields: dict[str, str] = {}
    for line in clean.splitlines():
        m = _re.match(r"\s*(.+?)\s{2,}(.+)", line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        if key in ("Email", "Subscription", "Subscription Status"):
            fields[key.lower().replace(" ", "_")] = val

    email = fields.get("email", "")
    sub = fields.get("subscription", "")
    status = fields.get("subscription_status", "")

    if email:
        detail = ", ".join(filter(None, [sub, status]))
        return f"{email} ({detail})" if detail else email
    return "Authenticated"


def check_ccproxy_auth(provider: str = "claude_api") -> tuple[bool, str]:
    """Check if ccproxy has valid OAuth credentials.

    Args:
        provider: ccproxy provider name ("claude_api" or "codex").

    Returns:
        (is_valid, message) tuple.
    """
    try:
        exe = _ccproxy_exe() or "ccproxy"
        result = subprocess.run(
            [exe, "auth", "status", provider],
            capture_output=True,
            text=True,
            timeout=10,
        )
        import re as _re

        raw = (result.stdout + result.stderr).strip()
        clean = _re.sub(r"\x1b\[[0-9;]*m", "", raw)

        # Filter out structlog warning/noise lines, keep only status lines
        status_lines = [
            line
            for line in clean.splitlines()
            if line.strip()
            and not _re.match(r"\d{4}-\d{2}-\d{2}", line.strip())
            and "warning" not in line.lower()
            and "plugin" not in line.lower()
        ]
        status_msg = " ".join(status_lines).strip()

        # ccproxy auth status may exit 0 even when not authenticated —
        # detect failure by checking output content
        if result.returncode != 0 or "not authenticated" in clean.lower():
            return False, status_msg or "Not authenticated"

        summary = _summarize_auth_output(result.stdout)
        return True, summary or "Authenticated"
    except FileNotFoundError:
        return False, "ccproxy not found"
    except subprocess.TimeoutExpired:
        return False, "Auth check timed out"
    except Exception as exc:
        return False, f"Auth check failed: {exc}"


# =============================================================================
# Process management
# =============================================================================


def is_ccproxy_running(port: int) -> bool:
    """Check if ccproxy is already serving on the given port."""
    import httpx

    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/health/live", timeout=2.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        return False


def start_ccproxy(port: int) -> subprocess.Popen:
    """Start ccproxy serve as a background process.

    Args:
        port: Port number for the proxy server.

    Returns:
        The Popen handle for the ccproxy process.

    Raises:
        RuntimeError: If ccproxy fails to become healthy within 10 seconds.
        FileNotFoundError: If ccproxy binary is not found.
    """
    exe = _ccproxy_exe() or "ccproxy"
    proc = subprocess.Popen(
        [exe, "serve", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for health (ccproxy can take up to ~11s on first start)
    deadline = time.monotonic() + 30
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
    raise RuntimeError("ccproxy did not become healthy within 30 seconds")


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


def ensure_ccproxy(port: int) -> subprocess.Popen | None:
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


def setup_ccproxy_env(port: int) -> None:
    """Set environment variables for Anthropic ccproxy routing.

    Force-sets ``ANTHROPIC_BASE_URL`` and ``ANTHROPIC_API_KEY`` so that
    downstream LangChain/Anthropic clients route through ccproxy.

    Always overrides existing values — when this function is called,
    we've decided to use ccproxy, so env must point to it.
    """
    os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}/claude"
    os.environ["ANTHROPIC_API_KEY"] = "ccproxy-oauth"


def setup_codex_env(port: int) -> None:
    """Set environment variables for OpenAI/Codex ccproxy routing.

    Force-sets ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` so that
    downstream LangChain/OpenAI clients route through ccproxy's Codex
    endpoint.

    Always overrides existing values — when this function is called,
    we've decided to use ccproxy, so env must point to it.
    """
    os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/codex/v1"
    os.environ["OPENAI_API_KEY"] = "ccproxy-oauth"


# =============================================================================
# High-level orchestration
# =============================================================================


def _patch_ccproxy_oauth_header() -> None:
    """Auto-patch ccproxy's adapter to send the correct OAuth beta header.

    ccproxy 0.2.4 hardcodes ``computer-use-2025-01-24`` as the
    ``anthropic-beta`` header, which causes two problems:
    - Missing ``oauth-2025-04-20`` → 401 from Anthropic
    - ``computer-use-2025-01-24`` incompatible with OAuth auth → 400

    The ccproxy binary may use a different Python environment than the one
    running EvoScientist, so we resolve the adapter path via the ccproxy
    binary's shebang line rather than the current Python's import system.

    This patch is idempotent and places the header AFTER cli_headers so it
    cannot be overridden.
    """
    import pathlib
    import re

    try:
        ccproxy_bin = _ccproxy_exe()
        if not ccproxy_bin:
            return

        # Find the Python interpreter used by the ccproxy binary via shebang
        shebang = pathlib.Path(ccproxy_bin).read_text().splitlines()[0]
        python_exe = shebang.lstrip("#!").strip()

        # Ask that Python where ccproxy's adapter lives
        result = subprocess.run(
            [
                python_exe, "-c",
                "import inspect, ccproxy.plugins.claude_api.adapter as m; print(inspect.getfile(m))",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return
        src_file = pathlib.Path(result.stdout.strip())
        if not src_file.exists():
            return

        text = src_file.read_text()

        # Check if already correctly patched (oauth header set after cli_headers)
        correct = 'filtered_headers["anthropic-beta"] = "oauth-2025-04-20"'
        cli_marker = "cli_headers = self._collect_cli_headers()"
        if correct in text:
            # Verify it's placed after cli_headers
            if text.index(correct) > text.index(cli_marker):
                return  # Already correctly patched

        # Move/replace anthropic-beta assignment to after cli_headers loop,
        # and set only oauth-2025-04-20 (computer-use-* is incompatible).
        # Step 1: remove any existing filtered_headers["anthropic-beta"] line
        patched = re.sub(
            r'\s*filtered_headers\["anthropic-beta"\]\s*=\s*"[^"]*"\n',
            "\n",
            text,
        )
        # Step 2: insert correct assignment after the cli_headers block
        insert_after = 'filtered_headers[lk] = value\n'
        replacement = (
            'filtered_headers[lk] = value\n\n'
            '        # oauth-2025-04-20: required for OAuth Bearer token auth (Anthropic 2026-03)\n'
            '        filtered_headers["anthropic-beta"] = "oauth-2025-04-20"\n'
        )
        patched = patched.replace(insert_after, replacement, 1)

        if patched == text:
            return

        src_file.write_text(patched)
        for pyc in src_file.parent.glob("__pycache__/adapter*.pyc"):
            pyc.unlink(missing_ok=True)
        logger.info("Auto-patched ccproxy adapter: set anthropic-beta=oauth-2025-04-20")
    except Exception as exc:
        logger.warning("Could not auto-patch ccproxy adapter: %s", exc)


def maybe_start_ccproxy(config: EvoScientistConfig) -> subprocess.Popen | None:
    """High-level: conditionally start ccproxy based on config.

    Checks ``config.anthropic_auth_mode`` and ``config.openai_auth_mode``:
    - ``oauth``: ccproxy must work — raises on failure.
    - ``api_key``: no-op for that provider.

    When either provider uses OAuth, ccproxy is started (single process
    serves both providers). Environment variables are set for each
    provider that uses OAuth.

    Args:
        config: An ``EvoScientistConfig`` instance.

    Returns:
        Popen handle if we started ccproxy, None otherwise.
    """
    anthropic_oauth = getattr(config, "anthropic_auth_mode", "api_key") == "oauth"
    openai_oauth = getattr(config, "openai_auth_mode", "api_key") == "oauth"

    if not anthropic_oauth and not openai_oauth:
        return None

    if not is_ccproxy_available():
        raise RuntimeError(
            "ccproxy is required for OAuth mode but not found. "
            "Install it with: uv sync --extra oauth "
            "or pip install 'evoscientist[oauth]'"
        )

    # Check auth for each provider that uses OAuth
    if anthropic_oauth:
        authed, msg = check_ccproxy_auth("claude_api")
        if not authed:
            raise RuntimeError(
                f"ccproxy Anthropic OAuth not authenticated: {msg}\n"
                "Run: ccproxy auth login claude_api"
            )

    if openai_oauth:
        authed, msg = check_ccproxy_auth("codex")
        if not authed:
            raise RuntimeError(
                f"ccproxy Codex OAuth not authenticated: {msg}\n"
                "Run: ccproxy auth login codex"
            )

    port = config.ccproxy_port
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid ccproxy port: {port}. Must be between 1 and 65535.")

    # Auto-patch ccproxy adapter to fix OAuth header compatibility
    _patch_ccproxy_oauth_header()

    # Start ccproxy (single process serves both providers)
    proc = ensure_ccproxy(port)

    # Set environment for each OAuth provider
    if anthropic_oauth:
        setup_ccproxy_env(port)
    if openai_oauth:
        setup_codex_env(port)

    if proc:
        logger.info("Started ccproxy on port %d", port)
    else:
        logger.info("Reusing existing ccproxy on port %d", port)
    return proc
