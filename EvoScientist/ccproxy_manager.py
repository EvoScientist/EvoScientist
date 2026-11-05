"""ccproxy lifecycle management for OAuth-based Anthropic access.

Provides functions to start/stop/health-check ccproxy, which allows
users with a Claude Pro/Max subscription to use EvoScientist without
a separate API key by reusing Claude Code's OAuth tokens.

ccproxy is invoked via subprocess (not Python imports) so the
``ccproxy-api`` package is truly optional at runtime.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from EvoScientist.config import EvoScientistConfig

logger = logging.getLogger(__name__)

_CCPROXY_AUTH_TIMEOUT_SECONDS = 30
_CCPROXY_HEALTH_TIMEOUT_SECONDS = 180


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


def _is_editable_install() -> bool:
    """Return True if EvoScientist was installed in editable/development mode.

    Checks all matching distributions because a stale ``.egg-info`` in the
    project root can shadow the real ``dist-info`` in site-packages.
    """
    try:
        import importlib.metadata as _meta
        import json

        for dist in _meta.distributions():
            name = dist.metadata.get("Name", "")
            if name.lower() != "evoscientist":
                continue
            direct_url = dist.read_text("direct_url.json")
            if direct_url is not None:
                data = json.loads(direct_url)
                if data.get("dir_info", {}).get("editable", False) is True:
                    return True
    except Exception:
        pass
    return False


def _oauth_install_hint() -> str:
    """Return the appropriate install command depending on install method."""
    if _is_editable_install():
        return "uv sync --extra oauth or pip install -e '.[oauth]'"
    return "pip install 'evoscientist[oauth]'"


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
        provider: ccproxy provider name.

    Returns:
        (is_valid, message) tuple.
    """
    try:
        exe = _ccproxy_exe() or "ccproxy"
        result = subprocess.run(
            [exe, "auth", "status", provider],
            capture_output=True,
            text=True,
            # ccproxy's CLI initializes its full plugin system on every
            # invocation — a cold start takes ~10s on Apple Silicon, so a
            # 10s timeout made OAuth startup fail intermittently with
            # "Auth check timed out".
            timeout=_CCPROXY_AUTH_TIMEOUT_SECONDS,
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
        # trust_env=False: a loopback probe must not go through the
        # environment's or the system's proxy.
        resp = httpx.get(
            f"http://127.0.0.1:{port}/health/live", timeout=2.0, trust_env=False
        )
        return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        # Also another program on the port that does not speak HTTP.
        return False


def start_ccproxy(port: int, *, detached: bool = False) -> subprocess.Popen:
    """Start ccproxy serve as a background process.

    Args:
        port: Port number for the proxy server.
        detached: Run it in its own session (process group on Windows), so
            it outlives this process and the terminal's signals, as the
            background langgraph dev server does.

    Returns:
        The Popen handle for the ccproxy process.

    Raises:
        RuntimeError: If ccproxy fails to become healthy within
            ``_CCPROXY_HEALTH_TIMEOUT_SECONDS``.
        FileNotFoundError: If ccproxy binary is not found.
    """
    proc = spawn_ccproxy(port, detached=detached)
    wait_for_ccproxy(proc, port)
    return proc


def spawn_ccproxy(
    port: int, *, detached: bool = False, output: Any = subprocess.DEVNULL
) -> subprocess.Popen:
    """Start ``ccproxy serve`` without waiting for it to serve
    (:func:`wait_for_ccproxy`). *output* is where its stdout and stderr go, as
    ``Popen`` takes it: a file, or ``DEVNULL``. See :func:`start_ccproxy`
    for *detached*."""
    exe = _ccproxy_exe() or "ccproxy"
    cmd = [exe, "serve", "--port", str(port)]
    logger.warning(
        "Starting ccproxy on port %d; first startup may take up to %d seconds",
        port,
        _CCPROXY_HEALTH_TIMEOUT_SECONDS,
    )
    spawn: dict = {}
    if detached:
        if os.name == "nt":
            # A console of its own, without a window: closing the terminal
            # does not end it.
            spawn["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            ) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            spawn["start_new_session"] = True
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=output,
        **spawn,
    )


def wait_for_ccproxy(proc: subprocess.Popen, port: int) -> None:
    """Wait until the ccproxy *proc* serves on *port*.

    Raises:
        RuntimeError: When it exits first, or does not serve within
            ``_CCPROXY_HEALTH_TIMEOUT_SECONDS``; it is stopped then, with
            what it started (:func:`stop_ccproxy_tree`).
    """
    deadline = time.monotonic() + _CCPROXY_HEALTH_TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"ccproxy exited with code {proc.returncode} before it answered"
                )
            if is_ccproxy_running(port):
                return
            time.sleep(0.3)
    except BaseException:
        # Ctrl+C or an error during the wait: the caller never gets the
        # handle, so nobody else could stop this process.
        stop_ccproxy_tree(proc)
        raise

    # Timed out — clean up
    stop_ccproxy_tree(proc)
    raise RuntimeError(
        "ccproxy did not become healthy within "
        f"{_CCPROXY_HEALTH_TIMEOUT_SECONDS} seconds"
    )


def stop_ccproxy_tree(proc: subprocess.Popen | None) -> None:
    """Stop the ccproxy *proc* and what it started (on Windows
    ``ccproxy.exe`` is a launcher whose child serves), then reap it. Safe to
    call with None or a process already reaped."""
    import psutil

    if proc is None or proc.poll() is not None:
        return
    # Not reaped (poll() just said it runs), so its PID is still its own.
    try:
        end_tree(psutil.Process(proc.pid))
    except psutil.Error:
        # Cannot be found or inspected: stop it as a plain child.
        stop_ccproxy(proc)
        return
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=2)


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


def ensure_ccproxy(port: int, *, detached: bool = False) -> subprocess.Popen | None:
    """Ensure ccproxy is running — reuse existing or start new
    (see :func:`start_ccproxy` for *detached*).

    Returns:
        Popen handle if we started a new process, None if already running.
    """
    if is_ccproxy_running(port):
        logger.debug("ccproxy already running on port %d", port)
        return None
    return start_ccproxy(port, detached=detached)


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
    from .config.settings import record_env_origin

    os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}/claude"
    os.environ["ANTHROPIC_API_KEY"] = "ccproxy-oauth"
    record_env_origin("ANTHROPIC_BASE_URL", "ccproxy")
    record_env_origin("ANTHROPIC_API_KEY", "ccproxy")


# =============================================================================
# High-level orchestration
# =============================================================================


def check_ccproxy_ready() -> None:
    """Raise :class:`RuntimeError`, saying what to do, when ccproxy is not
    installed or not signed in to Anthropic."""
    if not is_ccproxy_available():
        raise RuntimeError(
            "ccproxy is required for OAuth mode but not found. "
            f"Install it with: {_oauth_install_hint()}"
        )

    authed, msg = check_ccproxy_auth("claude_api")
    if not authed:
        raise RuntimeError(
            f"ccproxy Anthropic OAuth not authenticated: {msg}\n"
            "Run: ccproxy auth login claude_api"
        )


def maybe_start_ccproxy(
    config: EvoScientistConfig, *, lasting: bool = False
) -> subprocess.Popen | None:
    """Start ccproxy when Anthropic uses OAuth.

    OpenAI's OAuth mode signs in to ChatGPT directly and needs no proxy.

    Args:
        config: An ``EvoScientistConfig`` instance.
        lasting: The ccproxy outlives this session, for every session and
            the background server to share: start it detached and record it
            for ``EvoSci ccproxy stop`` (:func:`stop_recorded_ccproxy`), and
            return None so the caller does not stop it.

    Returns:
        Popen handle if we started ccproxy for this session to stop, None
        otherwise.
    """
    if not uses_oauth(config):
        return None

    check_ccproxy_ready()

    port = config.ccproxy_port
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid ccproxy port: {port}. Must be between 1 and 65535.")

    if lasting:
        try:
            with _lasting_start_lock():
                proc = ensure_ccproxy(port, detached=True)
                if proc is not None:
                    # Recorded at once, under the lock: an interrupt during
                    # the check below leaves a ccproxy that a stop finds.
                    _record_lasting(proc.pid)
                    if not _still_running(proc):
                        _forget_lasting(proc.pid)
        except OSError as exc:
            # The lock timing out, too: a filelock Timeout is an OSError.
            raise RuntimeError(f"Could not start ccproxy: {exc}") from exc
    else:
        proc = ensure_ccproxy(port, detached=False)
    setup_ccproxy_env(port)

    if proc:
        logger.info("Started ccproxy on port %d", port)
    else:
        logger.info("Reusing existing ccproxy on port %d", port)
    return None if lasting else proc


def _still_running(proc: subprocess.Popen) -> bool:
    """Whether *proc* keeps running: one that lost the port to another
    ccproxy started at the same time answered the health probe through the
    winner's, and exits; the winner records itself."""
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        return True
    return False


def _lasting_record() -> Path:
    from . import paths

    return paths.SERVER_HOME / "ccproxy.pid"


def _lasting_start_lock():
    """Held across the check for a running ccproxy and the start of a lasting
    one: two sessions starting together would each spawn one, and the loser
    can pass its health probe through the winner's port, then die on its bind
    after being recorded."""
    from filelock import FileLock

    from .langgraph_dev.server_home import ensure_server_home

    return FileLock(
        str(ensure_server_home() / "ccproxy.lock"),
        timeout=_CCPROXY_HEALTH_TIMEOUT_SECONDS + 10,
    )


def _record_lasting(pid: int) -> None:
    from .langgraph_dev.server_home import ensure_server_home

    try:
        # Created as the server creates it: readable by this user only.
        ensure_server_home()
        _lasting_record().write_text(str(pid))
    except OSError:
        logger.warning("Could not record the ccproxy pid", exc_info=True)


def _forget_lasting(pid: int) -> None:
    """Remove the record of *pid*, when it is still the one recorded."""
    if recorded_ccproxy_pid() != pid:
        return
    try:
        _lasting_record().unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove the ccproxy pid record", exc_info=True)


def uses_oauth(config: object) -> bool:
    """Whether Anthropic uses OAuth, and so routes through ccproxy."""
    return getattr(config, "anthropic_auth_mode", "api_key") == "oauth"


def recorded_ccproxy_pid() -> int | None:
    """The pid of the ccproxy a session started for every session, if one
    is recorded."""
    try:
        return int(_lasting_record().read_text().strip())
    except (OSError, ValueError):
        return None


def _recorded_process():
    """The recorded ccproxy's process: ``None`` when it is gone (its pid
    unused, or used by another program now), ``...`` when it cannot be
    inspected."""
    pid = recorded_ccproxy_pid()
    if pid is None:
        return None
    return _ccproxy_process(pid)


def _ccproxy_process(pid: int):
    """*pid*'s process while it runs ccproxy: ``None`` when it is gone or
    runs another program, ``...`` when it cannot be inspected."""
    import psutil

    try:
        proc = psutil.Process(pid)
        cmdline = proc.cmdline()
    except psutil.AccessDenied:
        return ...
    except psutil.NoSuchProcess:
        return None
    return proc if any("ccproxy" in part for part in cmdline) else None


def recorded_ccproxy_running() -> int | None:
    """The recorded ccproxy's pid while it runs, else None."""
    proc = _recorded_process()
    if proc is None:
        return None
    return recorded_ccproxy_pid() if proc is ... else proc.pid


def stop_recorded_ccproxy() -> int | None:
    """Stop the ccproxy a session started for every session (``EvoSci
    ccproxy stop``, ``EvoSci service stop``, ``EvoSci update``); its pid, or
    None when none runs. A pid whose process is not ccproxy (reused) is left
    alone. Every server start stops it too."""
    record = _lasting_record()
    proc = _recorded_process()
    if proc is ...:
        # Kept: it may still be ours, and a later stop can try again.
        return None
    record.unlink(missing_ok=True)
    if proc is None:
        return None
    return proc.pid if end_tree(proc) else None


def end_ccproxy(pid: int, created: float | None, port: int) -> bool:
    """End the ccproxy *pid* (and what it started) when that process still
    is the one created at *created*, or serves on *port* (its creation time
    moves with a clock step, and may not have been readable), and runs
    ccproxy; whether it did."""
    from .langgraph_dev.manager import _same_process

    proc = _ccproxy_process(pid)
    if proc is None or proc is ...:
        return False
    same = created is not None and _same_process(pid, created)
    if not same and not _serves_on(proc, port):
        return False
    return end_tree(proc)


def _serves_on(proc, port: int) -> bool:
    """Whether the ccproxy *proc* was started to serve on *port*."""
    import psutil

    try:
        cmdline = proc.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return any(
        flag == "--port" and value == str(port)
        for flag, value in itertools.pairwise(cmdline)
    )


def end_tree(proc) -> bool:
    """Terminate the psutil process *proc* and its descendants, killing what
    is left after 5 seconds; False when it was already gone."""
    import psutil

    try:
        tree = [*proc.children(recursive=True), proc]
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        tree = [proc]
    for member in tree:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            member.terminate()
    _gone, alive = psutil.wait_procs(tree, timeout=5)
    for member in alive:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            member.kill()
    return True
