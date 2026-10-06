"""MCP (Model Context Protocol) client integration.

Loads MCP server configurations from YAML, connects via langchain-mcp-adapters,
and routes the resulting LangChain tools to the appropriate agents.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import re
import shutil
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import yaml

from ..runtime import AsyncRuntime, AsyncRuntimeError

logger = logging.getLogger(__name__)


# =============================================================================
# Windows MCP SDK patch — drop the blocking os.access from stdio command resolution
# =============================================================================
#
# On Windows, ``mcp.client.stdio`` resolves the server command on every stdio
# session open via ``get_windows_executable_command()``, which calls
# ``shutil.which()`` → ``os.access()`` — a blocking syscall. ``langgraph dev``
# enables ``blockbuster`` by default, which flags that ``os.access()`` as an
# illegal blocking call inside the event loop (one of the two root causes of
# issue #283; the other is the Selector-loop subprocess fallback handled by
# ``EvoScientist._winloop``).
#
# ``_resolve_command()`` (below) already resolves bare command names to their
# absolute paths at connection-build time — a sync context where blocking is
# fine. So by the time the stdio transport asks for the executable, the command
# is already absolute and there is nothing left to look up. We short-circuit the
# SDK's resolver for absolute paths, avoiding the ``os.access()`` entirely; bare
# commands (which shouldn't reach here, but might via a transport we don't build)
# still fall through to the original SDK behaviour.


def _patch_mcp_windows_command_resolver() -> None:
    """Make the MCP SDK's stdio command resolver skip ``os.access`` for absolute
    paths.

    Idempotent. A no-op when the MCP SDK is absent (it is an optional
    dependency). If the SDK is present but its resolver can't be located, we log
    a warning rather than failing silently — a silent no-op here would let the
    blocking ``os.access`` quietly return after an SDK refactor.
    """
    try:
        import mcp.client.stdio as _stdio_mod
    except ImportError:
        return  # MCP SDK not installed — nothing to patch.

    original = getattr(_stdio_mod, "get_windows_executable_command", None)
    if not callable(original):
        logger.warning(
            "MCP SDK layout changed: mcp.client.stdio.get_windows_executable_command "
            "is missing; the Windows os.access fast-path was NOT applied. MCP stdio "
            "tool calls may trip blocking-call detection (blockbuster) on Windows."
        )
        return

    if getattr(original, "_evosci_absolute_fast_path", False):
        return  # Already patched.

    def _patched_get_windows_executable_command(command: str) -> str:
        # Absolute path → already resolved, no filesystem probe needed.
        if os.path.isabs(command):
            return command
        return original(command)

    _patched_get_windows_executable_command._evosci_absolute_fast_path = True  # type: ignore[attr-defined]

    # Patch the name in the stdio module (the call site resolves it from this
    # module's globals) and, best-effort, the origin module in case anything
    # imports it from there directly.
    _stdio_mod.get_windows_executable_command = _patched_get_windows_executable_command
    try:
        import mcp.os.win32.utilities as _win32_utils

        _win32_utils.get_windows_executable_command = (
            _patched_get_windows_executable_command
        )
    except ImportError:
        pass  # Origin module path differs in this SDK version; stdio patch suffices.

    logger.debug("Applied MCP Windows command-resolver os.access fast-path patch")


_patch_mcp_windows_command_resolver()


# =============================================================================
# Windows MCP SDK patch — give stdio subprocesses a real stderr file descriptor
# =============================================================================
#
# ``mcp.client.stdio.stdio_client`` forwards the parent process's ``stderr``
# (``errlog``, defaulting to ``sys.stderr``) to the MCP server subprocess via
# ``anyio.open_process`` → ``subprocess.Popen(stderr=...)``. ``Popen`` resolves
# that to an OS handle by calling ``errlog.fileno()``.
#
# Under the Textual TUI (and any non-console host), ``sys.stderr`` is
# redirected to ``textual.app._PrintCapture``, whose ``fileno()`` returns
# ``-1``. ``subprocess.Popen`` dutifully converts fd ``-1`` into an invalid
# Windows handle and the child inherits a broken stderr pipe — every stdio
# MCP server then fails to spawn with ``OSError: [Errno 9] Bad file
# descriptor``. HTTP/SSE servers are unaffected (no subprocess), which is why
# only the stdio server (e.g. arxiv) shows up as failed in the loader. See
# issue #418; the same class of bug is tracked upstream as
# modelcontextprotocol/python-sdk#1103.
#
# We can't pass ``errlog`` through ``langchain-mcp-adapters`` (it calls
# ``stdio_client(server_params)`` with no ``errlog``), and the SDK's default
# is bound once at import time — which may itself already capture a
# redirected ``sys.stderr``. So we wrap ``stdio_client`` to swap in a safe
# ``errlog`` at call time whenever the configured one has no usable fileno.
# The substitute is ``sys.__stderr__`` (the real console handle) when
# available, otherwise a discarded ``os.devnull`` handle.


def _stdio_errlog_is_usable(errlog: object) -> bool:
    """Return ``True`` if *errlog* can back a subprocess ``stderr`` pipe.

    A usable errlog exposes a ``fileno()`` that resolves to a live OS file
    descriptor. Textual's ``_PrintCapture`` and similar redirected streams
    return ``-1`` (or raise), so they are rejected here. A closed stream may
    still report its former (positive) fd, so we additionally ``os.fstat``
    the descriptor to confirm it is still open.
    """
    fileno = getattr(errlog, "fileno", None)
    if not callable(fileno):
        return False
    try:
        fd = fileno()
    except Exception:
        return False
    if not isinstance(fd, int) or fd < 0:
        return False
    try:
        os.fstat(fd)
    except (OSError, OverflowError):
        return False
    return True


def _safe_stdio_errlog() -> tuple[Any, bool]:
    """Return a ``(stream, opened_by_us)`` pair for a usable stderr.

    Prefers ``sys.__stderr__`` (the original console handle, so the server's
    diagnostic output still lands where the user expects — note
    ``sys.__stderr__`` is the process's original handle and stays usable even
    while the Textual TUI controls the screen, since Textual only redirects
    ``sys.stderr``). Falls back to an ``os.devnull`` handle when even
    ``__stderr__`` is unavailable (e.g. in a GUI/pythonw host with no console).

    The second element is ``True`` when *we* allocated the stream (the
    ``os.devnull`` case) and therefore own its lifecycle; it is ``False`` for
    ``sys.__stderr__``, which is process-owned and must never be closed here.
    Callers use that flag to decide whether to close the stream after the
    stdio session exits.
    """
    dunder = getattr(sys, "__stderr__", None)
    if dunder is not None and _stdio_errlog_is_usable(dunder):
        return dunder, False
    # Last resort: discard the server's stderr so the spawn still succeeds.
    return open(os.devnull, "w", encoding="utf-8", errors="replace"), True


def _patch_mcp_stdio_errlog_safe() -> None:
    """Wrap the SDK's ``stdio_client`` to guarantee a usable ``errlog``.

    Idempotent. A no-op when the MCP SDK is absent (optional dependency).
    When the caller already supplied a usable ``errlog`` it is forwarded
    unchanged; only the unsafe default (redirected ``sys.stderr``) is
    replaced. This keeps the patch transparent for embedders that pass their
    own ``errlog`` explicitly.

    The wrapper is installed on both ``mcp.client.stdio.stdio_client`` and
    ``langchain_mcp_adapters.sessions.stdio_client``: the adapter binds the
    name via a ``from … import`` at its module load, so updating only the
    SDK module would leave an already-imported adapter pointing at the
    unwrapped function.
    """
    try:
        import mcp.client.stdio as _stdio_mod
    except ImportError:
        return  # MCP SDK not installed — nothing to patch.

    original = getattr(_stdio_mod, "stdio_client", None)
    if original is None:
        # The SDK renamed/removed stdio_client — nothing to wrap. Log so a
        # future SDK refactor doesn't silently drop this guard.
        logger.warning(
            "MCP SDK layout changed: mcp.client.stdio.stdio_client is missing; "
            "the Windows stdio errlog safety patch was NOT applied. MCP stdio "
            "tool loading may fail with [Errno 9] under a redirected stderr."
        )
        return

    if getattr(original, "_evosci_errlog_safe", False):
        return  # Already patched.

    @wraps(original)
    def _stdio_client_safe(server: Any, errlog: Any = ..., *args: Any, **kwargs: Any):
        # When the caller didn't supply a usable errlog we allocate a fallback
        # stream (sys.__stderr__ or os.devnull). The SDK never closes a
        # caller-provided errlog, so a devnull fallback would leak its fd on
        # every MCP reload. We allocate the fallback inside the async context
        # manager below so it is closed on exit — and, if the CM is discarded
        # before being entered, Python finalises the async generator and runs
        # the same ``finally``. ``errlog`` is forwarded by keyword so a future
        # SDK that inserts a positional parameter before it can't mis-bind it.
        caller_errlog = errlog
        needs_fallback = errlog is ... or not _stdio_errlog_is_usable(errlog)

        @asynccontextmanager
        async def _close_owned_errlog():
            if needs_fallback:
                # ``opened_by_us`` is True only for the os.devnull case;
                # sys.__stderr__ is process-owned and must not be closed.
                errlog, opened_by_us = _safe_stdio_errlog()
            else:
                errlog, opened_by_us = caller_errlog, False
            try:
                # Construct inside the try so a failure here still reaches the
                # finally and closes a wrapper-owned fallback stream.
                # Forward by keyword: robust against future SDK signature changes
                # that insert a positional parameter before ``errlog``.
                cm = original(server, *args, errlog=errlog, **kwargs)
                async with cm as streams:
                    yield streams
            finally:
                if opened_by_us:
                    close = getattr(errlog, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            logger.debug(
                                "Failed to close fallback stdio errlog", exc_info=True
                            )

        return _close_owned_errlog()

    _stdio_client_safe._evosci_errlog_safe = True  # type: ignore[attr-defined]
    _stdio_mod.stdio_client = _stdio_client_safe

    # langchain-mcp-adapters binds stdio_client via a ``from`` import at its
    # module load, so a pre-imported adapter keeps the unwrapped reference.
    # Re-bind it too (best-effort; ignore if the layout differs).
    try:
        import langchain_mcp_adapters.sessions as _adapter_sessions

        if getattr(_adapter_sessions, "stdio_client", None) is original:
            _adapter_sessions.stdio_client = _stdio_client_safe
    except ImportError:
        pass  # Adapter not installed — nothing extra to rebind.

    logger.debug("Applied MCP stdio errlog safety patch")


_patch_mcp_stdio_errlog_safe()


def _mcp_major_version() -> int | None:
    """Return the installed MCP SDK's major version, or None if unknown."""
    try:
        from importlib.metadata import version

        return int(version("mcp").split(".")[0])
    except Exception:
        return None


def _patch_mcp_stdio_kill_on_cancel() -> None:
    """Kill a stdio server whose shutdown wait is cancelled.

    Idempotent. A no-op when the MCP SDK is absent (optional dependency).
    In mcp 1.x the stdio shutdown closes the server's stdin, waits up to 2s
    for it to exit, and kills the process tree only when that wait times
    out. A cancellation that lands inside the wait (for example our
    ``get_tools`` deadline) skips the kill, and a server that ignores stdin
    closing is then waited on with nothing left to stop it. Wrapping the
    process's ``wait`` kills the tree in that case before the cancellation
    carries on.

    The SDK's own 2s limit also cancels ``wait``, and its kill fallback
    calls ``wait`` again, so each process is killed at most once: whichever
    of the wrapper and the SDK gets there first does the kill, and the
    other does nothing.

    Skipped on mcp 2.x, which runs its whole shutdown shielded and polls
    ``returncode`` instead of calling ``wait``.
    """
    major = _mcp_major_version()
    if major is not None and major >= 2:
        return

    try:
        import anyio
        import mcp.client.stdio as _stdio_mod
    except ImportError:
        return  # MCP SDK not installed — nothing to patch.

    create = getattr(_stdio_mod, "_create_platform_compatible_process", None)
    terminate = getattr(_stdio_mod, "_terminate_process_tree", None)
    if create is None or terminate is None:
        logger.debug(
            "MCP SDK layout changed: stdio process helpers are missing; "
            "the stdio kill-on-cancel patch was not applied"
        )
        return

    if getattr(create, "_evosci_kill_on_cancel", False):
        return  # Already patched.

    def _claim_kill(process: Any) -> bool:
        """Mark ``process`` as killed; False if it already was."""
        if getattr(process, "_evosci_killed", False):
            return False
        try:
            process._evosci_killed = True
        except AttributeError:
            pass  # Can't track it; kill as the SDK would.
        return True

    @wraps(create)
    async def _create_killable_process(*args: Any, **kwargs: Any):
        process = await create(*args, **kwargs)
        wait = process.wait

        async def _wait_or_kill():
            try:
                return await wait()
            except anyio.get_cancelled_exc_class():
                if _claim_kill(process):
                    # Bounded by the SDK's own termination timeout.
                    with anyio.CancelScope(shield=True):
                        try:
                            await terminate(process)
                        except Exception:
                            logger.debug(
                                "Failed to kill cancelled MCP stdio server",
                                exc_info=True,
                            )
                raise

        try:
            process.wait = _wait_or_kill
        except AttributeError:
            logger.debug("MCP stdio process does not allow wrapping wait()")
        return process

    @wraps(terminate)
    async def _terminate_once(process: Any, *args: Any, **kwargs: Any):
        if _claim_kill(process):
            await terminate(process, *args, **kwargs)

    _create_killable_process._evosci_kill_on_cancel = True  # type: ignore[attr-defined]
    _stdio_mod._create_platform_compatible_process = _create_killable_process
    _stdio_mod._terminate_process_tree = _terminate_once
    logger.debug("Applied MCP stdio kill-on-cancel patch")


_patch_mcp_stdio_kill_on_cancel()


# =============================================================================
# Constants
# =============================================================================

# Regex for ${VAR} env var interpolation
ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")

# Supported transport protocols
VALID_TRANSPORTS = {"stdio", "http", "streamable_http", "sse", "websocket"}

# URL-based transports (share the same connection shape)
_URL_TRANSPORTS = {"http", "streamable_http", "sse", "websocket"}

# Upper bound on simultaneous ``get_tools`` attempts in :func:`_load_tools`.
# Keeps stdio-server fleets from spawning 20+ subprocesses at once while
# still parallelizing the common 3–7 server case to completion.
_MAX_CONCURRENT_CONNECTIONS = 8

# Per-server time limit for ``get_tools`` inside ``langgraph dev`` (see
# :func:`_get_tools_timeout`).  A server that hasn't answered by then is
# skipped like one that failed, so a single slow server can't hold up the
# rest.  The server builds its graphs, and so loads MCP tools, before
# answering ``/ok``, so this is sized for the 60s health budget in
# ``start_langgraph_dev`` while still leaving room for a first ``npx -y``
# launch that downloads its package.  Shutting down a stdio server that timed
# out adds up to ~4s on top (the MCP SDK's termination waits).
#
# It is not a hard bound on the whole load: the limit starts once a server
# gets a slot under ``_MAX_CONCURRENT_CONNECTIONS``, so with more than that
# many servers hanging, the ones queued behind them can still push the load
# past 60s.
_SERVER_GET_TOOLS_TIMEOUT_SECONDS = 40

# Healthy remote servers usually list tools within a few seconds; 20s leaves
# room for hosted cold starts without holding up CLI/TUI startup for 40s.
# Unlike stdio servers, they need no first-launch package download.
_NETWORK_GET_TOOLS_TIMEOUT_SECONDS = 20

# Env vars forwarded to stdio MCP subprocesses on top of the MCP SDK's
# minimal default set (HOME/PATH/USER/…). Without this, servers behind
# a proxy or with a custom CA bundle silently fail with long timeouts.
# User-provided ``env`` still wins via dict merge.
_STDIO_FORWARDED_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
)


def _get_mcp_config_dir() -> Path:
    """Get the MCP configuration directory, respecting XDG_CONFIG_HOME."""
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "evoscientist"
    return Path.home() / ".config" / "evoscientist"


# User-level config path
USER_CONFIG_DIR = _get_mcp_config_dir()
USER_MCP_CONFIG = USER_CONFIG_DIR / "mcp.yaml"


# =============================================================================
# Environment variable interpolation
# =============================================================================


def _interpolate_env(value: str) -> str:
    """Replace ``${VAR}`` patterns with environment variable values.

    Missing variables are replaced with an empty string and a warning is logged.
    """

    def _replace(match: re.Match) -> str:
        var = match.group(1)
        val = os.environ.get(var)
        if val is None:
            logger.warning("MCP config: env var $%s is not set", var)
            return ""
        return val

    return ENV_VAR_RE.sub(_replace, value)


def _interpolate_value(value: Any) -> Any:
    """Recursively interpolate env vars in strings, dicts, and lists."""
    if isinstance(value, str):
        return _interpolate_env(value)
    if isinstance(value, dict):
        return {k: _interpolate_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_value(v) for v in value]
    return value


# =============================================================================
# User config persistence
# =============================================================================


def _load_user_config() -> dict[str, Any]:
    """Load the user-level MCP config, returning an empty dict if absent."""
    if USER_MCP_CONFIG.is_file():
        try:
            data = yaml.safe_load(USER_MCP_CONFIG.read_text(encoding="utf-8")) or {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _save_user_config(config: dict[str, Any]) -> None:
    """Write *config* to the user-level MCP config file."""
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    USER_MCP_CONFIG.write_text(
        yaml.dump(config, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


# =============================================================================
# CRUD operations
# =============================================================================


def add_mcp_server(
    name: str,
    transport: str,
    *,
    command: str | None = None,
    args: list[str] | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    tools: list[str] | None = None,
    expose_to: list[str] | None = None,
) -> dict[str, Any]:
    """Add or replace an MCP server in the user config.

    Returns the server entry that was written.
    """
    if transport not in VALID_TRANSPORTS:
        raise ValueError(
            f"Unknown transport {transport!r}. "
            f"Must be one of: {', '.join(sorted(VALID_TRANSPORTS))}"
        )

    entry: dict[str, Any] = {"transport": transport}

    if transport == "stdio":
        if not command:
            raise ValueError("stdio transport requires a command")
        entry["command"] = command
        entry["args"] = args or []
        if env:
            entry["env"] = env
    else:
        if not url:
            raise ValueError(f"{transport} transport requires a url")
        entry["url"] = url
        if headers:
            entry["headers"] = headers

    if tools:
        entry["tools"] = tools
    if expose_to:
        entry["expose_to"] = expose_to

    user_cfg = _load_user_config()
    user_cfg[name] = entry
    _save_user_config(user_cfg)
    return entry


def edit_mcp_server(name: str, **fields: Any) -> dict[str, Any]:
    """Update fields on an existing MCP server entry.

    Only the provided *fields* are changed; everything else is preserved.
    Passing ``None`` for a field removes it.

    Returns the updated entry.

    Raises:
        KeyError: if *name* doesn't exist in the user config.
        ValueError: on invalid transport or missing required fields.
    """
    user_cfg = _load_user_config()
    if name not in user_cfg:
        raise KeyError(f"MCP server {name!r} not found in user config")

    entry = user_cfg[name]

    for key, value in fields.items():
        if value is None:
            entry.pop(key, None)
        else:
            entry[key] = value

    # Re-validate after edits
    transport = entry.get("transport", "")
    if transport and transport not in VALID_TRANSPORTS:
        raise ValueError(
            f"Unknown transport {transport!r}. "
            f"Must be one of: {', '.join(sorted(VALID_TRANSPORTS))}"
        )
    if transport == "stdio" and not entry.get("command"):
        raise ValueError("stdio transport requires a command")
    if transport in _URL_TRANSPORTS and not entry.get("url"):
        raise ValueError(f"{transport} transport requires a url")

    user_cfg[name] = entry
    _save_user_config(user_cfg)
    return entry


def remove_mcp_server(name: str) -> bool:
    """Remove an MCP server from the user config.

    Returns True if removed, False if it didn't exist.
    """
    user_cfg = _load_user_config()
    if name not in user_cfg:
        return False
    del user_cfg[name]
    _save_user_config(user_cfg)
    return True


# =============================================================================
# CLI argument parsing
# =============================================================================


def _infer_transport(target: str) -> str:
    """Return transport type inferred from *target* URL scheme."""
    if target.startswith(("ws://", "wss://")):
        return "websocket"
    if target.startswith(("http://", "https://")):
        return "http"
    return "stdio"


def build_mcp_add_kwargs(
    name: str,
    target: str,
    extra_args: list[str] | None = None,
    transport: str | None = None,
    tools: list[str] | None = None,
    expose_to: list[str] | None = None,
    headers: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    """Build kwargs dict for :func:`add_mcp_server` from structured parameters.

    If *transport* is ``None`` it is inferred from *target* (URL → ``http``,
    otherwise ``stdio``).
    """
    if transport is None:
        transport = _infer_transport(target)
    kwargs: dict = {"name": name, "transport": transport}
    if transport == "stdio":
        kwargs["command"] = target
        kwargs["args"] = list(extra_args) if extra_args else []
        if env:
            kwargs["env"] = env
    else:
        kwargs["url"] = target
        if headers:
            kwargs["headers"] = headers
    if tools:
        kwargs["tools"] = tools
    if expose_to:
        kwargs["expose_to"] = expose_to
    return kwargs


def build_mcp_edit_fields(
    transport: str | None = None,
    command: str | None = None,
    url: str | None = None,
    tools: str | None = None,
    expose_to: str | None = None,
    headers: list[str] | None = None,
    env: list[str] | None = None,
) -> dict:
    """Build fields dict for :func:`edit_mcp_server` from structured parameters.

    *tools* and *expose_to* accept the string ``"none"`` to clear the field,
    or a comma-separated list.  *headers* and *env* are lists of
    ``"Key:Value"`` / ``"KEY=VALUE"`` strings respectively.
    """
    fields: dict = {}
    if transport is not None:
        fields["transport"] = transport
    if command is not None:
        fields["command"] = command
    if url is not None:
        fields["url"] = url
    if tools is not None:
        fields["tools"] = (
            None
            if tools == "none"
            else [t.strip() for t in tools.split(",") if t.strip()]
        )
    if expose_to is not None:
        fields["expose_to"] = (
            None
            if expose_to == "none"
            else [a.strip() for a in expose_to.split(",") if a.strip()]
        )
    if headers:
        hdr: dict[str, str] = {}
        for h in headers:
            if ":" in h:
                k, v = h.split(":", 1)
                hdr[k.strip()] = v.strip()
        if hdr:
            fields["headers"] = hdr
    if env:
        env_dict: dict[str, str] = {}
        for e in env:
            if "=" in e:
                k, v = e.split("=", 1)
                env_dict[k.strip()] = v.strip()
        if env_dict:
            fields["env"] = env_dict
    return fields


def parse_mcp_add_args(tokens: list[str]) -> dict:
    """Parse CLI tokens for ``/mcp add`` into kwargs for :func:`add_mcp_server`.

    Syntax::

        <name> <command-or-url> [extra-args...]
            [--transport T] [--tools t1,t2] [--expose-to a1,a2]
            [--header Key:Value]... [--env KEY=VALUE]...

    Transport defaults to ``stdio`` for commands and ``http`` for URLs.
    """
    if len(tokens) < 2:
        raise ValueError(
            "Usage: <name> <command-or-url> [args...]\n"
            "  Options: --transport T  --tools t1,t2  --expose-to agent1,agent2  --header Key:Value  --env KEY=VALUE"
        )

    name = tokens[0]

    positional: list[str] = []
    transport: str | None = None
    tools: list[str] | None = None
    expose_to: list[str] | None = None
    headers: dict[str, str] = {}
    env: dict[str, str] = {}

    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("--transport", "-T") and i + 1 < len(tokens):
            transport = tokens[i + 1]
            i += 2
        elif tok == "--tools" and i + 1 < len(tokens):
            tools = [t.strip() for t in tokens[i + 1].split(",") if t.strip()]
            i += 2
        elif tok == "--expose-to" and i + 1 < len(tokens):
            expose_to = [a.strip() for a in tokens[i + 1].split(",") if a.strip()]
            i += 2
        elif tok == "--header" and i + 1 < len(tokens):
            kv = tokens[i + 1]
            if ":" in kv:
                k, v = kv.split(":", 1)
                headers[k.strip()] = v.strip()
            i += 2
        elif tok == "--env" and i + 1 < len(tokens):
            kv = tokens[i + 1]
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k.strip()] = v.strip()
            i += 2
        elif tok == "--env-ref" and i + 1 < len(tokens):
            env[tokens[i + 1]] = "${" + tokens[i + 1] + "}"
            i += 2
        elif tok == "--":
            i += 1  # skip -- separator (used by shells, not meaningful here)
        else:
            positional.append(tok)
            i += 1

    if not positional:
        raise ValueError("A command or URL is required after the server name")

    return build_mcp_add_kwargs(
        name=name,
        target=positional[0],
        extra_args=positional[1:] or None,
        transport=transport,
        tools=tools,
        expose_to=expose_to,
        headers=headers or None,
        env=env or None,
    )


def parse_mcp_edit_args(tokens: list[str]) -> tuple[str, dict]:
    """Parse CLI tokens for ``/mcp edit`` into (name, fields).

    Syntax::

        <name> [--transport T] [--command C] [--url U]
               [--tools t1,t2] [--tools none] [--expose-to a1,a2]
               [--header Key:Value]... [--env KEY=VALUE]...

    ``--tools none`` and ``--expose-to none`` clear those fields.
    """
    if not tokens:
        raise ValueError(
            "Usage: <name> [--transport T] [--command C] [--url U] "
            "[--tools t1,t2] [--expose-to a1,a2] [--header K:V] [--env K=V]"
        )

    name = tokens[0]

    # Parse tokens into raw values
    transport_val: str | None = None
    command_val: str | None = None
    url_val: str | None = None
    args_val: list[str] | None = None
    tools_val: str | None = None
    expose_to_val: str | None = None
    header_list: list[str] = []
    env_list: list[str] = []

    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--transport" and i + 1 < len(tokens):
            transport_val = tokens[i + 1]
            i += 2
        elif tok == "--command" and i + 1 < len(tokens):
            command_val = tokens[i + 1]
            i += 2
        elif tok == "--url" and i + 1 < len(tokens):
            url_val = tokens[i + 1]
            i += 2
        elif tok == "--args" and i + 1 < len(tokens):
            args_val = tokens[i + 1].split(",")
            i += 2
        elif tok == "--tools" and i + 1 < len(tokens):
            tools_val = tokens[i + 1]
            i += 2
        elif tok == "--expose-to" and i + 1 < len(tokens):
            expose_to_val = tokens[i + 1]
            i += 2
        elif tok == "--header" and i + 1 < len(tokens):
            header_list.append(tokens[i + 1])
            i += 2
        elif tok == "--env" and i + 1 < len(tokens):
            env_list.append(tokens[i + 1])
            i += 2
        else:
            i += 1

    fields = build_mcp_edit_fields(
        transport=transport_val,
        command=command_val,
        url=url_val,
        tools=tools_val,
        expose_to=expose_to_val,
        headers=header_list or None,
        env=env_list or None,
    )
    if args_val is not None:
        fields["args"] = args_val

    if not fields:
        raise ValueError(
            "No fields to edit. Use --transport, --command, --url, --tools, --expose-to, etc."
        )

    return name, fields


# =============================================================================
# Config loading & merging
# =============================================================================


def load_mcp_config() -> dict[str, Any]:
    """Load MCP configuration from user config.

    Reads ``~/.config/evoscientist/mcp.yaml`` and interpolates ``${VAR}``
    environment variable references.

    Returns an empty dict if no servers are configured (MCP is optional).
    """
    if not USER_MCP_CONFIG.is_file():
        return {}

    try:
        data = yaml.safe_load(USER_MCP_CONFIG.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return {}
    except Exception as exc:
        logger.warning("Failed to load MCP config %s: %s", USER_MCP_CONFIG, exc)
        return {}

    return _interpolate_value(data)


def _resolve_command(command: str) -> str:
    """Resolve a stdio command to its full path.

    Checks PATH first, then the current Python environment's bin directory
    (handles conda/venv envs where newly installed binaries may not be on PATH).
    Returns the original command string if not found (let the OS report the error).
    """
    if os.path.isabs(command):
        return command
    found = shutil.which(command)
    if found:
        return found
    candidate = os.path.join(os.path.dirname(sys.executable), command)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return command


_NODE_COMMANDS = frozenset({"node", "npx", "npm"})


def _ensure_node_for_stdio(config: dict[str, Any]) -> None:
    """Install the private Node when a stdio server runs ``node`` / ``npx`` /
    ``npm`` and that command is not on PATH.

    Covers users who never ran ``EvoSci setup``. A failed install is logged and
    the server then fails to start as it would without Node. Never runs inside
    ``langgraph dev`` (``EVOSCIENTIST_SERVER_PROCESS`` set): a download there would
    race the server's health deadline with its progress hidden in the server
    log, so ``start_langgraph_dev`` runs this before spawning instead.
    """
    if os.environ.get("EVOSCIENTIST_SERVER_PROCESS") == "1":
        return
    missing = [
        name
        for name, server in config.items()
        if server.get("transport") == "stdio"
        and (command := str(server.get("command", "")))
        and not os.path.isabs(command)
        and Path(command).stem.lower() in _NODE_COMMANDS
        and shutil.which(command) is None
    ]
    if not missing:
        return
    from ..setup.node import activate_runtime, ensure_node, log_progress
    from ..setup.protocol import StageError

    try:
        ensure_node(progress=log_progress(logger))
    except StageError as exc:
        logger.warning(
            f"MCP servers {', '.join(missing)} need Node.js, and installing it "
            f"failed: {exc.message}. Run 'EvoSci setup' to retry."
        )
        return
    except Exception as exc:
        # Unpacking can still raise outside StageError (zipfile's
        # NotImplementedError / RuntimeError, lzma.LZMAError). This hook runs
        # before every MCP load; letting it raise would drop all servers, not
        # just the Node ones.
        logger.warning(
            f"MCP servers {', '.join(missing)} need Node.js, and installing it "
            f"failed unexpectedly: {exc!r}. Run 'EvoSci setup' to retry.",
            exc_info=True,
        )
        return
    activate_runtime()


def _build_connections(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert YAML config to ``MultiServerMCPClient`` connections format.

    Unknown transports are skipped with a warning.
    """
    connections: dict[str, dict[str, Any]] = {}

    for name, server in config.items():
        transport = server.get("transport", "")

        if transport == "stdio":
            conn: dict[str, Any] = {
                "transport": "stdio",
                "command": _resolve_command(server.get("command", "")),
                "args": server.get("args", []),
            }
            forwarded = {
                k: os.environ[k] for k in _STDIO_FORWARDED_ENV_VARS if k in os.environ
            }
            user_env = server.get("env") or {}
            merged = {**forwarded, **user_env}
            if merged:
                conn["env"] = merged
            connections[name] = conn

        elif transport in _URL_TRANSPORTS:
            conn = {
                "transport": transport,
                "url": server.get("url", ""),
            }
            if "headers" in server:
                conn["headers"] = server["headers"]
            connections[name] = conn

        else:
            logger.warning(
                "MCP server %r: unknown transport %r, skipping", name, transport
            )

    return connections


# =============================================================================
# Tool loading, filtering & routing
# =============================================================================


def _filter_tools(tools: list, allowed_names: list[str] | None) -> list:
    """Filter tools by allowlist with wildcard support.

    If *allowed_names* is ``None``, all tools pass through.

    Supports glob-style wildcards:
    - ``*`` matches any sequence of characters
    - ``?`` matches any single character
    - ``[seq]`` matches any character in seq
    - ``[!seq]`` matches any character not in seq

    Examples:
    - ``*_exa`` matches ``web_search_exa``, ``get_code_context_exa``
    - ``read_*`` matches ``read_file``, ``read_directory``
    - ``tool_[0-9]`` matches ``tool_1``, ``tool_2``, etc.
    """
    if allowed_names is None:
        return tools

    # Check if any pattern contains wildcard characters
    has_wildcards = any(
        any(char in pattern for char in "*?[]") for pattern in allowed_names
    )

    if not has_wildcards:
        # Fast path: exact matching with set lookup
        allowed_set = set(allowed_names)
        return [t for t in tools if t.name in allowed_set]

    # Wildcard matching: check each tool against all patterns
    filtered = []
    for tool in tools:
        if any(fnmatch.fnmatch(tool.name, pattern) for pattern in allowed_names):
            filtered.append(tool)
    return filtered


def _route_tools(
    config: dict[str, Any],
    server_tools: dict[str, list],
) -> dict[str, list]:
    """Group filtered tools by target agent.

    Args:
        config: Full MCP config dict (server name -> server settings).
        server_tools: server name -> list of LangChain tools from that server.

    Returns:
        Dict mapping agent name -> list of tools. Key ``"main"`` targets the
        main EvoScientist agent; other keys match subagent names.
    """
    by_agent: dict[str, list] = {}

    for server_name, tools in server_tools.items():
        server_cfg = config.get(server_name, {})

        # Apply tool name filter
        allowed = server_cfg.get("tools")  # None means all
        filtered = _filter_tools(tools, allowed)

        # Determine target agents
        expose_to = server_cfg.get("expose_to", ["main"])
        if isinstance(expose_to, str):
            expose_to = [expose_to]

        for agent_name in expose_to:
            by_agent.setdefault(agent_name, []).extend(filtered)

    return by_agent


ProgressCallback = Callable[[str, str, str], None]
"""Per-server progress callback: ``(event, server_name, detail)``.

- ``event="start"``   — connection attempt has begun.  ``detail`` is empty.
- ``event="success"`` — tools fetched.  ``detail`` is the count as a string.
- ``event="error"``   — failed.  ``detail`` is the exception message.
"""


def _get_tools_timeout(transport: str) -> float | None:
    """The default ``get_tools`` limit for this transport and process.

    URL transports always have a discovery deadline. Stdio servers have a
    deadline only inside ``langgraph dev`` (``EVOSCIENTIST_SERVER_PROCESS=1``),
    whose health check gives up after 60s. CLI/TUI stdio discovery may wait
    as long as needed for first-launch package downloads.
    """
    if transport in _URL_TRANSPORTS:
        return _NETWORK_GET_TOOLS_TIMEOUT_SECONDS
    if os.environ.get("EVOSCIENTIST_SERVER_PROCESS") == "1":
        return _SERVER_GET_TOOLS_TIMEOUT_SECONDS
    return None


async def _load_tools(
    config: dict[str, Any],
    *,
    on_progress: ProgressCallback | None = None,
    timeout: float | None = None,
) -> dict[str, list]:
    """Connect to MCP servers and retrieve tools.

    Returns a dict of server name -> list of LangChain tools.

    Args:
        timeout: Seconds each server gets to return its tools before it is
            skipped like a failed one. With ``None``, network servers use the
            default discovery budget. Stdio servers wait as long as needed in
            the CLI/TUI, or use the server health budget in ``langgraph dev``.

    Raises:
        ImportError: if ``langchain-mcp-adapters`` is not installed.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        raise ImportError(
            "MCP servers are configured but langchain-mcp-adapters is not installed.\n"
            "Install with: pip install langchain-mcp-adapters"
        ) from None

    # May download Node (in the CLI process only); off the event loop so the
    # running session stays responsive.
    await asyncio.to_thread(_ensure_node_for_stdio, config)
    connections = _build_connections(config)
    if not connections:
        return {}

    client = MultiServerMCPClient(connections)  # type: ignore[invalid-argument-type]

    def _report(event: str, name: str, detail: str = "") -> None:
        if on_progress is None:
            return
        try:
            on_progress(event, name, detail)
        except Exception:
            # Progress callbacks are UI glue — never let their bugs break
            # the actual MCP load.
            logger.debug("MCP progress callback raised", exc_info=True)

    # Cap in-flight connections so a user with many servers doesn't
    # spawn all their stdio subprocesses at once (fd/ulimit pressure,
    # load spikes).  The cap still parallelizes ~an order of magnitude
    # better than the old serial loop.
    sem = asyncio.Semaphore(_MAX_CONCURRENT_CONNECTIONS)

    async def _fetch(name: str) -> tuple[str, list]:
        async with sem:
            _report("start", name)
            server_timeout = (
                timeout
                if timeout is not None
                else _get_tools_timeout(connections[name]["transport"])
            )
            deadline = asyncio.timeout(server_timeout)
            try:
                async with deadline:
                    tools = await client.get_tools(server_name=name)
                logger.info("MCP server %r: loaded %d tool(s)", name, len(tools))
                _report("success", name, str(len(tools)))
                return name, tools
            except Exception as exc:
                # Only our own deadline gets the "timed out" message.  A
                # TimeoutError raised inside ``get_tools`` (a websocket
                # handshake, ``socket.timeout``) keeps its real error.
                if isinstance(exc, TimeoutError) and deadline.expired():
                    detail = f"timed out after {server_timeout}s"
                else:
                    detail = str(exc)
                # When the caller wired up ``on_progress`` they own the
                # user-facing display; downgrade the logger so we don't
                # double-print.
                if on_progress is None:
                    logger.warning(
                        "MCP server %r: failed to load tools: %s", name, detail
                    )
                else:
                    logger.debug(
                        "MCP server %r: failed to load tools: %s", name, detail
                    )
                _report("error", name, detail)
                return name, []

    # ``return_exceptions=False`` is fine because ``_fetch`` already
    # swallows errors per server.
    results = await asyncio.gather(*(_fetch(name) for name in connections))
    return dict(results)


async def aload_mcp_tools(
    config: dict[str, Any] | None = None,
    *,
    on_progress: ProgressCallback | None = None,
) -> dict[str, list]:
    """Async version of :func:`load_mcp_tools`.

    Prefer this when already inside an async context (e.g. Jupyter, async CLI).

    Args:
        config: Optional pre-loaded MCP config dict.  When ``None``,
            loads from ``~/.config/evoscientist/mcp.yaml``.
        on_progress: Optional callback invoked per server with
            ``(event, server_name, detail)``.  See :data:`ProgressCallback`.
    """
    if config is None:
        config = load_mcp_config()
    if not config:
        return {}
    try:
        server_tools = await _load_tools(config, on_progress=on_progress)
    except Exception as exc:
        logger.warning("MCP tool loading failed: %s", exc)
        return {}
    return _route_tools(config, server_tools)


def load_mcp_tools(
    config: dict[str, Any] | None = None,
    *,
    on_progress: ProgressCallback | None = None,
    runtime: AsyncRuntime | None = None,
) -> dict[str, list]:
    """Load MCP tools and return them grouped by target agent.

    This is the main synchronous entry point. It:
    1. Loads user config from ``~/.config/evoscientist/mcp.yaml``
    2. Connects to each configured MCP server
    3. Filters tools per server allowlist
    4. Routes tools to target agents

    Args:
        config: Optional pre-loaded MCP config dict.  When ``None``,
            loads from ``~/.config/evoscientist/mcp.yaml``.  Passing a
            pre-loaded config avoids duplicate env-var interpolation
            warnings when the caller has already loaded the config.
        on_progress: Optional callback invoked per server with
            ``(event, server_name, detail)``.  See :data:`ProgressCallback`.
        runtime: Runtime that owns MCP discovery work. When omitted, this
            function creates one scoped to this call. The returned adapters
            open a fresh MCP session for each tool call and do not retain the
            discovery loop.

    Returns:
        Dict mapping agent name -> list of LangChain ``BaseTool`` objects.
        Key ``"main"`` = main agent. Other keys = subagent names.
        Returns empty dict if no MCP servers are configured.
    """
    if config is None:
        config = load_mcp_config()
    if not config:
        return {}

    if runtime is None:
        with AsyncRuntime(thread_name="evosci-mcp-runtime") as owned_runtime:
            return load_mcp_tools(
                config,
                on_progress=on_progress,
                runtime=owned_runtime,
            )

    try:
        server_tools = runtime.run_sync(
            lambda: _load_tools(config, on_progress=on_progress)
        )
    except AsyncRuntimeError as exc:
        # A bridge lifecycle/call-site error is not an MCP availability
        # failure.  In particular, hiding a running-loop violation here makes
        # callers cache an empty tool set for the rest of the process.
        if "cannot block a running event loop" in str(exc):
            raise AsyncRuntimeError(
                "load_mcp_tools() cannot run inside an async context; use "
                "`await aload_mcp_tools(config, on_progress=...)` instead"
            ) from exc
        raise
    except Exception as exc:
        logger.warning("MCP tool loading failed: %s", exc)
        return {}

    return _route_tools(config, server_tools)
