"""The research environment setup stage.

When no usable ``python`` is on PATH, the agent's shell gets a virtual
environment under ``<DATA_DIR>/envs/default`` with numpy, pandas, matplotlib
and scipy, so ``python script.py`` and ``pip install`` work as the prompts
teach. A user's own ``python`` (conda, venv, system) always wins.

The environment reaches only the agent's ``execute`` and ``run_in_background``
commands, through :func:`research_env_overrides`; EvoScientist's own child
processes keep their PATH.

A venv is used rather than the Python EvoScientist runs on: uv-managed Pythons
are marked ``EXTERNALLY-MANAGED``, so pip refuses to install into them. A venv
hard-codes its own path, so it is built in place, and the ready marker is
written only after the import check passed.
"""

from __future__ import annotations

import configparser
import functools
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from .protocol import Emitter, StageError, StageResult, make_event

logger = logging.getLogger(__name__)

PACKAGES = ("numpy", "pandas", "matplotlib", "scipy")
CN_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"

_READY_MARKER = ".evoscientist-ready"
_PYTHON_PROBE_TIMEOUT = 10
_VENV_TIMEOUT = 300
_PIP_TIMEOUT = 1800
# The first matplotlib import builds its font cache.
_IMPORT_TIMEOUT = 300
_IMPORT_CHECK = (
    f"import platform, {', '.join(PACKAGES)}; print(platform.python_version())"
)
_WINDOWSAPPS_RE = re.compile(r"[\\/]microsoft[\\/]windowsapps[\\/]", re.IGNORECASE)
# Publisher ids of the Python Software Foundation's packages: the Python
# install manager and the Microsoft Store CPython builds. Windows derives a
# publisher id from the package's signing certificate.
_PSF_PUBLISHER_IDS = frozenset({"3847v3x7pw1km", "qbz5n2kfra8p0"})
# Win32 constants for reading an app execution alias.
_APPEXECLINK_TAG = 0x8000001B
_FSCTL_GET_REPARSE_POINT = 0x000900A8
_FILE_READ_ATTRIBUTES = 0x80
_FILE_SHARE_ALL = 0x7
_OPEN_EXISTING = 3
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000

ProgressFn = Callable[[float, str], None]


# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
def env_dir() -> Path:
    """``<DATA_DIR>/envs/default``, read at call time so an overridden DATA_DIR applies."""
    from .. import paths

    return paths.DATA_DIR / "envs" / "default"


def _bin_dir(env: Path) -> Path:
    return env / "Scripts" if os.name == "nt" else env / "bin"


def _env_python(env: Path) -> Path:
    return _bin_dir(env) / ("python.exe" if os.name == "nt" else "python")


def _pip_config(env: Path) -> Path:
    # pip reads a site config from ``sys.prefix``, which is the venv.
    return env / ("pip.ini" if os.name == "nt" else "pip.conf")


def is_ready(env: Path | None = None) -> bool:
    """The environment passed its import check and its ``python`` still exists."""
    env = env or env_dir()
    return (env / _READY_MARKER).is_file() and _env_python(env).is_file()


# --------------------------------------------------------------------------- #
# Subprocesses
# --------------------------------------------------------------------------- #
def _run(cmd: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` capturing combined output. Raises OSError / SubprocessError."""
    return subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        errors="replace",
        timeout=timeout,
    )


def _runs(
    cmd: Sequence[str], timeout: float
) -> subprocess.CompletedProcess[str] | None:
    """``cmd``'s result when it exits 0, else None (also when it cannot start)."""
    try:
        result = _run(cmd, timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return result if result.returncode == 0 else None


def _last_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else "no output"


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #
def _parse_app_exec_link(data: bytes) -> str | None:
    """The package family name from an app execution alias reparse buffer.

    Layout: ULONG tag, USHORT data length, USHORT reserved, then the data:
    ULONG version followed by NUL-separated UTF-16 strings, the first being
    the package family name.
    """
    if len(data) < 12:
        return None
    tag, length = struct.unpack_from("<IH", data)
    if tag != _APPEXECLINK_TAG:
        return None
    strings = data[12 : 8 + length].decode("utf-16-le", errors="replace")
    return strings.split("\0", 1)[0] or None


def _app_exec_link_package(path: str) -> str | None:
    """The package behind a Windows app execution alias, read without running it.

    None when ``path`` is not such an alias or cannot be read.
    """
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.DeviceIoControl.restype = wintypes.BOOL
    kernel32.DeviceIoControl.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    handle = kernel32.CreateFileW(
        path,
        _FILE_READ_ATTRIBUTES,
        _FILE_SHARE_ALL,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_OPEN_REPARSE_POINT | _FILE_FLAG_BACKUP_SEMANTICS,
        None,
    )
    if handle in (None, wintypes.HANDLE(-1).value):
        return None
    try:
        buffer = ctypes.create_string_buffer(16 * 1024)
        returned = wintypes.DWORD()
        ok = kernel32.DeviceIoControl(
            handle,
            _FSCTL_GET_REPARSE_POINT,
            None,
            0,
            buffer,
            len(buffer),
            ctypes.byref(returned),
            None,
        )
    finally:
        kernel32.CloseHandle(handle)
    if not ok:
        return None
    return _parse_app_exec_link(buffer.raw[: returned.value])


def _is_untrusted_alias(path: str) -> bool:
    """True for a ``WindowsApps`` alias that must never run.

    Only aliases of Python Software Foundation packages (the Python install
    manager, Store CPython) are real Pythons. Any other, including the Store's
    ``python`` alias, which opens the Microsoft Store when run, or an alias
    whose package cannot be read, is never run.
    """
    if not _WINDOWSAPPS_RE.search(path):
        return False
    package = _app_exec_link_package(path)
    if package is None:
        return True
    name, _, publisher = package.rpartition("_")
    return not (
        name.startswith("PythonSoftwareFoundation.") and publisher in _PSF_PUBLISHER_IDS
    )


def find_usable_python() -> str | None:
    """The ``python`` the agent's shell would run, as an absolute path, if it
    runs; else None.

    Our own environment is left out of the search. A ``WindowsApps`` alias
    that is not a Python Software Foundation package is never run (see
    :func:`_is_untrusted_alias`); when it comes first on PATH the shell would
    run it too, so there is no usable ``python``.
    """
    own_bin = os.path.normcase(str(_bin_dir(env_dir())))
    search = os.pathsep.join(
        p
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p and os.path.normcase(p.rstrip("\\/")) != own_bin
    )
    found = shutil.which("python", path=search)
    if found is None or _is_untrusted_alias(found):
        return None
    # A relative PATH entry means another file in the agent's working dir.
    found = os.path.abspath(found)
    if _runs([found, "-c", "import sys"], _PYTHON_PROBE_TIMEOUT) is None:
        return None
    return found


def _import_check(env: Path) -> str | None:
    """The environment's Python version when all packages import, else None."""
    result = _runs([str(_env_python(env)), "-c", _IMPORT_CHECK], _IMPORT_TIMEOUT)
    return _last_line(result.stdout) if result is not None else None


# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #
def _site_index_url(env: Path) -> str | None:
    """``global.index-url`` from the environment's own pip config, if set."""
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(_pip_config(env), encoding="utf-8")
    except configparser.Error:
        return None
    return parser.get("global", "index-url", fallback=None)


def _sync_pip_config(env: Path, mirror: str) -> None:
    """Bring the environment's ``global.index-url`` in line with ``mirror``.

    Uses ``pip config --site``, which changes only that entry, and runs pip
    only when the entry has to change. Without ``mirror: cn`` only our own
    mirror entry is removed; an index the user set there stays.
    """
    current = _site_index_url(env)
    if mirror == "cn":
        if current == CN_INDEX_URL:
            return
        action = ["set", "global.index-url", CN_INDEX_URL]
    else:
        if current != CN_INDEX_URL:
            return
        action = ["unset", "global.index-url"]
    cmd = [str(_env_python(env)), "-m", "pip", "config", "--site", *action]
    try:
        result = _run(cmd, _PYTHON_PROBE_TIMEOUT * 3)
    except (OSError, subprocess.SubprocessError) as exc:
        raise StageError(
            "install_failed", f"Could not update the pip config: {exc}"
        ) from exc
    if result.returncode != 0:
        raise StageError(
            "install_failed",
            f"Could not update the pip config: {_last_line(result.stdout)}",
        )


def _create_venv(env: Path) -> None:
    cmd = [sys.executable, "-m", "venv", str(env)]
    try:
        result = _run(cmd, _VENV_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        raise StageError("install_failed", f"Could not create {env}: {exc}") from exc
    if result.returncode == 0:
        logger.info(f"python -m venv output:\n{result.stdout}")
        return
    # The error event keeps one line; the cause can be earlier in the output.
    logger.warning(f"python -m venv failed:\n{result.stdout}")
    message = f"Could not create {env}: {_last_line(result.stdout)}"
    if "ensurepip" in result.stdout:
        message += (
            " This Python has no ensurepip; on Debian and Ubuntu install the"
            " python3-venv package."
        )
    raise StageError("install_failed", message)


def _pip_install(env: Path, mirror: str) -> None:
    """Install :data:`PACKAGES` without upgrading what is already there."""
    cmd = [
        str(_env_python(env)),
        "-m",
        "pip",
        "install",
        "--only-binary",
        ":all:",
        "--disable-pip-version-check",
        "--no-input",
        *PACKAGES,
    ]
    if mirror == "cn":
        cmd += ["--index-url", CN_INDEX_URL]
    try:
        result = _run(cmd, _PIP_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        raise StageError("install_failed", f"pip install failed: {exc}") from exc
    if result.returncode == 0:
        logger.info(f"pip install output:\n{result.stdout}")
    else:
        # The error event keeps pip's last line; network errors (offline, a
        # blocked index) come earlier in the output.
        logger.warning(f"pip install failed:\n{result.stdout}")
        raise StageError(
            "install_failed", f"pip install failed: {_last_line(result.stdout)}"
        )


def _repair(env: Path, mirror: str, report: ProgressFn) -> str | None:
    """Try to fix a ready environment in place; its version, or None to rebuild.

    A Python that does not start (e.g. its base interpreter was removed) cannot
    be repaired. Otherwise missing packages are installed; a pip failure raises
    and leaves the environment and its marker as they are, so the agent keeps a
    mostly working Python.
    """
    python = str(_env_python(env))
    if _runs([python, "-c", "import sys"], _PYTHON_PROBE_TIMEOUT) is None:
        return None
    # Before pip, so the config follows the mirror even when pip fails.
    _sync_pip_config(env, mirror)
    report(0.2, "Installing missing packages")
    _pip_install(env, mirror)
    report(0.9, "Checking the packages")
    return _import_check(env)


def _build(env: Path, mirror: str, report: ProgressFn) -> str:
    try:
        (env / _READY_MARKER).unlink(missing_ok=True)
        if env.exists():
            shutil.rmtree(env)
    except OSError as exc:
        raise StageError("install_failed", f"Could not remove {env}: {exc}") from exc
    report(0.1, "Creating the virtual environment")
    _create_venv(env)
    _sync_pip_config(env, mirror)
    report(0.2, f"Installing {', '.join(PACKAGES)}")
    _pip_install(env, mirror)
    report(0.9, "Checking the packages")
    version = _import_check(env)
    if version is None:
        raise StageError(
            "probe_failed", f"{', '.join(PACKAGES)} do not import in {env}."
        )
    return version


def ensure_research_env(mirror: str, progress: ProgressFn | None = None) -> str:
    """Make ``envs/default`` ready and return its Python version.

    A ready environment that still passes the import check is kept as it is,
    including the agent's own installs, and needs no network. One that fails
    it is repaired in place when its Python still starts, and rebuilt
    otherwise. The pip config follows ``mirror`` on every run.
    """
    report = progress or (lambda _f, _m: None)
    env = env_dir()

    from filelock import FileLock

    try:
        env.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(env.parent / f"{env.name}.lock")):
            if is_ready(env):
                version = _import_check(env)
                if version is not None:
                    _sync_pip_config(env, mirror)
                    return version
                version = _repair(env, mirror, report)
                if version is not None:
                    return version
            version = _build(env, mirror, report)
            (env / _READY_MARKER).write_text(version, encoding="utf-8")
            return version
    except OSError as exc:
        raise StageError(
            "install_failed",
            f"Could not set up the research environment in {env}: {exc}",
        ) from exc


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
@functools.cache
def _agent_python() -> tuple[str | None, Path | None]:
    """The agent shell's ``python`` and the research environment it comes from.

    Decided once per process, the first time it is needed (an agent is built,
    or a server is started or reused): ``(system python, None)``,
    ``(env python, env)`` or ``(None, None)``. Logs the result once at INFO,
    for people reading the log.
    """
    system = find_usable_python()
    if system is not None:
        logger.info(f"Agent shell python: {system}")
        return system, None
    env = env_dir()
    if is_ready(env):
        logger.info(f"Agent shell python: research environment {env}")
        return str(_env_python(env)), env
    logger.info("Agent shell python: none")
    return None, None


MISSING_PYTHON_HINT = (
    "The agent's shell has no usable `python`. Run `EvoSci setup`, then restart "
    f"EvoScientist, to give it a Python with {', '.join(PACKAGES)}."
)


def missing_python_hint() -> str | None:
    """The setup hint when the agent's shell has no ``python``, else None.

    For callers that show it themselves (the TUI, the WebUI launcher,
    ``EvoSci deploy``); it is not logged here.
    """
    return MISSING_PYTHON_HINT if agent_python() is None else None


@functools.cache
def _log_missing_python_hint() -> None:
    """Log the hint once per process where the agent is built, so the Rich
    CLI, ``-p`` and ``EvoSci serve`` show it."""
    hint = missing_python_hint()
    if hint is not None:
        logger.warning(hint)


def research_env_overrides() -> dict[str, str] | None:
    """Env overrides that put the research environment first for the agent's shell.

    None when a usable ``python`` is on PATH or the environment is not ready.
    ``PATH`` is built from the current ``os.environ`` on every call, so changes
    made after the decision (e.g. the private Node from ``activate_runtime``)
    are kept.
    """
    _log_missing_python_hint()
    _python, env = _agent_python()
    if env is None:
        return None
    bin_dir = str(_bin_dir(env))
    path = os.environ.get("PATH", "")
    return {
        "PATH": f"{bin_dir}{os.pathsep}{path}" if path else bin_dir,
        "VIRTUAL_ENV": str(env),
    }


def agent_python() -> str | None:
    """The ``python`` the agent's shell resolves, or None when it has none."""
    return _agent_python()[0]


# Sidecar key for the ``python`` a langgraph dev server's agents got at launch.
SIDECAR_KEY = "agent_python"


def python_drift_message(sidecar: dict) -> str | None:
    """A warning when a reused server's agents run another ``python`` than ours.

    The server's backends fix the agent shell's ``python`` when it starts, so
    after a reuse its agents (async sub-agents, and the WebUI's main agent)
    keep it. None when both match, or when the sidecar has no record (a
    server started by an older version).
    """
    if SIDECAR_KEY not in sidecar:
        return None
    recorded, current = sidecar[SIDECAR_KEY], agent_python()
    if recorded == current:
        return None
    return (
        f"The running langgraph dev gives its agents {recorded or 'no python'}, "
        f"but this session resolves {current or 'no python'}. Its agents keep "
        "the server's python until 'EvoSci server stop' and a restart."
    )


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
def run_stage(emit: Emitter, mirror: str) -> StageResult:
    def report(fraction: float, message: str) -> None:
        emit(make_event("research-env", "running", progress=fraction, message=message))

    report(0.0, "Checking for a usable python")
    system = find_usable_python()
    if system is not None:
        return StageResult(
            f"Using {system}",
            {"reason": "system_python", "python": system},
            status="skipped",
        )
    version = ensure_research_env(mirror, report)
    env = env_dir()
    return StageResult(
        f"Using Python {version} in {env}",
        {"source": "venv", "path": str(env), "python": version},
    )
