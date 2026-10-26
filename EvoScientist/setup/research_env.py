"""The research environment setup stage.

When no usable ``python`` is on PATH, the agent's shell gets a virtual
environment under ``<DATA_DIR>/envs/default`` with numpy, pandas, matplotlib
and scipy, so ``python script.py`` and ``pip install`` work as the prompts
teach. A user's own ``python`` (conda, venv, system) wins whenever it is
usable: Python 3.9 or newer, not EXTERNALLY-MANAGED (PEP 668) outside a venv,
and not the environment EvoScientist's installer made (see
:func:`find_usable_python`).

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
import json
import locale
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

from ._install import (
    kill_tree,
    remove_stale_temp_dirs,
    same_path,
    stage_lock,
    tree_signature,
)
from .protocol import (
    CN_MIRROR_HINT,
    HEARTBEAT_INTERVAL,
    Emitter,
    Heartbeat,
    StageError,
    StageResult,
    StepStalled,
    make_event,
    wait_with_heartbeat,
)

logger = logging.getLogger(__name__)

# The versions the environment gets, tested together. The newest set with
# wheels for Python 3.11-3.14 on every supported platform (numpy 2.5 and scipy
# 1.18 dropped 3.11). Update at release time, like the package's constraints.
PINS = {
    "numpy": "2.4.6",
    "pandas": "3.0.6",
    "matplotlib": "3.11.2",
    "scipy": "1.17.1",
}
PACKAGES = tuple(PINS)
CN_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"

_READY_MARKER = ".evoscientist-ready"
_PYTHON_PROBE_TIMEOUT = 10
# Prints the interpreter's `major.minor`, its `sys.prefix`, and 1 when it is
# EXTERNALLY-MANAGED (PEP 668) outside a venv, else 0. Written so Python 2,
# which some systems still install as `python`, runs it too; the rules are
# applied in :func:`_unusable_reason`.
_SYSTEM_PYTHON_PROBE = (
    "import os, sys\n"
    "managed = 0\n"
    "if sys.version_info[0] >= 3:\n"
    "    import sysconfig\n"
    "    managed = int(sys.prefix == getattr(sys, 'base_prefix', sys.prefix)"
    " and os.path.isfile(os.path.join(sysconfig.get_path('stdlib'),"
    " 'EXTERNALLY-MANAGED')))\n"
    "sys.stdout.write('%d.%d\\n%s\\n%d\\n'"
    " % (sys.version_info[0], sys.version_info[1], sys.prefix, managed))\n"
)
# The oldest `python` of the user's own that the agent keeps: older ones
# (Python 2.7 on CentOS 7 or early macOS, 3.8) fail the agent's `pip install`
# of current packages.
_MIN_PYTHON = (3, 9)
# Files that mark an environment EvoScientist's installer made (uv tool
# install; the Docker image's /opt/venv). uv builds both without pip.
_MANAGED_ENV_MARKERS = ("uv-receipt.toml", ".evoscientist-managed")
# Seconds venv creation and pip may go without printing anything or writing
# a file before they are stopped; no limit while they make progress.
_SILENCE_LIMIT = 300
# pip's temporary folders next to the environment; removed after each run,
# and by the next run under the lock when a run was killed.
_PIP_TEMP_PREFIX = ".pip-"
# How often a watched step repeats its last progress line.
_BEAT_INTERVAL = HEARTBEAT_INTERVAL
# The first matplotlib import builds its font cache, printing nothing.
_IMPORT_TIMEOUT = 300
# Prints the Python version, then one "failed <name>: <error>" line per
# package that does not import; exits 1 if any failed. Each package is tried
# on its own, so a repair can reinstall only what is broken.
_IMPORT_CHECK = (
    "import importlib, platform, sys\n"
    "print(f'version {platform.python_version()}', flush=True)\n"
    "failed = False\n"
    f"for name in {PACKAGES!r}:\n"
    "    try:\n"
    "        importlib.import_module(name)\n"
    "    except Exception as exc:\n"
    "        failed = True\n"
    "        print(f'failed {name}: {exc!r}', flush=True)\n"
    "sys.exit(1 if failed else 0)\n"
)
# Its lines are tagged and flushed one by one: stderr shares the pipe, and a
# working environment can warn at startup or on import (a broken .pth line, a
# pandas warning about an optional dependency the agent installed).
_VERSION_LINE_RE = re.compile(r"^version (\S+)$", re.MULTILINE)
_FAILED_LINE_RE = re.compile(r"^failed (\w+):", re.MULTILINE)
# pip's last words when no candidate matches a requirement, pinned or not;
# "from versions" lists the compatible versions the index offers ("none" when
# it was not reached, or has no wheel for this interpreter).
_NO_MATCH_RE = re.compile(
    r"satisfies the requirement (?P<name>[A-Za-z0-9_.-]+)(?:==\S+)? "
    r"\(from versions: (?P<versions>[^)]*)\)"
)
# pip's warning for each failed connection to the index (DNS failure,
# timeout, refusal), printed before it gives up.
_CONNECTION_FAILED = "after connection broken by"
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
class _Stalled(subprocess.SubprocessError):
    """A watched step showed no sign of work for its silence limit."""


def _stop(proc: subprocess.Popen[str]) -> None:
    """Kill ``proc``'s tree and reap it."""
    kill_tree(proc.pid)
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        logger.warning(
            f"{proc.args[0]} (pid {proc.pid}) did not stop after being killed"
        )


def _run(
    cmd: Sequence[str],
    timeout: float,
    *,
    beat: Callable[[], None] | None = None,
    watched: Sequence[Path] = (),
    held_open: Sequence[Path] = (),
    env: Mapping[str, str] | None = None,
    label: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` capturing combined output. Raises OSError / SubprocessError.

    Without ``beat``, ``timeout`` limits the whole run. With it, the step is
    watched instead: ``beat()`` runs every few seconds, and ``timeout`` is a
    silence limit, counted while the output and the files under ``watched``
    stay unchanged; past it the process tree is stopped and :class:`_Stalled`
    raised, naming the step as ``label`` (default: the program's name). pip
    prints nothing while it downloads a wheel when its output is piped, so the
    files it writes are the better sign of work. Folders in ``held_open``
    count too, for files that grow while held open (pip's downloads): each
    file there is asked for its size (see :func:`tree_signature`).

    No console window on Windows: the python probes also run when an agent is
    built, and a process without a console (the desktop app) would otherwise
    flash one for every probe.
    """
    if beat is None:
        return subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            errors="replace",
            timeout=timeout,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    output: list[bytes] = []
    assert proc.stdout is not None
    fd = proc.stdout.fileno()

    def read() -> None:
        # Raw chunks, so output without a newline (a progress dot) counts as
        # activity at once; reading by line would hide it until the line ends.
        while chunk := os.read(fd, 65536):
            output.append(chunk)

    # Only collects: every line to the caller comes from this thread.
    reader = threading.Thread(target=read, daemon=True)
    reader.start()

    def poll(seconds: float) -> int | None:
        try:
            return proc.wait(seconds)
        except subprocess.TimeoutExpired:
            return None

    def activity() -> tuple[object, ...]:
        return (
            sum(map(len, output)),
            *(tree_signature(p) for p in watched),
            *(tree_signature(p, exact=True) for p in held_open),
        )

    try:
        code = wait_with_heartbeat(
            poll, beat, activity, silence_limit=timeout, interval=_BEAT_INTERVAL
        )
    except StepStalled as exc:
        _stop(proc)
        silent = (
            "printed and wrote nothing" if watched or held_open else "printed nothing"
        )
        raise _Stalled(
            f"{label or Path(cmd[0]).name} {silent} for {timeout:g} s and was stopped"
        ) from exc
    except BaseException:
        _stop(proc)
        raise
    finally:
        # A grandchild that keeps the pipe open must not hold the stage.
        reader.join(30)
        if not reader.is_alive():
            proc.stdout.close()
    # As text=True would: the locale's encoding, bad bytes replaced, any line
    # ending turned into "\n".
    text = b"".join(output).decode(locale.getpreferredencoding(False), "replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return subprocess.CompletedProcess(list(cmd), code, text)


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


def _unusable_reason(probe_output: str) -> str | None:
    """Why a probed ``python`` is not usable for the agent, or None if it is."""
    # The probe's own three lines come last: stderr shares the pipe, and a
    # working interpreter can print warnings at startup (a broken .pth line).
    lines = probe_output.splitlines()[-3:]
    try:
        major, minor = (int(part) for part in lines[0].split("."))
        prefix, managed = lines[1], lines[2].strip() == "1"
    except (IndexError, ValueError):
        return f"unexpected probe output {probe_output!r}"
    if (major, minor) < _MIN_PYTHON:
        return f"Python {major}.{minor} is older than {'.'.join(map(str, _MIN_PYTHON))}"
    if managed:
        return "it is EXTERNALLY-MANAGED (PEP 668), so pip refuses to install into it"
    # EvoScientist's own environment when its installer made it: no pip there.
    if same_path(prefix, sys.prefix) and any(
        os.path.isfile(os.path.join(prefix, name)) for name in _MANAGED_ENV_MARKERS
    ):
        return "it is the environment EvoScientist's installer made, without pip"
    return None


def find_usable_python() -> str | None:
    """The ``python`` the agent's shell would run, as an absolute path, if it
    runs, is Python 3.9 or newer, lets pip install into it (no PEP 668 marker
    outside a venv) and is not the environment EvoScientist's installer made
    (a uv tool environment, the Docker image's ``/opt/venv``); else None.

    Only the first ``python`` on PATH counts, as in the shell. When it is our
    own environment's (activated by hand, or a nested ``EvoSci`` in the
    agent's shell), it is not a system python: the environment is then
    checked, injected or repaired like on any other run. A ``WindowsApps``
    alias that is not a Python Software Foundation package is never run (see
    :func:`_is_untrusted_alias`); when it comes first on PATH the shell would
    run it too, so there is no usable ``python``.
    """
    found = shutil.which("python")
    if found is None or _is_untrusted_alias(found):
        return None
    # Absolute, so the reported and recorded path does not depend on a cwd; a
    # relative PATH entry resolves against this process's working directory.
    found = os.path.abspath(found)
    own_bin = os.path.normcase(str(_bin_dir(env_dir())))
    if os.path.normcase(os.path.dirname(found)) == own_bin:
        return None
    result = _runs([found, "-c", _SYSTEM_PYTHON_PROBE], _PYTHON_PROBE_TIMEOUT)
    if result is None:
        logger.info(f"{found} is not usable: it did not run")
        return None
    reason = _unusable_reason(result.stdout)
    if reason is not None:
        logger.info(f"{found} is not usable: {reason}")
        return None
    return found


class ImportCheck(NamedTuple):
    """The result of :func:`_import_check`."""

    # None when the check did not run, or Python failed before the imports
    version: str | None
    failed: tuple[str, ...]  # the packages that do not import

    @property
    def ok(self) -> bool:
        return self.version is not None and not self.failed


def _import_check(env: Path, beat: Callable[[], None] | None = None) -> ImportCheck:
    """Import each package in the environment's Python, one by one.

    Isolated (``-I``) like every command that maintains the environment: the
    caller's cwd and ``PYTHON*`` variables could shadow a package and send a
    healthy environment to a rebuild. The output is logged on failure, since a
    shadowing file, a broken wheel and a missing system library look the same
    from the exit code.
    """
    cmd = [str(_env_python(env)), "-I", "-c", _IMPORT_CHECK]
    try:
        result = _run(cmd, _IMPORT_TIMEOUT, beat=beat, label="The import check")
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f"Import check in {env} failed: {exc}")
        return ImportCheck(None, PACKAGES)
    version_line = _VERSION_LINE_RE.search(result.stdout)
    version = version_line[1] if version_line else None
    failed = tuple(
        name for name in _FAILED_LINE_RE.findall(result.stdout) if name in PINS
    )
    if (
        version is None
        or result.returncode not in (0, 1)
        or (result.returncode == 1 and not failed)
    ):
        # Python itself failed (e.g. before the first import), or crashed
        # midway so the failed lines are only a part (the check exits only
        # 0 or 1); treat every package as failed so a repair reinstalls them
        # all, never none or too few.
        version, failed = None, PACKAGES
    check = ImportCheck(version, failed)
    if not check.ok:
        logger.warning(f"Import check in {env} failed:\n{result.stdout}")
    return check


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
    cmd = [str(_env_python(env)), "-I", "-m", "pip", "config", "--site", *action]
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


def _create_venv(env: Path, beat: Callable[[], None]) -> None:
    cmd = [sys.executable, "-I", "-m", "venv", str(env)]
    try:
        result = _run(
            cmd, _SILENCE_LIMIT, beat=beat, watched=(env,), label="python -m venv"
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StageError("install_failed", f"Could not create {env}: {exc}") from exc
    if result.returncode == 0:
        logger.info(f"python -m venv output:\n{result.stdout}")
        return
    # The error event keeps one line; the cause can be earlier in the output.
    logger.warning(f"python -m venv failed:\n{result.stdout}")
    message = f"Could not create {env}: {_last_line(result.stdout)}"
    # Debian's venv prints this when ensurepip is missing; other failures of
    # the pip bootstrap also mention ensurepip, but on any OS.
    if "ensurepip is not" in result.stdout:
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        message += (
            " This Python has no ensurepip; on Debian and Ubuntu install the"
            f" python{version}-venv package."
        )
    raise StageError("install_failed", message)


def _pip_install(
    env: Path,
    mirror: str,
    packages: Sequence[str],
    beat: Callable[[], None],
    *,
    force: bool = False,
    unpinned: Sequence[str] = (),
) -> list[str]:
    """Install ``packages`` at their :data:`PINS`; returns the ones installed
    without the pin. Those in ``unpinned`` (an earlier call's answer) start
    without it, so pip does not fail on their pin again.

    Without ``--upgrade``, so packages already there stay as they are unless a
    missing one requires a change. ``force`` reinstalls the packages
    themselves (``--force-reinstall --no-deps``), for a package whose files
    are broken although pip counts it as installed. When the pinned version
    of a package has no wheel for this interpreter (pip's "from versions:"
    lists others), that package is installed without the pin and pip takes
    its newest wheel, once per package. "from versions: none" means the index
    was not reached (or has nothing for this interpreter), which no fallback
    fixes.

    pip gets a temporary folder of its own (``TMPDIR`` / ``TEMP``), where it
    downloads: its growth, and the environment's, show that pip still works
    while it prints nothing.
    """
    requirements = {
        name: name if name in unpinned else f"{name}=={PINS[name]}" for name in packages
    }
    unpinned = [name for name in unpinned if name in requirements]
    tmp = Path(tempfile.mkdtemp(prefix=_PIP_TEMP_PREFIX, dir=env.parent))
    child_env = {**os.environ, "TMPDIR": str(tmp), "TEMP": str(tmp), "TMP": str(tmp)}
    try:
        return _pip_install_in(
            env, mirror, requirements, unpinned, beat, force, tmp, child_env
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _pip_install_in(
    env: Path,
    mirror: str,
    requirements: dict[str, str],
    unpinned: list[str],
    beat: Callable[[], None],
    force: bool,
    tmp: Path,
    child_env: Mapping[str, str],
) -> list[str]:
    while True:
        cmd = [
            str(_env_python(env)),
            "-I",
            "-m",
            "pip",
            "install",
            "--only-binary",
            ":all:",
            "--disable-pip-version-check",
            "--no-input",
            *(["--force-reinstall", "--no-deps"] if force else []),
            *requirements.values(),
        ]
        if mirror == "cn":
            cmd += ["--index-url", CN_INDEX_URL]
        try:
            result = _run(
                cmd,
                _SILENCE_LIMIT,
                beat=beat,
                watched=(env,),
                held_open=(tmp,),
                env=child_env,
                label="pip",
            )
        except _Stalled as exc:
            message = f"pip install failed: {exc}."
            if mirror != "cn":
                message += f" {CN_MIRROR_HINT}"
            raise StageError("install_failed", message) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise StageError("install_failed", f"pip install failed: {exc}") from exc
        if result.returncode == 0:
            logger.info(f"pip install output:\n{result.stdout}")
            return unpinned
        # The error event keeps one line; the cause (e.g. network errors) can
        # be earlier in the output.
        logger.warning(f"pip install failed:\n{result.stdout}")
        match = _NO_MATCH_RE.search(result.stdout)
        name = match["name"].lower() if match else None
        if (
            match is not None
            and match["versions"].strip() != "none"
            and name in requirements
            and name not in unpinned
        ):
            logger.warning(
                f"{requirements[name]} has no wheel for this Python; installing "
                f"the newest wheel of {name} instead."
            )
            requirements[name] = name
            unpinned.append(name)
            continue
        sentences = [f"pip install failed: {_last_line(result.stdout).rstrip('.')}."]
        if _CONNECTION_FAILED in result.stdout:
            sentences.append("The package index could not be reached.")
        elif match is not None and match["versions"].strip() == "none":
            sentences.append(
                f"The package index has no wheel of {match['name']} for this"
                " Python, or did not answer."
            )
        # On every pip failure while the mirror is off, as for download
        # errors: a wheel download that stalls mid-file ends with a traceback,
        # and an index answering with an HTTP error says "none", neither with
        # a connection warning.
        if mirror != "cn":
            sentences.append(CN_MIRROR_HINT)
        raise StageError("install_failed", " ".join(sentences))


class EnvResult(NamedTuple):
    """A ready research environment: its Python version, and the packages it
    has without their pin (see :func:`_pip_install`), which were never tested
    together with the pins."""

    version: str
    unpinned: list[str]


def _read_unpinned(env: Path) -> list[str]:
    """The unpinned packages recorded in the ready marker.

    A marker written by an older version holds only the Python version, and
    counts as none.
    """
    try:
        data = json.loads((env / _READY_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    unpinned = data.get("unpinned") if isinstance(data, dict) else None
    if not isinstance(unpinned, list):
        return []
    return [name for name in unpinned if name in PINS]


def _write_ready_marker(env: Path, result: EnvResult) -> None:
    """Mark the environment ready, after its import check passed."""
    (env / _READY_MARKER).write_text(
        json.dumps({"python": result.version, "unpinned": result.unpinned}),
        encoding="utf-8",
    )


def _repair(
    env: Path,
    mirror: str,
    failed: Sequence[str],
    report: Heartbeat,
    *,
    force: bool,
) -> EnvResult | None:
    """Try to fix a ready environment in place; None to rebuild it.

    A Python that does not start (e.g. its base interpreter was removed) cannot
    be repaired. Otherwise only the ``failed`` packages are installed again, at
    their pins, so a newer version of another package that the agent installed
    stays: first, with ``force``, their own files (pip counts a package with
    broken files as installed), then whatever they need that is missing. A pip failure raises
    and leaves the environment and its marker as they are, so the agent keeps
    a mostly working Python.
    """
    python = str(_env_python(env))
    if _runs([python, "-I", "-c", "import sys"], _PYTHON_PROBE_TIMEOUT) is None:
        return None
    # Before pip, so the config follows the mirror even when pip fails.
    _sync_pip_config(env, mirror)
    # Below the build's first step (0.1): a repair that does not help falls
    # back to a rebuild, and progress must not jump backwards.
    report(0.03, f"Installing {', '.join(failed)}")
    still_unpinned = [name for name in _read_unpinned(env) if name not in failed]
    # Only when the check ran far enough to name the failed packages: after a
    # crashed or stalled check, all four count as failed and would all be
    # downloaded again for nothing.
    unpinned = (
        _pip_install(env, mirror, failed, report.beat, force=True) if force else []
    )
    # Packages whose pin has no wheel here start unpinned in the second call.
    unpinned = _pip_install(env, mirror, failed, report.beat, unpinned=unpinned)
    report(0.06, "Checking the packages")
    check = _import_check(env, report.beat)
    if not check.ok:
        return None
    # A package reinstalled at its pin is no longer unpinned.
    return EnvResult(check.version, still_unpinned + unpinned)


def _build(env: Path, mirror: str, report: Heartbeat) -> EnvResult:
    try:
        (env / _READY_MARKER).unlink(missing_ok=True)
        if env.exists():
            shutil.rmtree(env)
    except OSError as exc:
        raise StageError("install_failed", f"Could not remove {env}: {exc}") from exc
    try:
        report(0.1, "Creating the virtual environment")
        _create_venv(env, report.beat)
        _sync_pip_config(env, mirror)
        report(0.2, f"Installing {', '.join(PACKAGES)}")
        unpinned = _pip_install(env, mirror, PACKAGES, report.beat)
        report(0.9, "Checking the packages")
        check = _import_check(env, report.beat)
        if not check.ok:
            failed = ", ".join(check.failed) or ", ".join(PACKAGES)
            raise StageError("probe_failed", f"{failed} do not import in {env}.")
    except BaseException:
        # Nothing uses an environment without its ready marker, and the next
        # run would delete it anyway; free the space now.
        shutil.rmtree(env, ignore_errors=True)
        raise
    return EnvResult(check.version, unpinned)


def ensure_research_env(mirror: str, progress: ProgressFn | None = None) -> EnvResult:
    """Make ``envs/default`` ready; returns its version and unpinned packages.

    A ready environment that still passes the import check is kept as it is,
    including the agent's own installs, and needs no network. One that fails
    it is repaired in place when its Python still starts, and rebuilt
    otherwise. The pip config follows ``mirror`` on every run.
    """
    report = Heartbeat(progress or (lambda _f, _m: None))
    env = env_dir()

    try:
        env.parent.mkdir(parents=True, exist_ok=True)
        lock = env.parent / f"{env.name}.lock"
        with stage_lock(lock, "the research environment", progress):
            remove_stale_temp_dirs(env.parent, _PIP_TEMP_PREFIX)
            if is_ready(env):
                # Gives the heartbeat a line to repeat while the check runs.
                report(0.02, "Checking the packages")
                check = _import_check(env, report.beat)
                if check.ok:
                    _sync_pip_config(env, mirror)
                    return EnvResult(check.version, _read_unpinned(env))
                result = _repair(
                    env, mirror, check.failed, report, force=check.version is not None
                )
                if result is not None:
                    _write_ready_marker(env, result)
                    return result
            result = _build(env, mirror, report)
            _write_ready_marker(env, result)
            return result
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
    env_python = str(_env_python(env))
    # The marker alone is not enough: on Windows the venv's python.exe stays
    # in place when its base interpreter is removed, but no longer starts.
    if is_ready(env) and _runs([env_python, "-c", "import sys"], _PYTHON_PROBE_TIMEOUT):
        logger.info(f"Agent shell python: research environment {env}")
        return env_python, env
    logger.info("Agent shell python: none")
    return None, None


# Shown through ``agent_shell.setup_hint``, which joins it with the Git Bash
# hint on Windows.
MISSING_PYTHON_HINT = (
    "The agent's shell has no usable `python`. Run `EvoSci setup`, then restart "
    f"EvoScientist, to give it a Python with {', '.join(PACKAGES)}."
)


def research_env_overrides() -> dict[str, str] | None:
    """Env overrides that put the research environment first for the agent's shell.

    None when a usable ``python`` is on PATH or the environment is not ready.
    ``PATH`` is built from the current ``os.environ`` on every call, so changes
    made after the decision (e.g. the private Node from ``activate_runtime``)
    are kept. Called where the agent is built, so it also logs the setup hint
    for the agent's shell once (:func:`~EvoScientist.agent_shell.log_setup_hint`).
    """
    from ..agent_shell import log_setup_hint

    log_setup_hint()
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
    # normcase: on Windows two terminals can spell the same path differently.
    if recorded == current or (
        isinstance(recorded, str)
        and current is not None
        and os.path.normcase(recorded) == os.path.normcase(current)
    ):
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
    result = ensure_research_env(mirror, report)
    env = env_dir()
    detail = {"source": "venv", "path": str(env), "python": result.version}
    message = f"Using Python {result.version} in {env}"
    if result.unpinned:
        # Their newest wheels were never tested together with the pins.
        detail["unpinned"] = result.unpinned
        message += f" ({', '.join(result.unpinned)} without the pinned version)"
    return StageResult(message, detail)
