"""The Node.js setup stage.

A system Node at :data:`MIN_SYSTEM_NODE` or newer is used as it is. Otherwise a
private Node :data:`NODE_VERSION` is installed under ``<DATA_DIR>/tools/`` and
recorded in ``tools/node.json``. The record is written only after the archive
passed its checksum and the installed ``node --version`` ran, so a failed or
partial download never replaces a working Node.

EvoScientist never edits shell rc files and never touches a system Node, nvm,
fnm or conda install: the private Node reaches child processes only through
:func:`activate_runtime`, which changes ``PATH`` inside this process.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._install import (
    is_under,
    move_into_place,
    prepend_to_path,
    remove_stale_temp_dirs,
    tools_dir,
)
from .download import download, fetch_text, verify_sha256_from_sums
from .protocol import (
    Emitter,
    ProgressThrottle,
    StageError,
    StageResult,
    make_event,
)

logger = logging.getLogger(__name__)

NODE_VERSION = "24.21.0"
# (major, minor): the WebUI's Next.js declares ``engines.node >=20.9.0``.
MIN_SYSTEM_NODE = (20, 9)

SOURCES = {
    "default": "https://nodejs.org/dist",
    "cn": "https://cdn.npmmirror.com/binaries/node",
}
# npm registry for packages fetched with our `npx` under a mirror; the default
# mirror leaves npm's own setting alone.
NPM_REGISTRIES = {"cn": "https://registry.npmmirror.com"}

_OS = {"darwin": "darwin", "linux": "linux", "win32": "win"}
_ARCH = {"x86_64": "x64", "amd64": "x64", "arm64": "arm64", "aarch64": "arm64"}
# Node publishes .tar.xz for macOS / Linux and .zip for Windows. The glibc
# Linux builds do not run on musl (Alpine); nodejs.org publishes a musl build
# for x64 only (from 24.20.0).
_SUPPORTED = {
    "darwin-arm64",
    "darwin-x64",
    "linux-x64",
    "linux-arm64",
    "linux-x64-musl",
    "win-x64",
    "win-arm64",
}

_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")

ProgressFn = Callable[[float, str], None]


def log_progress(log: logging.Logger) -> ProgressFn:
    """A progress callback for on-demand installs that have no emitter.

    Logs at WARNING, the level the CLI shows by default, so a first WebUI start
    or MCP load does not look frozen during the download. Thinned by
    :class:`ProgressThrottle`.
    """
    throttle = ProgressThrottle()

    def report(fraction: float, message: str) -> None:
        pct = int(fraction * 100)
        if throttle.should_show("node", pct, message):
            log.warning(f"Installing Node.js: {message} ({pct}%)")

    return report


@dataclass(frozen=True)
class NodeInfo:
    """The Node that EvoScientist uses."""

    source: str  # "system" | "private"
    version: str
    path: Path  # the ``node`` executable

    def detail(self) -> dict[str, Any]:
        return {"source": self.source, "version": self.version, "path": str(self.path)}


# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
def _record_path() -> Path:
    return tools_dir() / "node.json"


def _bin_dir(install_dir: Path) -> Path:
    """The directory holding ``node``: the archive root on Windows, ``bin/`` elsewhere."""
    return install_dir if os.name == "nt" else install_dir / "bin"


def _node_exe(install_dir: Path) -> Path:
    return _bin_dir(install_dir) / ("node.exe" if os.name == "nt" else "node")


def _is_musl() -> bool:
    """True on a musl-based Linux (e.g. Alpine).

    Decided by the running interpreter's libc first: Debian and Ubuntu install
    the musl loader with their ``musl`` package on a glibc system, where the
    glibc build is the one that runs. The loader check covers the musl side,
    where ``libc_ver()`` reports nothing.
    """
    if sys.platform != "linux" or platform.libc_ver()[0] == "glibc":
        return False
    return any(Path("/lib").glob("ld-musl-*.so.1"))


def platform_id() -> str:
    """The Node archive platform for this machine, e.g. ``linux-x64``."""
    os_id = _OS.get(sys.platform)
    arch = _ARCH.get(platform.machine().lower())
    plat = f"{os_id}-{arch}"
    if os_id == "linux" and _is_musl():
        plat += "-musl"
    if os_id is None or arch is None or plat not in _SUPPORTED:
        libc = " (musl)" if plat.endswith("-musl") else ""
        raise StageError(
            "unsupported_platform",
            f"No Node.js build for {sys.platform} / {platform.machine()}{libc}.",
        )
    return plat


def archive_name(plat: str, version: str = NODE_VERSION) -> str:
    ext = ".zip" if plat.startswith("win-") else ".tar.xz"
    return f"node-v{version}-{plat}{ext}"


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #
def _parse_version(text: str) -> tuple[int, int, int] | None:
    match = _VERSION_RE.match(text.strip())
    if match is None:
        return None
    return int(match[1]), int(match[2]), int(match[3])


def _probe(exe: Path) -> tuple[int, int, int] | None:
    """Run ``<exe> --version``; return the parsed version, or None on any failure."""
    env = node_child_env(os.environ, private=True)
    try:
        result = subprocess.run(
            [str(exe), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return _parse_version(result.stdout)


def _system_node() -> NodeInfo | None:
    """A ``node`` on PATH, outside our tools dir, at MIN_SYSTEM_NODE or newer,
    with ``npx`` next to it.

    Every consumer runs ``npx``, and some distros package ``npm`` (which
    provides it) separately from ``node``; a Node without it does not count.
    """
    root = tools_dir()
    search = os.pathsep.join(
        p
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p and not is_under(Path(p), root)
    )
    found = shutil.which("node", path=search)
    if found is None or shutil.which("npx", path=str(Path(found).parent)) is None:
        return None
    version = _probe(Path(found))
    if version is None or version[:2] < MIN_SYSTEM_NODE:
        return None
    return NodeInfo("system", ".".join(map(str, version)), Path(found))


def _read_record() -> dict[str, str] | None:
    try:
        data = json.loads(_record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    path, version = data.get("path"), data.get("version")
    if not isinstance(path, str) or not path or not isinstance(version, str):
        return None
    return data


def _write_record(version: str, install_dir: Path) -> None:
    record = _record_path()
    tmp = record.with_name(record.name + ".tmp")
    tmp.write_text(
        json.dumps({"version": version, "path": str(install_dir)}), encoding="utf-8"
    )
    os.replace(tmp, record)


def _recorded_node() -> NodeInfo | None:
    """The recorded private Node, if its ``node --version`` still runs."""
    record = _read_record()
    if record is None:
        return None
    exe = _node_exe(Path(record["path"]))
    version = _probe(exe)
    if version is None:
        return None
    return NodeInfo("private", ".".join(map(str, version)), exe)


# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #
def _extract(archive: Path, dest: Path) -> None:
    """Unpack ``archive`` into ``dest``.

    ``.zip`` (Windows) goes through ``zipfile``, which strips absolute and
    ``..`` member paths and creates no symlinks. ``.tar.xz`` uses the ``data``
    filter, which keeps Node's relative ``bin/npm`` / ``bin/npx`` links and
    rejects members or links that escape ``dest``.
    """
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        return
    with tarfile.open(archive, "r:xz") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, filter="data")
            return
        # Python < 3.11.4 has no extraction filters: check each member by hand.
        root = dest.resolve()
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if not is_under(target, root):
                raise tarfile.TarError(f"member {member.name!r} escapes {dest}")
            if member.issym() or member.islnk():
                link_base = target.parent if member.issym() else root
                if os.path.isabs(member.linkname) or not is_under(
                    (link_base / member.linkname).resolve(), root
                ):
                    raise tarfile.TarError(f"link {member.name!r} escapes {dest}")
        tf.extractall(dest)


def _install(mirror: str, report: ProgressFn) -> NodeInfo:
    plat = platform_id()
    source = SOURCES.get(mirror, SOURCES["default"])
    filename = archive_name(plat)
    base_url = f"{source}/v{NODE_VERSION}"
    root = tools_dir()
    final = root / f"node-v{NODE_VERSION}"

    tmp = Path(tempfile.mkdtemp(prefix=".node-", dir=root))
    try:
        report(0.05, f"Downloading Node {NODE_VERSION}")
        sums = fetch_text(f"{base_url}/SHASUMS256.txt")
        archive = tmp / filename
        actual = download(
            f"{base_url}/{filename}",
            archive,
            lambda f: report(0.05 + 0.8 * f, f"Downloading Node {NODE_VERSION}"),
        )
        report(0.86, "Verifying checksum")
        verify_sha256_from_sums(actual, sums, filename)

        report(0.9, "Unpacking")
        unpack = tmp / "unpack"
        try:
            _extract(archive, unpack)
        except (tarfile.TarError, zipfile.BadZipFile) as exc:
            raise StageError(
                "download_failed", f"Could not unpack {filename}: {exc}"
            ) from exc
        except OSError as exc:
            raise StageError(
                "install_failed", f"Could not unpack {filename}: {exc}"
            ) from exc
        extracted = unpack / filename.removesuffix(".zip").removesuffix(".tar.xz")
        if not _node_exe(extracted).exists():
            raise StageError(
                "download_failed", f"{filename} does not contain the expected layout."
            )
        try:
            if final.exists():
                # Left behind by an earlier attempt whose probe failed.
                shutil.rmtree(final)
            move_into_place(extracted, final, what="Node")
        except OSError as exc:
            raise StageError(
                "install_failed", f"Could not move Node into {final}: {exc}"
            ) from exc

        report(0.96, "Checking the installed Node")
        exe = _node_exe(final)
        version = _probe(exe)
        if version is None:
            shutil.rmtree(final, ignore_errors=True)
            raise StageError("probe_failed", f"{exe} --version did not run.")
        _write_record(NODE_VERSION, final)
        return NodeInfo("private", ".".join(map(str, version)), exe)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _adopt_installed(root: Path) -> NodeInfo | None:
    """Record an intact ``node-v<NODE_VERSION>`` again after its record was dropped.

    A system Node drops the record but leaves the install on disk (and
    ``mcp.yaml`` may point into it); without this the next run with no system
    Node on PATH would delete and re-download it, or fail offline.
    """
    install_dir = root / f"node-v{NODE_VERSION}"
    exe = _node_exe(install_dir)
    if not exe.is_file():
        return None
    version = _probe(exe)
    if version is None:
        return None
    _write_record(NODE_VERSION, install_dir)
    return NodeInfo("private", ".".join(map(str, version)), exe)


def configured_mirror() -> str:
    from ..config import load_config

    return load_config().mirror


# The first failed install in this process; later calls raise it again without
# retrying. A new process (e.g. `EvoSci setup`) tries again.
_failed_install: StageError | None = None


def ensure_node(
    mirror: str | None = None, progress: ProgressFn | None = None
) -> NodeInfo:
    """Return a usable Node, installing the private one if needed.

    Order: a system Node >= MIN_SYSTEM_NODE with ``npx`` (the private record
    is then dropped, so the system Node is the one on PATH); the recorded
    private Node if it still runs (no network needed); otherwise download,
    verify, unpack, probe and record. Raises :class:`StageError` on failure,
    leaving any previous record in place; file-system errors in the tools dir
    are reported as ``install_failed``. After one failed install, later calls
    in the same process raise the same error without retrying.

    ``mirror`` defaults to the configured ``mirror``.
    """
    report = progress or (lambda _f, _m: None)

    system = _system_node()
    if system is not None:
        # Housekeeping only: a record we cannot remove must not fail a usable
        # system Node.
        with contextlib.suppress(OSError):
            _record_path().unlink(missing_ok=True)
        return system

    recorded = _recorded_node()
    if recorded is not None:
        return recorded

    global _failed_install
    if _failed_install is not None:
        # One process can ask several times (server spawn, MCP load, WebUI);
        # on a blocked network each attempt would wait out the timeouts again.
        raise _failed_install

    from filelock import FileLock

    root = tools_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(root / "node.lock")):
            # Another process may have finished the install while we waited.
            recorded = _recorded_node() or _adopt_installed(root)
            if recorded is not None:
                return recorded
            remove_stale_temp_dirs(root, ".node-")
            return _install(mirror or configured_mirror(), report)
    except StageError as exc:
        _failed_install = exc
        raise
    except OSError as exc:
        _failed_install = StageError(
            "install_failed", f"Could not install Node into {root}: {exc}"
        )
        raise _failed_install from exc


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
def activate_runtime() -> Path | None:
    """Put the recorded private Node's directory first on ``PATH``.

    Reads ``tools/node.json`` only (no subprocess), so it is cheap enough for
    CLI start-up. Child processes - the WebUI launcher, ``npx`` MCP servers,
    ``langgraph dev`` and the agent's shell - inherit the change. Returns the
    directory put on PATH, or None when no private Node is recorded.
    """
    record = _read_record()
    if record is None:
        return None
    # Absolute even for a record written from a relative DATA_DIR, so children
    # in other working dirs resolve the same `node`.
    install_dir = Path(record["path"]).resolve()
    bin_dir = _bin_dir(install_dir)
    if not _node_exe(install_dir).exists():
        return None
    prepend_to_path(bin_dir)
    return bin_dir


def is_private(exe: str | Path | None) -> bool:
    """True when ``exe`` lives under our tools dir."""
    return exe is not None and is_under(Path(exe), tools_dir())


def node_child_env(env: Mapping[str, str], *, private: bool) -> dict[str, str]:
    """Copy ``env`` for a child that runs the private ``node`` / ``npm`` / ``npx``.

    With ``private`` set, drops ``npm_config_*`` and ``NODE_OPTIONS`` so a
    user's global npm settings cannot break our Node. A system Node keeps the
    user's environment as it is.
    """
    if not private:
        return dict(env)
    return {
        k: v
        for k, v in env.items()
        if not k.lower().startswith("npm_config_") and k.upper() != "NODE_OPTIONS"
    }


DEFAULT_NPM_REGISTRY = "https://registry.npmjs.org"
_NPM_CONFIG_TIMEOUT = 10
# `npm config get registry` per npm executable; it starts a Node process, so
# ask once per process.
_npm_config_registry: dict[str, str | None] = {}


def configured_npm_registry(env: Mapping[str, str], node_exe: Path) -> str | None:
    """The npm registry the user configured, or None when they set none.

    ``npm_config_registry`` in ``env`` wins (the name is compared without
    case, as npm does). Otherwise asks the ``npm`` next to ``node_exe`` with
    ``npm config get registry``, which reads ``~/.npmrc`` and npm's global
    config; npm prints its default when nothing is set, so the default counts
    as unset. It runs in the tools dir so a project ``.npmrc`` in the current
    directory does not leak in. If it cannot run, the answer comes from
    ``env`` alone.
    """
    for key, value in env.items():
        if key.lower() == "npm_config_registry" and value.strip():
            return value.strip()

    npm = shutil.which("npm", path=str(Path(node_exe).parent))
    if npm is None:
        return None
    if npm not in _npm_config_registry:
        _npm_config_registry[npm] = _ask_npm_registry(npm, env)
    return _npm_config_registry[npm]


def _ask_npm_registry(npm: str, env: Mapping[str, str]) -> str | None:
    cwd = tools_dir()
    if not cwd.is_dir():
        cwd = Path.home()
    child_env = {k: v for k, v in env.items() if k.upper() != "NODE_OPTIONS"}
    try:
        result = subprocess.run(
            [npm, "config", "get", "registry"],
            capture_output=True,
            text=True,
            timeout=_NPM_CONFIG_TIMEOUT,
            cwd=cwd,
            env=child_env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(f"npm config get registry failed to run: {exc!r}")
        return None
    registry = result.stdout.strip()
    if result.returncode != 0 or not registry.startswith(("http://", "https://")):
        logger.debug(
            f"npm config get registry gave no registry (exit "
            f"{result.returncode}): {result.stdout!r} {result.stderr!r}"
        )
        return None
    if registry.rstrip("/") == DEFAULT_NPM_REGISTRY:
        return None
    return registry


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
def run_stage(emit: Emitter, mirror: str) -> StageResult:
    emit(make_event("node", "running", progress=0.0, message="Checking for Node.js"))

    def report(fraction: float, message: str) -> None:
        emit(make_event("node", "running", progress=fraction, message=message))

    info = ensure_node(mirror, report)
    if info.source == "system":
        message = f"Using system Node {info.version}"
    else:
        message = f"Using Node {info.version} from {info.path.parent}"
    return StageResult(message, info.detail())
