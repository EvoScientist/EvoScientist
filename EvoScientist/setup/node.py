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
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from ._install import (
    PolicyBlocked,
    atomic_write_text,
    is_policy_block,
    is_under,
    move_into_place,
    policy_error,
    prepend_to_path,
    remove_stale_temp_dirs,
    stage_lock,
    tools_dir,
    which_in,
    which_on_path,
)
from .download import download
from .protocol import (
    Emitter,
    Heartbeat,
    ProgressThrottle,
    StageError,
    StageResult,
    make_event,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

NODE_VERSION = "24.21.0"
# (major, minor): the WebUI's Next.js declares ``engines.node >=20.9.0``.
MIN_SYSTEM_NODE = (20, 9)

SOURCES = {
    "default": "https://nodejs.org/dist",
    "cn": "https://cdn.npmmirror.com/binaries/node",
}
# npm registry for `npx` / `npm` children under a mirror, unless the user set
# one; the default mirror leaves npm's own setting alone.
NPM_REGISTRIES = {"cn": "https://registry.npmmirror.com"}

_OS = {"darwin": "darwin", "linux": "linux", "win32": "win"}
_ARCH = {"x86_64": "x64", "amd64": "x64", "arm64": "arm64", "aarch64": "arm64"}
# SHA-256 of each supported platform's archive (.tar.xz for macOS / Linux,
# .zip for Windows), from the release's SHASUMS256.txt; npmmirror serves the
# same files. Update together with NODE_VERSION.
NODE_SHA256 = {
    "darwin-arm64": "6239d4cf92d864487ec8cd3615038f7b67e7f58b77b21cd2f09ea9fbd68065fe",
    "darwin-x64": "0ae5a24c24bb7d015cd816c5036b3f90f2945aa872fcf54e58da054753b3a299",
    "linux-x64": "fd8e59d5a511510f6a298afb548f18c7d2b1be404d8b4a27d94fbe49f56cb2d6",
    "linux-arm64": "6ad1325edbdb5649c379b75a237147a666c95d4f9ae8d340fef2d1575d289ad2",
    "win-x64": "158f7685b44de51f6c0df1d153526cbcd3e1bc739a8dfc607721cef75de9e541",
}
# The official Linux builds need glibc 2.28 or newer (Node's BUILDING.md).
MIN_GLIBC = (2, 28)
_TRY_YOUR_OWN = "Install Node.js 20.9 or newer yourself and run `EvoSci setup` again."
# Completes policy_error()'s "or ... and run `EvoSci setup` again".
_POLICY_ALTERNATIVE = "install Node.js 20.9 or newer yourself"

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


def _glibc_version() -> tuple[int, int] | None:
    """The C library's glibc version, or None when there is none (e.g. musl)."""
    try:
        text = os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, OSError, ValueError):
        return None
    match = re.match(r"glibc (\d+)\.(\d+)", text or "")
    return (int(match[1]), int(match[2])) if match else None


def platform_id() -> str:
    """The Node archive platform for this machine, e.g. ``linux-x64``.

    Raises ``unsupported_platform`` before any download where the official
    builds do not run or are not offered: Windows on arm64, a Linux without
    glibc (musl, e.g. Alpine) and glibc older than :data:`MIN_GLIBC`. A system
    Node is looked for first, so it is still used there.
    """
    machine = platform.machine()
    os_id = _OS.get(sys.platform)
    arch = _ARCH.get(machine.lower())
    if os_id == "win" and arch == "arm64":
        raise StageError(
            "unsupported_platform",
            "Windows on arm64 is not supported: no Node.js 20.9 or newer with npx "
            f"was found on PATH, and the private Node is x64 only. {_TRY_YOUR_OWN}",
        )
    if os_id == "linux":
        glibc = _glibc_version()
        if glibc is None:
            raise StageError(
                "unsupported_platform",
                "This Linux has no glibc (musl-based systems such as Alpine are "
                "not supported), so the official Node.js builds do not run. "
                "Install Node.js 20.9 or newer yourself, e.g. `apk add nodejs "
                "npm` on Alpine, and run `EvoSci setup` again.",
            )
        if glibc < MIN_GLIBC:
            found, need = (".".join(map(str, v)) for v in (glibc, MIN_GLIBC))
            raise StageError(
                "unsupported_platform",
                f"This Linux has glibc {found}; the official Node.js builds need "
                f"glibc {need} or newer. {_TRY_YOUR_OWN}",
            )
    plat = f"{os_id}-{arch}"
    if os_id is None or arch is None or plat not in NODE_SHA256:
        raise StageError(
            "unsupported_platform",
            f"No Node.js build for {sys.platform} / {machine}. {_TRY_YOUR_OWN}",
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
    """Run ``<exe> --version``; return the parsed version, or None on any failure.

    A program blocked by a Windows policy raises :class:`PolicyBlocked`
    instead, so a blocked system Node is skipped while a blocked private one
    is reported, not downloaded again.
    """
    env = node_child_env(os.environ, private=True)
    try:
        result = subprocess.run(
            [str(exe), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except OSError as exc:
        if is_policy_block(exc):
            raise PolicyBlocked(exe) from exc
        return None
    except subprocess.SubprocessError:
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
    found = which_on_path("node", outside=tools_dir())
    if found is None or which_in("npx", Path(found).parent) is None:
        return None
    try:
        version = _probe(Path(found))
    except PolicyBlocked as exc:
        logger.info(f"Not using {found}: a Windows policy blocked {exc.path}")
        return None
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
    atomic_write_text(
        _record_path(), json.dumps({"version": version, "path": str(install_dir)})
    )


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
def _ticking(members: Iterable[T], tick: Callable[[], None]) -> Iterator[T]:
    """``members``, calling ``tick`` before each one is unpacked."""
    for member in members:
        tick()
        yield member


def _extract(
    archive: Path, dest: Path, tick: Callable[[], None] = lambda: None
) -> None:
    """Unpack ``archive`` into ``dest``, calling ``tick`` between members.

    ``.zip`` (Windows) goes through ``zipfile``, which strips absolute and
    ``..`` member paths and creates no symlinks. ``.tar.xz`` uses the ``data``
    filter, which keeps Node's relative ``bin/npm`` / ``bin/npx`` links and
    rejects members or links that escape ``dest``. Both unpack the members as
    the iterator yields them, so ``tick`` runs while the work goes on.
    """
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest, members=_ticking(zf.namelist(), tick))
        return
    with tarfile.open(archive, "r:xz") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, members=_ticking(tf, tick), filter="data")
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
        tf.extractall(dest, members=_ticking(tf.getmembers(), tick))


def _install(mirror: str, report: Heartbeat) -> NodeInfo:
    plat = platform_id()
    source = SOURCES.get(mirror, SOURCES["default"])
    filename = archive_name(plat)
    base_url = f"{source}/v{NODE_VERSION}"
    expected = NODE_SHA256[plat]
    root = tools_dir()
    final = root / f"node-v{NODE_VERSION}"

    tmp = Path(tempfile.mkdtemp(prefix=".node-", dir=root))
    try:
        report(0.05, f"Downloading Node {NODE_VERSION}")
        archive = tmp / filename
        actual = download(
            f"{base_url}/{filename}",
            archive,
            lambda f: report(0.05 + 0.8 * f, f"Downloading Node {NODE_VERSION}"),
            cn_hint=mirror != "cn",
        )
        report(0.86, "Verifying checksum")
        if actual.lower() != expected:
            raise StageError(
                "checksum_mismatch",
                f"{filename} sha256 {actual} does not match the pinned {expected}.",
            )

        report(0.9, "Unpacking")
        unpack = tmp / "unpack"
        try:
            _extract(archive, unpack, report.tick)
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
        # A policy block propagates (see ensure_node) and leaves the install on
        # disk unrecorded: the next run adopts it, is blocked again and reports
        # it without a new download.
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
    report = Heartbeat(progress or (lambda _f, _m: None))

    system = _system_node()
    if system is not None:
        # Housekeeping only: a record we cannot remove must not fail a usable
        # system Node.
        with contextlib.suppress(OSError):
            _record_path().unlink(missing_ok=True)
        return system

    global _failed_install
    root = tools_dir()
    try:
        recorded = _recorded_node()
        if recorded is not None:
            return recorded
        if _failed_install is not None:
            # One process can ask several times (server spawn, MCP load,
            # WebUI); on a blocked network each attempt would wait out the
            # timeouts again.
            raise _failed_install
        root.mkdir(parents=True, exist_ok=True)
        with stage_lock(root / "node.lock", "Node.js", progress):
            # Another process may have finished the install while we waited.
            recorded = _recorded_node() or _adopt_installed(root)
            if recorded is not None:
                return recorded
            remove_stale_temp_dirs(root, ".node-")
            return _install(mirror or configured_mirror(), report)
    except PolicyBlocked as exc:
        # Our recorded, left-over or fresh Node is blocked: report it instead
        # of downloading the same blocked program again.
        _failed_install = policy_error(exc.path, _POLICY_ALTERNATIVE)
        raise _failed_install from exc
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
    ``langgraph dev`` - inherit the change; the agent's shell moves it last
    when a usable Node of the user's is on PATH (``agent_env``). Returns the
    directory put on PATH, or None when no private Node is recorded.
    """
    record = _read_record()
    if record is None:
        return None
    install_dir = Path(record["path"])
    bin_dir = _bin_dir(install_dir)
    if not _node_exe(install_dir).exists():
        return None
    prepend_to_path(bin_dir)
    return bin_dir


def is_private(exe: str | Path | None) -> bool:
    """True when ``exe`` lives under our tools dir."""
    return exe is not None and is_under(Path(exe), tools_dir())


# The user's npm network settings: a company or campus network may need its
# own registry, proxy or CA certificate, so these reach our Node's npm too.
NPM_NETWORK_VARS = frozenset(
    {
        "npm_config_registry",
        "npm_config_proxy",
        "npm_config_https_proxy",
        "npm_config_noproxy",
        "npm_config_cafile",
        "npm_config_ca",
        "npm_config_strict_ssl",
    }
)


def node_child_env(env: Mapping[str, str], *, private: bool) -> dict[str, str]:
    """Copy ``env`` for a child that runs the private ``node`` / ``npm`` / ``npx``.

    With ``private`` set, drops ``NODE_OPTIONS`` and the ``npm_config_*``
    variables other than :data:`NPM_NETWORK_VARS` (names compared without
    case, as npm does), so a user's global npm settings cannot break our
    Node. A system Node keeps the user's environment as it is.
    """
    if not private:
        return dict(env)
    return {
        k: v
        for k, v in env.items()
        if k.upper() != "NODE_OPTIONS"
        and (not k.lower().startswith("npm_config_") or k.lower() in NPM_NETWORK_VARS)
    }


def npm_network_env(env: Mapping[str, str]) -> dict[str, str]:
    """The :data:`NPM_NETWORK_VARS` set in ``env``, as written there."""
    return {k: v for k, v in env.items() if k.lower() in NPM_NETWORK_VARS}


_PROXY_VARS = ("http_proxy", "https_proxy")
_LOOPBACK = ("localhost", "127.0.0.1", "::1")


def node_network_env(env: Mapping[str, str], *, private: bool) -> dict[str, str]:
    """Variables to add for a private Node child so its own requests get
    through a company network. Empty for a system Node, which keeps the
    user's setup as it is (and may be too old to know these variables).

    ``NODE_USE_SYSTEM_CA=1`` (unless the user set it either way) makes Node
    trust the operating system's certificates on top of its bundled ones, as
    EvoScientist's Python downloads do: a proxy or firewall that inspects TLS
    re-signs every site with a company certificate that only the system store
    holds. :func:`node_child_env` removes ``NODE_OPTIONS``, so a
    ``--use-system-ca`` set there would not reach the child.

    Node's built-in ``fetch`` (and ``http``) ignore ``HTTPS_PROXY`` unless
    ``NODE_USE_ENV_PROXY=1`` is set (Node 24). It is set only when every
    proxy variable that is set is an ``http://`` or ``https://`` URL: Node
    exits at start for a value it refuses (``socks4://``, a malformed URL),
    and ``http`` ignores ``socks5://`` silently. Loopback then goes into each
    ``NO_PROXY`` / ``no_proxy`` already set, or ``NO_PROXY`` alone when none
    is (two names differing only in case would both land in one Windows
    environment block), so requests to this machine (Next.js calls its own
    ``localhost`` port) never go to the proxy.
    """
    if not private:
        return {}
    extra: dict[str, str] = {}
    if not any(k.upper() == "NODE_USE_SYSTEM_CA" for k in env):
        extra["NODE_USE_SYSTEM_CA"] = "1"
    proxies = [
        v.strip() for k, v in env.items() if k.lower() in _PROXY_VARS and v.strip()
    ]
    if not proxies or not all(_is_http_url(v) for v in proxies):
        return extra
    extra["NODE_USE_ENV_PROXY"] = "1"
    no_proxy_keys = [k for k in env if k.lower() == "no_proxy"] or ["NO_PROXY"]
    for key in no_proxy_keys:
        entries = [e.strip() for e in env.get(key, "").split(",") if e.strip()]
        extra[key] = ",".join(entries + [h for h in _LOOPBACK if h not in entries])
    return extra


def _is_http_url(value: str) -> bool:
    import urllib.parse

    try:
        parts = urllib.parse.urlsplit(value)
        return parts.scheme.lower() in ("http", "https") and bool(parts.hostname)
    except ValueError:
        return False


def npm_registry_override(env: Mapping[str, str], node_exe: Path) -> str | None:
    """npmmirror for ``npx`` children under ``mirror: cn``, unless the user
    configured a registry (environment or ``npm config get registry``)."""
    if configured_mirror() != "cn":
        return None
    if configured_npm_registry(env, node_exe) is not None:
        return None
    return NPM_REGISTRIES["cn"] + "/"


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

    npm = which_in("npm", Path(node_exe).parent)
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
    # On macOS and Linux `npm` is `#!/usr/bin/env node`: it runs the first
    # `node` on PATH, which is not ours when it was installed in this process.
    child_env["PATH"] = os.pathsep.join(
        filter(None, [str(Path(npm).parent), child_env.get("PATH")])
    )
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
