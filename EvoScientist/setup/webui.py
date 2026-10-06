"""The WebUI setup stage.

Installs the ``@evoscientist/webui`` front-end under ``<DATA_DIR>/tools/webui/<version>/``
and records it in ``tools/webui.json``. The version is the newest one inside
:data:`WEBUI_COMPAT`, the range of WebUI releases this core supports. The
tarball is checked against the ``sha512`` integrity npm publishes for it, the
unpacked copy is started once on a free loopback port and must answer, and only
then is it recorded, so a failed or partial install never replaces a working
copy.

The package ships a prebuilt Next.js standalone server: it needs only a Node,
no ``npm install`` and no network once installed.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._install import move_into_place, remove_stale_temp_dirs, tools_dir
from .download import download
from .protocol import Emitter, StageError, StageResult, make_event

logger = logging.getLogger(__name__)

PACKAGE = "@evoscientist/webui"
# The WebUI releases this core supports. A WebUI patch release reaches users
# without a core release; a breaking WebUI release needs a core change here.
WEBUI_COMPAT = ">=0.3,<0.4"
# Test-only override of the range (e.g. to install an older release and
# exercise the update path). Never saved to the config.
COMPAT_ENV = "EVOSCIENTIST_WEBUI_COMPAT"

_METADATA_TIMEOUT = 30
# The abbreviated document npm itself requests: a fraction of the full one and
# still carries ``dist.integrity`` / ``dist.tarball``. A registry that ignores
# the header answers with the full document, which has the same fields.
_METADATA_ACCEPT = "application/vnd.npm.install-v1+json; q=1.0, application/json; q=0.8"
_PROBE_TIMEOUT = 60.0
_LOGO = "/evoscientist-logo.png"
_TEMP_PREFIX = ".webui-"

ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class WebUIInfo:
    """An installed WebUI copy."""

    version: str
    path: Path  # the package root; the server is ``<path>/dist/server.js``

    @property
    def server_entry(self) -> Path:
        return self.path / "dist" / "server.js"

    def detail(self) -> dict[str, Any]:
        return {"version": self.version, "path": str(self.path)}


# --------------------------------------------------------------------------- #
# Range
# --------------------------------------------------------------------------- #
_warned_override: str | None = None


def compat_range() -> str:
    """The supported WebUI range: :data:`WEBUI_COMPAT`, or the test-only
    ``EVOSCIENTIST_WEBUI_COMPAT`` override when it is a valid range."""
    from packaging.specifiers import InvalidSpecifier, SpecifierSet

    global _warned_override
    override = os.environ.get(COMPAT_ENV, "").strip()
    if not override:
        return WEBUI_COMPAT
    try:
        SpecifierSet(override)
    except InvalidSpecifier:
        if _warned_override != override:
            _warned_override = override
            logger.warning(
                f"Ignoring {COMPAT_ENV}={override!r}: not a version range; "
                f"using {WEBUI_COMPAT}."
            )
        return WEBUI_COMPAT
    if _warned_override != override:
        _warned_override = override
        logger.warning(
            f"{COMPAT_ENV} overrides the supported WebUI range: {override} "
            f"instead of {WEBUI_COMPAT}. For testing only."
        )
    return override


def in_range(version: str, compat: str) -> bool:
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version

    try:
        parsed = Version(version)
    except InvalidVersion:
        return False
    return not parsed.is_prerelease and parsed in SpecifierSet(compat)


def pick_version(versions: Mapping[str, Any], compat: str) -> str:
    """The newest release in ``compat``; pre-releases and versions that are not
    valid PEP 440 (npm allows ``0.4.0-beta.1``) are skipped."""
    from packaging.version import Version

    candidates = [v for v in versions if in_range(v, compat)]
    if not candidates:
        raise StageError(
            "no_compatible_version",
            f"The registry has no {PACKAGE} release in the supported range {compat}.",
        )
    return max(candidates, key=Version)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def registry_url(mirror: str, node_exe: Path) -> str:
    """The one registry to read: the user's, else the mirror's, else npmjs.

    A priority, not a fallback: an error from the chosen registry is reported,
    never retried on another one.
    """
    from .node import DEFAULT_NPM_REGISTRY, NPM_REGISTRIES, configured_npm_registry

    chosen = (
        configured_npm_registry(os.environ, node_exe)
        or NPM_REGISTRIES.get(mirror)
        or DEFAULT_NPM_REGISTRY
    )
    return chosen.rstrip("/")


def fetch_metadata(registry: str) -> dict[str, Any]:
    url = f"{registry}/{PACKAGE.replace('/', '%2F')}"
    request = urllib.request.Request(url, headers={"Accept": _METADATA_ACCEPT})
    try:
        with urllib.request.urlopen(request, timeout=_METADATA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            hint = (
                " A registry that needs a token is not supported: EvoScientist "
                "reads it directly, not through npm."
            )
        elif exc.code == 404:
            hint = f" The registry does not host {PACKAGE}."
        else:
            hint = ""
        raise StageError(
            "download_failed",
            f"{registry} answered HTTP {exc.code} for {PACKAGE}.{hint}",
        ) from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise StageError(
            "download_failed", f"Could not read {PACKAGE} from {registry}: {exc}"
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("versions"), dict):
        raise StageError(
            "download_failed", f"{registry} sent no version list for {PACKAGE}."
        )
    return data


def verify_integrity(path: Path, dist: Mapping[str, Any], name: str) -> None:
    """Check ``path`` against the ``sha512`` SRI value in ``dist.integrity``.

    Fails closed: no ``sha512`` value is a failure too, never a skipped check.
    """
    integrity = dist.get("integrity")
    expected = [
        token.removeprefix("sha512-")
        for token in (integrity.split() if isinstance(integrity, str) else [])
        if token.startswith("sha512-")
    ]
    if not expected:
        raise StageError(
            "checksum_mismatch",
            f"The registry published no sha512 integrity for {name}.",
        )
    digest = hashlib.sha512()
    with open(path, "rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    if base64.b64encode(digest.digest()).decode() not in expected:
        raise StageError(
            "checksum_mismatch",
            f"{name} does not match the sha512 integrity the registry published.",
        )


# --------------------------------------------------------------------------- #
# Unpack and probe
# --------------------------------------------------------------------------- #
def extract_package(archive: Path, dest: Path) -> None:
    """Unpack the ``package/`` tree of an npm tarball into ``dest``.

    Regular files and directories only: links are skipped (the standalone
    server resolves modules by path), and a member that would land outside
    ``dest`` fails the install.
    """
    dest = dest.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            name = member.name.replace("\\", "/")
            if not name.startswith("package/"):
                continue
            rel = name[len("package/") :].strip("/")
            if not rel:
                continue
            target = (dest / rel).resolve()
            if target != dest and dest not in target.parents:
                raise StageError(
                    "download_failed",
                    f"Unsafe path in the WebUI archive: {member.name}",
                )
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            source = tf.extractfile(member)
            if source is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)


def _check_layout(root: Path, version: str) -> None:
    try:
        meta = json.loads((root / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StageError(
            "download_failed", f"The WebUI archive has no readable package.json: {exc}"
        ) from exc
    if meta.get("name") != PACKAGE or meta.get("version") != version:
        raise StageError(
            "download_failed",
            f"The WebUI archive holds {meta.get('name')}@{meta.get('version')}, "
            f"not {PACKAGE}@{version}.",
        )
    if not (root / "dist" / "server.js").is_file():
        raise StageError("download_failed", "The WebUI archive has no dist/server.js.")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get_status(url: str, timeout: float) -> int | None:
    """HTTP status of ``url`` without any proxy, or None when nothing answers."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, OSError, ValueError):
        return None


def server_env(node_exe: Path, port: int) -> dict[str, str]:
    """Environment for ``node dist/server.js``: the parent's, without secrets,
    and without npm / Node options when the Node is our private one."""
    from ..deploy.launcher import _scrubbed_env
    from .node import is_private, node_child_env

    env = _scrubbed_env({"PORT": str(port), "HOSTNAME": "127.0.0.1"})
    env["NODE_ENV"] = "production"
    return node_child_env(env, private=is_private(node_exe))


def probe(root: Path, node_exe: Path, timeout: float = _PROBE_TIMEOUT) -> None:
    """Start the copy at ``root`` on a free loopback port and require ``/`` and
    the static logo to answer 200; always stop the server again."""
    from ..deploy.launcher import _popen_group_kwargs, _stop_process_tree

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    kwargs = _popen_group_kwargs()
    if os.name == "nt":
        kwargs["creationflags"] = (
            kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
        )
    with tempfile.TemporaryFile() as log:
        try:
            proc = subprocess.Popen(
                [str(node_exe), str(root / "dist" / "server.js")],
                env=server_env(node_exe, port),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                **kwargs,
            )
        except OSError as exc:
            raise StageError(
                "probe_failed", f"Could not start the WebUI server: {exc}"
            ) from exc
        try:
            deadline = time.monotonic() + timeout
            status = None
            while time.monotonic() < deadline and proc.poll() is None:
                status = _get_status(f"{base}/", timeout=5)
                if status is not None:
                    break
                time.sleep(0.5)
            logo = _get_status(f"{base}{_LOGO}", timeout=10) if status == 200 else None
            exited = proc.poll() is not None
        finally:
            _stop_process_tree(proc)
        if status == 200 and logo == 200:
            return
        log.seek(0)
        tail = log.read()[-800:].decode("utf-8", "replace").strip()
    if status is None:
        what = "exited" if exited else "did not answer"
        reason = f"the WebUI server {what} within {timeout:.0f} s"
    elif status != 200:
        reason = f"/ answered HTTP {status}"
    else:
        reason = f"{_LOGO} answered {logo}"
    raise StageError(
        "probe_failed",
        f"The installed WebUI failed its check: {reason}. {tail}".strip(),
    )


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #
# ``tools/webui.json``:
#   version, path   the copy to run
#   previous        the version recorded before it (kept by the cleanup)
#   staged          {version, path}: downloaded in the background, not yet probed
#   rejected        versions whose probe failed; the background never fetches
#                   them again (``EvoSci setup`` may retry them)
def webui_root() -> Path:
    return tools_dir() / "webui"


def _record_path() -> Path:
    return tools_dir() / "webui.json"


def _lock_path() -> Path:
    return tools_dir() / "webui.lock"


def _read_record() -> dict[str, Any] | None:
    try:
        data = json.loads(_record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("version"), str) or not isinstance(
        data.get("path"), str
    ):
        return None
    return data


def _write_record(record: Mapping[str, Any]) -> None:
    from ._install import atomic_write_text

    atomic_write_text(_record_path(), json.dumps(record))


def _staged(record: Mapping[str, Any] | None) -> WebUIInfo | None:
    staged = (record or {}).get("staged")
    if not isinstance(staged, dict):
        return None
    version, path = staged.get("version"), staged.get("path")
    if not isinstance(version, str) or not isinstance(path, str):
        return None
    return WebUIInfo(version, Path(path))


def _rejected(record: Mapping[str, Any] | None) -> list[str]:
    rejected = (record or {}).get("rejected")
    return (
        [v for v in rejected if isinstance(v, str)]
        if isinstance(rejected, list)
        else []
    )


def _set_current(record: dict[str, Any] | None, info: WebUIInfo) -> dict[str, Any]:
    """``record`` with ``info`` as the copy to run; the old one becomes ``previous``."""
    new = dict(record or {})
    old = new.get("version")
    if isinstance(old, str) and old != info.version:
        new["previous"] = old
    new.update(info.detail())
    staged = _staged(new)
    if staged is not None and staged.version == info.version:
        new.pop("staged")
    return new


def recorded_webui(compat: str | None = None) -> WebUIInfo | None:
    """The recorded copy, if it is inside the range and its server is on disk."""
    record = _read_record()
    if record is None:
        return None
    info = WebUIInfo(record["version"], Path(record["path"]))
    if not in_range(info.version, compat or compat_range()):
        return None
    if not info.server_entry.is_file():
        return None
    return info


def _set_aside(path: Path) -> bool:
    """Move ``path`` out of the way with one rename, so the stale-temp cleanup
    deletes it later. False when the rename fails: on Windows that means a
    server still runs from it."""
    try:
        os.replace(
            path, Path(tempfile.mkdtemp(prefix=_TEMP_PREFIX, dir=path.parent)) / "old"
        )
    except OSError as exc:
        logger.debug(f"Could not set {path} aside: {exc!r}")
        return False
    return True


# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #
def _download_into_place(
    version: str,
    dist: Mapping[str, Any],
    report: ProgressFn,
    node_exe: Path | None,
) -> Path:
    """Download, verify and unpack ``version`` into ``tools/webui/<version>/``.

    With ``node_exe`` the copy is probed in its temporary directory before it
    moves into place, so the version directory never holds an unchecked copy
    that is recorded to run. Without it (a background download) the probe is
    left to the next launch.
    """
    root = webui_root()
    final = root / version
    name = f"{PACKAGE}@{version}"
    tarball = dist.get("tarball")
    if not isinstance(tarball, str) or not tarball:
        raise StageError(
            "download_failed", f"The registry gave no tarball URL for {name}."
        )

    tmp = Path(tempfile.mkdtemp(prefix=_TEMP_PREFIX, dir=root))
    try:
        archive = tmp / "webui.tgz"
        report(0.1, f"Downloading WebUI {version}")
        download(
            tarball,
            archive,
            lambda f: report(0.1 + 0.6 * f, f"Downloading WebUI {version}"),
        )
        report(0.72, "Verifying checksum")
        verify_integrity(archive, dist, name)

        report(0.75, "Unpacking")
        unpacked = tmp / "package"
        try:
            extract_package(archive, unpacked)
        except (tarfile.TarError, EOFError) as exc:
            raise StageError(
                "download_failed", f"Could not unpack {name}: {exc}"
            ) from exc
        _check_layout(unpacked, version)

        if node_exe is not None:
            report(0.85, "Checking the WebUI server")
            probe(unpacked, node_exe)

        if final.exists() and not _set_aside(final):
            # A copy that was never recorded, still held by a running server.
            raise StageError(
                "install_failed", f"{final} is in use and cannot be replaced."
            )
        move_into_place(unpacked, final, what="the WebUI")
        return final
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _promote_staged(compat: str, node_exe: Path, report: ProgressFn) -> None:
    """Make a copy staged by the background check the one to run, after it
    passes the probe. Call with the stage lock held.

    A staged copy outside the current range (the range may have moved since
    the download) is dropped; one that fails its probe is dropped and its
    version is not fetched in the background again.
    """
    record = _read_record()
    staged = _staged(record)
    if record is None or staged is None:
        return
    record = dict(record)
    record.pop("staged")
    if not in_range(staged.version, compat) or not staged.server_entry.is_file():
        logger.info(
            f"Dropping staged WebUI {staged.version}: not usable with {compat}."
        )
        _set_aside(staged.path)
    else:
        report(0.05, f"Checking WebUI {staged.version}")
        try:
            probe(staged.path, node_exe)
        except StageError as exc:
            logger.warning(
                f"WebUI {staged.version} failed its check and is not used; "
                f"keeping {record['version']}. {exc.message}"
            )
            record["rejected"] = sorted({*_rejected(record), staged.version})
            _set_aside(staged.path)
        else:
            record = _set_current(record, staged)
    _write_record(record)


# The first failed install in this process; later calls raise it again without
# retrying (the launch path may ask more than once).
_failed_install: StageError | None = None


def ensure_webui(
    node_exe: Path,
    mirror: str | None = None,
    progress: ProgressFn | None = None,
    *,
    refresh: bool = False,
) -> WebUIInfo:
    """Return the WebUI copy to run, installing it if needed.

    A copy staged by the background check is probed and, if it passes, takes
    over first. Without ``refresh`` the recorded copy is then used as it is,
    with no network. With ``refresh`` (``EvoSci setup``) the registry is asked
    for the newest version in range; when it cannot be reached and a copy is
    recorded, that copy is used. ``node_exe`` runs the probe and picks the
    ``npm`` whose registry setting counts. Raises :class:`StageError`; a
    failure leaves any previous record in place.
    """
    from .node import configured_mirror

    report = progress or (lambda _f, _m: None)
    compat = compat_range()
    recorded = recorded_webui(compat)
    if recorded is not None and not refresh and _staged(_read_record()) is None:
        return recorded

    global _failed_install
    if _failed_install is not None:
        raise _failed_install

    from filelock import FileLock

    root = webui_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(_lock_path())):
            remove_stale_temp_dirs(root, _TEMP_PREFIX)
            _promote_staged(compat, node_exe, report)
            recorded = recorded_webui(compat)
            if recorded is not None and not refresh:
                return recorded
            registry = registry_url(mirror or configured_mirror(), node_exe)
            report(0.02, f"Checking {registry} for WebUI releases in {compat}")
            try:
                metadata = fetch_metadata(registry)
            except StageError:
                if recorded is not None:
                    logger.info("Registry unreachable; keeping the recorded WebUI.")
                    return recorded
                raise
            version = pick_version(metadata["versions"], compat)
            if recorded is not None and recorded.version == version:
                return recorded
            dist = metadata["versions"][version].get("dist") or {}
            info = WebUIInfo(
                version, _download_into_place(version, dist, report, node_exe)
            )
            _write_record(_set_current(_read_record(), info))
            return info
    except StageError as exc:
        _failed_install = exc
        raise
    except OSError as exc:
        _failed_install = StageError(
            "install_failed", f"Could not install the WebUI into {root}: {exc}"
        )
        raise _failed_install from exc


# --------------------------------------------------------------------------- #
# Background update
# --------------------------------------------------------------------------- #
# How often a launch may ask the registry for a newer WebUI.
CHECK_INTERVAL = 86_400


def _check_stamp() -> Path:
    return tools_dir() / "webui.check.json"


def _checked_recently(now: float) -> bool:
    try:
        checked = json.loads(_check_stamp().read_text(encoding="utf-8"))["checked_at"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return isinstance(checked, (int, float)) and 0 <= now - checked < CHECK_INTERVAL


def stage_update(node_exe: Path, mirror: str | None = None) -> str | None:
    """Download a newer WebUI in range for the next launch; return its version.

    Runs at most once per :data:`CHECK_INTERVAL`, skips when another process
    holds the stage lock, never touches the copy that runs and never starts a
    process: the next launch probes the staged copy before it switches. Every
    failure is logged at debug level only.
    """
    from filelock import FileLock, Timeout

    from ._install import atomic_write_text
    from .node import configured_mirror

    now = time.time()
    if _checked_recently(now):
        return None
    try:
        with FileLock(str(_lock_path()), timeout=0):
            atomic_write_text(_check_stamp(), json.dumps({"checked_at": now}))
            record = _read_record()
            if record is None:
                return None
            compat = compat_range()
            registry = registry_url(mirror or configured_mirror(), node_exe)
            metadata = fetch_metadata(registry)
            version = pick_version(metadata["versions"], compat)
            staged = _staged(record)
            if (
                version == record["version"]
                or version in _rejected(record)
                or (staged is not None and staged.version == version)
            ):
                return None
            dist = metadata["versions"][version].get("dist") or {}
            path = _download_into_place(version, dist, lambda _f, _m: None, None)
            record = _read_record() or record
            if staged is not None and staged.path != path:
                _set_aside(staged.path)
            _write_record({**record, "staged": WebUIInfo(version, path).detail()})
            logger.debug(f"Staged WebUI {version} for the next launch.")
            return version
    except Timeout:
        return None
    except Exception as exc:
        logger.debug(f"Background WebUI update check failed: {exc!r}")
        return None


# --------------------------------------------------------------------------- #
# In use and cleanup
# --------------------------------------------------------------------------- #
_IN_USE = ".in-use"


def _process_start_ms(pid: int) -> int | None:
    import psutil

    try:
        return int(psutil.Process(pid).create_time() * 1000)
    except psutil.NoSuchProcess:
        return None


def mark_in_use(info: WebUIInfo, pid: int | None = None) -> Path | None:
    """Mark ``info`` as run by ``pid`` (default: this process) until
    :func:`release_in_use`. The name carries the process start time, so a
    reused pid does not keep a dead marker alive."""
    pid = os.getpid() if pid is None else pid
    start = _process_start_ms(pid)
    if start is None:
        return None
    marker = info.path / _IN_USE / f"{pid}-{start}"
    try:
        marker.parent.mkdir(exist_ok=True)
        marker.touch()
    except OSError as exc:
        logger.debug(f"Could not mark {info.path} in use: {exc!r}")
        return None
    return marker


def release_in_use(marker: Path | None) -> None:
    if marker is not None:
        with contextlib.suppress(OSError):
            marker.unlink()


def _marker_alive(name: str) -> bool:
    import psutil

    try:
        pid_text, start_text = name.split("-", 1)
        pid, start = int(pid_text), int(start_text)
    except ValueError:
        return False
    try:
        actual = _process_start_ms(pid)
    except psutil.Error:
        return True  # cannot tell (e.g. access denied): keep the copy
    return actual is not None and abs(actual - start) <= 1


def _in_use(version_dir: Path) -> bool:
    """True while a live process runs this copy; removes dead markers."""
    alive = False
    with contextlib.suppress(OSError):
        for marker in (version_dir / _IN_USE).iterdir():
            if _marker_alive(marker.name):
                alive = True
            else:
                with contextlib.suppress(OSError):
                    marker.unlink()
    return alive


def cleanup_old_versions() -> list[str]:
    """Delete installed versions other than the current, the previous and a
    staged one; return their names.

    A version a live process runs is kept. Each directory is first renamed
    with a single ``os.replace``: on Windows that fails while a server runs
    from it, and a renamed directory is never mistaken for an install, even if
    its removal stops halfway. Skips when another process holds the stage lock.
    """
    from filelock import FileLock, Timeout

    root = webui_root()
    removed: list[str] = []
    try:
        with FileLock(str(_lock_path()), timeout=0):
            record = _read_record()
            if record is None:
                return removed
            staged = _staged(record)
            keep = {record["version"], record.get("previous")}
            if staged is not None:
                keep.add(staged.version)
            for version_dir in sorted(root.iterdir()):
                if version_dir.name.startswith(".") or not version_dir.is_dir():
                    continue
                in_use = _in_use(version_dir)
                if version_dir.name in keep or in_use:
                    continue
                if _set_aside(version_dir):
                    removed.append(version_dir.name)
            remove_stale_temp_dirs(root, _TEMP_PREFIX)
    except Timeout:
        pass
    except OSError as exc:
        logger.debug(f"WebUI cleanup failed: {exc!r}")
    return removed


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
def run_stage(emit: Emitter, mirror: str) -> StageResult:
    from .node import ensure_node

    emit(make_event("webui", "running", progress=0.0, message="Checking for the WebUI"))

    def report(fraction: float, message: str) -> None:
        emit(make_event("webui", "running", progress=fraction, message=message))

    node = ensure_node(mirror)
    info = ensure_webui(node.path, mirror, report, refresh=True)
    return StageResult(f"Using WebUI {info.version} from {info.path}", info.detail())


__all__ = [
    "COMPAT_ENV",
    "PACKAGE",
    "WEBUI_COMPAT",
    "WebUIInfo",
    "cleanup_old_versions",
    "compat_range",
    "ensure_webui",
    "mark_in_use",
    "release_in_use",
    "run_stage",
    "stage_update",
]
