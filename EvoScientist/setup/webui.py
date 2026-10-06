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
def webui_root() -> Path:
    return tools_dir() / "webui"


def _record_path() -> Path:
    return tools_dir() / "webui.json"


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


def _write_record(info: WebUIInfo) -> None:
    from ._install import atomic_write_text

    atomic_write_text(_record_path(), json.dumps(info.detail()))


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


# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #
def _install(
    version: str, dist: Mapping[str, Any], node_exe: Path, report: ProgressFn
) -> WebUIInfo:
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

        report(0.85, "Checking the WebUI server")
        probe(unpacked, node_exe)

        if final.exists():
            # A copy left by an earlier run that was never recorded; set it
            # aside (a rename fails on Windows while a server runs from it)
            # and let the stale-temp cleanup remove it later.
            os.replace(
                final, Path(tempfile.mkdtemp(prefix=_TEMP_PREFIX, dir=root)) / "old"
            )
        move_into_place(unpacked, final, what="the WebUI")
        info = WebUIInfo(version, final)
        _write_record(info)
        return info
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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

    Without ``refresh`` the recorded copy is used as it is, with no network.
    With ``refresh`` (``EvoSci setup``) the registry is asked for the newest
    version in range; when it cannot be reached and a copy is recorded, that
    copy is used. ``node_exe`` runs the probe and picks the ``npm`` whose
    registry setting counts. Raises :class:`StageError`; a failure leaves any
    previous record in place.
    """
    from .node import configured_mirror

    report = progress or (lambda _f, _m: None)
    compat = compat_range()
    recorded = recorded_webui(compat)
    if recorded is not None and not refresh:
        return recorded

    global _failed_install
    if _failed_install is not None:
        raise _failed_install

    from filelock import FileLock

    root = webui_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(tools_dir() / "webui.lock")):
            recorded = recorded_webui(compat)
            if recorded is not None and not refresh:
                return recorded
            remove_stale_temp_dirs(root, _TEMP_PREFIX)
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
            return _install(version, dist, node_exe, report)
    except StageError as exc:
        _failed_install = exc
        raise
    except OSError as exc:
        _failed_install = StageError(
            "install_failed", f"Could not install the WebUI into {root}: {exc}"
        )
        raise _failed_install from exc


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
    "compat_range",
    "ensure_webui",
    "run_stage",
]
