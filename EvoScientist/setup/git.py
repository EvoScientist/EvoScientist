"""The Git for Windows setup stage (Windows only).

A system Git for Windows with its ``bin\\bash.exe`` is used as it is. Otherwise
PortableGit :data:`GIT_VERSION` is installed under ``<DATA_DIR>/tools/``. Either
way the chosen Git is recorded in ``tools/git.json`` (C07 reads the bash path
from it). For PortableGit the record is written only after the archive passed
its pinned SHA-256, the self-extractor finished its post-install step, and
``git --version`` and ``bash --version`` ran.

The archive is a 7-Zip self-extracting ``.exe`` that Python cannot unpack (it
uses LZMA ``lc=8``), so the verified file is run: hidden, because ``-y`` alone
still shows a progress window, and under our own timeout, because the
self-extractor waits for its post-install step without one and always exits 0.
Success is therefore judged on disk.

EvoScientist never edits the user's ``PATH`` or registry and never touches a
system Git for Windows: the private Git reaches child processes only through
:func:`activate_runtime`, which changes ``PATH`` inside this process.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
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
from .download import download
from .protocol import Emitter, StageError, StageResult, make_event

logger = logging.getLogger(__name__)

GIT_TAG = "v2.56.0.windows.1"
GIT_VERSION = "2.56.0.windows.1"

# Asset names are pinned in full: first builds drop the ".1" from them.
# Values: (asset, SHA-256, the architecture's internal prefix directory).
ASSETS = {
    "x64": (
        "PortableGit-2.56.0-64-bit.7z.exe",
        "eceb5e061aa90df2f69ddd3e90f0030e1b8037a7829934bc40e4be1caa1accc1",
        "ucrt64",
    ),
    "arm64": (
        "PortableGit-2.56.0-arm64.7z.exe",
        "edd9bd32aefa5d2bd4b938c38c18ceca306a7f6b29a6951cd6a4bb16d9d28d8f",
        "clangarm64",
    ),
}

SOURCES = {
    "default": "https://github.com/git-for-windows/git/releases/download",
    "cn": "https://registry.npmmirror.com/-/binary/git-for-windows",
}

_ARCH = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}

# 2.56 x64 unpacks to about 406 MB in 9611 files, next to the 60 MB archive.
MIN_FREE_BYTES = 600 * 1024 * 1024
# The extraction plus post-install took 19-28 s with antivirus; a post-install
# can hang (git-for-windows#3636).
SFX_TIMEOUT = 600
_PROBE_TIMEOUT = 30

_VERSION_RE = re.compile(r"^git version (\S+)")

ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class GitInfo:
    """The Git that EvoScientist uses."""

    source: str  # "system" | "portablegit"
    version: str  # e.g. "2.56.0.windows.1"
    git: Path  # <root>\cmd\git.exe
    bash: Path  # <root>\bin\bash.exe

    def detail(self) -> dict[str, Any]:
        return {"source": self.source, "version": self.version, "bash": str(self.bash)}


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #
def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run a probe without a console window; None if it could not run or failed.

    ``GIT_EXEC_PATH`` is dropped so ``git --exec-path`` reports the binary's own
    install, not an inherited override.
    """
    env = {k: v for k, v in os.environ.items() if k.upper() != "GIT_EXEC_PATH"}
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
            env=env,
            creationflags=_no_window(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result if result.returncode == 0 else None


def _git_version(git: Path) -> str | None:
    """The version of a Git for Windows build (``2.56.0.windows.1``), else None.

    MSYS2's and Cygwin's own git report no ``.windows.N`` and do not count.
    """
    result = _run([str(git), "--version"])
    match = _VERSION_RE.match(result.stdout.strip()) if result else None
    if match is None or ".windows." not in match[1]:
        return None
    return match[1]


def _bash_runs(bash: Path) -> bool:
    return bash.is_file() and _run([str(bash), "--version"]) is not None


def _root_from_exec_path(git: Path) -> Path | None:
    """The install root from ``git --exec-path`` (``<root>/<prefix>/libexec/git-core``).

    Git for Windows computes it from the running binary's location, so it names
    the real install even when PATH holds a launcher shim, and it works for
    every internal prefix (``mingw64``, ``ucrt64``, ``clangarm64``).
    """
    result = _run([str(git), "--exec-path"])
    if result is None:
        return None
    exec_path = Path(result.stdout.strip())
    if exec_path.parts[-2:] != ("libexec", "git-core"):
        return None
    return exec_path.parent.parent.parent


def _probe_root(root: Path, source: str, version: str | None = None) -> GitInfo | None:
    """A Git for Windows install at ``root`` with ``cmd\\git.exe`` and a running
    ``bin\\bash.exe``, or None."""
    git = root / "cmd" / "git.exe"
    bash = root / "bin" / "bash.exe"
    if not git.is_file():
        return None
    version = version or _git_version(git)
    if version is None or not _bash_runs(bash):
        return None
    return GitInfo(source, version, git, bash)


def _system_git() -> GitInfo | None:
    """A Git for Windows on PATH, outside our tools dir, with ``bin\\bash.exe``.

    Our own tools dir is skipped: :func:`activate_runtime` runs before
    ``EvoSci setup``, so the private PortableGit would otherwise be found.
    """
    root = tools_dir()
    search = os.pathsep.join(
        p
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p and not is_under(Path(p), root)
    )
    found = shutil.which("git", path=search)
    if found is None:
        return None
    version = _git_version(Path(found))
    if version is None:
        return None
    install = _root_from_exec_path(Path(found))
    if install is None or is_under(install, root):
        return None
    return _probe_root(install, "system", version)


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #
def _record_path() -> Path:
    return tools_dir() / "git.json"


def _read_record() -> dict[str, str] | None:
    try:
        data = json.loads(_record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    fields = ("version", "git", "bash", "source")
    if not all(isinstance(data.get(f), str) and data[f] for f in fields):
        return None
    if data["source"] not in ("system", "portablegit"):
        return None
    return data


def _write_record(info: GitInfo) -> None:
    record = _record_path()
    record.parent.mkdir(parents=True, exist_ok=True)
    tmp = record.with_name(record.name + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "version": info.version,
                "git": str(info.git),
                "bash": str(info.bash),
                "source": info.source,
            }
        ),
        encoding="utf-8",
    )
    os.replace(tmp, record)


def _recorded_portablegit() -> GitInfo | None:
    """The recorded PortableGit, if its probes still pass."""
    record = _read_record()
    if record is None or record["source"] != "portablegit":
        return None
    return _probe_root(Path(record["git"]).parent.parent, "portablegit")


def _adopt_installed(root: Path) -> GitInfo | None:
    """Record an intact ``git-<GIT_VERSION>`` again after its record was replaced.

    A system Git replaces the record but leaves PortableGit on disk; without
    this a later run without the system Git would download it again, or fail
    offline.
    """
    info = _probe_root(root / f"git-{GIT_VERSION}", "portablegit")
    if info is not None:
        _write_record(info)
    return info


# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #
def _arch() -> str:
    arch = _ARCH.get(platform.machine().lower())
    if arch is None:
        raise StageError(
            "unsupported_platform",
            f"No PortableGit build for Windows on {platform.machine()}.",
        )
    return arch


def _kill_tree(pid: int) -> None:
    import psutil

    try:
        proc = psutil.Process(pid)
        targets = [*proc.children(recursive=True), proc]
    except psutil.Error:
        return
    for target in targets:
        try:
            target.kill()
        except psutil.Error:
            pass


def _run_sfx(exe: Path, out: Path) -> None:
    """Run the verified self-extractor into ``out``, hidden and time-limited.

    It parses exactly ``-y`` and ``-o<dir>``: anything after them is appended
    to its post-install command, and a trailing backslash on the directory is
    kept doubled, so neither is passed.
    """
    argv = [str(exe), "-y", f"-o{str(out).rstrip(os.sep)}"]
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0  # SW_HIDE: also hides the progress window
        kwargs = {"startupinfo": startup, "creationflags": _no_window()}
    try:
        proc = subprocess.Popen(argv, **kwargs)
    except OSError as exc:
        raise StageError("install_failed", f"Could not run {exe.name}: {exc}") from exc
    try:
        code = proc.wait(timeout=SFX_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        _kill_tree(proc.pid)
        proc.wait()
        raise StageError(
            "install_failed",
            f"{exe.name} did not finish within {SFX_TIMEOUT} s and was stopped.",
        ) from exc
    if code != 0:
        raise StageError("install_failed", f"{exe.name} exited with code {code}.")


def _check_post_install(out: Path) -> None:
    """The self-extractor exits 0 whatever its post-install step did, and
    ``post-install.bat`` deletes itself unconditionally. ``etc\\post-install``
    is removed by the last post-install script, so it survives an aborted run."""
    for leftover in (out / "post-install.bat", out / "etc" / "post-install"):
        if leftover.exists():
            raise StageError(
                "probe_failed",
                f"PortableGit's post-install step did not finish ({leftover.name} "
                "is still there).",
            )


def _install(mirror: str, report: ProgressFn) -> GitInfo:
    asset, sha256, _prefix = ASSETS[_arch()]
    root = tools_dir()
    free = shutil.disk_usage(root).free
    if free < MIN_FREE_BYTES:
        raise StageError(
            "install_failed",
            f"PortableGit needs about {MIN_FREE_BYTES // 2**20} MB free in {root}; "
            f"{free // 2**20} MB are available.",
        )
    source = SOURCES.get(mirror, SOURCES["default"])
    final = root / f"git-{GIT_VERSION}"

    tmp = Path(tempfile.mkdtemp(prefix=".git-", dir=root))
    try:
        # Saved under a name without ".exe" until its hash matches, so a partial
        # or tampered download cannot be launched.
        part = tmp / f"{asset}.part"
        report(0.05, f"Downloading Git for Windows {GIT_VERSION}")
        actual = download(
            f"{source}/{GIT_TAG}/{asset}",
            part,
            lambda f: report(
                0.05 + 0.7 * f, f"Downloading Git for Windows {GIT_VERSION}"
            ),
        )
        report(0.76, "Verifying checksum")
        if actual.lower() != sha256:
            part.unlink(missing_ok=True)
            raise StageError(
                "checksum_mismatch",
                f"{asset} sha256 {actual} does not match the pinned {sha256}.",
            )
        exe = tmp / asset
        os.replace(part, exe)

        report(0.8, "Unpacking")
        out = tmp / "PortableGit"
        _run_sfx(exe, out)
        _check_post_install(out)
        try:
            if final.exists():
                # Left behind by an earlier attempt whose probe failed.
                shutil.rmtree(final)
            move_into_place(out, final, what="Git")
        except OSError as exc:
            raise StageError(
                "install_failed", f"Could not move Git into {final}: {exc}"
            ) from exc

        report(0.95, "Checking the installed Git")
        info = _probe_root(final, "portablegit")
        if info is None:
            shutil.rmtree(final, ignore_errors=True)
            raise StageError(
                "probe_failed",
                f"git --version or bash --version did not run in {final}.",
            )
        _write_record(info)
        return info
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ensure_git(mirror: str = "default", progress: ProgressFn | None = None) -> GitInfo:
    """Return the Git to use, installing PortableGit if needed.

    Order: a system Git for Windows with ``bin\\bash.exe``; the recorded
    PortableGit if its probes pass (no network needed); an intact
    ``git-<GIT_VERSION>`` left on disk; otherwise download, verify, extract,
    probe and record. The chosen Git is recorded either way. Raises
    :class:`StageError` on failure, leaving any previous record in place.
    """
    report = progress or (lambda _f, _m: None)
    root = tools_dir()
    try:
        system = _system_git()
        if system is not None:
            _write_record(system)
            return system

        recorded = _recorded_portablegit()
        if recorded is not None:
            return recorded

        from filelock import FileLock

        root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(root / "git.lock")):
            # Another process may have finished the install while we waited.
            recorded = _recorded_portablegit() or _adopt_installed(root)
            if recorded is not None:
                return recorded
            remove_stale_temp_dirs(root, ".git-")
            return _install(mirror, report)
    except OSError as exc:
        raise StageError(
            "install_failed", f"Could not set up Git in {root}: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
def activate_runtime() -> Path | None:
    """Put the recorded PortableGit's ``cmd`` directory first on ``PATH``.

    Windows only, and only for PortableGit: a system Git stays where the user's
    PATH puts it. Only ``cmd`` is added, as PortableGit's ``README.portable``
    suggests: ``usr\\bin`` would shadow Windows commands such as ``find`` and
    ``sort`` for the agent's ``cmd.exe`` shell. Reads ``tools/git.json`` only
    (no subprocess). Returns the directory put on PATH, or None.
    """
    if sys.platform != "win32":
        return None
    record = _read_record()
    if record is None or record["source"] != "portablegit":
        return None
    git = Path(record["git"]).resolve()
    if not git.is_file():
        return None
    prepend_to_path(git.parent)
    return git.parent


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
def run_stage(emit: Emitter, mirror: str) -> StageResult:
    emit(make_event("git", "running", progress=0.0, message="Checking for Git"))

    def report(fraction: float, message: str) -> None:
        emit(make_event("git", "running", progress=fraction, message=message))

    info = ensure_git(mirror, report)
    if info.source == "system":
        message = f"Using system Git for Windows {info.version}"
    else:
        message = f"Using PortableGit {info.version} from {info.git.parent.parent}"
    return StageResult(message, info.detail())
