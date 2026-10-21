"""Install helpers shared by the stages that unpack a tool into ``<DATA_DIR>/tools``."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from .protocol import HEARTBEAT_INTERVAL, StageError

logger = logging.getLogger(__name__)


def tools_dir() -> Path:
    """``<DATA_DIR>/tools``, read at call time so an overridden DATA_DIR applies."""
    from .. import paths

    return paths.DATA_DIR / "tools"


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _clean_entry(entry: str) -> str:
    """A ``PATH`` entry as the OS reads it. On Windows: without surrounding
    whitespace, one surrounding pair of quotes (Windows allows
    ``"C:\\Program Files\\Git\\cmd"`` in ``PATH``) or a trailing separator,
    normalised by ``os.path.normpath`` (``.`` and ``..`` collapsed, as Windows
    does, ``/`` turned into ``\\``). On POSIX only a trailing ``/`` is
    dropped: ``execvp`` and ``sh`` read the rest as written, ``..`` included."""
    if not _WINDOWS:
        return entry.rstrip("/") or entry
    entry = entry.strip()
    if len(entry) >= 2 and entry[0] == entry[-1] == '"':
        entry = entry[1:-1]
    return os.path.normpath(entry) if entry else entry


def same_path(a: str | Path, b: str | Path) -> bool:
    """True when ``a`` and ``b`` name the same location: cleaned like a
    ``PATH`` entry, links resolved, case folded where the OS ignores it."""

    def key(p: str | Path) -> str:
        return os.path.normcase(os.path.realpath(_clean_entry(str(p))))

    return key(a) == key(b)


def prepend_to_path(directory: Path) -> None:
    """Put ``directory`` first on this process's ``PATH``, dropping any other
    entry for it, so child processes inherit it.

    Only ``directory`` is resolved; the other entries are cleaned but not
    resolved, since resolving each of them is slow for a network path.
    """
    keys = {
        os.path.normcase(_clean_entry(str(directory))),
        os.path.normcase(os.path.realpath(directory)),
    }
    parts = [
        p
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p.strip() and os.path.normcase(_clean_entry(p)) not in keys
    ]
    os.environ["PATH"] = os.pathsep.join([str(directory), *parts])


# Read at call time, so tests can exercise the Windows lookup on any OS.
_WINDOWS = os.name == "nt"


def _command_names(name: str) -> list[str]:
    """File names that run as ``name``: on Windows ``name`` plus each
    ``PATHEXT`` extension (``name`` itself first when it already has one)."""
    if not _WINDOWS:
        return [name]
    exts = [
        e.lower()
        # An empty PATHEXT falls back too, as in shutil.which.
        for e in (os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD").split(";")
        if e
    ]
    names = [name] if os.path.splitext(name)[1].lower() in exts else []
    return names + [name + ext for ext in exts]


def which_in(name: str, directory: str | Path) -> str | None:
    """The program ``name`` in ``directory`` only, or None.

    Unlike ``shutil.which(name, path=directory)``, never looks in the current
    directory: on Windows that search comes first (always on Python 3.11, on
    3.12+ unless ``NoDefaultCurrentDirectoryInExePath`` is set) and its hit is
    returned, so a ``node.cmd`` in the folder EvoScientist runs from would win.
    """
    for candidate in _command_names(name):
        path = os.path.join(directory, candidate)
        if os.path.isfile(path) and (_WINDOWS or os.access(path, os.X_OK)):
            return path
    return None


def which_on_path(name: str, *, outside: Path | None = None) -> str | None:
    """The first program ``name`` on ``PATH``, or None.

    Never the current directory (see :func:`which_in`): empty and relative
    ``PATH`` entries are skipped, since both resolve against it. With
    ``outside``, entries under that folder are skipped too, e.g. our tools
    dir when looking for a system tool.
    """
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        directory = _clean_entry(entry)
        if not directory or not os.path.isabs(directory):
            continue
        if outside is not None and is_under(Path(directory), outside):
            continue
        found = which_in(name, directory)
        if found is not None:
            return found
    return None


# Pauses between attempts to move an unpacked tool into place (about 4 s).
_MOVE_RETRY_DELAYS = (0.1, 0.2, 0.5, 1.0, 2.0)


def move_into_place(src: Path, dst: Path, *, what: str) -> None:
    """``os.replace`` that retries briefly on ``PermissionError``.

    On Windows a scanner that opens a freshly written executable makes the
    rename of its folder fail with "Access is denied" (WinError 5) for a
    moment. Other errors, and a denial that outlasts the retries, raise.
    ``what`` names the tool in the warning, e.g. ``"Node"``.
    """
    for denied, delay in enumerate((*_MOVE_RETRY_DELAYS, None)):
        try:
            os.replace(src, dst)
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)
            continue
        if denied:
            logger.warning(
                f"Moving {what} into {dst} was denied {denied} time(s) before it "
                "succeeded (likely a file scanner holding one of its files)."
            )
        return


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so a reader sees the old or the new content.

    A temporary file with a unique name in the same directory (so concurrent
    writers never share one), then :func:`move_into_place`, which retries a
    Windows ``PermissionError`` for a moment (a virus scanner reading the new
    file). A failed write removes the temporary file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        move_into_place(Path(tmp), path, what=path.name)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def remove_stale_temp_dirs(root: Path, prefix: str) -> None:
    """Delete ``<prefix>*`` temp dirs left by an install that was killed.

    Call with the tool's install lock held, so no live install owns one.
    """
    for stale in root.glob(f"{prefix}*"):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)


@contextlib.contextmanager
def stage_lock(
    path: Path, what: str, report: Callable[[float, str], None] | None
) -> Iterator[None]:
    """Hold the file lock ``path`` for a stage; wait as long as it takes.

    While another process holds it (an ``EvoSci setup``, the WebUI launcher,
    an MCP load), ``report`` gets a line every :data:`HEARTBEAT_INTERVAL`
    seconds naming ``what`` is being installed, so a reader can tell a wait
    from a hang. The holder may run pip for many minutes, so there is no limit.
    """
    from filelock import FileLock, Timeout

    lock = FileLock(str(path))
    while True:
        try:
            lock.acquire(timeout=HEARTBEAT_INTERVAL)
            break
        except Timeout:
            if report is not None:
                report(
                    0.01,
                    f"Waiting for another EvoScientist process to finish installing {what}",
                )
    try:
        yield
    finally:
        lock.release()


def tree_signature(root: Path) -> tuple[int, int]:
    """(file count, total bytes) under ``root``; changes while files are written.

    Files that vanish or cannot be read mid-walk are skipped: the value only
    has to differ while a step makes progress.
    """
    files = size = 0
    stack = [root]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                else:
                    files += 1
                    size += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return files, size


def kill_tree(pid: int) -> None:
    """Kill process ``pid`` and all its descendants; gone ones are skipped."""
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


# Windows errors for a program blocked by a code-integrity policy:
# ERROR_ACCESS_DISABLED_BY_POLICY (AppLocker) and
# ERROR_SYSTEM_INTEGRITY_POLICY_VIOLATION (WDAC, also Smart App Control).
POLICY_WINERRORS = frozenset({1260, 4551})


class PolicyBlocked(Exception):
    """A Windows policy refused to start ``path``."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(str(path))
        self.path = str(path)


def is_policy_block(exc: OSError) -> bool:
    return getattr(exc, "winerror", None) in POLICY_WINERRORS


def policy_error(path: str | Path, alternative: str) -> StageError:
    """The readable error for a program of ours that a Windows policy blocked.

    ``probe_failed``, and no new download: the same program would be blocked
    again. ``alternative`` completes "or ..., and run `EvoSci setup` again".
    """
    return StageError(
        "probe_failed",
        f"A Windows policy (AppLocker, WDAC or Smart App Control) blocked {path}. "
        f"Ask your administrator to allow programs under {tools_dir()}, or "
        f"{alternative} and run `EvoSci setup` again.",
    )
