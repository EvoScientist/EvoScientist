"""Install helpers shared by the stages that unpack a tool into ``<DATA_DIR>/tools``."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def tools_dir() -> Path:
    """``<DATA_DIR>/tools`` as an absolute path, read at call time so an
    overridden DATA_DIR applies.

    Absolute because the recorded path is put on ``PATH`` for children that run
    in other working dirs; ``EVOSCIENTIST_DATA_DIR`` may be relative.
    """
    from .. import paths

    return (paths.DATA_DIR / "tools").resolve()


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def prepend_to_path(directory: Path) -> None:
    """Put ``directory`` first on this process's ``PATH``, dropping any other
    entry for it, so child processes inherit it."""
    key = os.path.normcase(str(directory))
    parts = [
        p
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p and os.path.normcase(p) != key
    ]
    os.environ["PATH"] = os.pathsep.join([str(directory), *parts])


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
