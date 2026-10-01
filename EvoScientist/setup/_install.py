"""Install helpers shared by the stages that unpack a tool into ``<DATA_DIR>/tools``."""

from __future__ import annotations

import logging
import os
import shutil
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


def remove_stale_temp_dirs(root: Path, prefix: str) -> None:
    """Delete ``<prefix>*`` temp dirs left by an install that was killed.

    Call with the tool's install lock held, so no live install owns one.
    """
    for stale in root.glob(f"{prefix}*"):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
