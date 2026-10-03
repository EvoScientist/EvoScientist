"""Run git for EvoScientist's own operations: skill installs and the MCP index.

Every git command for skills and the MCP index goes through :func:`run_git`,
so a missing git binary surfaces as one readable :class:`GitNotFoundError` on
every path instead of a raw ``FileNotFoundError``.
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

CLONE_TIMEOUT = 120  # seconds


def _install_hint() -> str:
    return "install Git from https://git-scm.com/downloads and try again"


class GitNotFoundError(RuntimeError):
    """git could not be started: not installed, not on PATH, or not runnable.

    A ``RuntimeError`` so the callers that already turn ``RuntimeError`` into a
    clean error result need no new ``except``.
    """

    def __init__(self) -> None:
        super().__init__(
            "git was not found on PATH. EvoScientist uses git to download "
            f"skills and the MCP server index; {_install_hint()}."
        )


def run_git(args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` and capture its text output.

    Raises :class:`GitNotFoundError` when git cannot be started.
    ``subprocess.TimeoutExpired`` propagates; a non-zero exit is returned for
    the caller to judge.
    """
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=timeout
        )
    except OSError as exc:
        # The user sees only the readable message; keep the real errno
        # (e.g. EACCES, EMFILE) in the log.
        logger.debug("could not start git", exc_info=True)
        raise GitNotFoundError() from exc


def clone_repo(repo: str, ref: str | None, dest: str) -> None:
    """Shallow-clone ``github.com/<repo>`` (at ``ref`` if given) into ``dest``.

    Raises :class:`GitNotFoundError` without git, and ``RuntimeError`` on a
    timeout or a failed clone.
    """
    args = ["clone", "--depth", "1"]
    if ref:
        args += ["--branch", ref]
    args += [f"https://github.com/{repo}.git", dest]
    try:
        result = run_git(args, timeout=CLONE_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"git clone timed out after {CLONE_TIMEOUT}s for {repo}"
        ) from e
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
