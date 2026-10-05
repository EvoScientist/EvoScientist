"""Run git for EvoScientist's own operations: skill installs and the MCP index.

Every git command for skills and the MCP index goes through :func:`run_git`,
so a missing git binary surfaces as one readable :class:`GitNotFoundError` on
every path instead of a raw ``FileNotFoundError``, and no command waits for
HTTPS credentials.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

CLONE_TIMEOUT = 120  # seconds

# A missing or private repository makes GitHub ask for credentials. Git would
# then open a credential helper window or wait for a username on the terminal.
# Terminal and askpass prompts are turned off (askpass programs run before git
# checks GIT_TERMINAL_PROMPT, so they are cleared too) and Git Credential
# Manager is told not to prompt, so a stored credential is still used (the
# user's own helpers, e.g. the one ``gh auth setup-git`` writes) and anything
# else fails at once. Only our own PortableGit, whose ``etc\gitconfig`` names
# the ``helper-selector`` picker, gets its helper list reset. This covers
# HTTPS, the only scheme our URLs use. A user's ``url.<ssh>.insteadOf`` rule
# turns them into SSH, whose own prompts (key passphrase, host key) these
# settings do not reach; ``core.sshCommand`` is left alone so the user's ssh
# setup keeps working.
_NON_INTERACTIVE = ["-c", "core.askPass="]
_PORTABLEGIT_ONLY = ["-c", "credential.helper="]
_ASKPASS_VARS = ("GIT_ASKPASS", "SSH_ASKPASS")


def _install_hint() -> str:
    if sys.platform == "win32":
        return (
            "install Git from https://git-scm.com/downloads, or run "
            "`EvoSci setup`, and try again"
        )
    return "install Git from https://git-scm.com/downloads and try again"


def _not_found_message() -> str:
    return (
        "git was not found on PATH. EvoScientist uses git to download "
        f"skills and the MCP server index; {_install_hint()}."
    )


class GitNotFoundError(RuntimeError):
    """git could not be started: not installed, not on PATH, or not runnable.

    A ``RuntimeError`` so the callers that already turn ``RuntimeError`` into a
    clean error result need no new ``except``. The message is a plain argument
    so the exception still pickles and copies.
    """

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or _not_found_message())


def _git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.upper() not in _ASKPASS_VARS}
    env["GIT_TERMINAL_PROMPT"] = "0"
    # Git Credential Manager answers from its store or fails, never prompts
    # (``0`` / ``false`` are the current spellings, ``never`` the legacy one).
    env["GCM_INTERACTIVE"] = "0"
    return env


def _portablegit_args() -> list[str]:
    """``-c credential.helper=`` when the git on PATH is our PortableGit."""
    if sys.platform != "win32":
        return []
    from .setup import git as setup_git

    return list(_PORTABLEGIT_ONLY) if setup_git.private_git_on_path() else []


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *_NON_INTERACTIVE, *_portablegit_args(), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_git_env(),
    )


def _activate_private_git() -> bool:
    """On Windows, put a PortableGit recorded since start-up on ``PATH``.

    The missing-git message tells Windows users to run ``EvoSci setup``; when
    they do that in another terminal, this process only learns about the new
    Git here. Returns True when a recorded PortableGit was put on ``PATH``.
    """
    if sys.platform != "win32":
        return False
    from .setup import git as setup_git

    return setup_git.activate_runtime() is not None


def run_git(args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` non-interactively and capture its text output.

    Raises :class:`GitNotFoundError` when git cannot be started (on Windows
    after one retry with a PortableGit recorded since start-up).
    ``subprocess.TimeoutExpired`` propagates; a non-zero exit is returned for
    the caller to judge.
    """
    try:
        return _run(args, timeout)
    except OSError as exc:
        logger.debug("could not start git", exc_info=True)
        missing = isinstance(exc, FileNotFoundError)
        if missing and _activate_private_git():
            try:
                return _run(args, timeout)
            except OSError:
                logger.debug("could not start the private git either", exc_info=True)
        # Only a missing binary gets the install hint; anything else (EACCES,
        # EMFILE, a Windows policy block) names its cause, because git is there.
        if missing:
            raise GitNotFoundError() from exc
        raise GitNotFoundError(
            f"git could not be started: {exc.strerror or exc}"
        ) from exc


def clone_repo(repo: str, ref: str | None, dest: str) -> None:
    """Shallow-clone ``github.com/<repo>`` (at ``ref`` if given) into ``dest``.

    Files keep the repository's line endings (``core.autocrlf=false``): CRLF
    breaks shell scripts in skills. Raises :class:`GitNotFoundError` without
    git, and ``RuntimeError`` on a timeout or a failed clone.
    """
    args = ["-c", "core.autocrlf=false", "clone", "--depth", "1"]
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
