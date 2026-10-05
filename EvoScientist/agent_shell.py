"""The shell that runs the agent's ``execute`` and ``run_in_background`` commands.

On macOS and Linux nothing changes: ``subprocess`` runs the command through
``/bin/sh`` (``shell=True``).

On Windows the prompts, skills and command checks all assume a POSIX shell, so
the command runs in the Git for Windows bash that ``EvoSci setup`` recorded in
``tools/git.json`` (its ``bin\\bash.exe``, which sets ``MSYSTEM``, ``HOME`` and
``PATH`` before starting ``usr\\bin\\bash.exe``). Without a recorded bash the
command keeps running in ``cmd.exe``. The choice is made once per process, so
the agent's prompt and its shell cannot disagree after a later ``EvoSci setup``.

The command reaches bash as a script file rather than through ``-c``:
``bin\\bash.exe`` hands its command line to the MSYS2 runtime, which splits it
with Cygwin's rules, not the MSVCRT rules ``subprocess`` quotes for. Through
``-c``, ``\\\\`` turns into ``\\``, a command without a space is split at
newlines and globbed, and an argument over about 8 KB is cut off without an
error.
"""

from __future__ import annotations

import functools
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .setup.git import GitInfo

logger = logging.getLogger(__name__)

_SCRIPT_PREFIX = "cmd-"
_SCRIPT_SUFFIX = ".sh"
# A script is deleted when its process ends; one left by a crash is removed by
# a later process once it is this old.
_STALE_SCRIPT_SECONDS = 24 * 60 * 60


@functools.cache
def agent_bash() -> GitInfo | None:
    """The recorded Git for Windows whose bash runs the agent's commands.

    None on macOS and Linux, and on Windows when ``EvoSci setup`` has not
    recorded a Git. Decided once per process and logged at INFO.
    """
    from .setup.git import recorded_git

    info = recorded_git()
    if info is not None:
        logger.info(f"Agent shell: {info.bash}")
    elif os.name == "nt":
        logger.info("Agent shell: cmd.exe (no Git for Windows recorded)")
    return info


def uses_bash() -> bool:
    """True when the agent's commands run in the recorded Git Bash (Windows)."""
    return agent_bash() is not None


def _script_dir() -> Path:
    return Path(tempfile.gettempdir()) / "evoscientist"


@functools.cache
def _sweep_stale_scripts() -> None:
    """Remove scripts left by a process that ended without deleting its own."""
    cutoff = time.time() - _STALE_SCRIPT_SECONDS
    try:
        scripts = list(_script_dir().glob(f"{_SCRIPT_PREFIX}*{_SCRIPT_SUFFIX}"))
    except OSError:
        return
    for script in scripts:
        try:
            if script.stat().st_mtime < cutoff:
                script.unlink()
        except OSError:
            pass


def _write_script(command: str) -> Path:
    _sweep_stale_scripts()
    directory = _script_dir()
    directory.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=_SCRIPT_PREFIX, suffix=_SCRIPT_SUFFIX, dir=directory
    )
    with os.fdopen(fd, "wb") as fh:
        fh.write(command.encode("utf-8"))
    return Path(name)


def _agent_gitconfig_text(root: Path) -> str:
    # Quoted: a path may contain `#` or `;`, which start a comment otherwise.
    # Forward slashes, so nothing in a Windows path needs escaping.
    shipped = (root / "etc" / "gitconfig").as_posix()
    return (
        "# Written by EvoScientist for the agent's shell (GIT_CONFIG_SYSTEM).\n"
        "[include]\n"
        f'\tpath = "{shipped}"\n'
        "[core]\n"
        "\tautocrlf = input\n"
        "[credential]\n"
        "\thelper =\n"
        "\thelper = manager\n"
    )


@functools.cache
def _agent_gitconfig(root: Path) -> Path | None:
    """The git config file for the agent's shell when it runs PortableGit.

    PortableGit's ``etc\\gitconfig`` sets ``core.autocrlf = true``, which
    checks out scripts with CRLF line endings that bash cannot run, and
    ``credential.helper = helper-selector``, a picker window that blocks the
    command until its timeout. This file includes that one and overrides the
    two; it is a system-level file, so the user's ``~/.gitconfig`` and the
    repository's config still win. Written once per process; None (and a
    WARNING) when it cannot be written.
    """
    from .setup._install import tools_dir

    path = tools_dir() / "git-agent.gitconfig"
    text = _agent_gitconfig_text(root)
    try:
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        logger.warning(
            f"Could not write {path}: {exc}. The agent's git uses PortableGit's "
            "own settings (CRLF checkouts, credential picker)."
        )
        return None
    return path


def _bash_env(env: dict[str, str] | None, info: GitInfo) -> dict[str, str]:
    """The bash child's environment: ``env`` (or ours) plus what bash needs.

    ``MSYS_NO_PATHCONV`` stops bash from rewriting arguments that look like
    POSIX paths (``"/hi"`` would reach a native program as
    ``C:/Program Files/Git/hi``). ``PYTHONIOENCODING``: a native ``python``
    writes to a pipe in the ANSI code page, while MSYS tools write UTF-8, and
    the output is decoded as UTF-8; a value the user set is kept. For
    PortableGit, ``GIT_CONFIG_SYSTEM`` points the agent's ``git`` at
    :func:`_agent_gitconfig`, unless the user set it (or
    ``GIT_CONFIG_NOSYSTEM``) themselves; a system Git for Windows is left as
    the user configured it.
    """
    child = dict(os.environ if env is None else env)
    keys = {key.upper() for key in child}
    child["MSYS_NO_PATHCONV"] = "1"
    if "PYTHONIOENCODING" not in keys:
        child["PYTHONIOENCODING"] = "utf-8"
    if info.source == "portablegit" and not keys & {
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_NOSYSTEM",
    }:
        gitconfig = _agent_gitconfig(info.git.parent.parent)
        if gitconfig is not None:
            child["GIT_CONFIG_SYSTEM"] = str(gitconfig)
    return child


@dataclass
class ShellLaunch:
    """How to start one agent command with :class:`subprocess.Popen`.

    Pass ``args``, ``shell`` and ``env`` to ``Popen``, OR :attr:`creationflags`
    into its flags, decode piped output with :attr:`text_options`, and call
    :meth:`cleanup` once the process has ended.
    """

    args: str | list[str]
    env: dict[str, str] | None
    script: Path | None = None

    @property
    def bash(self) -> bool:
        return self.script is not None

    @property
    def shell(self) -> bool:
        return not self.bash

    @property
    def creationflags(self) -> int:
        # ``shell=True`` hides cmd.exe's window itself (SW_HIDE); a plain argv
        # does not, and bash.exe is a console program.
        return getattr(subprocess, "CREATE_NO_WINDOW", 0) if self.bash else 0

    @property
    def text_options(self) -> dict[str, str]:
        """``Popen`` decoding options for piped output: UTF-8 under bash, else
        the locale default as before."""
        return {"encoding": "utf-8", "errors": "replace"} if self.bash else {}

    def cleanup(self) -> None:
        """Delete the script file, if any. Safe to call more than once."""
        if self.script is None:
            return
        try:
            self.script.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug(f"Could not delete {self.script}: {exc}")


def prepare(command: str, env: dict[str, str] | None) -> ShellLaunch:
    """How to run ``command`` in the agent's shell, with ``env`` as its environment.

    ``env`` None means "inherit ours", as for ``Popen``.
    """
    info = agent_bash()
    if info is None:
        return ShellLaunch(args=command, env=env)
    script = _write_script(command)
    # Forward slashes: the MSYS2 runtime would turn a ``\\`` into ``\``, and
    # the first argument must not look like one of bin\bash.exe's own options.
    return ShellLaunch(
        args=[str(info.bash), script.as_posix()],
        env=_bash_env(env, info),
        script=script,
    )
