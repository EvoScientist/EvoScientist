"""Environment overrides for the agent's shell commands.

Neutral home for :func:`agent_shell_env`, shared by every agent spawn site (the
``execute`` backend in :mod:`EvoScientist.EvoScientist` and
:mod:`EvoScientist.background` jobs). The bundled-interpreter layout it reads
lives in :mod:`EvoScientist.desktop.app_paths`, imported lazily so core modules
don't depend on the desktop package at import time.
"""

from __future__ import annotations

import os


def agent_shell_env() -> dict[str, str] | None:
    """Env overrides so the agent's shell commands use the bundled Python.

    Applied to every agent spawn site — the ``execute`` backend and
    ``run_in_background`` jobs — so both resolve the same interpreter. In the
    packaged Windows app a standalone CPython ships under ``runtime/python/``;
    prepend it to PATH so ``python``/``python3`` resolve to the bundled
    interpreter, not whatever is on the end-user's PATH. ``pip`` is NOT exposed
    on PATH (python-build-standalone puts ``pip.exe`` under a ``Scripts`` dir,
    which we do not add); agent code reaches it as ``python -m pip``.
    ``PYTHONUSERBASE`` points the user site at a per-user dir, and the bundle's
    own ``pip.ini`` (site config, read only by the bundled interpreter) makes
    its installs ``--user``, keeping them out of the pinned bundle tree
    (replaced wholesale on upgrade). A venv the agent creates has its own
    prefix, so it never sees that ``pip.ini`` and installs into itself.
    Returns ``None`` when no bundled Python is present (dev checkouts, Linux),
    leaving PATH untouched.
    """
    # Imported lazily: core modules (the backend, background jobs) call this, and
    # must not pull in the desktop package at import time.
    from .desktop import app_paths

    # Only the frozen desktop bundle (or an explicit override) ships a trusted
    # interpreter. In a dev checkout app_root() falls back to the cwd, so a
    # workspace-supplied runtime/python/ would be prepended to the agent shell's
    # PATH (execute runs shell=True) and could shadow real tools — don't trust it.
    if not (app_paths.is_frozen() or os.environ.get(app_paths.ENV_PYTHON_EXE)):
        return None
    py = app_paths.python_exe()
    if not py.exists():
        return None
    userbase = app_paths.user_pypackages_dir()
    userbase.mkdir(parents=True, exist_ok=True)
    existing_path = os.environ.get("PATH", "")
    py_dir = str(py.parent)
    return {
        "PATH": f"{py_dir}{os.pathsep}{existing_path}" if existing_path else py_dir,
        "PYTHONUSERBASE": str(userbase),
    }
