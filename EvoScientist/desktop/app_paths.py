"""Resolve the on-disk layout of the bundled desktop app.

The Windows installer produces a self-contained app directory. This module is
the single place that knows its shape, so the shell and the packaging scripts
agree. Expected installed layout::

    <app_root>/
      EvoScientist.exe            # windowed desktop shell (PyInstaller onedir)
      langgraph.exe              # bundled langgraph CLI, found by manager._langgraph_exe
      _internal/...              # bundled Python + all deps (PyInstaller)
      runtime/
        node/node.exe            # bundled Node runtime (bare binary)
      webui/
        dist/server.js           # pinned prebuilt @evoscientist/webui standalone
        dist/.next/ ...

``app_root`` is derived from the frozen executable when packaged, and from the
current directory (or an env override) in a dev checkout. Every path is
overridable by env var so the shell can be exercised on a dev machine that has
no bundle — point the overrides at a locally-extracted webui and a system node.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_APP_ROOT = "EVOSCIENTIST_DESKTOP_APP_ROOT"
ENV_WEBUI_DIR = "EVOSCIENTIST_DESKTOP_WEBUI_DIR"
ENV_NODE_EXE = "EVOSCIENTIST_DESKTOP_NODE_EXE"
ENV_PYTHON_EXE = "EVOSCIENTIST_DESKTOP_PYTHON_EXE"


def is_frozen() -> bool:
    """True when running from a PyInstaller/py2exe-style frozen build."""
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """Root of the installed app directory.

    Env override (:data:`ENV_APP_ROOT`) wins; then the frozen executable's
    directory; then the current working directory for a dev checkout.
    """
    override = os.environ.get(ENV_APP_ROOT)
    if override:
        return Path(override)
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def webui_dir() -> Path:
    """Directory holding the prebuilt WebUI (expects ``dist/server.js`` under it)."""
    override = os.environ.get(ENV_WEBUI_DIR)
    return Path(override) if override else app_root() / "webui"


def node_exe() -> Path:
    """Path to the bundled Node binary."""
    override = os.environ.get(ENV_NODE_EXE)
    if override:
        return Path(override)
    name = "node.exe" if os.name == "nt" else "node"
    return app_root() / "runtime" / "node" / name


def python_exe() -> Path:
    """Path to the bundled Python interpreter the agent's shell runs code with.

    This is a standalone CPython shipped under ``runtime/python/`` (see
    ``packaging/windows/assemble_bundle.py``), used so the agent never depends
    on whatever python happens to be on the end-user's PATH.
    """
    override = os.environ.get(ENV_PYTHON_EXE)
    if override:
        return Path(override)
    name = "python.exe" if os.name == "nt" else "python3"
    return app_root() / "runtime" / "python" / name


def user_pypackages_dir() -> Path:
    """Writable per-user dir for on-demand ``pip install``s by the agent.

    The bundled interpreter lives inside the app bundle, which is replaced
    wholesale on upgrade; routing runtime installs here via ``PYTHONUSERBASE``
    keeps them out of that tree. Lives under the existing per-user data root
    (``~/.evoscientist``).
    """
    from .. import paths

    return paths.DATA_DIR / "pypackages"


def desktop_log_path() -> Path:
    """Diagnostic log for the desktop shell (boot/shutdown/errors).

    Lives next to the backend's ``langgraph_dev.log`` in the config dir so the
    three app logs (desktop / webui / backend) are discoverable together.
    """
    from ..config.settings import get_config_dir

    return get_config_dir() / "desktop.log"


def webui_log_path() -> Path:
    """Captured stdout/stderr of the bundled WebUI (node) process.

    Node output is otherwise discarded on the windowed desktop app (no console),
    leaving a front-end failure with no diagnostic trace.
    """
    from ..config.settings import get_config_dir

    return get_config_dir() / "webui.log"
