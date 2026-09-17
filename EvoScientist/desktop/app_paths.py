"""Resolve the on-disk layout of the bundled desktop app.

The Windows installer produces a self-contained app directory. This module is
the single place that knows its shape, so the shell and the packaging scripts
agree. Expected installed layout::

    <app_root>/
      EvoScientist.exe            # windowed launcher -> python -m EvoScientist.desktop
      runtime/
        python/...               # bundled Python + the EvoScientist package
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
