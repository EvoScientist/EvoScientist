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
        python/python.exe        # standalone CPython for the agent's shell
      webui/
        dist/server.js           # prebuilt @evoscientist/webui standalone
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
    # Only the frozen desktop bundle (or an explicit override) ships a trusted
    # interpreter. In a dev checkout app_root() falls back to the cwd, so a
    # workspace-supplied runtime/python/ would be prepended to the agent shell's
    # PATH (execute runs shell=True) and could shadow real tools — don't trust it.
    if not (is_frozen() or os.environ.get(ENV_PYTHON_EXE)):
        return None
    py = python_exe()
    if not py.exists():
        return None
    userbase = user_pypackages_dir()
    userbase.mkdir(parents=True, exist_ok=True)
    existing_path = os.environ.get("PATH", "")
    py_dir = str(py.parent)
    return {
        "PATH": f"{py_dir}{os.pathsep}{existing_path}" if existing_path else py_dir,
        "PYTHONUSERBASE": str(userbase),
    }


def default_workspace() -> Path:
    """Default agent workspace for the desktop when the user picks none.

    A dedicated folder under the user's Documents, never Documents itself or the
    process cwd: the installer launches the app with its working directory set
    to Documents, so a cwd fallback would make the whole Documents tree the
    agent's root — its file tools, its ``execute`` shell, and the ``runs/`` /
    ``skills/`` / ``.langgraph_api`` / ``.bg_processes`` state it writes. A named
    subfolder keeps all of that in one place the user can find and delete.

    "Documents" is the real Documents known folder on Windows, which OneDrive
    folder backup redirects (e.g. ``%USERPROFILE%\\OneDrive\\Documents``) —
    the same folder Explorer shows and the installer's ``{userdocs}`` resolves.
    ``~/Documents`` is only the fallback (non-Windows, or the lookup failed).
    """
    docs = _windows_documents_dir() or Path(os.path.expanduser("~")) / "Documents"
    return docs / "EvoScientist"


def _windows_documents_dir() -> Path | None:
    """The Documents known folder via ``SHGetKnownFolderPath``, or None.

    Follows OneDrive Known Folder Move and any other redirection. Returns None
    off Windows or if the lookup fails, so the caller falls back to
    ``~/Documents``.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        # FOLDERID_Documents = {FDD39AD0-238F-46AF-ADB4-6C85480369C7}
        folder_id = _GUID(
            0xFDD39AD0,
            0x238F,
            0x46AF,
            (ctypes.c_ubyte * 8)(0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7),
        )
        path_ptr = ctypes.c_wchar_p()
        hr = ctypes.windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(folder_id), 0, None, ctypes.byref(path_ptr)
        )
        try:
            if hr != 0 or not path_ptr.value:
                return None
            return Path(path_ptr.value)
        finally:
            # Per the API contract the caller frees the string whether or not
            # the call succeeded.
            ctypes.windll.ole32.CoTaskMemFree(path_ptr)
    except Exception:
        return None


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
