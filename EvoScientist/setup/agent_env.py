"""The environment of the agent's ``execute`` and ``run_in_background`` commands.

:func:`agent_shell_overrides` starts from the research environment's
overrides (its ``python`` first, ``VIRTUAL_ENV``) and adds the Node rules:

- the private Node goes to the end of ``PATH`` when a usable Node of the
  user's own is on it, so a Node installed later (for example with nvm) wins
  inside the user's projects. The CLI's own children (``npx`` MCP servers,
  ``langgraph dev``) keep the private Node first, as
  :func:`~EvoScientist.setup.node.activate_runtime` set it, until the next
  ``ensure_node()`` (the WebUI launcher's, ``EvoSci setup``) finds the user's
  Node and drops the private record. With no usable Node of the user's, the
  private one stays first: the one that was there is the reason EvoScientist
  installed its own. It also stays first when ``PATH`` searches the working
  directory (an empty or relative entry), which would otherwise come before
  it. It then also gets ``NODE_USE_SYSTEM_CA=1``, as the
  private Node's other children do, so ``npx`` trusts a certificate that only
  the system store holds;
- under ``mirror: cn``, ``npx`` (``find-skills``) and ``npm`` use npmmirror
  unless the user configured a registry.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

from ._install import clean_path_entry, which_on_path


@functools.cache
def _user_node_usable() -> bool:
    """Whether a Node outside our tools dir passes the checks a system Node
    must pass to be used (``node.ensure_node``). Asked once per process: it
    runs ``node --version``."""
    from .node import _system_node

    return _system_node() is not None


def _private_node_dir() -> str | None:
    """The recorded private Node's folder, as :func:`activate_runtime` puts it
    on ``PATH`` (cleaned and case-folded), or None without a record."""
    from .node import _bin_dir, _read_record

    record = _read_record()
    if record is None:
        return None
    return os.path.normcase(clean_path_entry(str(_bin_dir(Path(record["path"])))))


def _node_dirs_last(path: str, ours_dir: str) -> str:
    """``path`` with the entries for ``ours_dir`` moved to the end. Entries are
    compared as written, not resolved: this runs for every background job, and
    resolving each entry is slow for a network path."""
    ours, others = [], []
    for entry in path.split(os.pathsep):
        mine = os.path.normcase(clean_path_entry(entry)) == ours_dir
        (ours if mine else others).append(entry)
    return os.pathsep.join(others + ours)


def _searches_the_current_directory(path: str) -> bool:
    """Whether ``path`` has an entry that resolves against the working
    directory: a relative one, or on POSIX an empty one (``a::b``). Moving the
    private Node behind it would let a ``node`` in the agent's workspace win.
    Windows ignores empty entries, and its ``PATH`` often ends with one."""
    for entry in path.split(os.pathsep):
        directory = clean_path_entry(entry)
        if directory and not os.path.isabs(directory):
            return True
        if not directory and os.name != "nt":
            return True
    return False


def agent_shell_overrides() -> dict[str, str] | None:
    """Environment overrides for the agent's shell, or None when there are none.

    Built from the current ``os.environ`` on each call; ``os.environ`` itself
    is not changed. Called where an agent is built and for each background job.
    """
    from .node import is_private, npm_registry_override
    from .research_env import research_env_overrides

    overrides = dict(research_env_overrides() or {})
    env = {**os.environ, **overrides}
    path = env.get("PATH", "")
    ours_dir = _private_node_dir()
    if ours_dir is not None and not _searches_the_current_directory(path):
        reordered = _node_dirs_last(path, ours_dir)
        if reordered != path and _user_node_usable():
            overrides["PATH"] = env["PATH"] = reordered
    node = which_on_path("node", path=env["PATH"]) if path else None
    if node is not None:
        registry = npm_registry_override(env, Path(node))
        if registry is not None:
            overrides["npm_config_registry"] = registry
        if is_private(node) and not any(k.upper() == "NODE_USE_SYSTEM_CA" for k in env):
            overrides["NODE_USE_SYSTEM_CA"] = "1"
    return overrides or None
