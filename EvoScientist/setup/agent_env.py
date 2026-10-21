"""The environment of the agent's ``execute`` and ``run_in_background`` commands.

:func:`agent_shell_overrides` starts from the research environment's
overrides (its ``python`` first, ``VIRTUAL_ENV``) and adds the Node rules:

- the private Node goes to the end of ``PATH`` when a usable Node of the
  user's own is on it, so a Node installed later (for example with nvm) wins
  inside the user's projects. EvoScientist's own children (the WebUI, ``npx``
  MCP servers, ``langgraph dev``) keep the private Node first, as
  :func:`~EvoScientist.setup.node.activate_runtime` set it. With no usable
  Node of the user's, the private one stays first: the one that was there is
  the reason EvoScientist installed its own;
- under ``mirror: cn``, ``npx`` (``find-skills``) and ``npm`` use npmmirror
  unless the user configured a registry.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

from ._install import clean_path_entry, is_under, tools_dir, which_in, which_on_path


@functools.cache
def _user_node_usable() -> bool:
    """Whether a Node outside our tools dir passes the checks a system Node
    must pass to be used (``node.ensure_node``). Asked once per process: it
    runs ``node --version``."""
    from .node import _system_node

    return _system_node() is not None


def _node_dirs_last(path: str) -> str:
    """``path`` with the entries that hold a Node under our tools dir moved to
    the end, in their order."""
    root = tools_dir()
    ours, others = [], []
    for entry in path.split(os.pathsep):
        directory = clean_path_entry(entry)
        mine = (
            bool(directory)
            and os.path.isabs(directory)
            and is_under(Path(directory), root)
            and which_in("node", directory) is not None
        )
        (ours if mine else others).append(entry)
    return os.pathsep.join(others + ours)


def agent_shell_overrides() -> dict[str, str] | None:
    """Environment overrides for the agent's shell, or None when there are none.

    Built from the current ``os.environ`` on each call; ``os.environ`` itself
    is not changed. Called where an agent is built and for each background job.
    """
    from .node import npm_registry_override
    from .research_env import research_env_overrides

    overrides = dict(research_env_overrides() or {})
    env = {**os.environ, **overrides}
    path = env.get("PATH", "")
    reordered = _node_dirs_last(path)
    if reordered != path and _user_node_usable():
        overrides["PATH"] = env["PATH"] = reordered
    node = which_on_path("node", path=env["PATH"]) if path else None
    if node is not None:
        registry = npm_registry_override(env, Path(node))
        if registry is not None:
            overrides["npm_config_registry"] = registry
    return overrides or None
