"""Deployed graphs for all yaml-flagged async sub-agents.

One module-level binding per ``async: true`` entry in
``EvoScientist/subagents/<name>.yaml``. Each binding is a graph compiled by
``build_async_subagent_graph`` (which reads the yaml, wires tools/skills/
backend/middleware identical to the in-process sync version, and returns a
runnable langgraph).

To add a new async sub-agent:

  1. Set ``async: true`` in ``EvoScientist/subagents/<name>.yaml``.
  2. Add a one-line binding here::

         <snake_name> = folder_checked(
             build_async_subagent_graph("<name>", **_session), works_in_folder=True
         )

  3. Register it in ``EvoScientist/langgraph_dev/langgraph.json``::

         "<name>": "EvoScientist.langgraph_dev.graphs:<snake_name>"

The deployed main agent (``EvoScientist_agent``) lives in ``main_graph.py``
because it follows a different mechanism (re-exporting a lazily-constructed
attribute), not the yaml-driven factory.
"""

from EvoScientist.langgraph_dev.folder_check import folder_checked
from EvoScientist.memory.agents import (
    build_autoskills_graph,
    build_memory_worker_graph,
    build_observation_linker_graph,
)
from EvoScientist.memory.types import MemorySourceType
from EvoScientist.paths import process_session_dirs
from EvoScientist.subagents._factory import build_async_subagent_graph
from EvoScientist.subagents.expert_container_async import (
    build_expert_container_async_graph,
)

# The server serves the workspace it was started for, and works in the run
# folder of a ``--mode=run`` session (the manager sets
# ``EVOSCIENTIST_WORKSPACE_DIR`` and ``EVOSCIENTIST_RUN_DIR`` on the
# subprocess). Memory graphs only need the workspace.
_dirs = process_session_dirs()
_workspace = _dirs.workspace
_session = {"workspace": _workspace, "work_dir": _dirs.work_dir}

# Graphs that work in the session's folder check a run's workspace and run
# folder; graphs built for the workspace root check the workspace only.
writing_agent = folder_checked(
    build_async_subagent_graph("writing-agent", **_session), works_in_folder=True
)
data_analysis_agent = folder_checked(
    build_async_subagent_graph("data-analysis-agent", **_session),
    works_in_folder=True,
)
# Scheduled tasks belong to the workspace, not to one run, so they always
# work in the workspace root.
scheduler = folder_checked(
    build_async_subagent_graph("scheduler", workspace=_workspace),
    works_in_folder=False,
)
# Generic async container for expert-skill dispatch. One graph, parameterised
# per invocation by the ``skill_name`` payload the main agent passes through
# ``EvoAsyncSubAgentMiddleware.start_async_task``. Any installed expert skill
# dispatches through this graph; the loader middleware resolves the skill
# body at model-call time.
expert_container_async = folder_checked(
    build_expert_container_async_graph(**_session), works_in_folder=True
)
evomemory_subagent_worker = folder_checked(
    build_memory_worker_graph(MemorySourceType.SUBAGENT, workspace=_workspace),
    works_in_folder=False,
)
evomemory_turn_worker = folder_checked(
    build_memory_worker_graph(MemorySourceType.TURN, workspace=_workspace),
    works_in_folder=False,
)
evomemory_observation_linker = folder_checked(
    build_observation_linker_graph(workspace=_workspace), works_in_folder=False
)
evomemory_autoskills = folder_checked(
    build_autoskills_graph(workspace=_workspace), works_in_folder=False
)
