"""Deployed graphs for all yaml-flagged async sub-agents.

One module-level binding per ``async: true`` entry in
``EvoScientist/subagents/<name>.yaml``. Each binding is a graph compiled by
``build_async_subagent_graph`` (which reads the yaml, wires tools/skills/
backend/middleware identical to the in-process sync version, and returns a
runnable langgraph).

To add a new async sub-agent:

  1. Set ``async: true`` in ``EvoScientist/subagents/<name>.yaml``.
  2. Add a one-line binding here::

         <snake_name> = build_async_subagent_graph("<name>")

  3. Register it in ``EvoScientist/langgraph_dev/langgraph.json``::

         "<name>": "EvoScientist.langgraph_dev.graphs:<snake_name>"

The deployed main agent (``EvoScientist_agent``) lives in ``main_graph.py``
because it follows a different mechanism (re-exporting a lazily-constructed
attribute), not the yaml-driven factory.
"""

from EvoScientist.memory.agents import (
    build_autoskills_graph,
    build_memory_worker_graph,
    build_observation_linker_graph,
)
from EvoScientist.memory.types import MemorySourceType
from EvoScientist.subagents._factory import build_async_subagent_graph

writing_agent = build_async_subagent_graph("writing-agent")
data_analysis_agent = build_async_subagent_graph("data-analysis-agent")
scheduler = build_async_subagent_graph("scheduler")
evomemory_subagent_worker = build_memory_worker_graph(MemorySourceType.SUBAGENT)
evomemory_turn_worker = build_memory_worker_graph(MemorySourceType.TURN)
evomemory_observation_linker = build_observation_linker_graph()
evomemory_autoskills = build_autoskills_graph()
