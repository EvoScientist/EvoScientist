"""Factories for the served graphs other than the main agent.

One module-level binding per graph registered in ``langgraph.json``; each is
the factory langgraph-api calls on every access, which returns the graph
built for the access's folders (see ``factories``). The builders live in
``factories.BUILDERS``.

To add a new async sub-agent:

  1. Set ``async: true`` in ``EvoScientist/subagents/<name>.yaml``.
  2. Add its builder to ``factories.BUILDERS`` and a binding here::

         <snake_name> = served_graph("<name>")

  3. Register it in ``EvoScientist/langgraph_dev/langgraph.json``::

         "<name>": "EvoScientist.langgraph_dev.graphs:<snake_name>"

  4. If it works in a run's folder (a ``--mode=run`` session's), not only
     in the project root, add ``"<name>"`` to
     ``EvoScientist.sessions.FOLDER_GRAPH_IDS``.

The main agent's factory lives in ``main_graph.py`` next to its stream
transformers.
"""

from EvoScientist.langgraph_dev.factories import served_graph

writing_agent = served_graph("writing-agent")
data_analysis_agent = served_graph("data-analysis-agent")
# Scheduled tasks belong to the workspace, not to one run, so they always work
# in the workspace root.
scheduler = served_graph("scheduler")
# Generic async container for expert-skill dispatch: any installed expert
# skill dispatches through this graph.
expert_container_async = served_graph("expert-container-async")
evomemory_subagent_worker = served_graph("evomemory-subagent-worker")
evomemory_turn_worker = served_graph("evomemory-turn-worker")
evomemory_observation_linker = served_graph("evomemory-observation-linker")
evomemory_autoskills = served_graph("evomemory-autoskills")
# Proactive pushes (#263): one tick of the workspace's proactive cron.
proactive = served_graph("proactive")
