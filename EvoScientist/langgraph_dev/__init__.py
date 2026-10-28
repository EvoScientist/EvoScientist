"""``langgraph dev`` deployment surface.

Holds everything needed to run the EvoScientist agent ecosystem on a local
``langgraph dev`` subprocess:

- ``manager`` — subprocess lifecycle (auto-start, port management, cleanup).
- ``langgraph.json`` — graph manifest consumed by ``langgraph dev --config``.
- ``factories`` — builds each served graph per project and run folder, and
  caches it.
- ``main_graph`` — the main agent's factory and its stream transformers.
- ``graphs`` — the factories of the other served graphs.

The graphs themselves are built from the canonical definitions (the main
agent, the yaml-defined sub-agents, the memory and proactive graphs), so this
package only owns the *deployment* concern. Adding a new async sub-agent is
described in ``graphs``.

``langgraph.json`` turns off langgraph-api's ``/mcp`` endpoint: listing its
tools reads the schema of every assistant, which builds a graph per workspace
assistant, and it would offer every internal graph (the memory workers
included) as a tool.
"""

# ``langgraph dev`` imports this package (to resolve the graphs listed in
# ``langgraph.json``) inside its own subprocess, before it creates the event
# loop that serves runs. Forcing the Proactor loop policy here — at import,
# ahead of loop creation — keeps MCP stdio subprocess spawning async on
# Windows so it isn't flagged as a blocking call by the dev runtime's
# ``blockbuster`` guard (see #283). This is the entrypoint where that error
# actually surfaces; the CLI entrypoint sets the same policy for its own
# process.
from .._winloop import ensure_proactor_event_loop_policy

ensure_proactor_event_loop_policy()
