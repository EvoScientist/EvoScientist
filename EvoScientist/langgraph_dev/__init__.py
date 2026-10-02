"""``langgraph dev`` deployment surface.

Holds everything needed to run the EvoScientist agent ecosystem on a local
``langgraph dev`` subprocess:

- ``manager`` — subprocess lifecycle (auto-start, port management, cleanup).
- ``langgraph.json`` — graph manifest consumed by ``langgraph dev --config``.
- ``main_graph`` — re-export of the lazy-loaded ``EvoScientist_agent``.
- ``graphs`` — module-level bindings for every yaml-flagged async sub-agent
  (``async: true`` in ``EvoScientist/subagents/<name>.yaml``).

The graphs themselves are built by ``EvoScientist.subagents._factory.
build_async_subagent_graph`` from the canonical yaml definitions, so this
package only owns the *deployment* concern. Adding a new async sub-agent
takes three steps: flip the yaml flag, add a one-line binding in
``graphs.py``, and register it in ``langgraph.json``.
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
