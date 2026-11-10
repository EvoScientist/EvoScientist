"""Sub-agent definitions (YAML, one file per agent).

Each ``<name>.yaml`` here describes one sub-agent in the form expected by
``EvoScientist.utils.load_subagents``. The directory is the canonical
single source of truth for sub-agent prompts, tools, skills, and metadata.

Optional ``async: true`` on a sub-agent's yaml routes it through
``langgraph dev`` as an AsyncSubAgent when ``config.enable_async_subagents``
is set; the server builds it with
``EvoScientist.subagents._factory.build_async_subagent_graph``.
(see ``EvoScientist/langgraph_dev/factories.py``)
"""
