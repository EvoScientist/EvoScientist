"""Stream-transformer registration for the served main graph.

langgraph-api enables only the stream modes that a graph's registered
transformers declare (there is no default ``custom`` mode). It discovers extra
transformers from a module-level ``stream_transformers`` symbol on the graph's
source module (``langgraph_api.graph._register_stream_transformers_from_module``).
``main_graph`` re-exports :func:`stream_transformers` so the symbol is present on
that module; the logic lives here so it can be unit-tested without building the
full agent that importing ``main_graph`` triggers.
"""

from __future__ import annotations

from langgraph.stream.transformers import CustomTransformer


def stream_transformers() -> list[type[CustomTransformer]]:
    """Opt the served graph into the ``custom`` stream mode.

    Without this the middleware event mirror's ``get_stream_writer()`` writes on
    the ``custom`` channel (``middleware/events.py``) hit a no-op writer and
    never reach the client. The local path opts in explicitly by passing
    ``transformers=[..., CustomTransformer]`` to the v3 stream framework; the
    server path has no equivalent call site, so declare it here. Returns a fresh
    list per call, as langgraph-api's contract requires.
    """
    return [CustomTransformer]
