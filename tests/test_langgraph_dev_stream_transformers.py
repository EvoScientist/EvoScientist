"""The served main graph opts into the ``custom`` stream mode.

langgraph-api enables only the stream modes that a graph's registered
transformers declare. The middleware event mirror
(``EvoScientist/middleware/events.py``) writes on the ``custom`` channel via
``get_stream_writer()``; without ``CustomTransformer`` registered for the served
graph those writes hit a no-op writer and never reach the client.

The factory lives in ``langgraph_dev.stream_transformers`` (re-exported by
``main_graph``) so it can be exercised here without building the full agent that
importing ``main_graph`` triggers. ``main_graph`` re-exports the symbol so
langgraph-api discovers it on the graph's source module.
"""

from __future__ import annotations

from langgraph.stream.transformers import CustomTransformer

from EvoScientist.langgraph_dev.stream_transformers import stream_transformers


def test_stream_transformers_returns_a_fresh_custom_factory_list():
    first = stream_transformers()
    assert first == [CustomTransformer]
    # A fresh list per call, as langgraph-api's contract requires.
    assert stream_transformers() == [CustomTransformer]
    assert stream_transformers() is not first


def test_custom_transformer_declares_the_custom_stream_mode():
    (transformer,) = stream_transformers()
    # The ``custom`` mode is what carries the middleware-event mirror to clients.
    assert "custom" in transformer.required_stream_modes


def test_langgraph_api_registration_stores_the_factory():
    """The factory satisfies langgraph-api's registration contract.

    Uses a stand-in module namespace so the registration path is covered
    without importing ``langgraph_api.graph`` (which needs a server config) or
    ``main_graph`` (which builds the agent). ``main_graph`` re-exports the same
    symbol, so this is what langgraph-api reads at deploy time.
    """
    from types import SimpleNamespace

    module = SimpleNamespace(stream_transformers=stream_transformers)

    factory = getattr(module, "stream_transformers", None)
    assert callable(factory)
    assert [t.required_stream_modes for t in factory()] == [("custom",)]
