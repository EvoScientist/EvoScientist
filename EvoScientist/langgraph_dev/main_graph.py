"""Deployed graph entry for the main EvoScientist agent.

Registered in ``langgraph.json`` as ``make_EvoScientist_agent`` — a
zero-arg **graph factory**, not a compiled graph. langgraph-api collects a
compiled graph into its ``GRAPHS`` registry once at startup and never
re-reads the source module, but it calls a factory on every request
(``langgraph_api.graph.get_graph`` → ``invoke_factory``). Since the deploy
subprocess is launched ``--no-reload``, the factory is the only way an
expert installed mid-session becomes dispatchable there without restarting
the server: see ``EvoScientist.EvoScientist._get_default_agent``, which
rebuilds itself when the expert registry changes and which the factory
simply forwards to.

The agent itself is exposed via ``__getattr__`` lazy loading in
``EvoScientist/EvoScientist.py`` so it doesn't construct on plain ``import
EvoScientist``; the factory triggers that load.

Before handing the graph over we upgrade its class in place to
``_EvoFilteredGraph``, which strips ``PrivateStateAttr``-marked fields
(currently just ``_quickjs_snapshot_payload``) from ``get_state`` /
``get_state_history`` responses. Upstream ``langchain_quickjs`` annotates
the field ``PrivateStateAttr = OmitFromSchema(input=True, output=True)``,
but LangGraph's ``_prepare_state_snapshot`` doesn't honor that on
checkpoint reads — every ``getState`` materializes the delta chain back
into a full ~1.4 MB blob, which the WebUI then downloads. The subclass
closes the gap without touching the middleware's write path, preserving
cross-turn REPL persistence as ``langchain-ai/deepagents#3064`` shipped it.
"""

import logging
import time

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import PregelTask, StateSnapshot

_logger = logging.getLogger(__name__)

_PRIVATE_STATE_FIELDS = frozenset({"_quickjs_snapshot_payload"})

# Sanity check on the LangGraph internals ``_strip_private`` scrubs. If any
# of these attributes disappear or get renamed in a future upstream bump,
# the assertion fires at import time — the deployment refuses to start,
# instead of silently degrading (the filter would ``.get()`` its way to a
# no-op and the private-field payload would come back on the wire without
# anyone noticing until a user reports slow thread switches again).
#
# Doesn't cover every internal we depend on — ``metadata["writes"]`` /
# ``metadata["counters_since_delta_snapshot"]`` dict keys aren't a canary
# target because ``dict.get`` already tolerates their absence. What we
# canary here is the ``NamedTuple`` field set: renames there would be the
# highest-impact silent regression.
_EXPECTED_SNAPSHOT_FIELDS = frozenset({"values", "metadata", "tasks"})
_EXPECTED_TASK_FIELDS = frozenset({"result", "state"})

_missing_snap = _EXPECTED_SNAPSHOT_FIELDS - set(StateSnapshot._fields)
_missing_task = _EXPECTED_TASK_FIELDS - set(PregelTask._fields)
if _missing_snap or _missing_task:
    raise RuntimeError(
        "LangGraph state shape drifted from the version _strip_private was "
        f"written against. Missing StateSnapshot fields: {_missing_snap or set()}. "
        f"Missing PregelTask fields: {_missing_task or set()}. Review "
        "_strip_private and re-verify against the current upstream shape "
        "before removing this assertion."
    )


def _strip_private(snap):
    """Strip ``PrivateStateAttr``-marked fields from a ``StateSnapshot``.

    Empirically verified against a live history response for a thread with
    a single touched turn: the private field leaks on four surfaces — three
    trivial, one heavy:

    * ``snap.values`` — the materialized channel state exposed as the main
      payload. For DeltaChannels this is the delta chain replayed into full
      bytes (~1.4 MB for the quickjs snapshot). ``get_state`` and every
      history entry.
    * ``snap.metadata['writes']`` — ``{node_name: {channel: value}}`` map of
      the raw writes that produced each checkpoint. On the ``after_agent``
      step that first snapshots the REPL, ``value`` is the encoded write
      record ``("snap", full_bytes)`` ≈ 1.4 MB.
    * ``snap.tasks[*].result`` — the return dict of each completed
      ``PregelTask``. ``after_agent`` returns
      ``{"_quickjs_snapshot_payload": ("snap", bytes)}``; this dict becomes
      the task's ``result`` field, which the API surfaces verbatim under
      ``tasks[*].result`` (``langgraph_api.state:106``). This is the
      dominant leak: 1.7 MB in the last history entry of any thread whose
      most-recent-in-window checkpoint had a snapshot anchor.
    * ``snap.metadata['counters_since_delta_snapshot']`` — DeltaChannel's
      snapshot cadence bookkeeping ``{channel: [count, superstep]}``. Tiny
      (~20 B) but exposes the private field name; strip for cleanliness.
    * ``snap.tasks[*].state`` (nested ``StateSnapshot``) — populated when the
      caller passes ``subgraphs=True``. Repeats all of the above surfaces
      for each subgraph task, so recurse into it. Not exercised by the
      current WebUI (which doesn't pass ``subgraphs=True`` on REST reads),
      but SDK / curl / gRPC callers can.
    """
    if snap is None:
        return snap
    values = {k: v for k, v in snap.values.items() if k not in _PRIVATE_STATE_FIELDS}
    metadata = snap.metadata
    if metadata:
        new_metadata = metadata
        if new_metadata.get("writes"):
            scrubbed_writes = {
                node: {
                    k: v for k, v in ch_writes.items() if k not in _PRIVATE_STATE_FIELDS
                }
                for node, ch_writes in new_metadata["writes"].items()
            }
            new_metadata = {**new_metadata, "writes": scrubbed_writes}
        if new_metadata.get("counters_since_delta_snapshot"):
            scrubbed_counters = {
                k: v
                for k, v in new_metadata["counters_since_delta_snapshot"].items()
                if k not in _PRIVATE_STATE_FIELDS
            }
            new_metadata = {
                **new_metadata,
                "counters_since_delta_snapshot": scrubbed_counters,
            }
        metadata = new_metadata
    tasks = snap.tasks
    if tasks:
        new_tasks = []
        changed = False
        for t in tasks:
            replace_kwargs: dict = {}
            result = getattr(t, "result", None)
            if isinstance(result, dict) and any(
                k in result for k in _PRIVATE_STATE_FIELDS
            ):
                replace_kwargs["result"] = {
                    k: v for k, v in result.items() if k not in _PRIVATE_STATE_FIELDS
                }
            # ``t.state`` is a ``RunnableConfig | StateSnapshot | None`` per
            # ``PregelTask``'s typing. When ``subgraphs=True`` on the caller,
            # this holds the subgraph's fully-materialized ``StateSnapshot`` —
            # which repeats the same four leak surfaces (``values``,
            # ``metadata.writes``, ``metadata.counters_since_delta_snapshot``,
            # ``tasks[*].result/state``). Recurse so the whole tree is clean.
            nested_state = getattr(t, "state", None)
            if isinstance(nested_state, StateSnapshot):
                scrubbed_state = _strip_private(nested_state)
                if scrubbed_state is not nested_state:
                    replace_kwargs["state"] = scrubbed_state
            if replace_kwargs:
                new_tasks.append(t._replace(**replace_kwargs))
                changed = True
            else:
                new_tasks.append(t)
        if changed:
            tasks = tuple(new_tasks)
    return snap._replace(values=values, metadata=metadata, tasks=tasks)


class _EvoFilteredGraph(CompiledStateGraph):
    """Filters ``PrivateStateAttr``-marked state fields from checkpoint reads.

    ``Pregel.copy`` uses ``self.__class__(**attrs)`` so this subclass
    survives the ``graph_obj.copy(update=...)`` call in
    ``langgraph_api.graph.get_graph`` that binds the checkpointer / store
    before yielding to endpoint handlers.

    **Known gap — streaming paths.** The overrides only cover ``get_state``
    / ``get_state_history``. On this compiled graph,
    ``self.output_channels`` correctly excludes ``_quickjs_snapshot_payload``
    (respects ``OmitFromSchema(output=True)``), but
    ``self.stream_channels_asis`` includes it alongside other private
    fields (``jump_to``, ``_summarization_event``) — the two lists are
    built by ``langgraph.graph.state``'s graph builder and only the first
    checks the output schema. So a client streaming with
    ``stream_mode="values"`` or ``stream_mode="events"`` (which fall back
    to ``stream_channels_asis`` when ``output_keys`` is ``None``) can pull
    the anchor blob in per-run event data. Empirically the WebUI's
    ``stream_mode=["updates"]`` path is clean, so this is transient per-run
    rather than the persistent per-getState download this PR targets.
    Filter here first; extend into the stream layer if a client relying on
    ``values`` / ``events`` reports it.
    """

    async def aget_state(self, config, *, subgraphs=False):
        return _strip_private(await super().aget_state(config, subgraphs=subgraphs))

    def get_state(self, config, *, subgraphs=False):
        return _strip_private(super().get_state(config, subgraphs=subgraphs))

    async def aget_state_history(self, config, **kw):
        async for snap in super().aget_state_history(config, **kw):
            yield _strip_private(snap)

    def get_state_history(self, config, **kw):
        for snap in super().get_state_history(config, **kw):
            yield _strip_private(snap)


def make_EvoScientist_agent() -> CompiledStateGraph:
    """Graph factory for the ``EvoScientist`` entry in ``langgraph.json``.

    Called by langgraph-api on every request that touches the graph
    (``get_graph`` → ``invoke_factory``), and it must not do its own
    caching: ``_get_default_agent`` already returns the same compiled
    object until the expert registry changes, at which point it rebuilds.
    Adding a second cache here would only be able to go stale.

    Steady-state cost is one skills-tree walk per request — what
    ``dispatchable_experts_token`` charges to answer "did the expert set
    change?". The walk cannot be replaced by a cheaper counter: an expert
    can be authored or edited straight onto the writable workspace skills
    tier without ever calling ``install_skill``, and that is a first-class
    flow (the agent itself writes there when asked for a new expert), not a
    corner case. Before this factory existed the equivalent walk ran on
    every *model call*, so this remains a large net reduction.

    A rebuild runs synchronously on the event loop, so the first request
    after the expert set changes pays for it. That is the price of the
    guarantee this factory exists to provide — returning the stale graph
    and rebuilding in the background would mean the very next turn still
    can't reach the new expert. It is logged with its elapsed time so the
    cost is visible rather than inferred.

    Zero-arg on purpose: ``langgraph_api._factory_utils._classify_factory``
    reads the signature, and any parameter would make it inject a config or
    a ``ServerRuntime`` we have no use for.
    """
    started = time.perf_counter()
    from EvoScientist.EvoScientist import _get_default_agent

    agent = _get_default_agent()
    # In-place ``__class__`` swap: the subclass adds only methods (no new
    # instance attributes) so the memory layout is identical and the swap
    # is safe. Constructing a fresh ``_EvoFilteredGraph`` via ``.copy()``
    # would require reproducing the deep-agent build pipeline; the swap
    # avoids that. Idempotent, so re-running it on the cached graph costs
    # one ``isinstance`` check.
    if not isinstance(agent, _EvoFilteredGraph):
        agent.__class__ = _EvoFilteredGraph
        elapsed_ms = (time.perf_counter() - started) * 1000
        _logger.info(
            "Built the main EvoScientist graph in %.0f ms.",
            elapsed_ms,
        )
    return agent


def _apply_filter_to_all_registered_graphs() -> None:
    """Extend the class swap to every graph registered in ``langgraph.json``.

    ``EvoScientist.py:_build_middleware_stack`` installs
    ``create_code_interpreter_middleware`` unconditionally — it's not gated
    on the ``for_async_subagent`` flag — so every subagent (sync ``task``
    dispatch and async ``start_async_task``) carries the QuickJS REPL and
    can produce ``_quickjs_snapshot_payload`` writes on its own checkpoint
    namespace.

    Async subagents get their own ``thread_id`` and their ``/threads/{id}/state``
    endpoint is served by their own compiled graph. Without swapping the
    class on those graphs, the filter ``make_EvoScientist_agent`` applies to
    the main agent doesn't reach that endpoint and any real code_interpreter
    touch inside a subagent leaks the anchor snapshot verbatim.

    Reads the graph registry straight from ``langgraph.json`` so a new
    subagent added to the config picks up the swap automatically — no
    hardcoded list to keep in sync. The main agent's own entry resolves to
    a factory rather than a compiled graph, so the ``isinstance`` guard
    below skips it; ``make_EvoScientist_agent`` swaps it instead, on every
    graph it returns.

    Idempotent (skips graphs already swapped) and safe on graphs that don't
    use the middleware — ``_strip_private`` returns snapshots unchanged when
    the private field is absent. Best-effort: if the config is unreadable
    or an entry can't be resolved, the deployment still starts — only the
    unresolvable subagents remain unfiltered.
    """
    import json
    from importlib import import_module
    from pathlib import Path

    config_path = Path(__file__).parent / "langgraph.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return

    for path in config.get("graphs", {}).values():
        # Format: "module.dotted.path:attr_name"
        if ":" not in path:
            continue
        module_path, attr = path.rsplit(":", 1)
        try:
            module = import_module(module_path)
        except ImportError:
            continue
        graph = getattr(module, attr, None)
        if isinstance(graph, CompiledStateGraph) and not isinstance(
            graph, _EvoFilteredGraph
        ):
            graph.__class__ = _EvoFilteredGraph


_apply_filter_to_all_registered_graphs()

# Build the main agent eagerly, at import. langgraph-api imports this module
# from a worker thread during startup (``_graph_from_spec`` runs under
# ``run_in_executor``), so paying the ~1 s construction here keeps it off the
# event loop and off the first request — the same startup profile this module
# had before it became a factory. Every later call is a cache hit unless the
# expert registry moved.
make_EvoScientist_agent()


__all__ = ["make_EvoScientist_agent"]
