"""Server-path delivery: commit the proactive push INTO the source thread.

The MVP appends the tagged ``AIMessage`` to the ORIGINAL thread via the server
gateway. This ``Deliverer`` does exactly that, reusing
:func:`commit.make_gateway_commit_fn` so the ConflictError defer-and-retry —
owned by ``run_proactive_check``'s ``commit_with_conflict_retry`` loop — drives
the retries: a ``ConflictError`` from the append propagates out of ``deliver``
and is retried by that outer loop.

Channel delivery is a SEPARATE serve-side concern: once the push is committed
here, the ``delivery_watcher`` (running in serve, holding the channel origins)
notices it and pushes it via ``publish_to_channel_origin``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from .commit import build_proactive_tag, make_gateway_commit_fn, proactive_message_id


class GatewayDeliverer:
    """Commit proactive replies into the source thread via the server gateway.

    ``gateway`` is the widened server gateway (``update_state_values`` accepting
    ``as_node``/``metadata``); ``target`` is its ``GraphTarget``. Metadata is left
    ``None`` — the server checkpointer stamps ``agent_name`` from ``graph_id`` on
    the commit, so the thread stays visible without forwarding it (verified: the
    server path accepts-but-ignores ``metadata``).
    """

    def __init__(self, gateway: Any, target: Any, *, as_node: str = "model") -> None:
        self._gateway = gateway
        self._target = target
        self._as_node = as_node

    async def deliver(
        self,
        source_thread_id: str,
        reply: str,
        proactive_id: str,
        *,
        source_messages: list | None = None,
    ) -> bool:
        message = AIMessage(
            content=reply,
            id=proactive_message_id(proactive_id),
            additional_kwargs=build_proactive_tag(proactive_id),
        )
        commit_fn = make_gateway_commit_fn(
            self._gateway, self._target, source_thread_id
        )
        # Append-conflict propagates (outer commit_with_conflict_retry retries);
        # clear-conflict is swallowed inside make_gateway_commit_fn.
        await commit_fn(message, None, self._as_node)
        return True
