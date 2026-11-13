"""Delete failures must stay visible, and full-ID retries must reach the server."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from EvoScientist.commands.implementation.session import DeleteCommand
from EvoScientist.gateway.composite import CompositeGraphGateway
from EvoScientist.gateway.server import LangGraphServerThreadStore
from EvoScientist.gateway.types import ThreadResolution

THREAD = "11111111-1111-1111-1111-111111111111"


@pytest.mark.asyncio
async def test_partial_delete_is_reported_and_retry_reaches_server():
    read = SimpleNamespace(
        resolve_thread=AsyncMock(
            side_effect=[ThreadResolution(THREAD, ()), ThreadResolution(None, ())]
        ),
        delete_thread=AsyncMock(side_effect=[True, False]),
    )
    execute = SimpleNamespace(
        delete_thread=AsyncMock(side_effect=[RuntimeError("offline"), True])
    )
    gateway = CompositeGraphGateway(read=read, execute=execute)
    ui = MagicMock()
    ctx = SimpleNamespace(graph_gateway=gateway, thread_id="other", ui=ui)
    command = DeleteCommand()
    await command.execute(ctx, [THREAD])
    messages = [c.args[0] for c in ui.append_system.call_args_list]
    assert not any(m.startswith("Deleted session") for m in messages)
    assert any("local" in m.lower() and "offline" in m for m in messages)
    await command.execute(ctx, [THREAD])
    assert execute.delete_thread.await_count == 2
    assert any(
        c.args[0].startswith("Deleted session") for c in ui.append_system.call_args_list
    )


@pytest.mark.asyncio
async def test_failed_run_cancellation_prevents_unreported_server_delete():
    client = SimpleNamespace(
        runs=SimpleNamespace(
            list=AsyncMock(side_effect=RuntimeError("cancel unavailable"))
        ),
        threads=SimpleNamespace(delete=AsyncMock()),
    )
    store = LangGraphServerThreadStore(client=client)
    with pytest.raises(RuntimeError, match="cancel unavailable"):
        await store.delete_thread(THREAD)
    client.threads.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_many_failure_keeps_server_registry_for_retry():
    client = SimpleNamespace(
        runs=SimpleNamespace(
            list=AsyncMock(return_value=[{"run_id": "r1"}]),
            cancel_many=AsyncMock(side_effect=RuntimeError("cancel rejected")),
        ),
        threads=SimpleNamespace(delete=AsyncMock()),
    )
    store = LangGraphServerThreadStore(client=client)
    with pytest.raises(RuntimeError, match="cancel rejected"):
        await store.delete_thread(THREAD)
    client.threads.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_unresolved_full_uuid_cannot_delete_current_session():
    gateway = SimpleNamespace(
        resolve_thread=AsyncMock(return_value=ThreadResolution(None, ())),
        delete_thread=AsyncMock(),
    )
    ui = MagicMock()
    ctx = SimpleNamespace(graph_gateway=gateway, thread_id=THREAD, ui=ui)
    await DeleteCommand().execute(ctx, [THREAD])
    gateway.delete_thread.assert_not_awaited()
    assert "Cannot delete the current session" in ui.append_system.call_args.args[0]


@pytest.mark.asyncio
async def test_unresolved_prefix_does_not_bypass_resolution():
    gateway = SimpleNamespace(
        resolve_thread=AsyncMock(return_value=ThreadResolution(None, ())),
        delete_thread=AsyncMock(),
    )
    ctx = SimpleNamespace(graph_gateway=gateway, thread_id="other", ui=MagicMock())
    await DeleteCommand().execute(ctx, [THREAD[:8]])
    gateway.delete_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_uuid_retry_compares_current_session_canonically():
    from uuid import UUID

    current = "abcdefab-cdef-abcd-efab-cdefabcdefab"
    gateway = SimpleNamespace(
        resolve_thread=AsyncMock(return_value=ThreadResolution(None, ())),
        delete_thread=AsyncMock(),
    )
    ctx = SimpleNamespace(
        graph_gateway=gateway, thread_id=current.upper(), ui=MagicMock()
    )
    await DeleteCommand().execute(ctx, [str(UUID(current))])
    gateway.delete_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_server_gateway_rejects_uuid_from_another_graph_before_delete():
    from EvoScientist.gateway.server import LangGraphServerGateway

    store = SimpleNamespace(
        resolve_thread_id_prefix=AsyncMock(return_value=(None, [])),
        delete_thread=AsyncMock(),
    )
    gateway = LangGraphServerGateway(thread_store=store)
    assert not await gateway.delete_thread(THREAD)
    store.delete_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_not_found_is_not_reported_as_missing_thread():
    import httpx
    from langgraph_sdk.errors import NotFoundError

    error = NotFoundError(
        "run gone",
        response=httpx.Response(
            404, request=httpx.Request("GET", "http://localhost/runs")
        ),
        body=None,
    )
    client = SimpleNamespace(
        runs=SimpleNamespace(list=AsyncMock(side_effect=error)),
        threads=SimpleNamespace(delete=AsyncMock()),
    )
    with pytest.raises(NotFoundError):
        await LangGraphServerThreadStore(client=client).delete_thread(THREAD)
    client.threads.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_server_gateway_retry_uses_configured_graph_resolution():
    from EvoScientist.gateway import GraphTarget
    from EvoScientist.gateway.server import LangGraphServerGateway

    store = SimpleNamespace(
        resolve_thread_id_prefix=AsyncMock(return_value=(THREAD, [])),
        delete_thread=AsyncMock(return_value=True),
    )
    gateway = LangGraphServerGateway(thread_store=store)
    assert await gateway.delete_thread(THREAD, GraphTarget(graph_id="custom"))
    store.resolve_thread_id_prefix.assert_awaited_once_with(THREAD, graph_id="custom")
    store.delete_thread.assert_awaited_once_with(THREAD)
