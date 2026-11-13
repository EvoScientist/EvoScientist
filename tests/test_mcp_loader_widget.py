"""Tests for ``cli/widgets/mcp_loader_widget``.

A cache-hit MCP retry emits progress events only for the servers it
actually reloads, so the servers served from the cache stay ``pending``
in the widget. ``mark_finished`` must count those as loaded instead of
leaving frozen spinners and an under-counted header behind.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from EvoScientist.cli.widgets.mcp_loader_widget import MCPLoaderWidget


class _WidgetApp(App):
    def compose(self) -> ComposeResult:
        yield MCPLoaderWidget(["good", "flaky", "other"])


@pytest.mark.asyncio
async def test_mark_finished_counts_cached_servers_as_loaded():
    app = _WidgetApp()
    async with app.run_test() as pilot:
        widget = app.query_one(MCPLoaderWidget)
        # Only the retried server reports: it failed again.
        widget.update_server("flaky", "error", "timed out")
        await pilot.pause()
        widget.mark_finished()
        await pilot.pause()

        assert widget._progress["good"] == ("ok", "")
        assert widget._progress["other"] == ("ok", "")
        assert widget._progress["flaky"] == ("error", "timed out")
        assert not widget.dismissed
        assert "2/3 loaded, 1 failed" in widget._build_renderable().plain


@pytest.mark.asyncio
async def test_mark_finished_counts_recovered_server_and_cached_ones():
    app = _WidgetApp()
    async with app.run_test() as pilot:
        widget = app.query_one(MCPLoaderWidget)
        widget.update_server("flaky", "ok", "2")
        await pilot.pause()
        widget.mark_finished()
        await pilot.pause()

        assert widget._progress["good"] == ("ok", "")
        assert widget._progress["flaky"] == ("ok", "2")
        assert "3/3 servers loaded" in widget._build_renderable().plain


@pytest.mark.asyncio
async def test_mark_finished_still_dismisses_when_nothing_reported():
    app = _WidgetApp()
    async with app.run_test() as pilot:
        widget = app.query_one(MCPLoaderWidget)
        widget.mark_finished()
        await pilot.pause()

        # A plain cached replay emits no events at all: the widget
        # drops itself instead of flashing a misleading header.
        assert widget.dismissed
