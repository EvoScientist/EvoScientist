"""Unit tests for PanelWidget (TUI in-eval fan-out live view).

Widget-mount / compose paths need a Textual App context and are exercised
through the interactive TUI end-to-end. Compositor-free tests cover
construction and state-transition helpers; ``TestPanelWidgetReentry`` runs
against a real Textual harness to probe the ``_is_active`` latch behaviour
after finalize (the ``Promise.allSettled`` → retry-failed-subset scenario).
"""

from __future__ import annotations

import unittest

import pytest

pytest.importorskip("textual")


class TestPanelWidgetState:
    """PanelWidget state transitions independent of the compositor."""

    def test_construction(self):
        from EvoScientist.cli.widgets.panel_widget import PanelWidget

        w = PanelWidget("ci_eval_1")
        assert w.eval_id == "ci_eval_1"
        assert w.dispatch_count == 0
        assert w._is_active is True

    def test_summary_counts_all_running(self):
        from EvoScientist.cli.widgets.panel_widget import PanelWidget, _DispatchRow

        w = PanelWidget("e1")
        w._rows["d1"] = _DispatchRow("innovator", "a")
        w._rows["d2"] = _DispatchRow("pragmatist", "b")
        running, ok, err = w._summary_counts()
        assert (running, ok, err) == (2, 0, 0)

    def test_summary_counts_mixed(self):
        from EvoScientist.cli.widgets.panel_widget import PanelWidget, _DispatchRow

        w = PanelWidget("e1")
        r1 = _DispatchRow("innovator", "a")
        r2 = _DispatchRow("pragmatist", "b")
        r3 = _DispatchRow("critic", "c")
        r1._status = "ok"
        r2._status = "err"
        w._rows["d1"] = r1
        w._rows["d2"] = r2
        w._rows["d3"] = r3
        running, ok, err = w._summary_counts()
        assert (running, ok, err) == (1, 1, 1)


class TestDispatchRow(unittest.TestCase):
    """_DispatchRow state transitions."""

    def test_construction_sets_running(self):
        from EvoScientist.cli.widgets.panel_widget import _DispatchRow

        row = _DispatchRow("idea-brainstorm", "innovator voice")
        assert row._subagent_type == "idea-brainstorm"
        assert row._label == "innovator voice"
        assert row._status == "running"
        assert row._duration_ms is None

    def test_elapsed_display_uses_recorded_duration(self):
        from EvoScientist.cli.widgets.panel_widget import _DispatchRow

        row = _DispatchRow("x", "y")
        row._duration_ms = 2500
        display = row._elapsed_display()
        assert "2.5" in display


class TestPanelWidgetReentry:
    """Regression: retry-failed-subset under the same eval_id re-arms the panel.

    The ``Promise.allSettled`` → retry-failed-subset pattern (encouraged by
    the DELEGATION_STRATEGY guardrail) can send a second wave of
    ``panel_dispatch_start`` events under the same ``eval_id`` after the
    panel has already finalized. ``start_dispatch`` must detect the
    finalized state and undo the three effects of ``_maybe_finalize``
    (latch, ``--completed`` class, interval timer) so the new row's
    spinner + elapsed keep ticking and the header refreshes on completion.
    """

    async def _boot_panel(self):
        from textual.app import App, ComposeResult

        from EvoScientist.cli.widgets.panel_widget import PanelWidget

        class _PanelApp(App[None]):
            def compose(self) -> ComposeResult:
                yield PanelWidget("eval-42")

        app = _PanelApp()
        return app

    def _header_text(self, panel) -> str:
        from textual.widgets import Static

        header = panel.query_one(".panel-header", Static)
        if header.size.height == 0:
            return ""
        return "".join(seg.text for seg in header.render_line(0)).strip()

    async def test_reentry_after_finalize_rearms_panel(self):
        from EvoScientist.cli.widgets.panel_widget import PanelWidget

        app = await self._boot_panel()
        async with app.run_test(size=(120, 20)) as pilot:
            panel = app.query_one(PanelWidget)
            await pilot.pause()

            # Wave 1: two dispatches, one settles ok, one settles err.
            await panel.start_dispatch("d1", "innovator", "brainstorm a")
            await panel.start_dispatch("d2", "pragmatist", "brainstorm b")
            await pilot.pause()
            panel.complete_dispatch("d1", 1200)
            panel.fail_dispatch("d2", 800, "boom")
            await pilot.pause()

            # Baseline: finalize fired.
            assert panel._is_active is False
            assert panel._timer_handle is None
            assert panel.has_class("--completed")
            header_after_wave1 = self._header_text(panel)
            assert "1 done" in header_after_wave1
            assert "1 failed" in header_after_wave1
            assert "running" not in header_after_wave1

            # Wave 2: eval retries the failed subset under the same eval_id.
            await panel.start_dispatch("d3", "pragmatist", "brainstorm b retry")
            await pilot.pause()

            # Panel re-armed: latch flipped, timer restarted, class dropped.
            assert panel._is_active is True
            assert panel._timer_handle is not None
            assert not panel.has_class("--completed")

            # Header shows the running count again.
            header_during_wave2 = self._header_text(panel)
            assert "1 running" in header_during_wave2
            assert "1 done" in header_during_wave2
            assert "1 failed" in header_during_wave2

            # d3 completes: header refreshes to 2 done, panel re-finalizes.
            panel.complete_dispatch("d3", 900)
            await pilot.pause()
            header_after_wave2 = self._header_text(panel)
            assert "2 done" in header_after_wave2
            assert "1 failed" in header_after_wave2
            assert "running" not in header_after_wave2
            assert panel._is_active is False
            assert panel._timer_handle is None
            assert panel.has_class("--completed")


class TestPanelWidgetFinalizeEmpty:
    """Regression: ``finalize_running`` stops the timer even with zero
    running rows.

    The timer is armed by ``on_mount`` (first-dispatch case) or re-armed
    by ``start_dispatch`` (retry-failed-subset case) before any row lands
    in the "running" state. If cancel arrives before the first row is
    registered, ``finalize_running``'s loop body never runs, so
    ``_maybe_finalize`` must be called unconditionally after the loop.
    """

    async def test_finalize_running_stops_timer_with_no_rows(self):
        from textual.app import App, ComposeResult

        from EvoScientist.cli.widgets.panel_widget import PanelWidget

        class _PanelApp(App[None]):
            def compose(self) -> ComposeResult:
                yield PanelWidget("eval-empty")

        app = _PanelApp()
        async with app.run_test(size=(120, 20)) as pilot:
            panel = app.query_one(PanelWidget)
            await pilot.pause()

            # Timer armed by on_mount, no dispatches yet.
            assert panel._is_active is True
            assert panel._timer_handle is not None
            assert panel._rows == {}

            panel.finalize_running()

            assert panel._is_active is False
            assert panel._timer_handle is None
            assert panel.has_class("--completed")
