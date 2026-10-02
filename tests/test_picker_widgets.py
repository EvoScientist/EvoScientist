"""Behavior tests for the picker/browser widgets and their shared engine.

Written against the pre-refactor SkillBrowserWidget / MCPBrowserWidget
behavior to pin the two-phase tag->checkbox flow, then kept green while
the widgets migrated onto the shared picker engine (PR #362 follow-up
discussion).  Widgets are driven without a running app: ``__init__`` and
the ``action_*`` methods only touch widget state, and DOM-dependent
rendering no-ops before ``compose()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar
from unittest.mock import MagicMock

from EvoScientist.cli.widgets.mcp_browser import MCPBrowserWidget
from EvoScientist.cli.widgets.picker_base import (
    first_selectable_index,
    move_selection,
)
from EvoScientist.cli.widgets.skill_browser import SkillBrowserWidget

_SKILL_INDEX = [
    {
        "name": "paper-writing",
        "description": "Write papers",
        "tags": ["writing"],
        "install_source": "src/paper-writing",
    },
    {
        "name": "paper-review",
        "description": "Review papers",
        "tags": ["writing", "review"],
        "install_source": "src/paper-review",
    },
    {
        "name": "experiment-pipeline",
        "description": "Run experiments",
        "tags": ["experiments"],
        "install_source": "src/experiment-pipeline",
    },
]


@dataclass
class _Server:
    name: str
    tags: list[str] = field(default_factory=list)
    description: str = ""
    label: str = ""


_SERVERS = [
    _Server("tavily", tags=["search"], description="Web search"),
    _Server("github", tags=["dev", "search"], description="GitHub API"),
    _Server("filesystem", tags=["dev"], description="Local files"),
]


def _skill_browser(**kwargs) -> SkillBrowserWidget:
    w = SkillBrowserWidget(_SKILL_INDEX, {"paper-review"}, **kwargs)
    w.post_message = MagicMock()
    w.focus = MagicMock()
    return w


def _mcp_browser(**kwargs) -> MCPBrowserWidget:
    w = MCPBrowserWidget(_SERVERS, {"github"}, **kwargs)
    w.post_message = MagicMock()
    w.focus = MagicMock()
    return w


class TestSelectionHelpers:
    """Pure navigation helpers shared by all pickers (headers skipped,
    wrapping cursor)."""

    _ITEMS: ClassVar[list[str]] = ["header", "item", "item", "header", "item"]

    @staticmethod
    def _selectable(item: str) -> bool:
        return item == "item"

    def test_first_selectable_skips_headers(self):
        assert first_selectable_index(self._ITEMS, self._selectable) == 1

    def test_first_selectable_empty_or_none_returns_zero(self):
        assert first_selectable_index([], self._selectable) == 0
        assert first_selectable_index(["header"], self._selectable) == 0

    def test_move_skips_headers_and_wraps(self):
        assert move_selection(self._ITEMS, 1, 1, self._selectable) == 2
        assert move_selection(self._ITEMS, 2, 1, self._selectable) == 4
        # wraps past the trailing header back to the first item
        assert move_selection(self._ITEMS, 4, 1, self._selectable) == 1
        assert move_selection(self._ITEMS, 1, -1, self._selectable) == 4

    def test_move_with_no_selectable_keeps_current(self):
        items = ["header", "header"]
        assert move_selection(items, 0, 1, self._selectable) == 0

    def test_move_single_selectable_stays_put(self):
        items = ["header", "item"]
        assert move_selection(items, 1, 1, self._selectable) == 1


class TestSkillBrowserFlow:
    def test_tag_items_sorted_by_count_with_all_first(self):
        w = _skill_browser()
        assert w._tag_items[0] == ("all", 3)
        # "writing" appears twice -> first among real tags
        assert w._tag_items[1] == ("writing", 2)

    def test_phase1_enter_filters_and_enters_phase2(self):
        w = _skill_browser()
        w._selected = 1  # "writing"
        w.action_confirm()
        assert w._phase == 2
        assert [s["name"] for s in w._filtered_items] == [
            "paper-writing",
            "paper-review",
        ]
        assert w._selected == 0

    def test_phase1_all_selects_everything(self):
        w = _skill_browser()
        w._selected = 0  # "all"
        w.action_confirm()
        assert len(w._filtered_items) == 3

    def test_space_toggles_uninstalled_only(self):
        w = _skill_browser()
        w.action_confirm()  # all -> phase 2
        w._selected = 0  # paper-writing (not installed)
        w.action_toggle()
        assert 0 in w._checked
        w.action_toggle()
        assert 0 not in w._checked
        w._selected = 1  # paper-review (installed)
        w.action_toggle()
        assert 1 not in w._checked

    def test_confirm_posts_confirmed_with_sources(self):
        w = _skill_browser()
        w.action_confirm()  # all -> phase 2
        w._selected = 0
        w.action_toggle()
        w._selected = 2
        w.action_toggle()
        w.action_confirm()
        msg = w.post_message.call_args[0][0]
        assert isinstance(msg, SkillBrowserWidget.Confirmed)
        assert msg.install_sources == ["src/paper-writing", "src/experiment-pipeline"]

    def test_escape_returns_to_phase1_then_cancels(self):
        w = _skill_browser()
        w.action_confirm()  # phase 2
        w.action_cancel()
        assert w._phase == 1
        w.action_cancel()
        msg = w.post_message.call_args[0][0]
        assert isinstance(msg, SkillBrowserWidget.Cancelled)

    def test_pre_filter_tag_skips_to_phase2(self):
        w = _skill_browser(pre_filter_tag="experiments")
        assert w._phase == 2
        assert [s["name"] for s in w._filtered_items] == ["experiment-pipeline"]
        # Esc cancels directly (no tag-picker phase to return to)
        w.action_cancel()
        msg = w.post_message.call_args[0][0]
        assert isinstance(msg, SkillBrowserWidget.Cancelled)

    def test_pre_filter_without_match_falls_back_to_phase1(self):
        w = _skill_browser(pre_filter_tag="nonexistent")
        assert w._phase == 1

    def test_move_wraps_around(self):
        w = _skill_browser()
        n = len(w._tag_items)
        w._selected = 0
        w.action_move_up()
        assert w._selected == n - 1
        w.action_move_down()
        assert w._selected == 0


class TestMountedBrowser:
    """End-to-end smoke test: the engine's mount hook, bindings and
    base-class CSS must all work on a real mounted widget."""

    async def test_skill_browser_keyboard_flow(self):
        from textual.app import App, ComposeResult

        posted: list = []

        class BrowserApp(App[None]):
            def compose(self) -> ComposeResult:
                yield SkillBrowserWidget(_SKILL_INDEX, {"paper-review"})

            def on_skill_browser_widget_confirmed(self, event) -> None:
                posted.append(event)

        app = BrowserApp()
        async with app.run_test(size=(80, 30)) as pilot:
            widget = app.query_one(SkillBrowserWidget)
            await pilot.pause()
            # Base CSS (via TagCheckboxBrowserBase type selector) applied
            assert widget.styles.border.top is not None
            # Phase 1 renders the tag rows
            assert widget._phase == 1
            # Enter on "all" -> phase 2; space toggles; enter confirms
            await pilot.press("enter")
            assert widget._phase == 2
            await pilot.press("space")
            await pilot.press("enter")
            await pilot.pause()
            assert len(posted) == 1
            assert posted[0].install_sources == ["src/paper-writing"]


class TestMCPBrowserFlow:
    def test_tag_items_sorted_by_count_with_all_first(self):
        w = _mcp_browser()
        assert w._tag_items[0] == ("all", 3)
        assert {t for t, _ in w._tag_items[1:]} == {"dev", "search"}

    def test_confirm_posts_confirmed_with_entries(self):
        w = _mcp_browser()
        w.action_confirm()  # all -> phase 2
        w._selected = 0  # tavily (not installed)
        w.action_toggle()
        w._selected = 1  # github (installed -> ignored)
        w.action_toggle()
        w.action_confirm()
        msg = w.post_message.call_args[0][0]
        assert isinstance(msg, MCPBrowserWidget.Confirmed)
        assert [e.name for e in msg.entries] == ["tavily"]

    def test_tag_filter_selects_matching_servers(self):
        w = _mcp_browser()
        tag_index = [t for t, _ in w._tag_items].index("dev")
        w._selected = tag_index
        w.action_confirm()
        assert w._phase == 2
        assert [e.name for e in w._filtered_items] == ["github", "filesystem"]

    def test_escape_returns_to_phase1_then_cancels(self):
        w = _mcp_browser()
        w.action_confirm()
        w.action_cancel()
        assert w._phase == 1
        w.action_cancel()
        msg = w.post_message.call_args[0][0]
        assert isinstance(msg, MCPBrowserWidget.Cancelled)
