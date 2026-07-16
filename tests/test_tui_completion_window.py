"""Tests for the TUI completion popup windowing (issue #354).

The ``#completions`` Static used to be clipped by a fixed CSS
``max-height: 15`` — the flattened command list for ``/`` renders ~30
rows (items + category headers + separators), so trailing categories
(``General``) were silently cut off. The fix renders a height-bounded
window that always keeps the selected item visible and marks hidden
items with overflow indicators.
"""

from __future__ import annotations

from EvoScientist.cli.tui_interactive import (
    _build_completion_rows,
    _render_completion_text,
    _window_completion_rows,
)
from EvoScientist.commands._completion_engine import (
    CompletionCandidate,
    compute_completions,
)


def _candidates(spec: list[tuple[str, str]]) -> list[CompletionCandidate]:
    """Build candidates from (text, category) pairs."""
    return [
        CompletionCandidate(
            text=text,
            description=f"desc of {text}",
            replace_start=0,
            replace_end=1,
            category=category,
        )
        for text, category in spec
    ]


class TestBuildCompletionRows:
    def test_headers_and_separators_between_categories(self):
        items = _candidates([("/a", "Session"), ("/b", "Session"), ("/c", "General")])
        rows = _build_completion_rows(items)
        kinds = [kind for kind, _payload, _idx in rows]
        assert kinds == ["header", "item", "item", "sep", "header", "item"]

    def test_item_rows_keep_original_indices(self):
        items = _candidates([("/a", "Session"), ("/b", "General")])
        rows = _build_completion_rows(items)
        item_indices = [idx for kind, _payload, idx in rows if kind == "item"]
        assert item_indices == [0, 1]

    def test_no_category_items_have_no_headers(self):
        items = _candidates([("@f1", ""), ("@f2", "")])
        rows = _build_completion_rows(items)
        assert [kind for kind, _p, _i in rows] == ["item", "item"]


class TestWindowCompletionRows:
    def test_fitting_rows_are_untouched(self):
        rows = _build_completion_rows(
            _candidates([("/a", "Session"), ("/b", "General")])
        )
        visible, above, below = _window_completion_rows(rows, -1, 10)
        assert visible == rows
        assert (above, below) == (0, 0)

    def test_no_selection_windows_from_top(self):
        items = _candidates([(f"/c{i}", "Session") for i in range(20)])
        rows = _build_completion_rows(items)
        visible, above, below = _window_completion_rows(rows, -1, 8)
        assert above == 0
        assert below > 0
        # First item visible at the top of the window
        first_items = [idx for kind, _p, idx in visible if kind == "item"]
        assert first_items[0] == 0

    def test_last_selection_windows_to_bottom(self):
        items = _candidates([(f"/c{i}", "Session") for i in range(20)])
        rows = _build_completion_rows(items)
        visible, above, below = _window_completion_rows(rows, 19, 8)
        assert below == 0
        assert above > 0
        assert any(kind == "item" and idx == 19 for kind, _p, idx in visible)

    def test_selected_item_always_visible(self):
        items = _candidates(
            [(f"/c{i}", cat) for i, cat in enumerate(["A"] * 7 + ["B"] * 7 + ["C"] * 6)]
        )
        rows = _build_completion_rows(items)
        for selected in range(len(items)):
            visible, above, below = _window_completion_rows(rows, selected, 9)
            assert any(
                kind == "item" and idx == selected for kind, _p, idx in visible
            ), f"selected {selected} not in window"
            # Window body + indicator rows never exceed the budget
            total = len(visible) + (1 if above else 0) + (1 if below else 0)
            assert total <= 9

    def test_hidden_counts_only_count_items(self):
        items = _candidates(
            [("/a", "A"), ("/b", "A"), ("/c", "B"), ("/d", "B"), ("/e", "C")]
        )
        rows = _build_completion_rows(items)  # 8 rows
        visible, above, below = _window_completion_rows(rows, 4, 5)
        assert above + below == 5 - sum(1 for k, _p, _i in visible if k == "item")


class TestRenderCompletionText:
    def test_tall_terminal_renders_all_commands(self):
        """Regression for #354: with enough rows every category renders."""
        result = compute_completions("/", 1)
        text = _render_completion_text(result.candidates, -1, 100).plain
        assert "General" in text
        assert "/help" in text
        for candidate in result.candidates:
            assert candidate.text in text

    def test_short_terminal_shows_overflow_indicator(self):
        result = compute_completions("/", 1)
        text = _render_completion_text(result.candidates, -1, 10).plain
        assert "more" in text
        assert "↓" in text  # ↓ indicator for hidden items below
        assert result.candidates[0].text in text

    def test_navigation_reaches_every_command(self):
        """Every candidate becomes visible when selected, even clipped ones."""
        result = compute_completions("/", 1)
        for i, candidate in enumerate(result.candidates):
            text = _render_completion_text(result.candidates, i, 10).plain
            assert candidate.text in text, f"{candidate.text} hidden when selected"

    def test_selected_candidate_marked(self):
        items = _candidates([("/a", "Session"), ("/b", "Session")])
        text = _render_completion_text(items, 1, 20).plain
        marked = [ln for ln in text.splitlines() if "▸" in ln]
        assert len(marked) == 1
        assert "/b" in marked[0]

    def test_no_indicators_when_everything_fits(self):
        items = _candidates([("/a", "Session"), ("/b", "General")])
        text = _render_completion_text(items, -1, 20).plain
        assert "more" not in text

    def test_lines_never_wrap_on_narrow_terminal(self):
        """A wrapped line would blow the height budget — long lines must
        crop, so visual rows always equal logical rows.

        Rendered via ``Console.render_lines`` (the pure-rich path).
        ``Console.print`` is unsuitable here: it re-joins Text objects
        and drops their no_wrap/overflow.
        """
        from rich.console import Console

        text = _render_completion_text(_long_candidates(), 0, 20)
        console = Console(width=40)
        rendered = console.render_lines(text, console.options, pad=False)
        assert len(rendered) == len(text.plain.splitlines())


def _long_candidates() -> list[CompletionCandidate]:
    """Three candidates whose lines far exceed a 40-column terminal."""
    return [
        CompletionCandidate(
            text=f"/long-command-{i}",
            description="x" * 120,
            replace_start=0,
            replace_end=1,
            category="Session",
        )
        for i in range(3)
    ]


def _completions_css() -> str:
    """Extract the real ``#completions`` CSS block from the TUI module."""
    import inspect
    import re

    import EvoScientist.cli.tui_interactive as tui_mod

    match = re.search(r"#completions \{[^}]*\}", inspect.getsource(tui_mod))
    assert match, "#completions CSS block not found in tui_interactive.py"
    return match.group(0)


class TestCompletionPopupCss:
    """Textual converts rich Text to Content and DROPS its no_wrap and
    overflow attributes, so cropping must be enforced by the widget CSS.
    """

    def test_css_disables_wrapping(self):
        css = _completions_css()
        assert "text-wrap: nowrap" in css
        assert "text-overflow: ellipsis" in css

    async def test_static_crops_long_lines_with_real_css(self):
        """Mount a Static with the real popup CSS and verify visual rows
        equal logical rows on a narrow terminal (no wrapping)."""
        from textual.app import App, ComposeResult
        from textual.widgets import Static

        class PopupApp(App[None]):
            CSS = _completions_css()

            def compose(self) -> ComposeResult:
                yield Static("", id="completions")

        text = _render_completion_text(_long_candidates(), 0, 20)
        logical_rows = len(text.plain.splitlines())

        app = PopupApp()
        async with app.run_test(size=(40, 30)) as pilot:
            widget = app.query_one("#completions", Static)
            widget.display = True
            widget.update(text)
            await pilot.pause()
            assert widget.size.height == logical_rows
            rendered = [
                "".join(seg.text for seg in widget.render_line(y))
                for y in range(widget.size.height)
            ]
            assert any("…" in line for line in rendered)
