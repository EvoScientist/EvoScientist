"""Regression tests for issue #301: banner position after /new.

PR #262 ("free-scrolling") replaced per-mount ``scroll_end()`` calls with an
explicit anchor engaged at the end of every stream.  The anchor keeps the
viewport pinned relative to the bottom of the previous content, so when the
user runs ``/new`` after a long conversation, ``clear_chat`` removes every
child while the container is still anchored.  Textual then computes a relative
scroll position against empty content — ``scroll_y`` ends up negative
(observed: ``-7``) and the welcome banner is pushed below the visible
viewport.

The fix in :meth:`EvoTextualInteractiveApp.clear_chat` resets the viewport to
the top after the wipe.  These tests boot the real TUI class via Textual's
pilot so they exercise the actual production code path.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("textual")


# ---------------------------------------------------------------------------
# Module access
#
# ``EvoTextualInteractiveApp`` is defined inside ``run_textual_interactive``,
# so it is not reachable as ``tui_interactive.EvoTextualInteractiveApp``.  We
# grab it by invoking the factory once with a patched ``App.run_async`` that
# captures the freshly-built instance.  The factory must be invoked from a
# fresh top-level event loop (it pulls in nest_asyncio and the global loop),
# so each test boots it via :func:`_capture_app`.
# ---------------------------------------------------------------------------


def _capture_app(monkeypatch) -> object:
    """Build an ``EvoTextualInteractiveApp`` without entering its main loop."""
    from textual.app import App

    from EvoScientist.cli import tui_interactive as tui_mod

    captured: dict = {}

    async def _no_run(self, *args, **kwargs):
        captured["app"] = self

    monkeypatch.setattr(App, "run_async", _no_run)

    fake_load_agent = AsyncMock(return_value=None)

    @asynccontextmanager
    async def _fake_checkpointer(*_a, **_k):
        yield None

    monkeypatch.setattr(
        "EvoScientist.cli.tui_interactive.get_checkpointer",
        _fake_checkpointer,
        raising=False,
    )

    class _FakeSuggester:
        def __init__(self, *_a, **_k):
            pass

    monkeypatch.setattr(
        "EvoScientist.cli.history_suggester.HistorySuggester", _FakeSuggester
    )

    monkeypatch.setattr(
        "EvoScientist.cli.tui_interactive._auto_start_channel",
        lambda *_a, **_k: None,
        raising=False,
    )

    monkeypatch.setattr("EvoScientist.cli.tui_interactive.mode", "dev", raising=False)
    # Note: ``create_session_workspace`` and ``load_agent`` are passed
    # as parameters to ``run_textual_interactive`` (the factory), so the
    # closure inside ``EvoTextualInteractiveApp`` uses the fakes directly
    # and the module-level ``create_session_workspace`` / ``load_agent``
    # symbols never get a chance to run.

    # The factory is synchronous at the outer level — it drives its own loop
    # via ``nest_asyncio`` + ``loop.run_until_complete`` internally.
    try:
        tui_mod.run_textual_interactive(
            show_thinking=False,
            channel_send_thinking=False,
            workspace_dir=None,
            workspace_fixed=False,
            mode="dev",
            model=None,
            provider=None,
            run_name="test-run",
            thread_id=None,
            load_agent=fake_load_agent,
            create_session_workspace=lambda *_a, **_k: str(Path.cwd()),
            config=None,
        )
    except SystemExit:
        pass

    app = captured.get("app")
    if app is None:
        raise RuntimeError(
            "EvoTextualInteractiveApp was not captured — _no_run was never "
            "called, meaning the factory crashed before constructing the app."
        )
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clear_chat_resets_scroll_after_long_anchored_conversation(
    monkeypatch, run_async
):
    """Repro of issue #301: clear after a long anchored stream → banner on top.

    Anchor is engaged (``_anchor_released`` False) at the end of every stream.
    Without the fix, ``clear_chat`` leaves ``scroll_y`` at an invalid value
    (negative or non-zero) because Textual keeps the relative scroll pinned to
    the now-empty bottom of the previous content.
    """

    async def scenario():
        from textual.containers import VerticalScroll
        from textual.widgets import Static

        app = _capture_app(monkeypatch)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            chat = app.query_one("#chat", VerticalScroll)
            welcome = app.query_one("#welcome", Static)

            for i in range(80):
                await chat.mount(Static(f"prior message {i}\n" * 2))
            await pilot.pause()
            chat.scroll_end(animate=False)
            await pilot.pause()
            chat.anchor()
            await pilot.pause()

            assert chat.scroll_y > 0, "precondition: viewport must be scrolled"

            app.clear_chat()
            app._append_system("New session: tid", style="green")
            await pilot.pause()
            await pilot.pause()

            _assert_banner_at_top(
                chat, welcome, label="after /new on long anchored convo"
            )
            assert len(chat.children) == 2

    run_async(scenario())


def test_clear_chat_with_anchor_released_also_resets(monkeypatch, run_async):
    """User scrolled up (anchor released) before /new → still lands at top."""

    async def scenario():
        from textual.containers import VerticalScroll
        from textual.widgets import Static

        app = _capture_app(monkeypatch)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            chat = app.query_one("#chat", VerticalScroll)
            welcome = app.query_one("#welcome", Static)

            for i in range(80):
                await chat.mount(Static(f"msg {i}\n" * 2))
            await pilot.pause()
            chat.anchor()
            chat.scroll_to(y=80, animate=False)
            await pilot.pause()

            app.clear_chat()
            app._append_system("New session: tid", style="green")
            await pilot.pause()
            await pilot.pause()

            _assert_banner_at_top(
                chat, welcome, label="after /new with released anchor"
            )
            assert len(chat.children) == 2

    run_async(scenario())


def test_clear_chat_short_conversation_anchored(monkeypatch, run_async):
    """Even with a short conversation, anchor + clear should not push banner down."""

    async def scenario():
        from textual.containers import VerticalScroll
        from textual.widgets import Static

        app = _capture_app(monkeypatch)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            chat = app.query_one("#chat", VerticalScroll)
            welcome = app.query_one("#welcome", Static)

            # Just enough content to overflow the viewport.
            for i in range(30):
                await chat.mount(Static(f"short msg {i}\n" * 2))
            await pilot.pause()
            chat.scroll_end(animate=False)
            chat.anchor()
            await pilot.pause()

            app.clear_chat()
            app._append_system("New session: tid", style="green")
            await pilot.pause()
            await pilot.pause()

            _assert_banner_at_top(chat, welcome, label="after /new on short convo")

    run_async(scenario())


def test_clear_chat_then_full_user_turn_keeps_banner_at_top(monkeypatch, run_async):
    """Repro of the user-reported scenario: clear → mount welcome banner →
    mount new-session → mount user message → mount assistant reply, in a
    normal-sized terminal where the resulting content fits in the viewport.

    The original ``scroll_home`` fix was defeated by the anchor being
    re-engaged on subsequent mounts (e.g. when ``append_system`` runs in
    ``start_new_session``).  The current fix releases the anchor and forces
    an immediate scroll, so the banner sits at ``scroll_y == 0`` after the
    wipe even when more widgets are mounted afterwards.
    """

    async def scenario():
        from textual.containers import VerticalScroll
        from textual.widgets import Static

        app = _capture_app(monkeypatch)
        # Tall-ish terminal: welcome + a few messages must fit in the
        # viewport, mirroring the user's manual-test setup.
        async with app.run_test(size=(80, 40)) as pilot:
            await pilot.pause()
            chat = app.query_one("#chat", VerticalScroll)
            welcome = app.query_one("#welcome", Static)

            # Long conversation, then /new.
            for i in range(80):
                await chat.mount(Static(f"prior message {i}\n" * 2))
            await pilot.pause()
            chat.scroll_end(animate=False)
            chat.anchor()
            await pilot.pause()

            app.clear_chat()
            # Render the actual banner (not the empty placeholder) and add
            # the /new system message — this is exactly what
            # ``start_new_session`` does after clearing.
            app._render_welcome()
            app._append_system("New session: tid", style="green")
            await pilot.pause()
            await pilot.pause()

            # User types "hello" — _run_turn mounts UserMessage then calls
            # ``container.scroll_end(animate=False)`` (line 1305 in the
            # real code).  In a tall viewport this still lands at scroll_y
            # == 0 because content fits.
            from EvoScientist.cli.widgets.assistant_message import AssistantMessage
            from EvoScientist.cli.widgets.user_message import UserMessage

            await chat.mount(UserMessage("hello"))
            chat.scroll_end(animate=False)
            await pilot.pause()

            await chat.mount(
                AssistantMessage(
                    "Hello. What research problem are we working on today?"
                )
            )
            await pilot.pause()
            await pilot.pause()

            _assert_banner_at_top(
                chat,
                welcome,
                label=(
                    f"after full /new → user msg → reply "
                    f"(max={chat.max_scroll_y}, "
                    f"viewport={chat.scrollable_content_region.height}, "
                    f"content={chat.content_size.height})"
                ),
            )

    run_async(scenario())


def test_short_turn_keeps_banner_at_top_after_layout_refresh(monkeypatch, run_async):
    """Regression for the second symptom of issue #301: after a short
    user/assistant turn that fits in the viewport, end-of-stream
    ``_anchor_chat`` must NOT leave the chat anchored.

    Why: Textual's compositor (``textual._compositor``) writes ``scroll_y``
    via ``set_reactive`` (bypassing the validator) whenever a widget is
    anchored. If the anchored widget's content is shorter than the
    viewport, ``scroll_y`` goes negative on the next layout pass and the
    welcome banner drops below the visible region. The fix in
    ``_anchor_chat`` only engages the anchor when content actually
    overflows (``max_scroll_y > 0``).
    """

    async def scenario():
        from textual.containers import VerticalScroll
        from textual.widgets import Static

        from EvoScientist.cli.widgets.assistant_message import AssistantMessage
        from EvoScientist.cli.widgets.user_message import UserMessage

        app = _capture_app(monkeypatch)
        # Tall terminal: welcome + a short exchange fits with room to spare,
        # which is exactly the bug condition (content < viewport).
        async with app.run_test(size=(80, 40)) as pilot:
            await pilot.pause()
            chat = app.query_one("#chat", VerticalScroll)
            welcome = app.query_one("#welcome", Static)

            await chat.mount(UserMessage("hi"))
            await pilot.pause()
            await chat.mount(
                AssistantMessage("Hi. What are you looking to work on today?")
            )
            await pilot.pause()

            # End-of-stream re-anchor (matches _stream_with_widgets).
            app._anchor_chat(chat)
            await pilot.pause()

            # Any subsequent mount triggers a layout refresh — this is when
            # the compositor would push scroll_y negative without the fix.
            # In production this happens via Markdown re-renders, status-bar
            # updates, the system "usage" line, etc.
            await chat.mount(Static("trailing line\n"))
            await pilot.pause()
            await pilot.pause()

            _assert_banner_at_top(
                chat, welcome, label="after short turn + trailing mount"
            )

    run_async(scenario())


def test_long_turn_keeps_viewport_pinned_to_bottom(monkeypatch, run_async):
    """When the conversation overflows, ``_anchor_chat`` must still engage
    the anchor so streaming output remains visible. The issue #301 fix
    only suppresses anchoring when content fits — long content must
    continue to behave as before.
    """

    async def scenario():
        from textual.containers import VerticalScroll
        from textual.widgets import Static

        app = _capture_app(monkeypatch)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            chat = app.query_one("#chat", VerticalScroll)

            for i in range(50):
                await chat.mount(Static(f"prior message {i}\n" * 2))
            await pilot.pause()

            app._anchor_chat(chat)
            await pilot.pause()
            assert chat.scroll_y == chat.max_scroll_y, (
                "long content must anchor to bottom after _anchor_chat"
            )

            # Trailing mount must keep the viewport pinned to the new bottom.
            await chat.mount(Static("trailing line\n"))
            await pilot.pause()
            await pilot.pause()
            assert chat.scroll_y == chat.max_scroll_y, (
                "anchored viewport must follow new bottom after trailing mount"
            )
            assert chat.scroll_y > 0, "long content must have positive scroll_y"

    run_async(scenario())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_banner_at_top(chat, welcome, label: str = "") -> None:
    """Assert the welcome banner is the first child and visible near the top.

    Textual layout can leave ``scroll_y`` at 0 or 1 depending on runner/font
    metrics when the welcome + a single message fill the viewport edge.  The
    regression we care about is a negative ``scroll_y`` that pushes the banner
    out of view (issue #301), so we bound the value instead of requiring 0.
    """
    suffix = f" ({label})" if label else ""
    assert chat.children[0] is welcome, f"welcome must remain first child{suffix}"
    assert chat.scroll_y >= 0, (
        f"banner must not be pushed above viewport{suffix}, scroll_y={chat.scroll_y}"
    )
    assert chat.scroll_y <= 1, (
        f"banner must stay at top{suffix}, scroll_y={chat.scroll_y}"
    )
