"""Tests for the session-exit resume hint helper."""

from __future__ import annotations

import io

from rich.console import Console

from EvoScientist.cli.resume_hint import print_resume_hint


def _capture(thread_id: str | None) -> str:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=120, color_system=None)
    print_resume_hint(thread_id, console=console)
    return buf.getvalue()


def test_prints_goodbye_and_hint_for_thread_id():
    output = _capture("365cf731")
    assert "Goodbye!" in output
    assert "Resume this session with:" in output
    assert "EvoSci --resume 365cf731" in output


def test_none_thread_id_prints_only_goodbye():
    output = _capture(None)
    assert "Goodbye!" in output
    assert "Resume" not in output


def test_empty_thread_id_prints_only_goodbye():
    output = _capture("")
    assert "Goodbye!" in output
    assert "Resume" not in output
