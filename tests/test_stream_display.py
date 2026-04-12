"""Tests for Rich streaming display helpers."""

from EvoScientist.stream.display import resolve_final_status_footer


def test_resolve_final_status_footer_hides_footer_for_interactive_cli():
    assert resolve_final_status_footer(True, lambda: "footer") is None


def test_resolve_final_status_footer_keeps_footer_for_noninteractive():
    assert resolve_final_status_footer(False, lambda: "footer") == "footer"
