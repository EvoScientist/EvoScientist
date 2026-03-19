"""Tests for UI backend runtime selection."""

from dataclasses import dataclass

from EvoScientist.cli.tui_runtime import (
    normalize_ui_backend,
    resolve_ui_backend,
    run_streaming,
)


def test_normalize_ui_backend_defaults_to_cli():
    assert normalize_ui_backend(None) == "cli"
    assert normalize_ui_backend("") == "cli"


def test_normalize_ui_backend_accepts_known_values():
    assert normalize_ui_backend("cli") == "cli"
    assert normalize_ui_backend("tui") == "tui"
    assert normalize_ui_backend("TUI") == "tui"


def test_normalize_ui_backend_maps_legacy_values():
    assert normalize_ui_backend("textual") == "tui"
    assert normalize_ui_backend("Textual") == "tui"
    assert normalize_ui_backend("rich") == "cli"
    assert normalize_ui_backend("Rich") == "cli"


def test_normalize_ui_backend_unknown_falls_back_to_cli():
    assert normalize_ui_backend("unknown-ui") == "cli"


def test_resolve_ui_backend_falls_back_when_textual_unavailable(monkeypatch):
    monkeypatch.setattr(
        "EvoScientist.cli.tui_runtime._has_textual_support", lambda: False
    )
    assert resolve_ui_backend("tui") == "cli"


def test_resolve_ui_backend_keeps_tui_when_available(monkeypatch):
    monkeypatch.setattr(
        "EvoScientist.cli.tui_runtime._has_textual_support", lambda: True
    )
    assert resolve_ui_backend("tui") == "tui"


@dataclass
class _BrokenBackend:
    name: str = "tui"

    def run_streaming(self, **kwargs):
        raise RuntimeError("boom")


def test_run_streaming_falls_back_to_cli_on_runtime_error(monkeypatch):
    monkeypatch.setattr(
        "EvoScientist.cli.tui_runtime.get_backend", lambda *a, **k: _BrokenBackend()
    )

    class _RichStub:
        def run_streaming(self, **kwargs):
            return "fallback-ok"

    monkeypatch.setattr(
        "EvoScientist.cli.tui_runtime.RichStreamingBackend", lambda: _RichStub()
    )

    result = run_streaming(
        ui_backend="tui",
        agent=object(),
        message="hello",
        thread_id="t1",
        show_thinking=False,
        interactive=True,
    )
    assert result == "fallback-ok"
