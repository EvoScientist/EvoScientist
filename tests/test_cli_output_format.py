"""Tests for the --output-format stream-json CLI wiring.

Covers:
  1. The console singleton is redirected to stderr so stdout stays pure JSONL.
  2. --output-format is validated (stream-json requires -p; bad values rejected).

The dispatch itself runs the sink through ``GraphGateway.stream_events`` from the
single-shot path; that sourcing is exercised by the json_sink unit tests.
"""

import sys

import pytest

from EvoScientist.cli.commands import _resolve_stream_json_auto_mode


@pytest.mark.parametrize(
    ("auto_mode", "output_format", "expected"),
    [
        (None, "stream-json", True),  # default on for headless stream-json
        (None, "text", False),  # text keeps historical off-by-default
        (False, "stream-json", False),  # explicit --no-auto-mode wins
        (True, "text", True),  # explicit --auto-mode wins
    ],
)
def test_resolve_stream_json_auto_mode(auto_mode, output_format, expected):
    """auto-mode defaults on for stream-json when unset; explicit flags always win."""
    assert _resolve_stream_json_auto_mode(auto_mode, output_format) is expected


def _overrides_for(monkeypatch, argv):
    """Invoke the CLI and capture the cli_overrides handed to get_effective_config,
    aborting before the run actually starts."""
    from typer.testing import CliRunner

    import EvoScientist.config as cfg_mod
    from EvoScientist.cli._app import app
    from EvoScientist.stream.console import console

    captured: dict[str, object] = {}

    class _Stop(Exception):
        """Sentinel to short-circuit the callback once overrides are captured."""

    def _grab(overrides):
        """Record the overrides dict, then abort before the heavy run path."""
        captured["overrides"] = dict(overrides)
        raise _Stop

    monkeypatch.setattr(cfg_mod, "get_effective_config", _grab)
    original_console_file = console.file
    try:
        CliRunner().invoke(app, argv, catch_exceptions=True)
    finally:
        console.file = original_console_file
    return captured.get("overrides", {})


def test_stream_json_defaults_auto_mode_on_in_overrides(monkeypatch):
    """stream-json without the flag defaults auto-mode (and auto-approve) on."""
    overrides = _overrides_for(
        monkeypatch, ["-p", "hi", "--output-format", "stream-json"]
    )
    assert overrides.get("auto_mode") is True
    assert overrides.get("auto_approve") is True


def test_no_auto_mode_writes_explicit_false_override(monkeypatch):
    """Explicit --no-auto-mode must write auto_mode=False so it wins over a config
    that enables auto-mode (not silently fall back to the config default)."""
    overrides = _overrides_for(
        monkeypatch,
        ["-p", "hi", "--output-format", "stream-json", "--no-auto-mode"],
    )
    assert overrides.get("auto_mode") is False


def test_redirect_console_to_stderr():
    """redirect_console_to_stderr moves the shared console's output to stderr."""
    from EvoScientist.stream.console import console
    from EvoScientist.stream.json_sink import redirect_console_to_stderr

    original = console.file
    try:
        redirect_console_to_stderr()
        assert console.file is sys.stderr
    finally:
        console.file = original


# ── CLI validation (via typer CliRunner) ──


def _invoke(monkeypatch, argv):
    """Run the CLI app with argv under stubbed config/dispatch, returning
    (dispatch-calls, CliRunner result)."""
    from typer.testing import CliRunner

    import EvoScientist.cli.commands as cmds
    import EvoScientist.cli.interactive as interactive_mod
    import EvoScientist.config as cfg_mod
    from EvoScientist.cli._app import app
    from EvoScientist.config.settings import EvoScientistConfig

    calls: dict[str, object] = {}

    def _fake_config(overrides):
        """Return a minimal config honoring only the ui_backend override."""
        cfg = EvoScientistConfig()
        cfg.ui_backend = overrides.get("ui_backend") or "cli"
        return cfg

    monkeypatch.setattr(cfg_mod, "get_effective_config", _fake_config)
    monkeypatch.setattr(cfg_mod, "apply_config_to_env", lambda cfg: None)
    monkeypatch.setattr(cmds, "ensure_dirs", lambda: None)
    monkeypatch.setattr(cmds, "_ensure_async_subagent_server", lambda *a, **k: None)
    monkeypatch.setattr(
        interactive_mod,
        "cmd_interactive",
        lambda **kw: calls.__setitem__("dispatch", "interactive"),
    )
    monkeypatch.setattr(
        interactive_mod,
        "cmd_run",
        lambda *a, **kw: calls.__setitem__("dispatch", "run"),
    )

    result = CliRunner().invoke(app, argv, catch_exceptions=False)
    return calls, result


def test_stream_json_without_prompt_is_rejected(monkeypatch):
    """stream-json requires -p (single-shot); bare --output-format is rejected."""
    calls, result = _invoke(monkeypatch, ["--output-format", "stream-json"])
    assert result.exit_code != 0
    assert "dispatch" not in calls


def test_invalid_output_format_is_rejected(monkeypatch):
    """An unknown --output-format value is rejected before any dispatch."""
    calls, result = _invoke(monkeypatch, ["-p", "hi", "--output-format", "bogus"])
    assert result.exit_code != 0
    assert "dispatch" not in calls
