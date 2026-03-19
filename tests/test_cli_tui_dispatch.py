"""Tests for CLI interactive UI backend dispatch."""

from EvoScientist.cli.interactive import cmd_interactive


def test_cmd_interactive_dispatches_to_textual(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_resolve_ui_backend(value, *, warn_fallback=False):
        captured["resolved_input"] = value
        captured["warn_fallback"] = warn_fallback
        return "tui"

    def _fake_run_textual_interactive(**kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "EvoScientist.cli.interactive.resolve_ui_backend",
        _fake_resolve_ui_backend,
    )
    monkeypatch.setattr(
        "EvoScientist.cli.interactive.run_textual_interactive",
        _fake_run_textual_interactive,
    )

    cmd_interactive(
        show_thinking=True,
        channel_send_thinking=True,
        workspace_dir="/tmp/workspace",
        workspace_fixed=True,
        mode="daemon",
        model="demo-model",
        provider="demo-provider",
        run_name="demo-run",
        thread_id="thread-1",
        ui_backend="tui",
    )

    assert captured["resolved_input"] == "tui"
    assert captured["warn_fallback"] is True

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["workspace_dir"] == "/tmp/workspace"
    assert kwargs["workspace_fixed"] is True
    assert kwargs["mode"] == "daemon"
    assert kwargs["model"] == "demo-model"
    assert kwargs["provider"] == "demo-provider"
    assert kwargs["run_name"] == "demo-run"
    assert kwargs["thread_id"] == "thread-1"
    assert kwargs["channel_send_thinking"] is True
    assert callable(kwargs["load_agent"])
    assert callable(kwargs["create_session_workspace"])
