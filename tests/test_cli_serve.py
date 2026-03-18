"""Tests for CLI serve/channel glue behavior."""

from __future__ import annotations

from types import SimpleNamespace

from EvoScientist.cli import commands


def _make_config(
    *,
    default_workdir: str = "",
    channel_send_thinking: bool = True,
):
    return SimpleNamespace(
        channel_enabled="telegram",
        default_workdir=default_workdir,
        channel_send_thinking=channel_send_thinking,
        provider="anthropic",
        anthropic_auth_mode="api_key",
        openai_auth_mode="api_key",
    )


def _run_serve_once(
    monkeypatch,
    config,
    *,
    workdir: str | None = None,
    no_thinking: bool = False,
    cwd: str | None = None,
):
    import EvoScientist.config as config_mod

    order: list[tuple[str, str | None]] = []
    captured: dict[str, object] = {}

    def _fake_set_workspace_root(path):
        order.append(("set_workspace_root", str(path)))

    def _fake_ensure_dirs():
        order.append(("ensure_dirs", None))

    def _fake_load_agent(workspace_dir=None, checkpointer=None, config=None):
        captured["workspace_dir"] = workspace_dir
        return object()

    def _fake_start_channels_bus_mode(cfg, agent, thread_id, *, send_thinking=None):
        captured["started"] = True
        captured["send_thinking"] = send_thinking
        captured["thread_id"] = thread_id

    def _fake_channels_stop():
        captured["stopped"] = True

    class _InterruptQueue:
        """A fake queue whose get() immediately raises KeyboardInterrupt."""

        def get(self, timeout=None):
            raise KeyboardInterrupt()

    monkeypatch.setattr(commands, "set_workspace_root", _fake_set_workspace_root)
    monkeypatch.setattr(commands, "ensure_dirs", _fake_ensure_dirs)
    monkeypatch.setattr(commands, "_load_agent", _fake_load_agent)
    monkeypatch.setattr(
        commands, "_start_channels_bus_mode", _fake_start_channels_bus_mode
    )
    monkeypatch.setattr(commands, "_channels_stop", _fake_channels_stop)
    monkeypatch.setattr(commands, "_message_queue", _InterruptQueue())

    monkeypatch.setattr(config_mod, "get_effective_config", lambda *_a, **_k: config)
    monkeypatch.setattr(config_mod, "apply_config_to_env", lambda _cfg: None)

    if cwd is not None:
        monkeypatch.setattr(commands.os, "getcwd", lambda: cwd)

    commands.serve(no_thinking=no_thinking, workdir=workdir)
    return order, captured


def test_serve_workdir_has_highest_priority_and_sets_root_before_ensure(
    monkeypatch, tmp_path
):
    cfg_ws = tmp_path / "cfg_ws"
    cli_ws = tmp_path / "cli_ws"
    config = _make_config(default_workdir=str(cfg_ws), channel_send_thinking=True)

    order, captured = _run_serve_once(
        monkeypatch,
        config,
        workdir=str(cli_ws),
    )

    expected = str(cli_ws.resolve())
    assert captured["workspace_dir"] == expected
    assert any(step == ("set_workspace_root", expected) for step in order)
    set_idx = next(i for i, step in enumerate(order) if step[0] == "set_workspace_root")
    ensure_idx = next(i for i, step in enumerate(order) if step[0] == "ensure_dirs")
    assert set_idx < ensure_idx


def test_serve_uses_config_default_workdir_when_no_cli_workdir(monkeypatch, tmp_path):
    cfg_ws = tmp_path / "cfg_ws"
    config = _make_config(default_workdir=str(cfg_ws), channel_send_thinking=True)

    order, captured = _run_serve_once(monkeypatch, config)

    expected = str(cfg_ws.resolve())
    assert captured["workspace_dir"] == expected
    assert ("set_workspace_root", expected) in order


def test_serve_uses_cwd_when_no_workdir_config(monkeypatch, tmp_path):
    cwd = str(tmp_path.resolve())
    config = _make_config(default_workdir="", channel_send_thinking=True)

    order, captured = _run_serve_once(
        monkeypatch,
        config,
        cwd=cwd,
    )

    assert captured["workspace_dir"] == cwd
    assert ("set_workspace_root", cwd) in order


def test_serve_channel_thinking_respects_config_and_no_thinking(monkeypatch, tmp_path):
    ws = str((tmp_path / "ws").resolve())

    _, captured_default = _run_serve_once(
        monkeypatch,
        _make_config(default_workdir=ws, channel_send_thinking=True),
    )
    assert captured_default["send_thinking"] is True

    _, captured_cfg_off = _run_serve_once(
        monkeypatch,
        _make_config(default_workdir=ws, channel_send_thinking=False),
    )
    assert captured_cfg_off["send_thinking"] is False

    _, captured_cli_off = _run_serve_once(
        monkeypatch,
        _make_config(default_workdir=ws, channel_send_thinking=True),
        no_thinking=True,
    )
    assert captured_cli_off["send_thinking"] is False
