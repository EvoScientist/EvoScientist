"""Tests for how the CLI entry point picks the session's workspace and work dir."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from EvoScientist.cli import commands
from EvoScientist.paths import Workspace


def _config(
    *, default_workdir: str = "", default_mode: str = "daemon", ui_backend: str = "cli"
):
    return SimpleNamespace(
        default_workdir=default_workdir,
        default_mode=default_mode,
        anthropic_auth_mode="api_key",
        openai_auth_mode="api_key",
        show_thinking=True,
        channel_send_thinking=True,
        ui_backend=ui_backend,
        model="test-model",
        provider="anthropic",
    )


def _run(
    monkeypatch,
    config,
    *,
    workdir: str | None = None,
    use_cwd: bool = False,
    mode: str | None = None,
    name: str | None = None,
) -> dict:
    """Drive the main callback up to ``cmd_interactive`` and capture its inputs."""
    import EvoScientist.cli.interactive as interactive_mod
    import EvoScientist.config as config_mod

    captured: dict = {}

    def _fake_get_effective_config(cli_overrides=None):
        merged = vars(config).copy()
        merged.update(cli_overrides or {})
        return SimpleNamespace(**merged)

    def _fake_ensure_server(cfg, *, dirs, backend=None):
        captured["server_dirs"] = dirs

    def _fake_cmd_interactive(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(config_mod, "get_effective_config", _fake_get_effective_config)
    monkeypatch.setattr(config_mod, "apply_config_to_env", lambda _cfg: None)
    monkeypatch.setattr(config_mod, "resolve_gateway_backend", lambda *_a: "local")
    monkeypatch.setattr(commands, "_get_cli_async_runtime", lambda _ctx: None)
    order: list[str] = []
    captured["order"] = order
    monkeypatch.setattr(commands, "ensure_dirs", lambda: order.append("ensure_dirs"))
    monkeypatch.setattr(
        commands, "reload_env_dirs", lambda: order.append("reload_env_dirs")
    )
    monkeypatch.setattr(commands, "_ensure_async_subagent_server", _fake_ensure_server)
    monkeypatch.setattr(interactive_mod, "cmd_interactive", _fake_cmd_interactive)

    commands._main_callback(
        SimpleNamespace(invoked_subcommand=None),
        version=None,
        mode=mode,
        name=name,
        prompt=None,
        thread_id=None,
        workdir=workdir,
        use_cwd=use_cwd,
        no_thinking=False,
        auto_approve=False,
        auto_mode=None,
        ask_user=False,
        dangerous=False,
        auth_mode=None,
        ui=None,
        host=None,
        output_format=None,
    )
    return captured


def test_cwd_is_the_default_workspace(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    got = _run(monkeypatch, _config())
    assert got["dirs"].workspace == Workspace(tmp_path)
    assert got["dirs"].work_dir == Workspace(tmp_path).root
    assert got["dirs"].run_dir is None
    assert got["server_dirs"] == got["dirs"]
    assert got["order"] == ["reload_env_dirs", "ensure_dirs"]


def test_symlinked_start_folder_gives_one_spelling(monkeypatch, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    daemon = _run(monkeypatch, _config(default_workdir=str(link)))
    run = _run(monkeypatch, _config(default_workdir=str(link), default_mode="run"))
    assert daemon["dirs"].work_dir == daemon["dirs"].workspace.root
    assert run["dirs"].run_dir.parent == daemon["dirs"].work_dir / "runs"


def test_use_cwd_ignores_default_workdir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    got = _run(
        monkeypatch, _config(default_workdir=str(tmp_path / "cfg")), use_cwd=True
    )
    assert got["dirs"].workspace == Workspace(tmp_path)
    assert got["mode"] is None


def test_workdir_is_created_and_used(monkeypatch, tmp_path):
    target = tmp_path / "proj"
    got = _run(monkeypatch, _config(), workdir=str(target))
    assert target.is_dir()
    assert got["dirs"].workspace == Workspace(target)
    assert got["dirs"].work_dir == Workspace(target).root
    assert got["dirs"].run_dir is None
    assert got["mode"] is None


def test_default_workdir_from_config(monkeypatch, tmp_path):
    target = tmp_path / "cfg"
    target.mkdir()
    monkeypatch.chdir(tmp_path)
    got = _run(monkeypatch, _config(default_workdir=str(target)))
    assert got["dirs"].workspace == Workspace(target)
    assert got["dirs"].work_dir == Workspace(target).root
    assert got["dirs"].run_dir is None
    assert got["mode"] == "daemon"


def test_run_mode_works_in_a_run_folder_of_the_workspace(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    got = _run(monkeypatch, _config(), mode="run", name="exp")
    workspace = got["dirs"].workspace
    assert workspace == Workspace(tmp_path)
    assert got["dirs"].run_dir == workspace.runs_dir / "exp"
    assert got["dirs"].work_dir == got["dirs"].run_dir
    assert got["dirs"].run_dir.is_dir()
    assert got["server_dirs"] == got["dirs"]
    assert got["mode"] == "run"


def test_run_mode_deduplicates_run_names(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs" / "exp").mkdir(parents=True)
    got = _run(monkeypatch, _config(), mode="run", name="exp")
    assert got["dirs"].run_dir == got["dirs"].workspace.runs_dir / "exp_1"


def test_default_mode_run_uses_default_workdir_as_root(monkeypatch, tmp_path):
    target = tmp_path / "cfg"
    target.mkdir()
    got = _run(monkeypatch, _config(default_workdir=str(target), default_mode="run"))
    assert got["dirs"].workspace == Workspace(target)
    assert got["dirs"].run_dir.parent == target.resolve() / "runs"


def test_mode_cannot_combine_with_workdir(monkeypatch, tmp_path):
    import typer

    with pytest.raises(typer.BadParameter):
        _run(monkeypatch, _config(), mode="run", workdir=str(tmp_path))
