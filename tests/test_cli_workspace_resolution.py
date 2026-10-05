"""Tests for how the CLI entry point picks the session's workspace and work dir."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from EvoScientist.cli import commands
from EvoScientist.paths import SessionDirs, Workspace
from tests.fakes import FakeGraphGateway, FakeThreadStore


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


def test_cwd_is_default_workspace(monkeypatch, tmp_path):
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


def test_run_mode_uses_run_dir_under_workspace(monkeypatch, tmp_path):
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


def _run_prompt(
    monkeypatch,
    config,
    *,
    thread_id: str | None,
    stored: dict | None,
    backend: str = "local",
) -> dict:
    """Drive the main callback through a ``-p`` run and capture its folders."""
    import EvoScientist.cli.interactive as interactive_mod
    import EvoScientist.config as config_mod
    import EvoScientist.gateway as gateway_mod
    import EvoScientist.sessions as sessions_mod

    captured: dict = {"synced": [], "prespawned": []}
    gateway = FakeGraphGateway(
        generated_thread_ids=["new-thread"],
        thread_store=FakeThreadStore(resolved_thread_id=thread_id, metadata=stored),
    )

    def _fake_get_effective_config(cli_overrides=None):
        merged = vars(config).copy()
        merged.update(cli_overrides or {})
        return SimpleNamespace(**merged)

    @asynccontextmanager
    async def _fake_checkpointer():
        yield None

    async def _fake_sync(_config, *, dirs, backend=None, **_kwargs):
        captured["synced"].append(dirs)

    def _fake_load_agent(**kwargs):
        captured["agent_workspace"] = kwargs["workspace"]
        captured["agent_work_dir"] = kwargs["work_dir"]
        return object()

    def _fake_cmd_run(_agent, _prompt, *, thread_id, dirs, **_kwargs):
        captured["thread_id"] = thread_id
        captured["run_dirs"] = dirs

    monkeypatch.setattr(config_mod, "get_effective_config", _fake_get_effective_config)
    monkeypatch.setattr(config_mod, "apply_config_to_env", lambda _cfg: None)
    monkeypatch.setattr(config_mod, "resolve_gateway_backend", lambda *_a: backend)
    monkeypatch.setattr(
        commands,
        "_get_cli_async_runtime",
        lambda _ctx: SimpleNamespace(run_sync=lambda fn: asyncio.run(fn())),
    )
    monkeypatch.setattr(commands, "ensure_dirs", lambda: None)
    monkeypatch.setattr(
        commands,
        "_ensure_async_subagent_server",
        lambda _config, *, dirs, backend=None: captured["prespawned"].append(dirs),
    )
    monkeypatch.setattr(commands, "_sync_background_agent_server_workspace", _fake_sync)
    monkeypatch.setattr(commands, "_load_agent", _fake_load_agent)
    monkeypatch.setattr(
        gateway_mod,
        "create_runtime_gateways_for_config",
        lambda *_a, **_k: SimpleNamespace(graph_gateway=gateway),
    )
    monkeypatch.setattr(sessions_mod, "get_checkpointer", _fake_checkpointer)
    monkeypatch.setattr(interactive_mod, "cmd_run", _fake_cmd_run)

    commands._main_callback(
        SimpleNamespace(invoked_subcommand=None),
        version=None,
        mode=None,
        name=None,
        prompt="hello",
        thread_id=thread_id,
        workdir=None,
        use_cwd=False,
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


def test_one_shot_resume_uses_thread_workspace(monkeypatch, tmp_path):
    launch, other = tmp_path / "launch", tmp_path / "other"
    launch.mkdir()
    monkeypatch.chdir(launch)

    got = _run_prompt(
        monkeypatch,
        _config(),
        thread_id="t-other",
        stored={"workspace_dir": other.as_posix()},
    )

    restored = SessionDirs(Workspace(other))
    assert got["thread_id"] == "t-other"
    assert got["agent_workspace"] == Workspace(other)
    assert Path(got["agent_work_dir"]) == other.resolve()
    assert got["run_dirs"] == restored
    assert got["synced"] == [restored]
    # The server is started once, for the thread's folders.
    assert got["prespawned"] == []


def test_one_shot_resume_on_server_backend_starts_server_first(monkeypatch, tmp_path):
    """The server gateway backend is built against a running server."""
    launch, other = tmp_path / "launch", tmp_path / "other"
    launch.mkdir()
    monkeypatch.chdir(launch)

    got = _run_prompt(
        monkeypatch,
        _config(),
        thread_id="t-other",
        stored={"workspace_dir": other.as_posix()},
        backend="langgraph_server",
    )

    assert got["prespawned"] == [SessionDirs(Workspace(launch))]
    assert got["synced"] == [SessionDirs(Workspace(other))]


def test_one_shot_resume_uses_thread_run_dir(monkeypatch, tmp_path):
    root, launch = tmp_path / "proj", tmp_path / "launch"
    run = root / "runs" / "exp"
    launch.mkdir()
    monkeypatch.chdir(launch)

    got = _run_prompt(
        monkeypatch,
        _config(),
        thread_id="t-run",
        stored={"workspace_dir": root.as_posix(), "run_dir": run.as_posix()},
    )

    assert got["agent_workspace"] == Workspace(root)
    assert Path(got["agent_work_dir"]) == run.resolve()
    assert got["run_dirs"] == SessionDirs(Workspace(root), run)


def test_one_shot_resume_without_stored_dirs_keeps_launch_dirs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    got = _run_prompt(monkeypatch, _config(), thread_id="t-bare", stored=None)
    assert got["agent_workspace"] == Workspace(tmp_path)
    assert got["run_dirs"] == SessionDirs(Workspace(tmp_path))


def test_one_shot_new_thread_keeps_launch_dirs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    got = _run_prompt(monkeypatch, _config(), thread_id=None, stored=None)
    assert got["thread_id"] == "new-thread"
    assert got["run_dirs"] == SessionDirs(Workspace(tmp_path))
    assert got["prespawned"] == [SessionDirs(Workspace(tmp_path))]
    assert got["synced"] == []


def test_webui_mode_run_works_in_root(monkeypatch, tmp_path):
    """Run mode is a CLI/TUI feature: the WebUI starts in the root, no run folder."""
    import EvoScientist.deploy.webui as webui_mod

    monkeypatch.chdir(tmp_path)
    launched: list[str] = []
    monkeypatch.setattr(
        webui_mod,
        "run_webui",
        lambda _config, *, workspace_dir: launched.append(workspace_dir),
    )

    _run(monkeypatch, _config(ui_backend="webui"), mode="run")

    assert launched == [str(tmp_path.resolve())]
    assert not (tmp_path / "runs").exists()
