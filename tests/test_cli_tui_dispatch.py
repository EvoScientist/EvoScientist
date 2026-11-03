"""Tests for CLI interactive UI backend dispatch."""

from types import SimpleNamespace

import pytest

from EvoScientist.cli.commands import _is_fresh_interactive_session
from EvoScientist.cli.interactive import cmd_interactive
from EvoScientist.langgraph_dev.registration import ServerExposure
from EvoScientist.paths import SessionDirs, Workspace


@pytest.mark.parametrize(
    ("prompt", "thread_id", "expected"),
    [
        (None, None, True),  # bare `EvoSci` → fresh → WebUI launches
        ("what is 1+1", None, False),  # `-p` one-shot → terminal (Rich CLI)
        (None, "47bcffcd", False),  # `--resume <id>` → terminal (Rich CLI)
        ("hi", "47bcffcd", False),  # both → terminal
        ("", None, True),  # empty `-p` is falsy → treated as fresh
    ],
)
def test_is_fresh_interactive_session(prompt, thread_id, expected):
    """WebUI only launches for a fresh interactive session; `-p` / `--resume`
    fall back to the terminal."""
    assert _is_fresh_interactive_session(prompt, thread_id) is expected


def _invoke_main(monkeypatch, argv):
    """Invoke the EvoSci main callback with ui_backend=webui and all heavy setup
    mocked. Returns (calls, result): calls["dispatch"] is "webui" if run_webui
    ran, or ("cli", <ui_backend>) if cmd_interactive ran."""
    from typer.testing import CliRunner

    import EvoScientist.cli.commands as cmds
    import EvoScientist.cli.interactive as interactive_mod
    import EvoScientist.config as cfg_mod
    import EvoScientist.deploy.webui as webui_mod
    from EvoScientist.cli._app import app
    from EvoScientist.config.settings import EvoScientistConfig

    calls: dict[str, object] = {}

    def _fake_config(overrides):
        cfg = EvoScientistConfig()
        calls["overrides"] = dict(overrides or {})
        # Apply every override the real merge would, so tests can assert on
        # flags (like --host) that reach the config rather than the callback.
        for key, value in (overrides or {}).items():
            setattr(cfg, key, value)
        # Mirror the real --ui override; default to webui for this test.
        cfg.ui_backend = overrides.get("ui_backend") or "webui"
        return cfg

    def _fake_run_webui(config, **_kw):
        calls["dispatch"] = "webui"
        calls["webui_config"] = config

    monkeypatch.setattr(cfg_mod, "get_effective_config", _fake_config)
    monkeypatch.setattr(cfg_mod, "apply_config_to_env", lambda cfg: None)
    monkeypatch.setattr(cmds, "ensure_dirs", lambda: None)
    monkeypatch.setattr(cmds, "_ensure_async_subagent_server", lambda *a, **k: None)
    monkeypatch.setattr(webui_mod, "run_webui", _fake_run_webui)
    monkeypatch.setattr(
        interactive_mod,
        "cmd_interactive",
        lambda **kw: calls.__setitem__("dispatch", ("cli", kw.get("ui_backend"))),
    )

    result = CliRunner().invoke(app, argv, catch_exceptions=False)
    return calls, result


def test_main_callback_launches_webui_for_fresh_session(monkeypatch):
    """Bare `EvoSci` with ui_backend=webui opens the browser app."""
    calls, result = _invoke_main(monkeypatch, [])
    assert result.exit_code == 0
    assert calls.get("dispatch") == "webui"


def test_main_callback_resume_falls_back_to_cli(monkeypatch):
    """`EvoSci --resume <id>` with ui_backend=webui does NOT open the browser;
    it resumes the conversation in the Rich CLI (ui_backend forced to 'cli')."""
    calls, result = _invoke_main(monkeypatch, ["--resume", "abc123"])
    assert result.exit_code == 0
    assert calls.get("dispatch") == ("cli", "cli")


# =============================================================================
# --host override
# =============================================================================


def test_host_flag_drives_both_servers(monkeypatch):
    """One flag, both halves. In WebUI mode the front-end and backend are two
    halves of one surface, so `--host` has to move them together — widening
    only one leaves the UI loading but unable to reach the agent."""
    calls, result = _invoke_main(monkeypatch, ["--host", "0.0.0.0"])

    assert result.exit_code == 0
    cfg = calls["webui_config"]
    assert cfg.webui_host == "0.0.0.0"
    assert cfg.langgraph_dev_host == "0.0.0.0"


def test_host_flag_is_stripped(monkeypatch):
    calls, result = _invoke_main(monkeypatch, ["--host", "  192.168.1.5  "])

    assert result.exit_code == 0
    assert calls["webui_config"].langgraph_dev_host == "192.168.1.5"


def test_blank_host_flag_leaves_config_defaults(monkeypatch):
    """An all-whitespace value must not write an unusable empty host into the
    override dict, where it would beat the config file."""
    calls, result = _invoke_main(monkeypatch, ["--host", "   "])

    assert result.exit_code == 0
    assert "langgraph_dev_host" not in calls["overrides"]
    assert calls["webui_config"].langgraph_dev_host == "127.0.0.1"


def test_no_host_flag_leaves_config_defaults(monkeypatch):
    calls, result = _invoke_main(monkeypatch, [])

    assert result.exit_code == 0
    assert "webui_host" not in calls["overrides"]
    assert calls["webui_config"].webui_host == "127.0.0.1"


def _run_ensure_backend(monkeypatch, config, *, exposure=None):
    """Drive ``_ensure_async_subagent_server`` with a server whose
    registration answered *exposure* (``None``: no server registered the
    project) and capture console output."""
    import EvoScientist.cli.commands as cmds
    from EvoScientist.langgraph_dev import manager

    printed: list[str] = []

    def ensure(config, *, workspace_dir, backend=None):
        manager.SERVER_EXPOSURE = exposure
        return exposure is not None

    monkeypatch.setattr(manager, "ensure_langgraph_dev", ensure)
    monkeypatch.setattr(
        cmds.console, "print", lambda *a, **k: printed.append(str(a[0]) if a else "")
    )
    monkeypatch.setattr(
        cmds.console,
        "status",
        lambda *a, **k: __import__("contextlib").nullcontext(),
    )
    cmds._ensure_async_subagent_server(
        config, dirs=SessionDirs(Workspace("/tmp/workspace"))
    )
    return printed


@pytest.mark.parametrize("exposed", ["0.0.0.0", "192.168.1.5", "::"])
def test_cli_mode_warns_on_public_backend_bind(monkeypatch, exposed):
    """The server is shared by every UI mode and project, so a plain `EvoSci`
    session must warn too, whichever session bound it to the network; the
    session's own host setting says nothing about it."""
    config = SimpleNamespace(langgraph_dev_host="127.0.0.1")
    printed = _run_ensure_backend(
        monkeypatch, config, exposure=ServerExposure.of(exposed, False)
    )

    assert any("PUBLIC BIND" in line for line in printed)


@pytest.mark.parametrize("loopback", ["127.0.0.1", "::1", "localhost"])
def test_cli_mode_silent_on_loopback_backend_bind(monkeypatch, loopback):
    config = SimpleNamespace(langgraph_dev_host=loopback)
    printed = _run_ensure_backend(
        monkeypatch, config, exposure=ServerExposure.of(loopback, False)
    )

    assert not any("PUBLIC" in line for line in printed)


def test_cli_mode_warns_on_tunnel_of_shared_server(monkeypatch):
    """A session in another project that uses a server some EvoSci deploy
    started with a tunnel is exposed through it too."""
    config = SimpleNamespace(langgraph_dev_host="127.0.0.1")
    printed = _run_ensure_backend(
        monkeypatch, config, exposure=ServerExposure.of("127.0.0.1", True)
    )

    assert any("PUBLIC TUNNEL" in line for line in printed)


def test_no_warning_when_project_not_registered(monkeypatch):
    """ensure_langgraph_dev fails soft (async degrades to in-process): a
    server that does not serve the project does not expose it."""
    config = SimpleNamespace(langgraph_dev_host="0.0.0.0")
    printed = _run_ensure_backend(monkeypatch, config, exposure=None)

    assert not any("PUBLIC" in line for line in printed)


@pytest.mark.parametrize("drift", [None, "its agents keep /conda/bin/python"])
def test_cli_prints_the_server_python_warning(monkeypatch, drift):
    from EvoScientist.langgraph_dev import manager

    monkeypatch.setattr(manager, "AGENT_SHELL_DRIFT", drift)
    config = SimpleNamespace(langgraph_dev_host="127.0.0.1")
    printed = _run_ensure_backend(monkeypatch, config, exposure=ServerExposure())

    assert any("/conda/bin/python" in line for line in printed) is (drift is not None)


def test_background_agent_server_starts_even_when_async_subagents_disabled(
    monkeypatch, workspace
):
    import EvoScientist.cli.commands as cmds

    calls = []

    def fake_ensure(config, *, workspace_dir, backend=None):
        calls.append((config, workspace_dir, backend))

    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.ensure_langgraph_dev",
        fake_ensure,
    )

    config = SimpleNamespace(enable_async_subagents=False)
    cmds._ensure_async_subagent_server(
        config, dirs=SessionDirs(workspace), backend="langgraph_server"
    )

    assert calls == [(config, workspace.root, "langgraph_server")]


async def test_resume_workspace_sync_runs_even_when_async_subagents_disabled(
    monkeypatch, workspace
):
    import EvoScientist.cli.commands as cmds

    calls = []

    def fake_ensure(config, *, workspace_dir, backend=None):
        calls.append((config, workspace_dir, backend))

    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.ensure_langgraph_dev",
        fake_ensure,
    )

    run_dir = workspace.root / "runs" / "resumed"
    config = SimpleNamespace(enable_async_subagents=False)
    await cmds._register_project(
        config,
        dirs=SessionDirs(workspace, run_dir),
        backend="langgraph_server",
    )

    # The server serves the project; a run names its run folder itself.
    assert calls == [(config, workspace.root, "langgraph_server")]


def test_cmd_interactive_dispatches_to_textual(monkeypatch, workspace):
    captured: dict[str, object] = {}
    captured_kwargs: list[dict[str, object]] = []
    effective_config = SimpleNamespace(langgraph_dev_port=9999)

    def _fake_resolve_ui_backend(value, *, warn_fallback=False):
        captured["resolved_input"] = value
        captured["warn_fallback"] = warn_fallback
        return "tui"

    def _fake_run_textual_interactive(**kwargs: object):
        captured_kwargs.append(kwargs)

    monkeypatch.setattr(
        "EvoScientist.cli.interactive.resolve_ui_backend",
        _fake_resolve_ui_backend,
    )
    monkeypatch.setattr(
        "EvoScientist.cli.interactive.run_textual_interactive",
        _fake_run_textual_interactive,
    )

    dirs = SessionDirs(workspace)
    cmd_interactive(
        show_thinking=True,
        channel_send_thinking=True,
        mode="daemon",
        model="demo-model",
        provider="demo-provider",
        run_name="demo-run",
        thread_id="thread-1",
        ui_backend="tui",
        config=effective_config,
        dirs=dirs,
    )

    assert captured["resolved_input"] == "tui"
    assert captured["warn_fallback"] is True

    assert len(captured_kwargs) == 1
    kwargs = captured_kwargs[0]
    assert kwargs["dirs"] is dirs
    assert kwargs["mode"] == "daemon"
    assert kwargs["model"] == "demo-model"
    assert kwargs["provider"] == "demo-provider"
    assert kwargs["run_name"] == "demo-run"
    assert kwargs["thread_id"] == "thread-1"
    assert kwargs["config"] is effective_config
    assert kwargs["channel_send_thinking"] is True
    assert callable(kwargs["load_agent"])
    assert callable(kwargs["create_run_dir"])


def test_main_callback_ccproxy_failure_keeps_install_hint(monkeypatch):
    """The hint's ``[oauth]`` extra is text, not markup."""
    import EvoScientist.ccproxy_manager as ccproxy_mod
    import EvoScientist.cli.commands as cmds

    def missing(config, *, lasting):
        raise RuntimeError("Install it with: pip install 'evoscientist[oauth]'")

    monkeypatch.setattr(cmds, "uses_oauth", lambda _config: True)
    monkeypatch.setattr(ccproxy_mod, "maybe_start_ccproxy", missing)

    _calls, result = _invoke_main(monkeypatch, [])

    assert result.exit_code == 1
    assert "evoscientist[oauth]" in result.output


@pytest.mark.parametrize(
    ("argv", "noted_for"),
    [([], ("langgraph_server", True)), (["--resume", "abc123"], ("local", False))],
    ids=["webui", "terminal"],
)
def test_main_callback_says_where_dangerous_applies(monkeypatch, argv, noted_for):
    """The WebUI's turns run in the background server; the terminal's on the
    session's own gateway."""
    import EvoScientist.cli.commands as cmds

    noted: list = []
    monkeypatch.setattr(
        cmds,
        "_print_dangerous_flag_note",
        lambda backend, *, webui=False: noted.append((backend, webui)),
    )

    _calls, result = _invoke_main(monkeypatch, argv)

    assert result.exit_code == 0
    assert noted == [noted_for]
