"""Tests for the headless standalone channel runner's dev-server startup."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import pytest

import EvoScientist.channels.standalone as standalone
from EvoScientist.paths import Workspace


@pytest.fixture(autouse=True)
def _restore_http_log_levels():
    """Reset process-global httpx/httpcore levels after every test."""
    loggers = [logging.getLogger(n) for n in ("httpx", "httpcore")]
    saved = [lg.level for lg in loggers]
    yield
    for lg, level in zip(loggers, saved, strict=False):
        lg.setLevel(level)


def _patch_manager(monkeypatch, *, gateway_backend, default_workdir):
    """Stub the dev-server spawn + workspace helpers; return (config, ensure_calls).

    ``ensure_calls`` captures each ``ensure_langgraph_dev`` invocation so a test
    can assert whether (and with what workspace) the dev server was ensured.
    """
    import EvoScientist.langgraph_dev.manager as manager_mod
    import EvoScientist.paths as paths_mod

    ensure_calls: list[dict] = []
    monkeypatch.setattr(
        manager_mod,
        "ensure_langgraph_dev",
        lambda cfg, *, workspace_dir, backend=None: ensure_calls.append(
            {"config": cfg, "workspace_dir": workspace_dir, "backend": backend}
        ),
    )
    # Avoid creating the real data dirs.
    monkeypatch.setattr(paths_mod, "ensure_dirs", lambda: None)
    config = SimpleNamespace(
        gateway_backend=gateway_backend,
        default_workdir=default_workdir,
    )
    return config, ensure_calls


def test_ensure_dev_server_spawns_on_server_backend(monkeypatch, tmp_path):
    config, ensure_calls = _patch_manager(
        monkeypatch, gateway_backend="langgraph_server", default_workdir=str(tmp_path)
    )

    standalone._ensure_standalone_dev_server(
        config, workspace_dir=str(tmp_path), backend="langgraph_server"
    )

    assert len(ensure_calls) == 1
    assert ensure_calls[0]["workspace_dir"] == str(tmp_path)
    # The resolved backend is forwarded so the manager spawns in full mode.
    assert ensure_calls[0]["backend"] == "langgraph_server"


def test_ensure_dev_server_noop_on_local_backend(monkeypatch, tmp_path):
    config, ensure_calls = _patch_manager(
        monkeypatch, gateway_backend="local", default_workdir=str(tmp_path)
    )

    standalone._ensure_standalone_dev_server(
        config, workspace_dir=str(tmp_path), backend="local"
    )

    assert ensure_calls == []


def test_run_standalone_dev_server_falls_back_to_cwd(monkeypatch):
    import EvoScientist.config as config_mod

    config, ensure_calls = _patch_manager(
        monkeypatch, gateway_backend="langgraph_server", default_workdir=""
    )
    monkeypatch.setattr(config_mod, "get_effective_config", lambda: config)
    monkeypatch.setattr(standalone.asyncio, "run", lambda coro: coro.close())

    standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert len(ensure_calls) == 1
    assert ensure_calls[0]["workspace_dir"] == os.getcwd()


def test_run_standalone_uses_default_workdir(monkeypatch, tmp_path):
    import EvoScientist.config as config_mod

    config, ensure_calls = _patch_manager(
        monkeypatch, gateway_backend="langgraph_server", default_workdir=str(tmp_path)
    )
    monkeypatch.setattr(config_mod, "get_effective_config", lambda: config)
    seen: dict[str, object] = {}

    def _fake_async_main(*args, workspace, **kwargs):
        seen["workspace"] = workspace
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(standalone, "_async_main", _fake_async_main)
    monkeypatch.setattr(standalone.asyncio, "run", lambda coro: coro.close())

    standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert ensure_calls[0]["workspace_dir"] == str(Workspace(tmp_path).root)
    assert seen["workspace"] == Workspace(tmp_path)


def test_run_standalone_ensures_dev_server_only_with_agent(monkeypatch):
    """``run_standalone`` resolves config and ensures the dev server iff use_agent."""
    import EvoScientist.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "get_effective_config",
        lambda: SimpleNamespace(gateway_backend="local", default_workdir=""),
    )
    ensure_configs: list[object] = []
    monkeypatch.setattr(
        standalone,
        "_ensure_standalone_dev_server",
        lambda cfg, *, workspace_dir, backend=None: ensure_configs.append(cfg),
    )
    monkeypatch.setattr(standalone.asyncio, "run", lambda coro: coro.close())

    standalone.run_standalone(channel=None, bus=None, use_agent=True)
    assert len(ensure_configs) == 1

    ensure_configs.clear()
    standalone.run_standalone(channel=None, bus=None, use_agent=False)
    assert ensure_configs == []


def _standalone_ready(monkeypatch, *, signed_in, **config_fields):
    """Stub everything ``run_standalone`` starts; return (ensure_calls, status_calls)."""
    import EvoScientist.config as config_mod
    from EvoScientist.llm.chatgpt_oauth import ChatGPTLogin

    config, ensure_calls = _patch_manager(
        monkeypatch, gateway_backend="langgraph_server", default_workdir=""
    )
    for name, value in config_fields.items():
        setattr(config, name, value)
    status_calls: list[bool] = []

    def _status():
        status_calls.append(True)
        return ChatGPTLogin(signed_in=signed_in)

    monkeypatch.setattr(config_mod, "get_effective_config", lambda: config)
    monkeypatch.setattr("EvoScientist.llm.chatgpt_oauth.status", _status)
    monkeypatch.setattr(standalone.asyncio, "run", lambda coro: coro.close())
    return ensure_calls, status_calls


def test_run_standalone_without_chatgpt_sign_in_exits_before_anything_starts(
    monkeypatch, caplog
):
    from EvoScientist.llm.chatgpt_oauth import SIGN_IN_HINT

    ensure_calls, _ = _standalone_ready(
        monkeypatch, signed_in=False, openai_auth_mode="oauth"
    )

    with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exc_info:
        standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert exc_info.value.code == 1
    assert [record.getMessage() for record in caplog.records] == [SIGN_IN_HINT]
    assert ensure_calls == []


def test_run_standalone_names_an_unreadable_token_before_the_hint(monkeypatch, caplog):
    from EvoScientist.llm.chatgpt_oauth import SIGN_IN_HINT, ChatGPTLogin

    _standalone_ready(monkeypatch, signed_in=False, openai_auth_mode="oauth")
    monkeypatch.setattr(
        "EvoScientist.llm.chatgpt_oauth.status",
        lambda: ChatGPTLogin(signed_in=False, problem="token file is unreadable"),
    )

    with caplog.at_level("ERROR"), pytest.raises(SystemExit):
        standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert [record.getMessage() for record in caplog.records] == [
        "ChatGPT: token file is unreadable",
        SIGN_IN_HINT,
    ]


def test_run_standalone_starts_when_signed_in(monkeypatch):
    ensure_calls, status_calls = _standalone_ready(
        monkeypatch, signed_in=True, openai_auth_mode="oauth"
    )

    standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert len(ensure_calls) == 1
    assert status_calls == [True]


@pytest.mark.parametrize("fields", [{}, {"openai_auth_mode": "api_key"}])
def test_run_standalone_does_not_look_at_the_sign_in_in_api_key_mode(
    monkeypatch, fields
):
    ensure_calls, status_calls = _standalone_ready(
        monkeypatch, signed_in=False, **fields
    )

    standalone.run_standalone(channel=None, bus=None, use_agent=True)

    assert len(ensure_calls) == 1
    assert status_calls == []


def test_run_standalone_quiets_httpx(monkeypatch):
    """HTTP client logs must not expose channel tokens in request URLs."""
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.DEBUG)

    info_enabled_at_start: dict[str, bool] = {}

    def _run(coro):
        # The channel's first request happens inside asyncio.run.
        for name in ("httpx", "httpcore"):
            info_enabled_at_start[name] = logging.getLogger(name).isEnabledFor(
                logging.INFO
            )
        coro.close()

    monkeypatch.setattr(standalone.asyncio, "run", _run)

    standalone.run_standalone(channel=None, bus=None, use_agent=False)

    assert info_enabled_at_start == {"httpx": False, "httpcore": False}
