"""Tests for ``EvoSci deploy`` command flow.

Verifies the orchestration:
- workspace resolution (CLI > config > cwd)
- port resolution (CLI > config > default)
- host resolution (CLI > config > default) + the public-bind warning
- the refusal of a running server or an occupied port
- no ccproxy of its own (the server starts it)
- the start ``ensure_server`` makes
- clean shutdown via signal handler / KeyboardInterrupt
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import typer

from EvoScientist.deploy import server as deploy_server
from tests.fakes.langgraph_dev import Server


@pytest.fixture(autouse=True)
def _isolated_runtime(runtime_paths):
    """The server's files in the test's server home."""


def _make_config(
    *,
    default_workdir: str = "",
    langgraph_dev_port: int = 6174,
    langgraph_dev_host: str = "127.0.0.1",
    anthropic_auth_mode: str = "api_key",
    openai_auth_mode: str = "api_key",
    log_level: str = "warning",
    langgraph_dev_jobs_per_worker: int = 10,
    langgraph_dev_file_persistence: bool = True,
    dangerous_mode: bool = False,
):
    return SimpleNamespace(
        default_workdir=default_workdir,
        langgraph_dev_port=langgraph_dev_port,
        langgraph_dev_host=langgraph_dev_host,
        anthropic_auth_mode=anthropic_auth_mode,
        openai_auth_mode=openai_auth_mode,
        log_level=log_level,
        langgraph_dev_jobs_per_worker=langgraph_dev_jobs_per_worker,
        langgraph_dev_file_persistence=langgraph_dev_file_persistence,
        dangerous_mode=dangerous_mode,
        provider="anthropic",
        auxiliary_provider="",
        model_fallbacks="",
        ccproxy_port=8000,
    )


def _run_deploy_once(
    monkeypatch,
    config,
    *,
    workdir: str | None = None,
    port: int | None = None,
    host: str | None = None,
    debug: bool = False,
    cwd: str | None = None,
    port_occupied: bool = False,
    langgraph_dev_running: bool = True,
    tunnel: bool = False,
    tunnel_url: str | None = None,
    tunnel_error: str | None = None,
    printed: list | None = None,
    server: Server | None = None,
    on_wait=None,
    on_stop=None,
):
    """Run ``deploy()`` end-to-end with all external dependencies mocked.
    *port_occupied*: something holds the port, this data folder's server
    when *langgraph_dev_running*, another program otherwise. *server* is the
    fake the start goes through (a fresh one by default). Serving ends with
    a Ctrl+C, after *on_wait* runs; *on_stop* runs inside the server's stop.
    Returns a ``captured`` dict with observation points; *printed* collects
    the console output, also of a run that exits."""
    import EvoScientist.config as config_mod

    captured: dict[str, Any] = {
        "ccproxy_started": False,
        "langgraph_dev_started": False,
        "langgraph_dev_stopped": False,
        "workspace_passed": None,
        "port_passed": None,
        "host_passed": None,
        "printed": [] if printed is None else printed,
        "stops": [],
    }

    def _fake_get_effective_config(cli_overrides=None):
        captured["cli_overrides"] = dict(cli_overrides or {})
        merged = vars(config).copy()
        merged.update(cli_overrides or {})
        return SimpleNamespace(**merged)

    monkeypatch.setattr(config_mod, "get_effective_config", _fake_get_effective_config)
    monkeypatch.setattr(config_mod, "apply_config_to_env", lambda _cfg: None)

    monkeypatch.setattr(deploy_server, "console", _SilentConsole(captured["printed"]))

    # langgraph_dev.manager mocks
    from EvoScientist.langgraph_dev import manager as lgm

    server = server or Server(monkeypatch, pid=99999)
    captured["server"] = server
    # For a test whose deploy() raises.
    _run_deploy_once.captured = captured
    effective_port = port if port is not None else config.langgraph_dev_port
    if port_occupied and langgraph_dev_running:
        server.run(4242, port=effective_port, lasting=True)
    elif port_occupied:
        server.occupied.add(effective_port)

    def _fake_stop_langgraph_dev(proc=None, *, clean=True):
        captured["langgraph_dev_stopped"] = True
        captured["stops"].append(
            {
                "proc": proc,
                "clean": clean,
                "handlers": dict(captured["signal_handlers"]),
            }
        )
        if on_stop is not None:
            on_stop()

    monkeypatch.setattr(lgm, "stop_langgraph_dev", _fake_stop_langgraph_dev)
    from EvoScientist.langgraph_dev import registration

    # The started server's answer: the options it started with.
    def _fake_register_folder(workspace, **_k):
        captured["workspace_passed"] = str(workspace.root)
        spawn = server.spawns[0]
        return registration.ServerExposure.of(spawn["host"], spawn["tunnel"])

    monkeypatch.setattr(registration, "register_folder", _fake_register_folder)

    # Never poll a real log for the tunnel URL in tests.
    def _fake_read_tunnel_url(*_a, **_k):
        if tunnel_error is not None:
            raise lgm.TunnelFailedError(tunnel_error)
        return tunnel_url

    monkeypatch.setattr(lgm, "read_tunnel_url", _fake_read_tunnel_url, raising=False)

    # A ccproxy start of its own would show here.
    from EvoScientist import ccproxy_manager as ccp

    def _fake_spawn_ccproxy(*_a, **_kw):
        captured["ccproxy_started"] = True

    monkeypatch.setattr(ccp, "spawn_ccproxy", _fake_spawn_ccproxy)
    # The server's ccproxy answers.
    monkeypatch.setattr(ccp, "is_ccproxy_running", lambda port: True)

    # signal mock: a table of the handlers deploy() installs, so tests can
    # read what was in place at any point without touching the real ones.
    import signal

    handlers: dict = {}
    captured["signal_handlers"] = handlers
    captured["installed"] = []

    def _fake_signal(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        captured["installed"].append((signum, handler))
        return previous

    monkeypatch.setattr(signal, "signal", _fake_signal)

    # Serving ends as a Ctrl+C would end it, unless the test says otherwise.
    def _fake_wait_for_stop():
        captured["handlers_while_serving"] = dict(handlers)
        if on_wait is not None:
            on_wait()
        raise KeyboardInterrupt

    monkeypatch.setattr(deploy_server, "_wait_for_stop", _fake_wait_for_stop)

    # os.makedirs / os.getcwd
    import os

    # Recorded on the helper too, so a test whose deploy() exits can read it.
    made_dirs: list[str] = []
    captured["made_dirs"] = made_dirs
    _run_deploy_once.made_dirs = made_dirs
    _run_deploy_once.printed = captured["printed"]
    monkeypatch.setattr(os, "makedirs", lambda path, *a, **k: made_dirs.append(path))
    if cwd is not None:
        monkeypatch.setattr(os, "getcwd", lambda: cwd)

    try:
        deploy_server.deploy(
            workdir=workdir, port=port, host=host, debug=debug, tunnel=tunnel
        )
    finally:
        if server.spawns:
            spawn = server.spawns[0]
            captured["langgraph_dev_started"] = True
            captured["lasting"] = spawn["lasting"]
            captured["allowed_origins_passed"] = spawn["allowed_origins"]
            captured["port_passed"] = spawn["port"]
            captured["host_passed"] = spawn["host"]
            captured["jobs_per_worker_passed"] = spawn["jobs_per_worker"]
            captured["file_persistence_passed"] = spawn["file_persistence"]
            captured["tunnel_passed"] = spawn["tunnel"]
    return captured


class _SilentConsole:
    """Stand-in for the Rich console — swallows all output so test runs
    don't spew ANSI to the captured pytest output (but doesn't break the
    code paths that call ``console.print`` / ``console.status``).

    Optionally records what was printed so tests can assert on banners
    (e.g. the public-bind warning) without letting them reach the terminal.
    """

    def __init__(self, sink: list | None = None):
        self._sink = sink

    def print(self, *args, **kwargs):
        if self._sink is not None:
            # A panel records the text inside it.
            self._sink.append(" ".join(str(getattr(a, "renderable", a)) for a in args))

    def status(self, *args, **kwargs):
        sink = self._sink

        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def update(self_inner, text):
                if sink is not None:
                    sink.append(f"status: {text}")

        return _Ctx()


# =============================================================================
# Tests
# =============================================================================


def test_deploy_tunnel_default_off(monkeypatch, tmp_path):
    """Without ``--tunnel``, the server starts without one."""
    config = _make_config(default_workdir=str(tmp_path))
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["tunnel_passed"] is False


def test_deploy_tunnel_flag_passed_through(monkeypatch, tmp_path):
    """``--tunnel`` reaches the server's start."""
    config = _make_config(default_workdir=str(tmp_path))
    captured = _run_deploy_once(
        monkeypatch,
        config,
        tunnel=True,
        tunnel_url="https://demo-xyz.trycloudflare.com",
    )

    assert captured["tunnel_passed"] is True
    assert any("PUBLIC TUNNEL" in line for line in captured["printed"])


def test_deploy_reports_a_tunnel_failure(monkeypatch, tmp_path):
    """A failure cloudflared logged is reported with its reason, not as a URL
    the user should look for in the log."""
    config = _make_config(default_workdir=str(tmp_path))
    captured = _run_deploy_once(
        monkeypatch,
        config,
        tunnel=True,
        tunnel_error="failed to request quick Tunnel: i/o timeout",
    )

    printed = "\n".join(captured["printed"])
    assert (
        "Cloudflare quick tunnel failed: failed to request quick Tunnel: i/o timeout"
        in printed
    )
    assert "tunnel up" not in printed
    assert "not detected" not in printed


def test_deploy_reports_a_tunnel_url_that_never_came(monkeypatch, tmp_path):
    config = _make_config(default_workdir=str(tmp_path))
    captured = _run_deploy_once(monkeypatch, config, tunnel=True)

    printed = "\n".join(captured["printed"])
    assert "Tunnel URL not detected" in printed
    assert "quick tunnel failed" not in printed


def test_deploy_workdir_cli_arg_beats_config(monkeypatch, tmp_path):
    cli_ws = tmp_path / "cli_ws"
    cfg_ws = tmp_path / "cfg_ws"
    config = _make_config(default_workdir=str(cfg_ws))
    captured = _run_deploy_once(monkeypatch, config, workdir=str(cli_ws))

    assert captured["workspace_passed"] == str(cli_ws)


def test_deploy_workdir_config_beats_cwd(monkeypatch, tmp_path):
    cfg_ws = tmp_path / "cfg_ws"
    config = _make_config(default_workdir=str(cfg_ws))
    captured = _run_deploy_once(monkeypatch, config, cwd="/tmp/should_not_be_used")

    assert captured["workspace_passed"] == str(cfg_ws)


def test_deploy_workdir_falls_back_to_cwd(monkeypatch, tmp_path):
    config = _make_config(default_workdir="")
    cwd = str(tmp_path / "cwd")
    captured = _run_deploy_once(monkeypatch, config, cwd=cwd)

    assert captured["workspace_passed"] == cwd


def test_deploy_port_cli_arg_beats_config(monkeypatch, tmp_path):
    config = _make_config(default_workdir=str(tmp_path), langgraph_dev_port=6174)
    captured = _run_deploy_once(monkeypatch, config, port=7000)

    assert captured["port_passed"] == 7000


def test_deploy_port_defaults_to_config(monkeypatch, tmp_path):
    config = _make_config(default_workdir=str(tmp_path), langgraph_dev_port=6543)
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["port_passed"] == 6543


def test_deploy_host_cli_arg_beats_config(monkeypatch, tmp_path):
    config = _make_config(default_workdir=str(tmp_path), langgraph_dev_host="127.0.0.1")
    captured = _run_deploy_once(monkeypatch, config, host="192.168.1.5")

    assert captured["host_passed"] == "192.168.1.5"


def test_deploy_host_defaults_to_config(monkeypatch, tmp_path):
    config = _make_config(
        default_workdir=str(tmp_path), langgraph_dev_host="192.168.1.5"
    )
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["host_passed"] == "192.168.1.5"


def test_deploy_passes_backend_allowed_origins(monkeypatch, tmp_path):
    config = _make_config(default_workdir=str(tmp_path))
    config.backend_allowed_origins = "https://smith.langchain.com"
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["allowed_origins_passed"] == "https://smith.langchain.com"


def test_deploy_host_falls_back_when_config_lacks_field(monkeypatch, tmp_path):
    """A config object missing the field entirely still resolves to the
    module default rather than passing None down to socket.bind()."""
    config = _make_config(default_workdir=str(tmp_path))
    del config.langgraph_dev_host
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["host_passed"] == "127.0.0.1"


def test_deploy_blank_host_keeps_config_value(monkeypatch, tmp_path):
    """``--host "  "`` means "not passed" (as in serve), so it must not discard
    the configured bind."""
    config = _make_config(
        default_workdir=str(tmp_path), langgraph_dev_host="192.168.1.5"
    )
    captured = _run_deploy_once(monkeypatch, config, host="   ")

    assert captured["host_passed"] == "192.168.1.5"


def test_deploy_blank_host_and_blank_config_use_default(monkeypatch, tmp_path):
    """Whitespace on both sides would reach socket.bind() as an empty string
    and raise an opaque gaierror; it degrades to the default instead."""
    config = _make_config(default_workdir=str(tmp_path), langgraph_dev_host="  ")
    captured = _run_deploy_once(monkeypatch, config, host="   ")

    assert captured["host_passed"] == "127.0.0.1"


def test_deploy_host_public_opt_in(monkeypatch, tmp_path):
    """``--host 0.0.0.0`` must actually widen the bind even though the config
    default keeps the unauthenticated backend on loopback."""
    config = _make_config(default_workdir=str(tmp_path), langgraph_dev_host="127.0.0.1")
    captured = _run_deploy_once(monkeypatch, config, host="0.0.0.0")

    assert captured["host_passed"] == "0.0.0.0"


def test_deploy_host_is_stripped(monkeypatch, tmp_path):
    config = _make_config(default_workdir=str(tmp_path))
    captured = _run_deploy_once(monkeypatch, config, host="  0.0.0.0  ")

    assert captured["host_passed"] == "0.0.0.0"


@pytest.mark.parametrize(
    ("host", "port", "overrides"),
    [
        (None, None, {}),
        (
            "  0.0.0.0  ",
            6190,
            {"langgraph_dev_host": "0.0.0.0", "langgraph_dev_port": 6190},
        ),
        ("   ", None, {}),
    ],
)
def test_deploy_bind_flags_are_config_overrides(
    monkeypatch, tmp_path, host, port, overrides
):
    """As ``EvoSci --host`` passes it, so the config fingerprint recorded for
    the server matches a session started with the same bind."""
    config = _make_config(default_workdir=str(tmp_path))
    captured = _run_deploy_once(monkeypatch, config, host=host, port=port)

    assert captured["cli_overrides"] == overrides


def test_deploy_config_host_is_stripped(monkeypatch, tmp_path):
    """Stripping must not depend on the value arriving via --host.
    ``EvoScientistConfig.__post_init__`` normalizes these fields, but deploy()
    reads through ``getattr`` and is handed duck-typed config objects that
    never run it — an unstripped value would reach socket.bind()."""
    config = _make_config(
        default_workdir=str(tmp_path), langgraph_dev_host="  192.168.1.5  "
    )
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["host_passed"] == "192.168.1.5"


def test_deploy_whitespace_config_host_collapses_to_default(monkeypatch, tmp_path):
    config = _make_config(default_workdir=str(tmp_path), langgraph_dev_host="   ")
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["host_passed"] == "127.0.0.1"


def test_deploy_padded_loopback_config_host_suppresses_warning(monkeypatch, tmp_path):
    """Consequence of the unstripped path: `is_loopback_host` would not match
    " 127.0.0.1 ", so a padded loopback config would print a false PUBLIC BIND
    warning while binding a value socket.bind() rejects outright."""
    config = _make_config(
        default_workdir=str(tmp_path), langgraph_dev_host=" 127.0.0.1 "
    )
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["host_passed"] == "127.0.0.1"
    assert not any("PUBLIC BIND" in line for line in captured["printed"])


@pytest.mark.parametrize("exposed_host", ["0.0.0.0", "192.168.1.5", "::"])
def test_deploy_warns_on_public_bind(monkeypatch, tmp_path, exposed_host):
    config = _make_config(default_workdir=str(tmp_path))
    captured = _run_deploy_once(monkeypatch, config, host=exposed_host)

    assert any("PUBLIC BIND" in line for line in captured["printed"]), (
        f"binding {exposed_host} reaches other machines and MUST warn"
    )


@pytest.mark.parametrize("loopback_host", ["127.0.0.1", "::1", "localhost"])
def test_deploy_no_warning_on_loopback_bind(monkeypatch, tmp_path, loopback_host):
    config = _make_config(default_workdir=str(tmp_path))
    captured = _run_deploy_once(monkeypatch, config, host=loopback_host)

    assert not any("PUBLIC BIND" in line for line in captured["printed"]), (
        f"{loopback_host} is unreachable off-box — warning would be noise"
    )


@pytest.mark.parametrize("bad_port", [0, -1, 70000])
def test_deploy_refuses_invalid_port(monkeypatch, tmp_path, bad_port):
    """CLI must reject out-of-range ports (port=0 was the original silent-fail
    case: ``port or default`` treated 0 as falsy)."""
    config = _make_config(default_workdir=str(tmp_path))
    with pytest.raises(typer.Exit) as exc:
        _run_deploy_once(monkeypatch, config, port=bad_port)
    assert exc.value.exit_code == 1


def test_deploy_starts_no_ccproxy_with_anthropic_oauth(monkeypatch, tmp_path):
    """The server it starts runs its own."""
    config = _make_config(default_workdir=str(tmp_path), anthropic_auth_mode="oauth")
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["ccproxy_started"] is False
    assert captured["langgraph_dev_started"] is True


def test_deploy_needs_no_ccproxy_for_openai_oauth(monkeypatch, tmp_path):
    from EvoScientist.llm.chatgpt_oauth import ChatGPTLogin

    monkeypatch.setattr(
        "EvoScientist.llm.chatgpt_oauth.status",
        lambda: ChatGPTLogin(signed_in=True, plan="plus"),
    )
    config = _make_config(default_workdir=str(tmp_path), openai_auth_mode="oauth")
    captured = _run_deploy_once(monkeypatch, config)

    assert captured["ccproxy_started"] is False
    assert captured["langgraph_dev_started"] is True


def test_deploy_refuses_openai_oauth_without_sign_in(monkeypatch, tmp_path):
    from EvoScientist.llm.chatgpt_oauth import ChatGPTLogin

    monkeypatch.setattr(
        "EvoScientist.llm.chatgpt_oauth.status",
        lambda: ChatGPTLogin(signed_in=False),
    )
    config = _make_config(default_workdir=str(tmp_path), openai_auth_mode="oauth")

    with pytest.raises(typer.Exit) as exc_info:
        _run_deploy_once(monkeypatch, config)

    assert exc_info.value.exit_code == 1
    assert _run_deploy_once.made_dirs == []


@pytest.mark.parametrize(
    ("anthropic", "openai", "expected"),
    [
        ("api_key", "api_key", "API key"),
        ("oauth", "api_key", "OAuth (Anthropic via ccproxy) + API key (OpenAI)"),
        ("api_key", "oauth", "API key (Anthropic) + ChatGPT sign-in (OpenAI)"),
        ("oauth", "oauth", "OAuth (Anthropic via ccproxy) + ChatGPT sign-in (OpenAI)"),
    ],
)
def test_describe_auth(anthropic, openai, expected):
    config = _make_config(anthropic_auth_mode=anthropic, openai_auth_mode=openai)

    assert deploy_server._describe_auth(config) == expected


def test_deploy_refuses_this_data_folders_running_server(monkeypatch, tmp_path):
    """deploy owns the server it starts: one already running for this data
    folder is refused, whatever its port, and nothing is started."""
    config = _make_config(default_workdir=str(tmp_path))
    printed: list = []
    with pytest.raises(typer.Exit) as exc:
        _run_deploy_once(
            monkeypatch,
            config,
            port=7000,
            port_occupied=True,
            langgraph_dev_running=True,
            printed=printed,
        )
    assert exc.value.exit_code == 1
    output = " ".join(" ".join(printed).split())
    assert "4242" in output
    assert "EvoSci server stop" in output


@pytest.mark.parametrize("lasting", [True, False])
def test_deploy_names_who_runs_recorded_server(monkeypatch, tmp_path, lasting):
    """It names the recorded server's pid, and whether the sessions share
    it; ``EvoSci server status`` reads the record, so it always describes
    this server."""
    server = Server(monkeypatch, pid=99999)
    server.run(4242, lasting=lasting)
    printed: list = []
    with pytest.raises(typer.Exit):
        _run_deploy_once(
            monkeypatch,
            _make_config(default_workdir=str(tmp_path)),
            printed=printed,
            server=server,
        )
    output = " ".join(" ".join(printed).split())
    assert "4242" in output
    assert "EvoSci server status" in output
    assert ("not the background service" in output) is not lasting
    assert server.spawns == []


def test_deploy_prints_no_banner_for_refused_start(monkeypatch, tmp_path):
    printed: list = []
    with pytest.raises(typer.Exit):
        _run_deploy_once(
            monkeypatch,
            _make_config(default_workdir=str(tmp_path)),
            port_occupied=True,
            printed=printed,
        )
    assert not any("Workspace:" in line for line in printed)


def test_deploy_refuses_when_port_occupied_by_foreign(monkeypatch, tmp_path):
    """If port is occupied but /ok doesn't respond, treat as foreign process
    and refuse."""
    config = _make_config(default_workdir=str(tmp_path))
    printed: list = []
    with pytest.raises(typer.Exit) as exc:
        _run_deploy_once(
            monkeypatch,
            config,
            port_occupied=True,
            langgraph_dev_running=False,  # foreign process holds the port
            printed=printed,
        )
    assert exc.value.exit_code == 1
    assert any("occupied by another process" in p for p in printed)


def test_deploy_refuses_another_data_folders_server(monkeypatch, tmp_path):
    server = Server(monkeypatch, pid=99999)
    server.answering.add(6174)
    server.whose[6174] = "other"
    printed: list = []
    with pytest.raises(typer.Exit):
        _run_deploy_once(
            monkeypatch,
            _make_config(default_workdir=str(tmp_path)),
            printed=printed,
            server=server,
        )
    assert any("another data folder" in p for p in printed)
    assert server.spawns == []


def test_deploy_reports_server_that_exits(monkeypatch, tmp_path):
    server = Server(monkeypatch, pid=99999)
    server.exit_code = 1
    printed: list = []
    with pytest.raises(typer.Exit):
        _run_deploy_once(
            monkeypatch,
            _make_config(default_workdir=str(tmp_path)),
            printed=printed,
            server=server,
        )
    assert any("langgraph dev startup failed" in p for p in printed)


def test_deploy_waits_for_another_start(monkeypatch, tmp_path):
    """Another start holds the start lock: deploy waits for it, says so,
    then starts."""
    import threading

    from filelock import FileLock

    from EvoScientist.langgraph_dev import manager as lgm
    from tests.fakes.langgraph_dev import LockHolder

    holder = LockHolder()
    threading.Timer(0.05, holder.done).start()
    captured = _run_deploy_once(
        monkeypatch, _make_config(default_workdir=str(tmp_path))
    )
    assert captured["langgraph_dev_started"] is True
    assert any("Waiting for another EvoSci process" in p for p in captured["printed"])
    lock = FileLock(str(lgm.RUNTIME.lock_file))
    lock.acquire(timeout=0.5)
    lock.release()


def test_deploy_gives_up_on_a_start_lock_held_too_long(monkeypatch, tmp_path):
    from tests.fakes.langgraph_dev import LockHolder

    holder = LockHolder()
    try:
        with pytest.raises(typer.Exit) as exc:
            _run_deploy_once(monkeypatch, _make_config(default_workdir=str(tmp_path)))
    finally:
        holder.done()
    assert exc.value.exit_code == 1
    assert any("for over" in p for p in _run_deploy_once.printed)


@pytest.mark.parametrize("step", ["mkdir", "acquire"])
def test_deploy_start_lock_os_error_is_startup_failed(monkeypatch, tmp_path, step):
    """An OSError from the start lock ends in the startup failed message, not
    in a traceback."""
    from tests.fakes.langgraph_dev import break_start_lock

    server = Server(monkeypatch, pid=99999)
    break_start_lock(monkeypatch, tmp_path, step)
    with pytest.raises(typer.Exit) as exc:
        _run_deploy_once(
            monkeypatch, _make_config(default_workdir=str(tmp_path)), server=server
        )
    assert exc.value.exit_code == 1
    assert server.spawns == []
    assert any("langgraph dev startup failed" in p for p in _run_deploy_once.printed)


def test_deploy_stops_its_server_when_it_ends(monkeypatch, tmp_path):
    captured = _run_deploy_once(
        monkeypatch, _make_config(default_workdir=str(tmp_path))
    )

    server = captured["server"]
    assert [(s["proc"], s["clean"]) for s in captured["stops"]] == [
        (server.procs[0], True)
    ]
    assert any("Stopped." in p for p in captured["printed"])


def test_deploy_stop_ignores_the_signals_that_end_it(monkeypatch, tmp_path):
    """A second Ctrl+C (or a SIGTERM) during the stop must not cut it short;
    the handlers are put back after it."""
    import signal

    captured = _run_deploy_once(
        monkeypatch, _make_config(default_workdir=str(tmp_path))
    )

    during = captured["stops"][0]["handlers"]
    assert during[signal.SIGINT] is signal.SIG_IGN
    assert during[signal.SIGTERM] is signal.SIG_IGN
    assert captured["signal_handlers"][signal.SIGINT] is signal.SIG_DFL
    assert captured["signal_handlers"][signal.SIGTERM] is signal.SIG_DFL


def test_deploy_signals_interrupt_while_it_serves(monkeypatch, tmp_path):
    """SIGTERM ends deploy as Ctrl+C does, so the stop runs."""
    import signal

    captured = _run_deploy_once(
        monkeypatch, _make_config(default_workdir=str(tmp_path))
    )

    serving = captured["handlers_while_serving"]
    for sig in (signal.SIGINT, signal.SIGTERM):
        assert serving[sig] is signal.default_int_handler
        with pytest.raises(KeyboardInterrupt):
            serving[sig](sig, None)


@pytest.mark.parametrize(
    "ending",
    [
        # The registration failed.
        typer.Exit(1),
        # A Ctrl+C before the Ready banner.
        KeyboardInterrupt(),
    ],
    ids=["exit", "interrupt"],
)
def test_deploy_stops_its_server_when_it_ends_before_serving(
    monkeypatch, tmp_path, ending
):
    def _serve(*_a):
        raise ending

    monkeypatch.setattr(deploy_server, "_serve", _serve)
    with pytest.raises(type(ending)):
        _run_deploy_once(monkeypatch, _make_config(default_workdir=str(tmp_path)))

    captured = _run_deploy_once.captured
    assert [s["proc"] for s in captured["stops"]] == captured["server"].procs


def test_deploy_stops_the_server_an_interrupted_start_left(monkeypatch):
    """No handle came back from the start: the stop takes the manager's,
    without the clean stop a server that never became ready does not need."""
    from EvoScientist.langgraph_dev import manager as lgm

    left = object()
    monkeypatch.setattr(lgm, "_PROCESS", left)
    stops: list = []
    monkeypatch.setattr(
        lgm, "stop_langgraph_dev", lambda proc, *, clean: stops.append((proc, clean))
    )
    monkeypatch.setattr(deploy_server, "console", _SilentConsole([]))

    deploy_server._stop_own_server(None)

    assert stops == [(left, False)]


def test_deploy_refused_start_stops_nothing(monkeypatch, tmp_path):
    with pytest.raises(typer.Exit):
        _run_deploy_once(
            monkeypatch,
            _make_config(default_workdir=str(tmp_path)),
            port_occupied=True,
        )
    assert _run_deploy_once.captured["stops"] == []


@pytest.mark.parametrize("hint", [None, "Run `EvoSci setup`, then restart."])
def test_deploy_prints_the_setup_hint(monkeypatch, tmp_path, hint):
    from EvoScientist import agent_shell

    monkeypatch.setattr(agent_shell, "setup_hint", lambda: hint)
    captured = _run_deploy_once(
        monkeypatch, _make_config(default_workdir=str(tmp_path))
    )
    shown = any("EvoSci setup" in line for line in captured["printed"])
    assert shown is (hint is not None)


def test_deploy_start_failure_prints_log_and_exit_code(monkeypatch, tmp_path):
    server = Server(monkeypatch, pid=99999)
    server.exit_code = 3
    printed: list = []
    with pytest.raises(typer.Exit):
        _run_deploy_once(
            monkeypatch,
            _make_config(default_workdir=str(tmp_path)),
            printed=printed,
            server=server,
        )
    output = " ".join(printed)
    assert "langgraph_dev.log" in output
    assert "Exit code: 3" in output
