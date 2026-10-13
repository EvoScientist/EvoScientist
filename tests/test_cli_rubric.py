"""--rubric option: parsing, validation, overrides and exit code."""

from __future__ import annotations

import logging
import re
import sys

import pytest
import typer

from EvoScientist.cli.commands import (
    RUBRIC_UNMET_EXIT_CODE,
    _grader_failure_on_one_line,
    _resolve_rubric_option,
    _rubric_exit_code,
    _rubric_exit_notice,
    _single_shot_agent_config,
)


def _plain(output: str) -> str:
    """CLI output without colour codes, which a CI terminal gets."""
    return re.sub(r"\x1b\[[0-9;]*m", "", output)


def test_literal_text_is_stripped():
    assert _resolve_rubric_option("  - a\n- b \n", prompt="go") == "- a\n- b"


def test_no_option_means_no_rubric():
    assert _resolve_rubric_option(None, prompt=None) is None


def test_rubric_needs_a_prompt():
    with pytest.raises(typer.BadParameter, match="requires -p"):
        _resolve_rubric_option("- a", prompt=None)


def test_at_path_reads_the_file_relative_to_cwd(tmp_path, monkeypatch):
    (tmp_path / "criteria.md").write_text("- from file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert _resolve_rubric_option("@criteria.md", prompt="go") == "- from file"


def test_at_path_expands_the_home_directory(tmp_path, monkeypatch):
    (tmp_path / "c.md").write_text("- home", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # where Windows takes ~ from
    assert _resolve_rubric_option("@~/c.md", prompt="go") == "- home"


@pytest.mark.parametrize("content", ["", "  \n\t\n"])
def test_empty_rubric_is_refused(tmp_path, content):
    path = tmp_path / "empty.md"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="is empty"):
        _resolve_rubric_option(f"@{path}", prompt="go")
    with pytest.raises(typer.BadParameter, match="is empty"):
        _resolve_rubric_option(content or " ", prompt="go")


def test_missing_file_and_directory_are_refused(tmp_path):
    with pytest.raises(typer.BadParameter, match="could not read"):
        _resolve_rubric_option(f"@{tmp_path / 'nope.md'}", prompt="go")
    with pytest.raises(typer.BadParameter, match="could not read"):
        _resolve_rubric_option(f"@{tmp_path}", prompt="go")


def test_non_utf8_file_is_refused(tmp_path):
    path = tmp_path / "latin.md"
    path.write_bytes(b"- caf\xe9")
    with pytest.raises(typer.BadParameter, match="could not read"):
        _resolve_rubric_option(f"@{path}", prompt="go")


@pytest.mark.parametrize(
    ("rubric", "result", "expected"),
    [
        (None, None, None),  # ungraded runs keep their exit code
        (None, "needs_revision", None),
        ("- a", "satisfied", None),
        ("- a", "max_iterations_reached", RUBRIC_UNMET_EXIT_CODE),
        ("- a", "grader_error", RUBRIC_UNMET_EXIT_CODE),
        ("- a", "failed", RUBRIC_UNMET_EXIT_CODE),
        ("- a", "some_new_verdict", RUBRIC_UNMET_EXIT_CODE),
        ("- a", None, RUBRIC_UNMET_EXIT_CODE),  # the run ended without a verdict
    ],
)
def test_exit_code(rubric, result, expected):
    assert _rubric_exit_code(rubric, result) == expected
    assert RUBRIC_UNMET_EXIT_CODE == 3


def test_background_dispatch_switch_also_covers_experts(tmp_path):
    from types import SimpleNamespace

    from EvoScientist.paths import Workspace
    from EvoScientist.subagents.expert_container_async import (
        build_expert_async_subagent_specs,
    )

    cfg = SimpleNamespace(enable_async_subagents=False)
    assert build_expert_async_subagent_specs(cfg, workspace=Workspace(tmp_path)) == []


def _invoke(monkeypatch, argv, config=None):
    """Run the CLI up to get_effective_config and return (overrides, result)."""
    from typer.testing import CliRunner

    import EvoScientist.config as cfg_mod
    from EvoScientist.cli._app import app
    from EvoScientist.stream.console import console

    captured: dict[str, object] = {}

    class _Stop(Exception):
        pass

    def _grab(overrides):
        captured["overrides"] = dict(overrides)
        if config is None:
            raise _Stop
        return config

    monkeypatch.setattr(cfg_mod, "get_effective_config", _grab)
    monkeypatch.setattr(cfg_mod, "apply_config_to_env", lambda _config: None)
    original = console.file
    try:
        result = CliRunner().invoke(app, argv, catch_exceptions=True)
    finally:
        console.file = original
    return captured.get("overrides"), result


@pytest.mark.parametrize("argv", [["-p", "go", "--rubric", "- a"], ["-p", "go"]])
def test_the_process_config_keeps_the_users_background_dispatch(monkeypatch, argv):
    overrides, _result = _invoke(monkeypatch, argv)
    assert "enable_async_subagents" not in overrides


def test_only_the_agents_copy_turns_background_dispatch_off():
    from EvoScientist.config import EvoScientistConfig

    config = EvoScientistConfig()
    agent_config = _single_shot_agent_config(config, "- a")
    assert agent_config is not config
    assert agent_config.enable_async_subagents is False
    assert config.enable_async_subagents is True
    assert _single_shot_agent_config(config, None) is config


def _single_shot(monkeypatch, tmp_path, *, load_agent, verdict=None):
    """Run ``-p go --rubric`` up to the agent load (or through a stubbed turn)."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    import EvoScientist.cli.commands as cmd_mod
    import EvoScientist.cli.interactive as interactive_mod
    import EvoScientist.gateway as gateway_mod
    import EvoScientist.sessions as sessions_mod
    from EvoScientist.config import EvoScientistConfig
    from EvoScientist.langgraph_dev.manager import _server_config_fingerprint

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    config = EvoScientistConfig(ui_backend="cli")
    seen: dict[str, object] = {"plain": _server_config_fingerprint(config)}

    @asynccontextmanager
    async def _checkpointer():
        yield MagicMock()

    gateways = SimpleNamespace(
        graph_gateway=SimpleNamespace(create_thread=AsyncMock(return_value="t1"))
    )
    monkeypatch.setattr(
        cmd_mod,
        "_ensure_async_subagent_server",
        lambda cfg, **_kwargs: seen.update(server=_server_config_fingerprint(cfg)),
    )
    monkeypatch.setattr(
        cmd_mod, "_load_agent", lambda **kwargs: load_agent(seen, kwargs)
    )
    monkeypatch.setattr(cmd_mod, "_wait_for_memory_workers", lambda *_a, **_k: None)
    monkeypatch.setattr(interactive_mod, "cmd_run", lambda *_a, **_k: verdict)
    monkeypatch.setattr(sessions_mod, "get_checkpointer", _checkpointer)
    monkeypatch.setattr(
        gateway_mod, "create_runtime_gateways_for_config", lambda *_a, **_k: gateways
    )
    _overrides, result = _invoke(
        monkeypatch,
        ["-p", "go", "--rubric", "- a", "--workdir", str(tmp_path / "ws")],
        config=config,
    )
    return config, seen, result


def test_a_graded_run_does_not_make_the_running_server_look_out_of_date(
    monkeypatch, tmp_path
):
    """The server check gets the user's own settings; only the agent's differ."""

    class _Loaded(Exception):
        pass

    def _load(seen, kwargs):
        seen["agent"] = kwargs["config"]
        raise _Loaded

    config, seen, _result = _single_shot(monkeypatch, tmp_path, load_agent=_load)

    assert seen["server"] == seen["plain"]
    assert seen["agent"].enable_async_subagents is False
    assert config.enable_async_subagents is True


@pytest.mark.parametrize(
    ("verdict", "line"),
    [
        (None, "The run stopped before the acceptance criteria were checked"),
        (
            "grader_error",
            "The acceptance criteria could not be checked: the grader failed",
        ),
        ("max_iterations_reached", "Acceptance criteria not met"),
    ],
)
def test_the_exit_line_says_whether_the_run_was_graded(
    monkeypatch, tmp_path, verdict, line
):
    import io
    from unittest.mock import MagicMock

    from EvoScientist.stream.console import console

    # The turn runs on the runtime's own thread, out of reach of a capture.
    printed = io.StringIO()
    monkeypatch.setattr(console, "file", printed)
    _config, _seen, result = _single_shot(
        monkeypatch, tmp_path, load_agent=lambda *_a: MagicMock(), verdict=verdict
    )

    assert result.exit_code == RUBRIC_UNMET_EXIT_CODE
    shown = " ".join(_plain(printed.getvalue()).split())
    assert f"{line} (exit code 3)." in shown
    assert ("not met" in shown) is (verdict == "max_iterations_reached")


def test_a_run_that_was_never_graded_does_not_say_the_criteria_failed():
    notice = _rubric_exit_notice(None, RUBRIC_UNMET_EXIT_CODE)
    assert notice == (
        "The run stopped before the acceptance criteria were checked (exit code 3)."
    )
    assert _rubric_exit_notice("max_iterations_reached", 3) == (
        "Acceptance criteria not met (exit code 3)."
    )


def test_a_rubric_that_could_not_be_evaluated_does_not_say_not_met():
    notice = _rubric_exit_notice("failed", RUBRIC_UNMET_EXIT_CODE)
    assert notice == "The acceptance criteria could not be evaluated (exit code 3)."


def test_a_grader_that_failed_does_not_say_the_criteria_failed():
    """Nothing was checked: the line says so instead of "not met"."""
    notice = _rubric_exit_notice("grader_error", RUBRIC_UNMET_EXIT_CODE)
    assert notice == (
        "The acceptance criteria could not be checked: the grader failed (exit code 3)."
    )


def _raised(logger_name: str, message: str) -> logging.LogRecord:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        return logging.LogRecord(
            logger_name, logging.ERROR, __file__, 1, message, None, sys.exc_info()
        )


@pytest.mark.parametrize(
    "message",
    ["RubricMiddleware grader failed (m)", "RubricMiddleware coverage retry raised"],
)
def test_the_console_gets_a_one_line_copy_of_a_grader_failure(message):
    record = _raised("deepagents.middleware.rubric", message)
    short = _grader_failure_on_one_line(record)
    assert short is not record
    assert short.exc_info is None
    assert short.getMessage() == f"{message}: RuntimeError: boom"
    # The record other handlers see keeps its traceback.
    assert record.exc_info is not None
    assert record.getMessage() == message


def test_other_records_reach_the_console_as_they_are():
    elsewhere = _raised("EvoScientist.cli", "RubricMiddleware grader failed (m)")
    another = _raised("deepagents.middleware.rubric", "on_evaluation callback raised")
    plain = logging.LogRecord(
        "deepagents.middleware.rubric",
        logging.ERROR,
        __file__,
        1,
        "grader failed",
        None,
        None,
    )
    for record in (elsewhere, another, plain):
        assert _grader_failure_on_one_line(record) is record


@pytest.fixture
def root_logging():
    """Let a test install the CLI's handler, then put the root logger back."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    genai = logging.getLogger("langchain_google_genai._function_utils")
    genai_level = genai.level
    yield root
    root.handlers[:] = handlers
    root.setLevel(level)
    genai.setLevel(genai_level)


@pytest.mark.parametrize(
    ("log_level", "traceback_on_console"), [("WARNING", False), ("DEBUG", True)]
)
def test_only_the_console_loses_the_grader_traceback(
    monkeypatch, root_logging, log_level, traceback_on_console
):
    from types import SimpleNamespace

    import EvoScientist.config as cfg_mod
    from EvoScientist.cli.commands import _configure_logging
    from EvoScientist.stream.console import console

    monkeypatch.setattr(
        cfg_mod,
        "get_effective_config",
        lambda *_a: SimpleNamespace(log_level=log_level),
    )
    _configure_logging()
    kept: list[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record):
            kept.append(record)

    root_logging.addHandler(_Keep())
    upstream = logging.getLogger("deepagents.middleware.rubric")
    with console.capture() as shown:
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            upstream.exception("RubricMiddleware grader failed (m)")

    printed = _plain(shown.get())
    assert "RubricMiddleware grader" in printed
    assert ("Traceback" in printed) is traceback_on_console
    assert ("RuntimeError: boom" in printed) is True
    assert kept[0].exc_info is not None


def test_rubric_without_prompt_is_a_usage_error(monkeypatch):
    overrides, result = _invoke(monkeypatch, ["--rubric", "- a"])
    assert overrides is None  # refused before the config is even loaded
    assert result.exit_code == 2
    assert "requires -p" in _plain(result.output)


def test_rubric_is_refused_on_the_server_gateway(monkeypatch):
    from types import SimpleNamespace

    config = SimpleNamespace(
        gateway_backend="langgraph_server",
        gateway_backend_single_shot="inherit",
    )
    _overrides, result = _invoke(
        monkeypatch, ["-p", "go", "--rubric", "- a"], config=config
    )
    assert result.exit_code == 2
    assert "local gateway" in _plain(result.output)


def test_runtime_fallback_keeps_the_rubric(monkeypatch):
    """When the first renderer fails, the retry must still be a graded run."""
    from EvoScientist.cli import tui_runtime

    class _FailingBackend:
        def run_streaming(self, **_kwargs):
            raise RuntimeError("renderer broke")

    seen: dict[str, object] = {}

    class _FallbackBackend:
        def run_streaming(self, **kwargs):
            seen.update(kwargs)
            return "ok"

    monkeypatch.setattr(tui_runtime, "get_backend", lambda *_a, **_k: _FailingBackend())
    monkeypatch.setattr(tui_runtime, "RichStreamingBackend", _FallbackBackend)

    result = tui_runtime.run_streaming(
        ui_backend="tui",
        agent=object(),
        message="go",
        thread_id="t",
        show_thinking=False,
        interactive=False,
        gateway=object(),
        rubric="- a",
    )

    assert result == "ok"
    assert seen["rubric"] == "- a"


def test_unknown_user_home_is_a_usage_error():
    with pytest.raises(typer.BadParameter, match="could not read"):
        _resolve_rubric_option("@~no-such-user-for-rubric/c.md", prompt="go")


def test_utf8_bom_is_not_part_of_the_rubric(tmp_path):
    path = tmp_path / "bom.md"
    path.write_bytes(b"\xef\xbb\xbf- a\n")
    assert _resolve_rubric_option(f"@{path}", prompt="go") == "- a"

    path.write_bytes(b"\xef\xbb\xbf")
    with pytest.raises(typer.BadParameter, match="is empty"):
        _resolve_rubric_option(f"@{path}", prompt="go")
