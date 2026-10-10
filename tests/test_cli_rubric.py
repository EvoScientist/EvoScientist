"""--rubric option: parsing, validation, overrides and exit code."""

from __future__ import annotations

import pytest
import typer

from EvoScientist.cli.commands import (
    RUBRIC_UNMET_EXIT_CODE,
    _resolve_rubric_option,
    _rubric_exit_code,
)


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


def test_rubric_turns_background_dispatch_off_for_this_run(monkeypatch):
    overrides, _result = _invoke(monkeypatch, ["-p", "go", "--rubric", "- a"])
    assert overrides["enable_async_subagents"] is False


def test_runs_without_rubric_leave_background_dispatch_alone(monkeypatch):
    overrides, _result = _invoke(monkeypatch, ["-p", "go"])
    assert "enable_async_subagents" not in overrides


def test_rubric_without_prompt_is_a_usage_error(monkeypatch):
    overrides, result = _invoke(monkeypatch, ["--rubric", "- a"])
    assert overrides is None  # refused before the config is even loaded
    assert result.exit_code == 2
    assert "requires -p" in result.output


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
    assert "local gateway" in result.output


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
