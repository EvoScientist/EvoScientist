"""Tests for the ``--prompt-file`` CLI flag.

``--prompt-file`` reads the query from a file instead of the command line,
which bypasses the Windows CreateProcess 32767-character command-line limit
for very long prompts (e.g. full papers embedded in the query).
"""

from __future__ import annotations

import io
import re
from types import SimpleNamespace

import typer
from rich.console import Console
from typer.testing import CliRunner

import EvoScientist.cli.interactive as interactive
from EvoScientist.cli import commands
from EvoScientist.cli._app import app
from EvoScientist.paths import SessionDirs, Workspace

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(result) -> str:
    """Strip ANSI escapes (Rich color codes break literal substring lookups)."""
    return _ANSI_RE.sub("", result.stdout + (result.stderr or ""))


def _plain_help() -> str:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    return _ANSI_RE.sub("", result.stdout)


def test_prompt_file_blank_errors(monkeypatch, tmp_path):
    def _stop(_rubric, *, prompt):
        raise typer.Exit()

    monkeypatch.setattr(commands, "_resolve_rubric_option", _stop)
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("  \n\n", encoding="utf-8")
    result = runner.invoke(app, ["--prompt-file", str(prompt_file)])
    assert result.exit_code != 0
    assert "--prompt-file is empty" in _plain(result)


def test_prompt_file_invalid_utf8_errors(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_bytes(b"\xff\xfe\x00bad")
    result = runner.invoke(app, ["--prompt-file", str(prompt_file)])
    assert result.exit_code != 0
    assert "UTF-8" in _plain(result)


def test_prompt_file_flag_listed_in_help():
    assert "--prompt-file" in _plain_help()


def test_prompt_file_missing_path_errors(tmp_path):
    """A nonexistent --prompt-file path fails fast with a clear message."""
    missing = tmp_path / "no-such-prompt.txt"
    result = runner.invoke(app, ["--prompt-file", str(missing)])
    assert result.exit_code != 0
    assert "--prompt-file not found" in _plain(result)


def test_empty_prompt_file_path_errors(monkeypatch):
    def _stop(_rubric, *, prompt):
        raise typer.Exit()

    monkeypatch.setattr(commands, "_resolve_rubric_option", _stop)
    result = runner.invoke(app, ["--prompt-file", ""])
    assert result.exit_code != 0
    assert "--prompt-file not found" in _plain(result)


def test_prompt_file_and_prompt_together_error(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt from file", encoding="utf-8")
    result = runner.invoke(app, ["-p", "inline", "--prompt-file", str(prompt_file)])
    assert result.exit_code != 0
    assert "not both" in _plain(result)


def test_prompt_file_text_becomes_the_prompt(monkeypatch, tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("\ufeffprompt from file\n", encoding="utf-8")
    seen = {}

    def _capture(_rubric, *, prompt):
        seen["prompt"] = prompt
        raise typer.Exit()

    monkeypatch.setattr(commands, "_resolve_rubric_option", _capture)
    result = runner.invoke(app, ["--prompt-file", str(prompt_file)])
    assert result.exit_code == 0
    assert seen["prompt"] == "prompt from file\n"


def test_single_shot_echo_cuts_a_long_prompt(monkeypatch, tmp_path):
    out = io.StringIO()
    sent = {}
    monkeypatch.setattr(interactive, "console", Console(file=out, width=200))
    monkeypatch.setattr(
        interactive, "run_streaming", lambda **kw: sent.update(message=kw["message"])
    )
    monkeypatch.setattr(commands, "_wait_for_memory_workers", lambda: None)
    prompt = "x" * 500 + "TAIL" * 250

    interactive.cmd_run(
        object(),
        prompt,
        "thread-1",
        dirs=SessionDirs(Workspace(tmp_path)),
        runtime_gateways=SimpleNamespace(graph_gateway=None),
    )

    shown = out.getvalue().replace("\n", "")
    assert "x" * 500 + "\u2026 (1500 chars total)" in shown
    assert "TAIL" not in shown
    assert sent["message"] == prompt
