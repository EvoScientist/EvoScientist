"""Tests for the ``--prompt-file`` CLI flag.

``--prompt-file`` reads the query from a file instead of the command line,
which bypasses the Windows CreateProcess 32767-character command-line limit
for very long prompts (e.g. full papers embedded in the query).
"""

from __future__ import annotations

import re

import typer
from typer.testing import CliRunner

from EvoScientist.cli import commands
from EvoScientist.cli._app import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(result) -> str:
    """Strip ANSI escapes (Rich color codes break literal substring lookups)."""
    return _ANSI_RE.sub("", result.stdout + (result.stderr or ""))


def _plain_help() -> str:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    return _ANSI_RE.sub("", result.stdout)


def test_prompt_file_flag_listed_in_help():
    assert "--prompt-file" in _plain_help()


def test_prompt_file_missing_path_errors(tmp_path):
    """A nonexistent --prompt-file path fails fast with a clear message."""
    missing = tmp_path / "no-such-prompt.txt"
    result = runner.invoke(app, ["--prompt-file", str(missing)])
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
