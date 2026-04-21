"""Tests for the ``--resume`` CLI flag (alias of ``--thread-id``)."""

from __future__ import annotations

from typer.testing import CliRunner

from EvoScientist.cli._app import app

runner = CliRunner()


def test_resume_flag_listed_in_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--resume" in result.stdout
    assert "--thread-id" in result.stdout


def test_thread_id_flag_still_works():
    """Backwards compatibility: --thread-id should remain a valid flag."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--thread-id" in result.stdout
