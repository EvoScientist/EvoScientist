"""Tests for ``EvoSci --version`` and ``EvoSci --version --json``."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version

import pytest
from typer.testing import CliRunner

from EvoScientist.cli._app import app
from EvoScientist.setup import webui
from EvoScientist.setup.protocol import PROTOCOL


@pytest.fixture
def evosci(monkeypatch):
    """Invoke the CLI with ``sys.argv`` set as a real run would set it: the
    eager ``--version`` callback reads the raw arguments."""
    from EvoScientist.stream.console import console

    monkeypatch.delenv(webui.COMPAT_ENV, raising=False)
    # --version --json redirects the shared console; undo it after the test.
    monkeypatch.setattr(console, "_file", None)
    runner = CliRunner()

    def invoke(*argv):
        monkeypatch.setattr(sys, "argv", ["EvoSci", *argv])
        return runner.invoke(app, list(argv))

    return invoke


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_prints_plain_text(evosci, flag):
    result = evosci(flag)
    assert result.exit_code == 0
    assert result.stdout == f"EvoScientist {version('EvoScientist')}\n"


@pytest.mark.parametrize(
    "argv", [("--version", "--json"), ("--json", "--version"), ("-V", "--json")]
)
def test_version_json_in_either_order(evosci, argv):
    result = evosci(*argv)
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "version": version("EvoScientist"),
        "protocol": PROTOCOL,
        "webui_compat": webui.WEBUI_COMPAT,
    }


def test_version_json_reports_the_range_in_effect(evosci, monkeypatch):
    monkeypatch.setenv(webui.COMPAT_ENV, ">=0.2,<0.4")
    monkeypatch.setattr(webui, "_warned_override", None)
    result = evosci("--version", "--json")
    assert result.exit_code == 0
    assert json.loads(result.stdout)["webui_compat"] == ">=0.2,<0.4"


def test_json_without_version_exits_2(evosci):
    result = evosci("--json")
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "--version" in result.stderr
