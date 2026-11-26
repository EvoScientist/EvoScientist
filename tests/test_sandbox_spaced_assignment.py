"""Regression tests for variables holding spaced workspace paths (#843)."""

from __future__ import annotations

import shlex

from EvoScientist import backends


def _posix(monkeypatch) -> None:
    monkeypatch.setattr(backends, "_is_windows", lambda: False)


def test_a_variable_assigned_a_spaced_absolute_path_is_quoted_at_use(monkeypatch):
    _posix(monkeypatch)
    command = "cd sub && D=/tmp/build; mkdir -p $D"
    prepared, error = backends.prepare_sandbox_command(command, "/W s", dangerous=True)
    assert error is None
    assert prepared == "cd sub && D='/W s'/tmp/build; mkdir -p \"$D\""
    # The shell reads the use as a single argument, whatever the value holds.
    assert shlex.split(prepared)[-1] == "$D"


def test_an_unquoted_removal_use_is_quoted_too(monkeypatch):
    _posix(monkeypatch)
    command = "cd sub && D=/tmp/build; rm -rf $D"
    prepared, error = backends.prepare_sandbox_command(command, "/W s", dangerous=True)
    assert error is None
    assert prepared == "cd sub && D='/W s'/tmp/build; rm -rf \"$D\""


def test_an_already_quoted_use_is_untouched(monkeypatch):
    _posix(monkeypatch)
    command = 'cd sub && D=/tmp/build; mkdir -p "$D"'
    prepared, error = backends.prepare_sandbox_command(command, "/W s", dangerous=True)
    assert error is None
    assert prepared == "cd sub && D='/W s'/tmp/build; mkdir -p \"$D\""


def test_a_relative_assignment_does_not_need_quoted_uses(monkeypatch):
    _posix(monkeypatch)
    command = "D=/tmp/build; mkdir -p $D"
    assert (
        backends.convert_virtual_paths_in_command(command, workspace_root="/W s")
        == "D=./tmp/build; mkdir -p $D"
    )
