"""``EvoSci`` writes redirected output as UTF-8 (#660).

On Windows, Python writes to a pipe or a file in the ANSI code page, so the
first character outside it stopped the command with ``UnicodeEncodeError``.
"""

from __future__ import annotations

import io
import sys

import pytest

import EvoScientist.cli as cli_pkg


class _Stream(io.TextIOWrapper):
    """A cp1252 stream, as Python opens a redirected stdout on Windows."""

    def __init__(self, *, tty: bool = False, errors: str = "strict") -> None:
        super().__init__(io.BytesIO(), encoding="cp1252", errors=errors)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_redirected_output_is_written_as_utf8(monkeypatch):
    from EvoScientist.cli import commands
    from EvoScientist.setup import git as setup_git
    from EvoScientist.setup import node as setup_node
    from EvoScientist.stream.console import console

    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    out, err = _Stream(), _Stream(errors="backslashreplace")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(console, "_file", None)
    monkeypatch.setattr(setup_node, "activate_runtime", lambda: None)
    monkeypatch.setattr(setup_git, "activate_runtime", lambda: None)
    monkeypatch.setattr(commands, "_configure_logging", lambda: None)
    monkeypatch.setattr(cli_pkg, "app", lambda: console.print("✓ ready"))
    cli_pkg.main()
    out.flush()
    assert out.buffer.getvalue().decode("utf-8").startswith("✓ ready")
    assert err.encoding == "utf-8"
    assert err.errors == "backslashreplace"


@pytest.mark.parametrize(
    ("platform", "tty", "env", "expected"),
    [
        ("linux", False, None, "utf-8"),  # a pipe or a file
        ("linux", False, "", "utf-8"),  # Python ignores an empty value
        ("win32", True, None, "utf-8"),  # isatty() is true for NUL
        ("linux", True, None, "cp1252"),  # a terminal
        ("win32", False, "cp1252", "cp1252"),  # the user's choice
    ],
)
def test_which_streams_are_switched(monkeypatch, platform, tty, env, expected):
    if env is None:
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    else:
        monkeypatch.setenv("PYTHONIOENCODING", env)
    out, err = _Stream(tty=tty), _Stream(tty=tty)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys, "platform", platform)
    cli_pkg._write_redirected_output_as_utf8()
    assert (out.encoding, err.encoding) == (expected, expected)


def test_each_stream_is_decided_on_its_own(monkeypatch):
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    out, err = _Stream(tty=False), _Stream(tty=True)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys, "platform", "linux")
    cli_pkg._write_redirected_output_as_utf8()
    assert (out.encoding, err.encoding) == ("utf-8", "cp1252")


def _closed() -> _Stream:
    stream = _Stream()
    stream.close()
    return stream


@pytest.mark.parametrize(
    "stream",
    [
        pytest.param(None, id="no-stdout"),
        pytest.param(io.StringIO(), id="no-reconfigure"),
        pytest.param(_closed(), id="closed"),
    ],
)
def test_a_stream_that_cannot_be_switched_is_left_alone(monkeypatch, stream):
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    monkeypatch.setattr(sys, "stdout", stream)
    cli_pkg._write_redirected_output_as_utf8()
