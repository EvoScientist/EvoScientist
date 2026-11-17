"""Regression coverage for bare allowed roots later in a shell word."""

import pytest

from EvoScientist.backends import validate_command


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        ('git commit -m "move /home/u/work/input.csv to /home/u/work"', None),
        ("echo '/home/u/work/a,/home/u/work'", None),
        ("cat /home/u/work/a,/home/u/work,/etc/passwd", "/etc/passwd"),
        ("cat /home/u/work/a,/home/u/work)/etc/passwd", "/etc/passwd"),
        ("cat /home/u/work/a,/home/u/work-extra", "/home/u/work-extra"),
    ],
    ids=[
        "quoted-message",
        "quoted-list",
        "comma-after-root",
        "paren-after-root",
        "prefix-only",
    ],
)
def test_later_bare_allowed_root(command, blocked):
    result = validate_command(
        command,
        allow_prefixes=("/home/u/work",),
        workspace_root="/home/u/work",
    )
    if blocked is None:
        assert result is None
    else:
        assert result is not None
        assert blocked in result
