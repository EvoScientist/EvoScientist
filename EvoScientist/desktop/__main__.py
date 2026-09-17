"""``python -m EvoScientist.desktop`` — launch the Windows desktop shell."""

from __future__ import annotations

import argparse

from .shell import run_desktop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="EvoScientist.desktop")
    parser.add_argument(
        "--workspace",
        default=None,
        help="Workspace directory (defaults to the configured default_workdir).",
    )
    args = parser.parse_args(argv)
    run_desktop(args.workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
