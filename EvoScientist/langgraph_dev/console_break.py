"""Ask a langgraph dev server on Windows to shut down cleanly: Ctrl+Break.

``TerminateProcess`` (psutil ``terminate()`` on Windows) runs no shutdown
code, so langgraph-api never puts a cut run back in the queue and the run
stays ``running`` for good. uvicorn handles ``SIGBREAK`` like SIGTERM, and
the server leads its own process group, so ``CTRL_BREAK_EVENT`` sent to that
group shuts down the server and its children and nothing else.

The event can only be sent from a process attached to the server's console.
A lasting server has a hidden console of its own, and attaching to it means
leaving the caller's console first, which would cut the CLI or TUI off its
terminal; so a short helper process does it::

    python -m EvoScientist.langgraph_dev.console_break <server pid>

The target must lead a process group (spawned with
``CREATE_NEW_PROCESS_GROUP``): for any other PID Windows sends the event to
every process on that console, the caller included. The caller checks this
before running the helper.

Exit codes: 0 sent, 2 bad arguments, 3 attach failed, 4 send failed; the
Windows error code goes to stderr.
"""

from __future__ import annotations

import sys

EXIT_SENT = 0
EXIT_USAGE = 2
EXIT_ATTACH_FAILED = 3
EXIT_SEND_FAILED = 4

_CTRL_BREAK_EVENT = 1


def _kernel32():
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.AttachConsole.argtypes = [wintypes.DWORD]
    k32.SetConsoleCtrlHandler.argtypes = [wintypes.LPVOID, wintypes.BOOL]
    k32.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
    return k32, ctypes.get_last_error


def send_break(pid: int, kernel32=None, get_last_error=None) -> int:
    """Attach to ``pid``'s console and send ``CTRL_BREAK_EVENT`` to its
    process group. Returns one of the exit codes."""
    if kernel32 is None:
        kernel32, get_last_error = _kernel32()
    kernel32.FreeConsole()
    if not kernel32.AttachConsole(pid):
        print(f"AttachConsole({pid}) failed: error {get_last_error()}", file=sys.stderr)
        return EXIT_ATTACH_FAILED
    # The helper is on that console now too; it must not end on the event.
    kernel32.SetConsoleCtrlHandler(None, True)
    if not kernel32.GenerateConsoleCtrlEvent(_CTRL_BREAK_EVENT, pid):
        print(
            f"GenerateConsoleCtrlEvent({pid}) failed: error {get_last_error()}",
            file=sys.stderr,
        )
        return EXIT_SEND_FAILED
    kernel32.FreeConsole()
    return EXIT_SENT


def main(argv: list[str]) -> int:
    try:
        (pid,) = argv
        pid = int(pid)
        if pid <= 0:  # group 0 is every process on the console
            raise ValueError(pid)
    except ValueError:
        print("usage: console_break <pid>", file=sys.stderr)
        return EXIT_USAGE
    return send_break(pid)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
