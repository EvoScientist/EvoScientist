"""``EvoSci login chatgpt`` / ``EvoSci logout chatgpt``: ChatGPT sign-in outside
the setup wizard.

With ``--json`` stdout carries one JSON object per line and nothing else (Rich
output goes to stderr), for programs such as the desktop app. Every line has
``"protocol": 1`` and an ``event``. Later versions may add fields and events,
never rename or remove them.

``login chatgpt --json`` events:

- ``url``: ``url``, the sign-in page to open in a browser. Sent once the
  callback port is bound; this mode never opens a browser itself.
- ``signed_in``: ``plan`` (may be null).
- ``error``: ``code``, ``message``. Codes: ``port_in_use`` (no ``url`` was
  sent), ``timed_out``, ``cancelled``, ``sign_in_failed``, ``internal_error``.

``logout chatgpt --json`` ends with ``signed_out`` (also when nobody was signed
in) or ``error`` (``internal_error``).

Every run ends with exactly one ``signed_in``, ``signed_out`` or ``error``
line, then exits (0, or 1 after ``error``). Start ``EvoSci`` itself (not
through a shell) with a stdin pipe: closing stdin or writing a
``{"command": "stop"}`` line cancels a sign-in that waits for the browser, as
does SIGTERM on macOS and Linux (a second SIGTERM can end the process at
once, with no terminal line).
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import traceback
from collections.abc import Callable
from typing import Any

import typer  # type: ignore[import-untyped]

from ._app import login_app, logout_app

Emit = Callable[[dict[str, Any]], None]

_JSON_HELP = "Print one JSON event per line for a program; never opens a browser"


def _event(name: str, **fields: Any) -> dict[str, Any]:
    from ..setup.protocol import PROTOCOL

    return {"protocol": PROTOCOL, "event": name, **fields}


def _internal_error(emit: Emit, exc: BaseException) -> int:
    traceback.print_exception(exc, file=sys.stderr)
    emit(_event("error", code="internal_error", message=str(exc) or type(exc).__name__))
    return 1


def run_login_json(emit: Emit, stop: threading.Event) -> int:
    """Run the JSON sign-in until it ends; returns the exit code."""
    try:
        import httpx

        from ..llm import chatgpt_oauth
    except Exception as exc:
        return _internal_error(emit, exc)

    def _on_url(url: str) -> None:
        emit(_event("url", url=url))

    try:
        state = chatgpt_oauth.sign_in(on_url=_on_url, stop=stop)
    except chatgpt_oauth.SignInError as exc:
        emit(_event("error", code=exc.code, message=str(exc)))
        return 1
    except KeyboardInterrupt:
        message = chatgpt_oauth.SIGN_IN_CANCELLED
        emit(_event("error", code="cancelled", message=message))
        return 1
    except httpx.HTTPError:
        message = chatgpt_oauth.SIGN_IN_INCOMPLETE
        emit(_event("error", code="sign_in_failed", message=message))
        return 1
    except Exception as exc:
        return _internal_error(emit, exc)
    if not state.signed_in:
        return _internal_error(emit, RuntimeError("No sign-in was stored."))
    emit(_event("signed_in", plan=state.plan))
    return 0


def run_logout_json(emit: Emit) -> int:
    """Remove the stored sign-in; returns the exit code."""
    try:
        from ..llm import chatgpt_oauth

        chatgpt_oauth.logout()
    except Exception as exc:
        return _internal_error(emit, exc)
    emit(_event("signed_out"))
    return 0


def _json_mode() -> Emit:
    from ..setup.protocol import JsonEmitter
    from ..stream.json_sink import redirect_console_to_stderr

    redirect_console_to_stderr()
    return JsonEmitter()


@login_app.command("chatgpt")
def login_chatgpt_command(
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """Sign in with a ChatGPT plan (used when openai_auth_mode is oauth)."""
    if not json_output:
        from ..config.onboard.helpers import _run_chatgpt_login

        raise typer.Exit(0 if _run_chatgpt_login() else 1)
    from ..setup.protocol import stdin_is_pipe, watch_stdin

    emit = _json_mode()
    stop = threading.Event()

    def _on_sigterm(_signum: int, _frame: Any) -> None:
        # Hand a second SIGTERM to the previous action: re-entering
        # Event.set() from a nested handler would deadlock on its lock.
        signal.signal(signal.SIGTERM, previous_sigterm)
        stop.set()

    # Read first, so a SIGTERM right after the install finds it; None when the
    # old action was not set from Python.
    previous_sigterm = signal.getsignal(signal.SIGTERM) or signal.SIG_DFL
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        if stdin_is_pipe():
            threading.Thread(
                target=watch_stdin,
                args=(os.dup(sys.stdin.fileno()), stop),
                name="login-json-stdin",
                daemon=True,
            ).start()
        code = run_login_json(emit, stop)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    raise typer.Exit(code)


@logout_app.command("chatgpt")
def logout_chatgpt_command(
    json_output: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """Remove the ChatGPT sign-in stored on this machine (nothing is revoked)."""
    if json_output:
        raise typer.Exit(run_logout_json(_json_mode()))
    from ..llm import chatgpt_oauth
    from ..stream.console import console

    removed = chatgpt_oauth.logout()
    console.print("Signed out of ChatGPT." if removed else "Not signed in to ChatGPT.")
