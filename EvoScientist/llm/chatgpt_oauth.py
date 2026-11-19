"""Sign in with a ChatGPT plan and call the Codex backend directly.

Thin wrapper around ``langchain-openai``'s ChatGPT OAuth support, and the only
module that imports its private names. Re-check them when the
``langchain-openai`` floor moves: ``_ChatOpenAICodex``,
``_FileChatGPTOAuthTokenProvider`` (with ``_read_from_disk`` and ``_cached``),
its two messages for a refused refresh (matched in ``_refresh_rejected``), and
the sign-in pieces ``_authorize`` puts together: ``_generate_pkce_pair``,
``_build_authorize_url``, ``_CallbackHandler`` (``server_result``,
``callback_path``), ``_ChatGPTToken``, ``_post_form`` and
``_token_from_response``.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import http.server
import logging
import os
import re
import secrets
import selectors
import socket
import sys
import threading
import time
import uuid
import warnings
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from filelock import FileLock
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai.chat_models.codex import _ChatOpenAICodex
from langchain_openai.chatgpt_oauth import (
    CHATGPT_CLIENT_ID,
    CHATGPT_TOKEN_URL,
    DEFAULT_REDIRECT_HOST,
    DEFAULT_REDIRECT_PATH,
    DEFAULT_REDIRECT_PORT,
    DEFAULT_SCOPE,
    _build_authorize_url,
    _CallbackHandler,
    _ChatGPTToken,
    _FileChatGPTOAuthTokenProvider,
    _generate_pkce_pair,
    _post_form,
    _token_from_response,
)

SIGN_IN_HINT = (
    'ChatGPT sign-in required (openai_auth_mode is "oauth"). '
    "Start EvoSci in a terminal to sign in."
)
SIGN_IN_AGAIN = (
    'ChatGPT sign-in required (openai_auth_mode is "oauth"). Run EvoSci again '
    "to sign in, or switch with: EvoSci config set openai_auth_mode api_key"
)
PORT_FORWARD_HINT = (
    "No browser on this machine? Forward the callback port, then open the "
    "link on your own computer: ssh -L 1455:localhost:1455 <this-host>"
)
PORT_IN_USE = (
    "Port 1455 is in use (another sign-in may be running). Close it and try again."
)
SIGN_IN_INCOMPLETE = "OpenAI did not complete the sign-in. Try again."
SIGN_IN_CANCELLED = "ChatGPT sign-in was cancelled."
SIGN_IN_TIMED_OUT = "ChatGPT sign-in timed out. Try again."
SIGN_IN_TIMEOUT = 300.0
# How often the callback wait looks at its stop event and deadline.
_POLL_SECONDS = 0.5
# A connection that sends nothing must not hold the wait past a cancel.
_CALLBACK_READ_SECONDS = 1.0
# How long the check for another program on the callback port waits.
_PROBE_SECONDS = 0.3
DEFAULT_INSTRUCTIONS = "You are a helpful assistant."
# Codex's default compaction window for this route; the backend takes more, but
# every turn resends the context against the plan's usage limit.
CONTEXT_WINDOW = 272_000


class NotSignedInError(RuntimeError):
    """No usable sign-in: no token file, an unreadable one, or a rejected refresh."""

    def __init__(self) -> None:
        super().__init__(SIGN_IN_HINT)


class SignInError(RuntimeError):
    """A sign-in that did not finish, with a code a program can act on."""

    def __init__(
        self,
        code: Literal["port_in_use", "timed_out", "cancelled", "sign_in_failed"],
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ChatGPTLogin:
    """Sign-in state read from disk."""

    signed_in: bool
    plan: str | None = None
    problem: str | None = None


class _DropNoFcntlWarning(logging.Filter):
    """Our lock already guards the token file where upstream's is a no-op."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("fcntl is unavailable")


logging.getLogger("langchain_openai.chatgpt_oauth").addFilter(_DropNoFcntlWarning())
logger = logging.getLogger(__name__)


def token_path() -> Path:
    """Token file under the data dir. Creates nothing."""
    from ..paths import DATA_DIR

    return DATA_DIR / "auth" / "chatgpt.json"


# Upstream's two messages for a refused refresh token; its error class alone
# does not tell (it also covers a malformed answer, and misses a plain 401).
_REFUSED = re.compile(
    r"ChatGPT refresh token is no longer valid \(`invalid_grant`\)"
    r"|OAuth request to \S+ failed with status 401\b"
)


def _refresh_rejected(exc: RuntimeError) -> bool:
    """Whether the token endpoint refused the refresh token itself."""
    return bool(_REFUSED.match(str(exc)))


def _token_lock(path: Path) -> FileLock:
    """The lock every writer of the token file holds: refresh, sign-in, removal."""
    # Not upstream's ``chatgpt.json.lock``: a second flock on the same file
    # from this process would wait on the first forever.
    return FileLock(str(path.with_name("chatgpt.lock")))


class _LockedTokenProvider:
    """Upstream's file token provider behind a lock that also works on Windows."""

    def __init__(self, path: Path) -> None:
        self._inner = _FileChatGPTOAuthTokenProvider(path=path)
        self._lock = _token_lock(path)

    def get_token(self) -> Any:
        if not self._inner.path.exists():
            raise NotSignedInError
        with self._lock:
            try:
                stored = self._inner._read_from_disk()
            except Exception as exc:  # whatever is in the file, it is not a token
                raise NotSignedInError from exc
            if stored is None:
                raise NotSignedInError
            # The file is the truth: upstream would keep serving its cached
            # token after another process signed in again.
            self._inner._cached = stored
            try:
                return self._inner.get_token()
            except RuntimeError as exc:
                if not _refresh_rejected(exc):
                    raise
                self._forget(stored)
                raise NotSignedInError from exc

    def _forget(self, rejected: Any) -> None:
        """Drop the rejected token so the next start asks to sign in again."""
        path = self._inner.path
        try:
            current = self._inner._read_from_disk()
        except Exception:
            return
        # Only a file put there by hand (copied into a container) can differ here.
        if current is None or current.refresh_token != rejected.refresh_token:
            return
        try:
            os.unlink(path)
        except FileNotFoundError:
            return
        except OSError as exc:
            # Left in place it would still read as signed in, with no prompt.
            raise RuntimeError(
                f"OpenAI rejected the ChatGPT sign-in stored at {path}, and the "
                f"file could not be removed ({exc}). Delete it and sign in again."
            ) from exc

    async def aget_token(self) -> Any:
        return await asyncio.to_thread(self.get_token)

    def get_access_token(self) -> str:
        return self.get_token().access_token

    async def aget_access_token(self) -> str:
        return (await self.aget_token()).access_token


class EvoChatOpenAICodex(_ChatOpenAICodex):
    """ChatGPT-plan model whose non-stream entry points read the stream.

    The backend only accepts streamed requests, so a copy with
    ``disable_streaming=True`` still gets a stream back.
    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return generate_from_stream(self._stream(messages, stop=stop, **kwargs))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await agenerate_from_stream(self._astream(messages, stop=stop, **kwargs))


def build_model(model_id: str, **kwargs: Any) -> EvoChatOpenAICodex:
    """Build the ChatGPT-plan model for ``model_id``.

    Raises:
        NotSignedInError: No token file exists.
    """
    path = token_path()
    if not path.exists():
        raise NotSignedInError
    kwargs.setdefault("instructions", DEFAULT_INSTRUCTIONS)
    with warnings.catch_warnings():
        # Upstream's one-time "experimental and unofficial" notice; onboarding shows ours.
        warnings.simplefilter("ignore", UserWarning)
        model = EvoChatOpenAICodex(
            model=model_id, token_provider=_LockedTokenProvider(path), **kwargs
        )
    model.profile = {**(model.profile or {}), "max_input_tokens": CONTEXT_WINDOW}
    return model


class _CallbackServer(http.server.HTTPServer):
    """A loopback callback server; it never shares its address."""

    # On Windows SO_REUSEADDR would let it bind a port another program listens on.
    allow_reuse_address = sys.platform != "win32"

    def server_bind(self) -> None:
        if self.address_family == socket.AF_INET6:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class _CallbackServer6(_CallbackServer):
    address_family = socket.AF_INET6


# Errors that mean this machine has no IPv6 loopback, not that ::1 is taken.
_NO_IPV6 = {
    code
    for code in (
        errno.EADDRNOTAVAIL,
        errno.EAFNOSUPPORT,
        getattr(errno, "WSAEADDRNOTAVAIL", None),
        getattr(errno, "WSAEAFNOSUPPORT", None),
    )
    if code is not None
}


def _ipv6_server(port: int, handler: type) -> _CallbackServer | None:
    """The callback server on ``[::1]``, or None without an IPv6 loopback.

    Browsers may try ``::1`` first for ``localhost``; serving it keeps the code
    from reaching another program there. Raises ``OSError`` when it is taken.
    """
    try:
        return _CallbackServer6(("::1", port), handler)
    except OSError as exc:
        if exc.errno in _NO_IPV6:
            return None
        raise


def _someone_listens(port: int) -> bool:
    """Whether another program already accepts connections on the loopback port.

    A browser could reach it instead of us, over a connection it keeps open.
    """
    for host in ("127.0.0.1", "::1"):
        try:
            with socket.create_connection((host, port), timeout=_PROBE_SECONDS):
                return True
        except OSError:
            continue
    return False


def _authorize(
    *,
    on_url: Callable[[str], None],
    stop: threading.Event | None = None,
    timeout: float = SIGN_IN_TIMEOUT,
    port: int = DEFAULT_REDIRECT_PORT,
) -> _ChatGPTToken:
    """Run the browser authorization and return the token, storing nothing.

    ``on_url`` gets the sign-in link once the callback port is bound.

    Raises:
        SignInError: The port is taken, the wait was stopped or timed out, or
            OpenAI's answer cannot be used.
    """
    redirect_uri = f"http://{DEFAULT_REDIRECT_HOST}:{port}{DEFAULT_REDIRECT_PATH}"
    state = secrets.token_urlsafe(32)
    verifier, challenge = _generate_pkce_pair()
    url = _build_authorize_url(
        client_id=CHATGPT_CLIENT_ID,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=challenge,
        scope=DEFAULT_SCOPE,
    )

    class _Handler(_CallbackHandler):
        server_result: ClassVar[dict[str, str]] = {}
        callback_path = DEFAULT_REDIRECT_PATH
        timeout = _CALLBACK_READ_SECONDS

        def end_headers(self) -> None:
            # The page's own requests (favicon) must not carry the code along.
            self.send_header("Referrer-Policy", "no-referrer")
            super().end_headers()

    result = _Handler.server_result

    def _answered() -> bool:
        return bool(result.get("code") or result.get("error"))

    if _someone_listens(port):
        raise SignInError("port_in_use", PORT_IN_USE)
    with contextlib.ExitStack() as cleanup:
        try:
            server = _CallbackServer((DEFAULT_REDIRECT_HOST, port), _Handler)
            cleanup.callback(server.server_close)
            server6 = _ipv6_server(port, _Handler)
        except OSError as exc:
            raise SignInError("port_in_use", PORT_IN_USE) from exc
        servers = [server]
        if server6 is not None:
            cleanup.callback(server6.server_close)
            servers.append(server6)
        selector = cleanup.enter_context(selectors.DefaultSelector())
        for each in servers:
            each.timeout = 0  # the selector does the waiting
            selector.register(each, selectors.EVENT_READ)
        on_url(url)
        deadline = time.monotonic() + timeout
        while not _answered():
            if stop is not None and stop.is_set():
                raise SignInError("cancelled", SIGN_IN_CANCELLED)
            if time.monotonic() >= deadline:
                raise SignInError("timed_out", SIGN_IN_TIMED_OUT)
            for key, _ in selector.select(_POLL_SECONDS):
                key.fileobj.handle_request()
                if _answered():
                    break  # a second answer in the same round must not overwrite it

    # The state is checked first: a mismatch means the answer is not for us.
    if result.get("state") != state:
        raise SignInError(
            "sign_in_failed",
            "The ChatGPT sign-in answer belongs to another sign-in (state "
            "mismatch). Try again.",
        )
    if "error" in result:
        raise SignInError(
            "sign_in_failed",
            f"OpenAI refused the sign-in ({result['error']}). Try again.",
        )
    try:
        payload = _post_form(
            CHATGPT_TOKEN_URL,
            {
                "grant_type": "authorization_code",
                "code": result["code"],
                "redirect_uri": redirect_uri,
                "client_id": CHATGPT_CLIENT_ID,
                "code_verifier": verifier,
            },
        )
        return _token_from_response(payload)
    except RuntimeError as exc:
        # Upstream's wording points at its own function; say it in ours.
        raise SignInError("sign_in_failed", SIGN_IN_INCOMPLETE) from exc


def status() -> ChatGPTLogin:
    """Read the sign-in state from disk. No network, no refresh."""
    path = token_path()
    try:
        path.stat()  # exists() would hide a folder we cannot reach
    except FileNotFoundError:
        return ChatGPTLogin(signed_in=False)
    except OSError:
        return ChatGPTLogin(signed_in=False, problem="token file is unreadable")
    try:
        token = _FileChatGPTOAuthTokenProvider(path=path)._read_from_disk()
    except Exception:  # whatever is in the file, it is not a token
        return ChatGPTLogin(signed_in=False, problem="token file is unreadable")
    if token is None:
        return ChatGPTLogin(signed_in=False)
    return ChatGPTLogin(signed_in=True, plan=token.plan_type)


def sign_in(
    *,
    on_url: Callable[[str], None],
    stop: threading.Event | None = None,
    timeout: float = SIGN_IN_TIMEOUT,
) -> ChatGPTLogin:
    """Run the browser authorization and store the token.

    ``on_url`` gets the sign-in link; showing or opening it is the caller's.
    """
    path = token_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Stored beside the token file under this attempt's own name, then swapped
    # in under the lock a refresh holds (the browser step takes minutes).
    pending = path.with_name(f"chatgpt.json.{uuid.uuid4().hex[:8]}.new")
    with _token_lock(path):
        pass  # a lock that cannot be taken should show before the browser step
    try:
        token = _authorize(on_url=on_url, stop=stop, timeout=timeout)
        _FileChatGPTOAuthTokenProvider(path=pending).save(token)
        with _token_lock(path):
            os.replace(pending, path)
    finally:
        # The second name is the lock file upstream keeps for a store path.
        for leftover in (pending, pending.with_name(pending.name + ".lock")):
            with contextlib.suppress(OSError):
                leftover.unlink()
    return status()


def login(*, open_browser: bool) -> ChatGPTLogin:
    """The terminal sign-in: print the link, open it when a browser is here.

    Without a local browser the user opens it elsewhere through a forwarded
    callback port.
    """
    if not open_browser:
        print(PORT_FORWARD_HINT)

    def _show(url: str) -> None:
        print(f"\nChatGPT sign-in: open the following URL in a browser:\n  {url}\n")
        if open_browser:
            try:
                webbrowser.open(url)
            except webbrowser.Error as exc:
                logger.warning(
                    "Could not launch a browser: %s. Copy the URL above.", exc
                )

    return sign_in(on_url=_show)


def logout() -> bool:
    """Remove the stored sign-in; False when there was none. Nothing is revoked."""
    path = token_path()
    if not path.parent.exists():
        return False
    with _token_lock(path):
        try:
            os.unlink(path)
        except FileNotFoundError:
            return False
    return True
