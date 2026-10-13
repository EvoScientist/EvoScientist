"""Sign in with a ChatGPT plan and call the Codex backend directly.

Thin wrapper around ``langchain-openai``'s ChatGPT OAuth support, and the only
module that imports its private names. Re-check them when the
``langchain-openai`` floor moves: ``_ChatOpenAICodex``,
``_FileChatGPTOAuthTokenProvider`` (with ``_read_from_disk`` and ``_cached``),
and its two messages for a refused refresh (matched in ``_refresh_rejected``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    _FileChatGPTOAuthTokenProvider,
    login_chatgpt,
)

SIGN_IN_HINT = (
    'ChatGPT sign-in required (openai_auth_mode is "oauth"). '
    "Start EvoSci in a terminal to sign in."
)
PORT_FORWARD_HINT = (
    "No browser on this machine? Forward the callback port, then open the "
    "link on your own computer: ssh -L 1455:localhost:1455 <this-host>"
)
PORT_IN_USE = (
    "Port 1455 is in use (another sign-in may be running). Close it and try again."
)
SIGN_IN_INCOMPLETE = "OpenAI did not complete the sign-in. Try again."
DEFAULT_INSTRUCTIONS = "You are a helpful assistant."


class NotSignedInError(RuntimeError):
    """No usable sign-in: no token file, an unreadable one, or a rejected refresh."""

    def __init__(self) -> None:
        super().__init__(SIGN_IN_HINT)


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
        return EvoChatOpenAICodex(
            model=model_id, token_provider=_LockedTokenProvider(path), **kwargs
        )


def status() -> ChatGPTLogin:
    """Read the sign-in state from disk. No network, no refresh."""
    path = token_path()
    if not path.exists():
        return ChatGPTLogin(signed_in=False)
    try:
        token = _FileChatGPTOAuthTokenProvider(path=path)._read_from_disk()
    except Exception:  # whatever is in the file, it is not a token
        return ChatGPTLogin(signed_in=False, problem="token file is unreadable")
    if token is None:
        return ChatGPTLogin(signed_in=False)
    return ChatGPTLogin(signed_in=True, plan=token.plan_type)


def login(*, open_browser: bool) -> ChatGPTLogin:
    """Run the browser sign-in and store the token.

    Upstream prints the link either way; without a local browser the user
    opens it elsewhere through a forwarded callback port.
    """
    path = token_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not open_browser:
        print(PORT_FORWARD_HINT)
    # Stored beside the token file under this attempt's own name, then swapped
    # in under the lock a refresh holds (the browser step takes minutes).
    pending = path.with_name(f"chatgpt.json.{uuid.uuid4().hex[:8]}.new")
    with _token_lock(path):
        pass  # a lock that cannot be taken should show before the browser step
    try:
        login_chatgpt(store_path=pending, open_browser=open_browser)
        with _token_lock(path):
            os.replace(pending, path)
    except RuntimeError as exc:
        if isinstance(exc.__cause__, OSError):
            raise RuntimeError(PORT_IN_USE) from exc
        if "login_chatgpt" in str(exc):
            # Upstream's advice is to call its Python function.
            raise RuntimeError(SIGN_IN_INCOMPLETE) from exc
        raise
    finally:
        # The second name is the lock file upstream keeps for a store path.
        for leftover in (pending, pending.with_name(pending.name + ".lock")):
            with contextlib.suppress(OSError):
                leftover.unlink()
    return status()
