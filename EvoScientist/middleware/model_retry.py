"""Retry transient model-call failures inside a wait budget.

Classification walks only ``__cause__`` and exception-group members, never
``__context__``: an error raised while another was being handled says nothing
about that other one. A fallback chain that ran out lists each model's failure
on the error it raises, and one transient model is enough to try again.

``ModelCallBoundaryMiddleware`` marks each provider call on the custom stream
so the stream decoder can drop what a failed call had already emitted.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langgraph.config import get_config
from langgraph.errors import GraphBubbleUp

logger = logging.getLogger(__name__)

INTERACTIVE_CAP_SECONDS = 10.0
UNATTENDED_CAP_SECONDS = 120.0
MIN_WAIT_SECONDS = 1.0
JITTER_FRACTION = 0.1
# Past every cap; a long run of short waits must not overflow the float.
MAX_BACKOFF_DOUBLINGS = 32
# A wait may end this late (timer granularity) before it counts as overrun.
DEADLINE_SLACK_SECONDS = 1.0
RETRYABLE_STATUSES = frozenset({408, 409, 425, 429})
_TRANSIENT_CLASSES = frozenset(
    {
        ("openai", "APIConnectionError"),
        ("openai", "APITimeoutError"),
        ("anthropic", "APIConnectionError"),
        ("anthropic", "APITimeoutError"),
        ("httpx", "TransportError"),
        ("httpx2", "TransportError"),
        ("httpcore", "NetworkError"),
        ("httpcore", "TimeoutException"),
        ("httpcore2", "NetworkError"),
        ("httpcore2", "TimeoutException"),
        # The NVIDIA client: requests for sync calls, aiohttp for async ones.
        ("requests", "ConnectionError"),
        ("requests", "Timeout"),
        ("aiohttp", "ClientConnectionError"),
    }
)
_EXCLUDED_CLASSES = frozenset(
    {
        ("httpx", "UnsupportedProtocol"),
        ("httpx", "LocalProtocolError"),
        ("httpx2", "UnsupportedProtocol"),
        ("httpx2", "LocalProtocolError"),
        # A certificate or TLS setup failure repeats on every attempt.
        ("requests", "SSLError"),
        ("aiohttp", "ClientSSLError"),
    }
)
_ANTHROPIC_TRANSIENT_TYPES = frozenset(
    {"overloaded_error", "api_error", "rate_limit_error"}
)
_OPENROUTER_CODE = re.compile(r"\(code: (\d+)\)")
_REASON_MAX = 200


def _iter_tree(exc: BaseException) -> Iterator[BaseException]:
    """Outside-in over ``__cause__`` and exception-group members, never ``__context__``."""
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(getattr(current, "exceptions", None) or ())
        cause = getattr(current, "__cause__", None)
        if cause is not None:
            pending.append(cause)


def _has_class(exc: BaseException, names: frozenset[tuple[str, str]]) -> bool:
    for base in type(exc).__mro__:
        root = (base.__module__ or "").split(".", 1)[0]
        if (root, base.__name__) in names:
            return True
    return False


def _status_of(exc: BaseException) -> int | None:
    from ..llm.errors import ProviderStreamError, _extract_status_code

    if isinstance(exc, ProviderStreamError):
        return exc.status_code
    return _extract_status_code(exc)


def _status_signal(status: int | None) -> bool | None:
    if status is None:
        return None
    if status in RETRYABLE_STATUSES or 500 <= status < 600:
        return True
    if 400 <= status < 500:
        return False
    return None  # 200 on a mid-stream error: no signal, keep walking


def _code_status(code: object) -> int | None:
    """An HTTP-like status from an error body's ``code``, or ``None`` for a symbolic code."""
    try:
        return int(str(code).strip())
    except (TypeError, ValueError):
        return None


def _message_status(exc: BaseException) -> int | None:
    """The ``(code: N)`` an OpenRouter error message carries, if any."""
    if not isinstance(exc, ValueError):
        return None
    match = _OPENROUTER_CODE.search(str(exc))
    return int(match.group(1)) if match else None


def _body_of(exc: BaseException) -> dict[str, Any] | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        return error if isinstance(error, dict) else body
    if (
        isinstance(exc, ValueError)
        and len(exc.args) == 1
        and isinstance(exc.args[0], dict)
    ):
        return exc.args[0]
    return None


def _error_finished(exc: BaseException) -> Any | None:
    """The error-finished AI message an exception carries, if any."""
    from ..llm.errors import ModelErrorFinishError

    if isinstance(exc, ModelErrorFinishError):
        return exc.ai_message
    message = getattr(exc, "ai_message", None)
    metadata = getattr(message, "response_metadata", None) or {}
    if str(metadata.get("finish_reason", "")).lower() == "error":
        return message
    return None


def _hard_exclusion(exc: BaseException) -> bool:
    from langchain_core.exceptions import ContextOverflowError
    from langgraph.errors import GraphBubbleUp

    from ..llm.errors import ProviderStreamError
    from .error_normalization import ModelOutputTruncatedError
    from .model_fallback import _is_non_fallbackable

    if isinstance(
        exc, GraphBubbleUp | ContextOverflowError | ModelOutputTruncatedError
    ):
        return True
    if isinstance(exc, Exception) and _is_non_fallbackable(exc) is not None:
        return True
    code = (
        exc.code if isinstance(exc, ProviderStreamError) else getattr(exc, "code", None)
    )
    body = _body_of(exc) or {}
    if code == "insufficient_quota" or body.get("code") == "insufficient_quota":
        return True
    if _has_class(exc, _EXCLUDED_CLASSES):
        return True
    # A deterministic status reported in the body or the message counts too.
    statuses = (_status_of(exc), _code_status(body.get("code")), _message_status(exc))
    return any(_status_signal(status) is False for status in statuses)


def _signal(exc: BaseException) -> bool | None:
    from langchain_core.exceptions import ModelError

    if _error_finished(exc) is not None:
        return True
    if isinstance(exc, ModelError):
        return bool(exc.is_retryable)
    status = _status_signal(_status_of(exc))
    if status is not None:
        return status
    if _has_class(exc, _TRANSIENT_CLASSES) or isinstance(
        exc, ConnectionError | TimeoutError
    ):
        return True
    body = _body_of(exc)
    if body is not None:
        if _status_signal(_code_status(body.get("code"))) is True:
            return True
        if body.get("type") == "server_error":
            return True
        if body.get("type") in _ANTHROPIC_TRANSIENT_TYPES:
            return True
    if _status_signal(_message_status(exc)) is True:
        return True
    return None


def _transient_alone(exc: BaseException) -> bool:
    """One failure judged by its own tree: exclusions anywhere in it win."""
    tree = list(_iter_tree(exc))
    if any(_hard_exclusion(item) for item in tree):
        return False
    for item in tree:
        signal = _signal(item)
        if signal is not None:
            return signal
    return False


def retryable_failures(exc: BaseException) -> list[BaseException]:
    """The failures that make another attempt worth it; empty when none does.

    After an exhausted fallback chain these are the models that failed
    transiently: the next attempt starts from the primary again.
    """
    from .model_fallback import chain_failures

    return [f for f in chain_failures(exc) or (exc,) if _transient_alone(f)]


def is_transient(exc: BaseException) -> bool:
    """Whether a failed attempt is worth retrying."""
    return bool(retryable_failures(exc))


def has_error_finish(exc: BaseException) -> bool:
    """Whether the failure is a response the provider ended with ``finish_reason == "error"``."""
    return any(_error_finished(item) is not None for item in _iter_tree(exc))


def _headers_of(exc: BaseException) -> list[Any]:
    found = []
    for holder in (
        getattr(exc, "response", None),
        exc,
        getattr(exc, "raw_response", None),
    ):
        headers = getattr(holder, "headers", None)
        if headers is not None and hasattr(headers, "get"):
            found.append(headers)
    return found


def _finite(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _parse_retry_after(headers: Any) -> float | None:
    millis = headers.get("retry-after-ms")
    if millis is not None:
        value = _finite(str(millis).strip())
        if value is not None and value > 0:
            return value / 1000.0
    raw = headers.get("retry-after")
    if raw is None:
        return None
    raw = str(raw).strip()
    value = _finite(raw)
    if value is not None:
        return value
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return (when - datetime.now(UTC)).total_seconds()


def retry_after_seconds(exc: BaseException) -> float | None:
    """A positive ``Retry-After`` from the tree, or ``None`` (absent, zero, past, malformed)."""
    for item in _iter_tree(exc):
        for headers in _headers_of(item):
            value = _parse_retry_after(headers)
            if value is not None and value > 0:
                return value
    return None


def describe_failure(exc: BaseException) -> str:
    from ..llm.errors import ProviderStreamError

    if isinstance(exc, ProviderStreamError):
        text = f"{exc.status_code}: {exc.message}" if exc.status_code else exc.message
    else:
        text = f"{type(exc).__name__}: {exc}"
    text = " ".join(text.split())
    return text if len(text) <= _REASON_MAX else text[: _REASON_MAX - 3] + "..."


# --- the middleware ---------------------------------------------------------


def _agent_name() -> str:
    """The name of the agent this call runs in, or "" outside a run."""
    try:
        metadata = get_config().get("metadata") or {}
    except RuntimeError:
        return ""
    name = metadata.get("lc_agent_name")
    return name if isinstance(name, str) else ""


def _emit(request: ModelRequest, event: dict[str, Any]) -> None:
    """Write one custom stream event; a run without a stream writer stays silent."""
    writer = getattr(getattr(request, "runtime", None), "stream_writer", None)
    if writer is None:
        return
    try:
        writer(event)
    except Exception:
        logger.debug(
            "%s: stream_writer raised; ignoring", event.get("type"), exc_info=True
        )


@dataclass(frozen=True)
class RetryTier:
    wait_budget_seconds: float
    unattended: bool

    @property
    def cap_seconds(self) -> float:
        return UNATTENDED_CAP_SECONDS if self.unattended else INTERACTIVE_CAP_SECONDS


def _wait_after(exc: BaseException, backoff: float) -> float:
    """The provider's ``Retry-After`` for *exc* when it sent one, else *backoff*."""
    wait = retry_after_seconds(exc)
    return max(round(backoff if wait is None else wait, 3), MIN_WAIT_SECONDS)


class ModelRetryMiddleware(AgentMiddleware):
    """Retry a transient model-call failure inside a per-call wait budget.

    Sits outside the fallback chain: one attempt is primary plus fallbacks.
    The budget starts at the first failure: the waits and the retried attempts
    count against it, the first attempt's own time does not.
    """

    name = "model_retry"

    def __init__(
        self,
        *,
        wait_budget_seconds: float,
        unattended: bool,
        honor_run_key: bool = True,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        sync_sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        super().__init__()
        self._tier = RetryTier(max(0.0, float(wait_budget_seconds)), bool(unattended))
        self._honor_run_key = honor_run_key
        self._clock = clock
        self._sleep = sleep
        self._sync_sleep = sync_sleep
        self._rng = rng

    def _effective_tier(self) -> RetryTier:
        if not self._honor_run_key:
            return self._tier
        from ..backends import MODEL_RETRY_RUN_KEY

        try:
            marker = (get_config().get("configurable") or {}).get(MODEL_RETRY_RUN_KEY)
        except RuntimeError:
            return self._tier
        if not isinstance(marker, dict):
            return self._tier
        budget = marker.get("wait_budget_seconds")
        # The config's rule: whole seconds as an int, anything else is the default.
        try:
            if isinstance(budget, bool) or not isinstance(budget, int):
                raise TypeError
            budget = float(max(0, budget))
        except (TypeError, OverflowError):
            budget = self._tier.wait_budget_seconds
        return RetryTier(budget, bool(marker.get("unattended", self._tier.unattended)))

    @staticmethod
    def _check_response(response: ModelResponse) -> ModelResponse:
        from ..llm.errors import ModelErrorFinishError
        from .error_normalization import _error_finished_message

        errored = _error_finished_message(response)
        if errored is not None:
            raise ModelErrorFinishError(errored)
        return response

    def _backoff(self, attempt: int, tier: RetryTier) -> float:
        base = float(2 ** min(attempt - 1, MAX_BACKOFF_DOUBLINGS))
        jittered = base * (1.0 + JITTER_FRACTION * (2.0 * self._rng() - 1.0))
        return min(jittered, tier.cap_seconds)

    def _plan_retry(
        self, exc: BaseException, attempt: int, tier: RetryTier, deadline: float
    ) -> tuple[float, dict[str, Any]] | None:
        """The wait and event for the next attempt, or ``None`` to give up now."""
        failures = [] if isinstance(exc, GraphBubbleUp) else retryable_failures(exc)
        if not failures:
            return None
        # Any model that may answer makes the attempt worth making, so the
        # shortest wait is sat out; one backoff draw, so a tie names the earlier model.
        backoff = self._backoff(attempt, tier)
        wait, failure = min(
            ((_wait_after(item, backoff), item) for item in failures),
            key=lambda planned: planned[0],
        )
        now = self._clock()
        if now + wait > deadline:
            logger.warning(
                "model call still failing after %d attempt%s, %.0f s after it first failed; giving up (%s)",
                attempt,
                "" if attempt == 1 else "s",
                now - (deadline - tier.wait_budget_seconds),
                describe_failure(failure),
            )
            return None
        left = deadline - now - wait
        event = {
            "type": "model_retry",
            "attempt": attempt + 1,
            "wait_seconds": round(wait, 1),
            "reason": describe_failure(failure),
            "budget_left_seconds": round(left, 1),
            "cause": "finish_reason" if has_error_finish(failure) else "exception",
            # A namespace is not always named after its agent downstream.
            "agent": _agent_name(),
        }
        logger.warning(
            "model call failed (%s); retrying in %.0f s (attempt %d, %.0f s of %.0f s budget left)",
            event["reason"],
            wait,
            attempt + 1,
            left,
            tier.wait_budget_seconds,
        )
        return wait, event

    def _overran(self, reason: str, tier: RetryTier, deadline: float) -> bool:
        """Whether the wait itself ran past the deadline (suspended process, stalled loop)."""
        late = self._clock() - deadline
        if late <= DEADLINE_SLACK_SECONDS:
            return False
        logger.warning(
            "model call retry wait ran %.0f s past the %.0f s budget; giving up (%s)",
            late,
            tier.wait_budget_seconds,
            reason,
        )
        return True

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        tier = self._effective_tier()
        if tier.wait_budget_seconds <= 0:
            return self._check_response(await handler(request))
        deadline: float | None = None
        attempt = 1
        while True:
            try:
                return self._check_response(await handler(request))
            except Exception as exc:
                if deadline is None:
                    deadline = self._clock() + tier.wait_budget_seconds
                plan = self._plan_retry(exc, attempt, tier, deadline)
                if plan is None:
                    raise
                wait, event = plan
                _emit(request, event)
                await self._sleep(wait)
                if self._overran(event["reason"], tier, deadline):
                    raise
                attempt += 1

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        tier = self._effective_tier()
        if tier.wait_budget_seconds <= 0:
            return self._check_response(handler(request))
        deadline: float | None = None
        attempt = 1
        while True:
            try:
                return self._check_response(handler(request))
            except Exception as exc:
                if deadline is None:
                    deadline = self._clock() + tier.wait_budget_seconds
                plan = self._plan_retry(exc, attempt, tier, deadline)
                if plan is None:
                    raise
                wait, event = plan
                _emit(request, event)
                self._sync_sleep(wait)
                if self._overran(event["reason"], tier, deadline):
                    raise
                attempt += 1


class ModelCallBoundaryMiddleware(AgentMiddleware):
    """Mark each provider call on the custom stream: a start, then how it ended.

    The stream decoder drops a failed call's output by these marks. They travel
    as custom events because the server transport strips message metadata.
    Mounted inside the fallback chain, so every model of a chain gets its own pair.
    """

    name = "model_call_boundary"

    @staticmethod
    def _start(request: ModelRequest) -> str:
        call = uuid.uuid4().hex[:12]
        _emit(request, {"type": "model_call", "phase": "start", "call": call})
        return call

    @staticmethod
    def _end(request: ModelRequest, call: str, status: str) -> None:
        _emit(
            request,
            {"type": "model_call", "phase": "end", "call": call, "status": status},
        )

    @staticmethod
    def _status(response: ModelResponse) -> str:
        from .error_normalization import _error_finished_message

        return "error" if _error_finished_message(response) is not None else "clean"

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        call = self._start(request)
        try:
            response = await handler(request)
        except BaseException:
            self._end(request, call, "failed")
            raise
        self._end(request, call, self._status(response))
        return response

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        call = self._start(request)
        try:
            response = handler(request)
        except BaseException:
            self._end(request, call, "failed")
            raise
        self._end(request, call, self._status(response))
        return response
