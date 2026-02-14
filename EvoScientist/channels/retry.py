"""Configurable exponential-backoff retry for async callables."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behaviour."""

    attempts: int = 3
    min_delay_s: float = 0.3
    max_delay_s: float = 30.0
    jitter: float = 0.1  # ±10 % random offset


@dataclass
class RetryInfo:
    """Information passed to the *on_retry* callback."""

    attempt: int
    max_attempts: int
    delay_s: float
    error: Exception
    label: str | None = None


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    config: RetryConfig = RetryConfig(),
    *,
    should_retry: Callable[[Exception, int], bool] | None = None,
    retry_after_s: Callable[[Exception], float | None] | None = None,
    on_retry: Callable[[RetryInfo], None] | None = None,
    label: str | None = None,
) -> T:
    """Execute *fn* with exponential-backoff retry.

    Parameters
    ----------
    fn:
        Zero-argument async factory — called on every attempt so the
        awaitable is always fresh.
    config:
        Retry timing / attempt parameters.
    should_retry:
        ``(exception, attempt) -> bool``.  Return ``False`` to abort
        immediately.  When *None* every exception is retried.
    retry_after_s:
        ``(exception) -> seconds | None``.  If the server provides a
        ``Retry-After`` value (e.g. HTTP 429), return it here.  The
        actual delay will be ``max(server_value, min_delay_s)``.
    on_retry:
        Optional callback invoked before each retry sleep.
    label:
        Human-readable label included in :class:`RetryInfo`.
    """
    last_exc: Exception | None = None
    for attempt in range(1, config.attempts + 1):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc

            if attempt >= config.attempts:
                raise

            if should_retry is not None and not should_retry(exc, attempt):
                raise

            # Compute delay
            server_delay: float | None = None
            if retry_after_s is not None:
                server_delay = retry_after_s(exc)

            if server_delay is not None:
                base_delay = max(server_delay, config.min_delay_s)
            else:
                base_delay = config.min_delay_s * (2 ** (attempt - 1))

            # Apply jitter
            jittered = base_delay * (1 + random.uniform(-config.jitter, config.jitter))

            # Clamp to [min_delay_s, max_delay_s]
            delay = max(config.min_delay_s, min(jittered, config.max_delay_s))

            if on_retry is not None:
                on_retry(RetryInfo(
                    attempt=attempt,
                    max_attempts=config.attempts,
                    delay_s=delay,
                    error=exc,
                    label=label,
                ))

            await asyncio.sleep(delay)

    # Should never reach here, but satisfy the type checker.
    assert last_exc is not None  # noqa: S101
    raise last_exc


# ── Presets ──────────────────────────────────────────────────────────

TELEGRAM_RETRY = RetryConfig(attempts=3, min_delay_s=0.4, max_delay_s=30.0, jitter=0.1)
DEFAULT_RETRY = RetryConfig()

# Discord, Slack, Teams, Feishu all use the same config (attempts=3,
# min_delay_s=0.5, max_delay_s=30.0, jitter=0.1) — close enough to
# DEFAULT_RETRY that separate presets add no value.  Channels that
# don't appear in RETRY_PRESETS already fall back to DEFAULT_RETRY.

RETRY_PRESETS: dict[str, RetryConfig] = {
    "telegram": TELEGRAM_RETRY,
}
