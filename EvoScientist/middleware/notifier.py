"""Notifier port for async-task / background-process notifications.

The ``async_watcher`` and ``background`` middleware need to (a) pre-cancel a
stale watcher, (b) spawn a watcher, and (c) enqueue a completion notification.
Those are infrastructure calls with behaviour, not display events — so they do
not belong on the :mod:`~EvoScientist.middleware.events` display sink.

Instead the composition root injects a :class:`NotifierPort`: a small,
structural interface implemented by ``EvoScientist.cli.async_notifier`` (the
module itself satisfies it — its public functions match these methods). The
middleware depends only on this port, never on ``EvoScientist.cli``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import asyncio


class NotifierPort(Protocol):
    """Behaviour the notifier layer exposes to middleware.

    ``EvoScientist.cli.async_notifier`` implements this structurally; the
    composition root passes that module in as the port.
    """

    def pre_cancel_watcher(self, task_id: str) -> None:
        """Cancel any in-flight watcher registered for ``task_id``.

        No-op when there is no live watcher. Swallows cancellation errors —
        a failed pre-cancel only risks a stale success notification, never a
        crash of the launching tool call.
        """
        ...

    def spawn_watcher(
        self,
        client: Any,
        thread_id: str,
        run_id: str,
        agent_name: str,
        prompt: str = "",
        origin_cli_thread_id: str | None = None,
    ) -> asyncio.Task[None]:
        """Spawn a run watcher on the caller's asyncio loop."""
        ...

    def enqueue_task_notification(self, notification: Any) -> None:
        """Route a completed-task notification onto the consumer queue."""
        ...

    def enqueue_bg_process_notification(
        self,
        *,
        task_id: str,
        agent_name: str,
        status: str,
        prompt: str = "",
        origin_cli_thread_id: str | None = None,
    ) -> None:
        """Build and enqueue a background-process completion notification.

        The notifier owns the notification type, so the background middleware
        never constructs it (and never imports the CLI layer).
        """
        ...
