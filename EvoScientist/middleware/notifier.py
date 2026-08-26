"""Notifier port for async-task / background-process notifications.

The ``background`` middleware needs to enqueue a completion notification when an
OS background process exits. That is an infrastructure call with behaviour, not
a display event — so it does not belong on the
:mod:`~EvoScientist.middleware.events` display sink.

Instead the composition root injects a :class:`NotifierPort`: a small,
structural interface implemented by ``EvoScientist.cli.async_notifier`` (the
module itself satisfies it — its public functions match these methods). The
middleware depends only on this port, never on ``EvoScientist.cli``.
"""

from __future__ import annotations

from typing import Any, Protocol


class NotifierPort(Protocol):
    """Behaviour the notifier layer exposes to middleware.

    ``EvoScientist.cli.async_notifier`` implements this structurally; the
    composition root passes that module in as the port.
    """

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
