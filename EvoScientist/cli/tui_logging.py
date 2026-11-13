"""Route runtime warnings to Textual while preserving CLI logging on exit."""

from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import monotonic
from typing import TYPE_CHECKING

from rich.logging import RichHandler

if TYPE_CHECKING:
    from textual.app import App


class TUIWarningHandler(logging.Handler):
    """Notify from any thread, remembering only the last 256 distinct messages."""

    def __init__(
        self,
        app: App,
        handlers: list[logging.Handler],
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        super().__init__()
        self.app = app
        self.handlers = handlers
        self.clock = clock
        self.seen: OrderedDict[tuple[int, str], None] = OrderedDict()
        self._last_toast: dict[str, float] = {}
        self._pending: list[logging.LogRecord] = []
        self.active = True

    def _forward(self, record: logging.LogRecord) -> None:
        for handler in self.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    def _notify(self, record: logging.LogRecord) -> bool:
        """Post the warning to the app; False if the app refused it."""
        if not self.active:
            return False
        from textual.notifications import Notification, Notify

        message = record.getMessage()
        exc = record.exc_info[1] if record.exc_info else None
        if exc is not None:
            try:
                text = str(exc)
            except Exception:  # a broken __str__ must not break the log call
                text = ""
            message = f"{message}: {type(exc).__name__}"
            if text and text not in message:
                message = f"{message}: {text}"
        # Keyed by level too, so an earlier warning never hides the same error.
        key = (record.levelno, message)
        if key in self.seen:
            return True
        throttle = getattr(record, "notify_throttle", None)
        if throttle is not None and record.levelno < logging.ERROR:
            # Records that opt in (a retry loop) toast once per key every 20 s;
            # errors and every other warning are never throttled.
            now = self.clock()
            last = self._last_toast.get(throttle)
            if last is not None and now - last < 20:
                # Throttled records are neither toasted nor forwarded into TUI stdout.
                return True
            self._last_toast[throttle] = now
        self.seen[key] = None
        if len(self.seen) > 256:
            self.seen.popitem(last=False)
        # App.notify drops post_message's result; a closing app refuses the post.
        severity = "error" if record.levelno >= logging.ERROR else "warning"
        notification = Notification(message, severity=severity, timeout=20)
        return self.app.post_message(Notify(notification))

    def emit(self, record: logging.LogRecord) -> None:
        with self.lock:
            if not self.active:
                self._forward(record)
            elif self._stopping():
                self._pending.append(record)
            elif record.levelno < logging.WARNING:
                self._forward(record)
            else:
                try:
                    posted = self._notify(record)
                except RuntimeError:
                    # The event loop closed between the check and the post.
                    posted = False
                if not posted:
                    self._pending.append(record)

    def _stopping(self) -> bool:
        # Once exit() has queued its stop marker, a posted notification is
        # never shown, though is_running stays true until the loop ends.
        # Relies on Textual's private App._exit (pinned textual<8.2.7); recheck on bump.
        # _exiting is set by the TUI before it stops channels.
        app = self.app
        return (
            not app.is_running
            or getattr(app, "_exit", False)
            or getattr(app, "_exiting", False)
        )


@contextmanager
def tui_warning_notifications(app: App) -> Iterator[None]:
    """Replace console handlers and the warning hook for one app lifetime."""
    root = logging.getLogger()
    handlers = root.handlers[:]
    showwarning = warnings.showwarning
    console_handlers = [h for h in handlers if isinstance(h, RichHandler)]
    handler = TUIWarningHandler(app, console_handlers)

    def show_warning(message, category, filename, lineno, file=None, line=None):
        if file is not None:
            showwarning(message, category, filename, lineno, file, line)
            return
        record = logging.LogRecord(
            "py.warnings", logging.WARNING, filename, lineno, str(message), (), None
        )
        handler.handle(record)

    try:
        root.handlers = [h for h in handlers if h not in console_handlers] + [handler]
        warnings.showwarning = show_warning
        yield
    finally:
        with handler.lock:
            handler.active = False
            warnings.showwarning = showwarning
            root.handlers = handlers
            pending, handler._pending = handler._pending, []
        try:
            for record in pending:
                try:
                    handler._forward(record)
                except Exception:
                    handler.handleError(record)
        finally:
            handler.close()
