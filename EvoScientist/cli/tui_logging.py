"""Route runtime warnings to Textual while preserving CLI logging on exit."""

from __future__ import annotations

import logging
import threading
import warnings
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from rich.logging import RichHandler

if TYPE_CHECKING:
    from textual.app import App


class TUIWarningHandler(logging.Handler):
    """Notify on the app thread, remembering only the last 256 distinct messages."""

    def __init__(self, app: App, handlers: list[logging.Handler]) -> None:
        super().__init__()
        self.app = app
        self.handlers = handlers
        self.app_thread = threading.get_ident()
        self.seen: OrderedDict[str, None] = OrderedDict()
        self.active = True

    def createLock(self) -> None:
        # call_from_thread waits for the UI. Holding a handler lock here would
        # deadlock if the UI logs too; deduplication happens only on the UI.
        self.lock = None

    def _forward(self, record: logging.LogRecord) -> None:
        for handler in self.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    def _notify(self, record: logging.LogRecord) -> None:
        if not self.active:
            self._forward(record)
            return
        from .commands import _grader_failure_on_one_line

        if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
            record = _grader_failure_on_one_line(record)
        message = record.getMessage()
        if message in self.seen:
            return
        self.seen[message] = None
        if len(self.seen) > 256:
            self.seen.popitem(last=False)
        self.app.notify(message, severity="warning", timeout=20, markup=False)

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno != logging.WARNING or not self.active:
            self._forward(record)
        elif threading.get_ident() == self.app_thread:
            self._notify(record)
        else:
            try:
                self.app.call_from_thread(self._notify, record)
            except RuntimeError:
                # Textual rejects the hop once its event loop has stopped.
                self._forward(record)


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
        handler.active = False
        warnings.showwarning = showwarning
        root.handlers = handlers
        handler.close()
