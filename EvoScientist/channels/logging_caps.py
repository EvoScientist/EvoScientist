"""Shared caps for loggers whose output can carry channel credentials.

HTTP request URLs can embed channel credentials: Telegram's Bot API puts the
bot token in the URL path, and DingTalk/WeCom send access_token as a query
parameter. ``httpx`` logs every request URL at INFO and ``python-telegram-bot``
logs its API URL at DEBUG, so these loggers are capped above the level at which
they print those URLs.

Both ``EvoSci serve`` (``cli/commands.py``) and the per-channel standalone
entry points (``channels/standalone.py``) apply the same caps. Keep them in one
place so a new cap or a level change needs a single edit (issue #666).
"""

import logging

# Logger name -> cap level. The cap sits above the level at which the logger
# prints the credential-bearing URL, which keeps the URL out of the log.
CREDENTIAL_URL_LOGGERS: dict[str, int] = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "telegram": logging.INFO,
}


def quiet_credential_url_loggers() -> None:
    """Cap the loggers that print credential-bearing HTTP URLs.

    Uses the fixed cap levels from :data:`CREDENTIAL_URL_LOGGERS`; the root
    handler's own level already governs whether a record at or above the cap
    is emitted, so no ``max(resolved_level, ...)`` is needed here.
    """
    for name, level in CREDENTIAL_URL_LOGGERS.items():
        logging.getLogger(name).setLevel(level)
