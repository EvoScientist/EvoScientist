"""Shared caps for loggers whose output can carry channel credentials.

HTTP request URLs can embed channel credentials: Telegram's Bot API puts the
bot token in the URL path, and DingTalk/WeCom send access_token as a query
parameter. ``httpx`` logs every request URL at INFO and ``python-telegram-bot``
logs its API URL at DEBUG. Feishu's long connection goes through ``websockets``,
which logs the handshake request path, query string included, at DEBUG. These
loggers are capped above the level at which they print those URLs.
"""

import logging

# Logger name -> cap level. The cap sits above the level at which the logger
# prints the credential-bearing URL, which keeps the URL out of the log.
CREDENTIAL_URL_LOGGERS: dict[str, int] = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "telegram": logging.INFO,
    "websockets": logging.INFO,
}


def quiet_credential_url_loggers() -> None:
    """Cap the loggers that print credential-bearing HTTP URLs."""
    for name, level in CREDENTIAL_URL_LOGGERS.items():
        logging.getLogger(name).setLevel(level)
