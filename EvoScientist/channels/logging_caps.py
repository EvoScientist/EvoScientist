"""Shared caps for loggers whose output can carry channel credentials.

HTTP request URLs can embed channel credentials: Telegram's Bot API puts the
bot token in the URL path, and DingTalk/WeCom send access_token as a query
parameter. ``httpx`` logs every request URL at INFO and ``python-telegram-bot``
logs its API URL at DEBUG. Feishu's long connection goes through ``websockets``,
which logs the handshake request path, query string included, at DEBUG.
``slack_sdk`` logs every Web API response body at DEBUG, and the Socket Mode
connection URL in it carries a ticket; it also logs a proxy URL it reads from
the environment. These loggers are capped above the level at which they print
those URLs.

Caps cannot reach ERROR records or tracebacks whose exception text quotes a
URL, such as a connect timeout to a ticketed WebSocket URL. The root handlers
therefore also scrub the userinfo and query of every URL they write.
"""

import logging
import re

# Logger name -> cap level. The cap sits above the level at which the logger
# prints the credential-bearing URL, which keeps the URL out of the log.
CREDENTIAL_URL_LOGGERS: dict[str, int] = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "telegram": logging.INFO,
    "websockets": logging.INFO,
    "slack_sdk": logging.INFO,
}


def quiet_credential_url_loggers() -> None:
    """Cap the loggers that print credential-bearing HTTP URLs."""
    for name, level in CREDENTIAL_URL_LOGGERS.items():
        logging.getLogger(name).setLevel(level)


# Everything between the scheme and the last "@" of the authority: a proxy
# password written with a raw "@" still ends up inside the match.
_URL_USERINFO_RE = re.compile(r"(://)[^\s/?#]*@")
# The query, up to whitespace or a quote, so a URL quoted inside an exception
# message keeps its closing quote. Commas and brackets are valid inside a query.
# The query is optional so that a URL without one is consumed in one step:
# requiring it rescans a long run of text once per "://" in it.
_URL_QUERY_RE = re.compile(r"(://[^\s?#]*)(\?[^\s'\"]*)?")


def scrub_url_secrets(text: str) -> str:
    """*text* with the userinfo and query of every URL replaced by ``***``.

    Proxy URLs carry a password in their userinfo, and connection tickets and
    access tokens ride in query strings.
    """
    if "://" not in text:
        return text
    text = _URL_USERINFO_RE.sub(r"\1***@", text)
    return _URL_QUERY_RE.sub(lambda m: m[1] + ("?***" if m[2] else ""), text)


class RedactingFormatter(logging.Formatter):
    """Format with *inner*, then scrub URL secrets from the whole text.

    Working on the formatted string covers the message, its arguments and the
    traceback, including exception text that third-party libraries build.
    """

    def __init__(self, inner: logging.Formatter | None = None) -> None:
        super().__init__()
        self._inner = inner

    def format(self, record: logging.LogRecord) -> str:
        if self._inner is not None:
            text = self._inner.format(record)
        else:
            text = super().format(record)
        return scrub_url_secrets(text)


def redact_root_handlers() -> None:
    """Scrub URL secrets from everything the root logger's handlers write."""
    for handler in logging.getLogger().handlers:
        if not isinstance(handler.formatter, RedactingFormatter):
            handler.setFormatter(RedactingFormatter(handler.formatter))
