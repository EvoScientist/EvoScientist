"""The standalone Telegram channel must never write the bot token to its log (#567).

The Bot API puts the token in the URL path, and httpx logs every request URL at INFO.
"""

import io
import logging

import pytest

from EvoScientist.channels.telegram import serve

TOKEN = "123456789:AAEhBP0av28cXN5wC-Tp1qYvZ_9sMUL3h8Q"
URL = f"https://api.telegram.org/bot{TOKEN}/getUpdates"


@pytest.fixture
def root_log():
    root = logging.getLogger()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    saved_handlers, saved_level = root.handlers[:], root.level
    httpx_level = logging.getLogger("httpx").level
    root.handlers = [handler]
    root.setLevel(logging.DEBUG)
    yield stream
    root.handlers, root.level = saved_handlers, saved_level
    logging.getLogger("httpx").setLevel(httpx_level)


def test_request_urls_are_not_logged_with_the_token(root_log):
    serve.protect_bot_token(TOKEN)
    logging.getLogger("httpx").info('HTTP Request: POST %s "HTTP/1.1 200 OK"', URL)
    logging.getLogger("telegram.ext").debug("polling %s", URL)
    out = root_log.getvalue()
    assert TOKEN not in out
    assert "getUpdates" in out  # the rest of the line is still there


def test_the_token_is_redacted_in_tracebacks_too(root_log):
    serve.protect_bot_token(TOKEN)
    try:
        raise RuntimeError(f"Client error '404 Not Found' for url '{URL}'")
    except RuntimeError:
        logging.getLogger("telegram.ext").exception("polling failed")
    out = root_log.getvalue()
    assert "polling failed" in out
    assert "RuntimeError" in out
    assert TOKEN not in out


def test_httpx_request_lines_are_quiet_like_evosci_serve(root_log):
    serve.protect_bot_token(TOKEN)
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
