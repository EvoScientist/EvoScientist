"""Tests for the credential-logging caps and URL scrubbing."""

import io
import logging
import sys

from EvoScientist.channels.logging_caps import (
    RedactingFormatter,
    redact_root_handlers,
    scrub_url_secrets,
)


def test_scrub_url_secrets_hides_query_and_userinfo():
    text = scrub_url_secrets(
        "timeout to wss://gw.example.com/connect?ticket=t-secret "
        "via http://user:p@ss-secret@proxy:3128"
    )

    assert "t-secret" not in text
    assert "ss-secret" not in text
    assert "wss://gw.example.com/connect?***" in text
    assert "http://***@proxy:3128" in text


def test_scrub_url_secrets_hides_query_past_a_comma():
    text = scrub_url_secrets("GET https://api/x?fields=a,b&access_token=t-secret")

    assert "t-secret" not in text


def test_scrub_url_secrets_keeps_closing_quote():
    assert scrub_url_secrets("url='wss://gw/c?ticket=x')").endswith("?***')")


def test_redacting_formatter_scrubs_traceback_text():
    try:
        raise TimeoutError("Connection timeout to host wss://gw/c?ticket=t-secret")
    except TimeoutError:
        exc_info = sys.exc_info()
    record = logging.LogRecord(
        "lib", logging.ERROR, __file__, 1, "Failed to connect", None, exc_info
    )

    text = RedactingFormatter().format(record)

    assert "Traceback" in text
    assert "t-secret" not in text


def test_redact_root_handlers_keeps_existing_format(monkeypatch):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    monkeypatch.setattr(logging.getLogger(), "handlers", [handler])

    redact_root_handlers()
    redact_root_handlers()
    logging.getLogger("lib").error("GET https://api/x?access_token=a-secret")

    assert stream.getvalue().startswith("[ERROR] GET https://api/x?***")
    assert "a-secret" not in stream.getvalue()
    assert not isinstance(handler.formatter._inner, RedactingFormatter)
