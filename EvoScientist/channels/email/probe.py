"""Email credential validation."""

import imaplib
import smtplib
import ssl
import logging

logger = logging.getLogger(__name__)


async def validate_email_imap(
    host: str, port: int, username: str, password: str,
    use_ssl: bool = True,
) -> tuple[bool, str]:
    """Validate IMAP credentials."""
    if not host or not username or not password:
        return False, "host, username, and password are required"

    import asyncio
    loop = asyncio.get_event_loop()

    def _check():
        try:
            if use_ssl:
                ctx = ssl.create_default_context()
                conn = imaplib.IMAP4_SSL(host, port, ssl_context=ctx)
            else:
                conn = imaplib.IMAP4(host, port)
            conn.login(username, password)
            conn.logout()
            return True, "IMAP credentials valid"
        except imaplib.IMAP4.error as e:
            return False, f"IMAP auth failed: {e}"
        except Exception as e:
            return False, f"IMAP error: {e}"

    return await loop.run_in_executor(None, _check)


async def validate_email_smtp(
    host: str, port: int, username: str, password: str,
    use_tls: bool = True,
) -> tuple[bool, str]:
    """Validate SMTP credentials."""
    if not host or not username or not password:
        return False, "host, username, and password are required"

    import asyncio
    loop = asyncio.get_event_loop()

    def _check():
        try:
            if use_tls:
                server = smtplib.SMTP(host, port, timeout=10)
                server.starttls()
            else:
                ctx = ssl.create_default_context()
                server = smtplib.SMTP_SSL(host, port, context=ctx, timeout=10)
            server.login(username, password)
            server.quit()
            return True, "SMTP credentials valid"
        except smtplib.SMTPAuthenticationError as e:
            return False, f"SMTP auth failed: {e}"
        except Exception as e:
            return False, f"SMTP error: {e}"

    return await loop.run_in_executor(None, _check)
