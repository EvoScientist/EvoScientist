"""Email channel for EvoScientist.

Uses IMAP polling for inbound + SMTP for outbound. Pure Python, no extra deps.

Usage in config:
    channel_enabled = "email"
    email_imap_host = "imap.gmail.com"
    email_smtp_host = "smtp.gmail.com"
    ...
"""

from .channel import EmailChannel, EmailConfig
from ..channel_manager import register_channel, _parse_csv

__all__ = ["EmailChannel", "EmailConfig"]


def create_from_config(config) -> EmailChannel:
    allowed = _parse_csv(getattr(config, "email_allowed_senders", ""))
    return EmailChannel(EmailConfig(
        imap_host=getattr(config, "email_imap_host", ""),
        imap_port=int(getattr(config, "email_imap_port", 993)),
        imap_username=getattr(config, "email_imap_username", ""),
        imap_password=getattr(config, "email_imap_password", ""),
        imap_mailbox=getattr(config, "email_imap_mailbox", "INBOX"),
        imap_use_ssl=getattr(config, "email_imap_use_ssl", True),
        smtp_host=getattr(config, "email_smtp_host", ""),
        smtp_port=int(getattr(config, "email_smtp_port", 587)),
        smtp_username=getattr(config, "email_smtp_username", ""),
        smtp_password=getattr(config, "email_smtp_password", ""),
        smtp_use_tls=getattr(config, "email_smtp_use_tls", True),
        from_address=getattr(config, "email_from_address", ""),
        poll_interval=int(getattr(config, "email_poll_interval", 30)),
        mark_seen=getattr(config, "email_mark_seen", True),
        max_body_chars=int(getattr(config, "email_max_body_chars", 12000)),
        subject_prefix=getattr(config, "email_subject_prefix", "Re: "),
        allowed_senders=allowed,
    ))


register_channel("email", create_from_config)
