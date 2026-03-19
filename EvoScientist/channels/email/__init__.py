"""Email channel for EvoScientist.

Uses IMAP polling for inbound + SMTP for outbound. Pure Python, no extra deps.

Usage in config:
    channel_enabled = "email"
    email_imap_host = "imap.gmail.com"
    email_smtp_host = "smtp.gmail.com"
    ...
"""

from ..channel_manager import _parse_csv, register_channel
from .channel import EmailChannel, EmailConfig

__all__ = ["EmailChannel", "EmailConfig"]


def create_from_config(config) -> EmailChannel:
    allowed = _parse_csv(config.email_allowed_senders)
    return EmailChannel(
        EmailConfig(
            imap_host=config.email_imap_host,
            imap_port=config.email_imap_port,
            imap_username=config.email_imap_username,
            imap_password=config.email_imap_password,
            imap_mailbox=config.email_imap_mailbox,
            imap_use_ssl=config.email_imap_use_ssl,
            smtp_host=config.email_smtp_host,
            smtp_port=config.email_smtp_port,
            smtp_username=config.email_smtp_username,
            smtp_password=config.email_smtp_password,
            smtp_starttls=config.email_smtp_use_tls,
            from_address=config.email_from_address,
            poll_interval=config.email_poll_interval,
            mark_seen=config.email_mark_seen,
            max_body_chars=config.email_max_body_chars,
            subject_prefix=config.email_subject_prefix,
            allowed_senders=allowed,
        )
    )


register_channel("email", create_from_config)
