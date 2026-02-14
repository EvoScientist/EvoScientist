"""Email channel — refactored with PollingMixin."""

import asyncio
import email as email_lib
import email.utils
import html
import imaplib
import logging
import re
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.utils import parseaddr
from pathlib import Path

from ..base import Channel, RawIncoming, ChannelError
from ..capabilities import EMAIL as EMAIL_CAPS
from ..mixins import PollingMixin
from ..config import BaseChannelConfig

logger = logging.getLogger(__name__)


def _decode_hdr(raw: str) -> str:
    try:
        return str(make_header(decode_header(raw))) if raw else ""
    except Exception:
        return raw or ""


def _strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<p[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"</p>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()







@dataclass
class EmailConfig(BaseChannelConfig):
    imap_host: str = ""; imap_port: int = 993; imap_username: str = ""; imap_password: str = ""
    imap_mailbox: str = "INBOX"; imap_use_ssl: bool = True
    smtp_host: str = ""; smtp_port: int = 587; smtp_username: str = ""; smtp_password: str = ""
    smtp_use_tls: bool = True; from_address: str = ""
    poll_interval: int = 30; mark_seen: bool = True; max_body_chars: int = 12000
    subject_prefix: str = "Re: "; allowed_senders: set[str] | None = None; text_chunk_limit: int = 4096


class EmailChannel(Channel, PollingMixin):
    capabilities = EMAIL_CAPS
    name = "email"
    _non_retryable_patterns = ("auth", "login", "credential")

    def __init__(self, config: EmailConfig):
        super().__init__(config)
        self._imap: imaplib.IMAP4_SSL | imaplib.IMAP4 | None = None

    async def start(self) -> None:
        cfg = self.config
        if not cfg.imap_host or not cfg.imap_username:
            raise ChannelError("Email imap_host and imap_username are required")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._connect_imap)
        self._running = True
        logger.info(f"Email channel started (IMAP: {cfg.imap_host}, poll {cfg.poll_interval}s)")
        await self._start_polling()

    def _connect_imap(self) -> None:
        cfg = self.config
        try:
            if cfg.imap_use_ssl:
                self._imap = imaplib.IMAP4_SSL(cfg.imap_host, cfg.imap_port, ssl_context=ssl.create_default_context())
            else:
                self._imap = imaplib.IMAP4(cfg.imap_host, cfg.imap_port)
            self._imap.login(cfg.imap_username, cfg.imap_password)
            self._imap.select(cfg.imap_mailbox)
        except Exception as e:
            raise ChannelError(f"IMAP failed: {e}")

    def _reconnect_imap(self) -> None:
        try:
            if self._imap:
                self._imap.noop()
                return
        except Exception:
            pass
        self._connect_imap()

    async def _poll_once(self) -> None:
        loop = asyncio.get_event_loop()
        messages = await loop.run_in_executor(None, self._fetch_unseen)
        for m in messages:
            await self._process_email(m)

    def _fetch_unseen(self) -> list[dict]:
        self._reconnect_imap()
        results = []
        try:
            st, data = self._imap.search(None, "UNSEEN")
            if st != "OK":
                return []
            for mid in data[0].split()[-20:]:
                st, msg_data = self._imap.fetch(mid, "(RFC822)")
                if st != "OK":
                    continue
                msg = email_lib.message_from_bytes(msg_data[0][1])
                from_name, from_addr = parseaddr(msg.get("From", ""))
                body = self._extract_body(msg)
                if len(body) > self.config.max_body_chars:
                    body = body[:self.config.max_body_chars] + "\n[...truncated]"
                # Extract attachments and inline images
                attachments = []
                if msg.is_multipart():
                    for part in msg.walk():
                        content_disp = str(part.get("Content-Disposition", ""))
                        content_type = part.get_content_type() or ""
                        is_attachment = "attachment" in content_disp
                        is_inline_image = (
                            "inline" in content_disp
                            and content_type.startswith("image/")
                        )
                        if is_attachment or is_inline_image:
                            filename = part.get_filename() or "attachment"
                            filename = _decode_hdr(filename)
                            payload_data = part.get_payload(decode=True)
                            if payload_data:
                                from ..base import MAX_ATTACHMENT_BYTES, MEDIA_DIR
                                if len(payload_data) > MAX_ATTACHMENT_BYTES:
                                    attachments.append({"annotation": f"[attachment: {filename} - too large ({len(payload_data)} bytes)]"})
                                else:
                                    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
                                    local_path = MEDIA_DIR / f"email_{mid.decode()}_{filename}"
                                    local_path.write_bytes(payload_data)
                                    label = "inline-image" if is_inline_image else "attachment"
                                    attachments.append({"path": str(local_path), "annotation": f"[{label}: {local_path}]"})
                if self.config.mark_seen:
                    self._imap.store(mid, "+FLAGS", "\\Seen")
                results.append({
                    "from_addr": from_addr, "from_name": _decode_hdr(from_name),
                    "subject": _decode_hdr(msg.get("Subject", "")), "body": body,
                    "message_id": msg.get("Message-ID", ""), "date": msg.get("Date", ""),
                    "references": msg.get("References", ""), "attachments": attachments,
                })
        except Exception as e:
            logger.error(f"IMAP fetch: {e}")
        return results

    def _extract_body(self, msg) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                if ct == "text/plain":
                    return self._decode_payload(part)
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    return _strip_html(self._decode_payload(part))
            return "[no text content]"
        text = self._decode_payload(msg)
        return _strip_html(text) if msg.get_content_type() == "text/html" else text

    @staticmethod
    def _decode_payload(part) -> str:
        payload = part.get_payload(decode=True)
        if not payload:
            return ""
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")

    async def _process_email(self, m: dict) -> None:
        subject = m["subject"]
        text = f"[邮件] 主题: {subject}\n\n{m['body']}" if subject else m["body"]
        try:
            ts = email_lib.utils.parsedate_to_datetime(m["date"])
        except Exception:
            ts = datetime.now()
        # Process attachments
        media_paths: list[str] = []
        annotations: list[str] = []
        for att in m.get("attachments", []):
            if att.get("path"):
                media_paths.append(att["path"])
            if att.get("annotation"):
                annotations.append(att["annotation"])
        await self._enqueue_raw(RawIncoming(
            sender_id=m["from_addr"], chat_id=m["from_addr"], text=text, timestamp=ts,
            message_id=m["message_id"],
            media_files=media_paths,
            content_annotations=annotations,
            metadata={"chat_id": m["from_addr"], "subject": subject,
                       "original_message_id": m["message_id"], "references": m["references"], "backend": "email"},
        ))

    # ── Send ──────────────────────────────────────────────────────

    def _is_ready(self) -> bool:
        return bool(self.config.smtp_host)

    async def _send_chunk(self, chat_id, formatted_text, raw_text, reply_to, metadata):
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, self._smtp_send_html, chat_id, formatted_text, raw_text, metadata or {},
            )
        except Exception as e:
            err_str = str(e).lower()
            # Only fall back to plain text for format-related errors, not server rejections
            if any(code in err_str for code in ("550", "553", "554", "auth", "rejected")):
                raise
            logger.warning(f"HTML email failed ({e}), falling back to plain text")
            await loop.run_in_executor(
                None, self._smtp_send, chat_id, raw_text, metadata or {},
            )

    def _smtp_send(self, to: str, content: str, meta: dict) -> None:
        cfg = self.config
        from_addr = cfg.from_address or cfg.smtp_username
        logger.debug(f"SMTP plain send: from={from_addr} to={to}")
        msg = EmailMessage()
        orig_subj = meta.get("subject", "")
        msg["Subject"] = f"{cfg.subject_prefix}{orig_subj}" if orig_subj and not orig_subj.lower().startswith("re:") else (orig_subj or "EvoScientist Reply")
        msg["From"] = from_addr
        msg["To"] = to
        orig_id = meta.get("original_message_id", "")
        if orig_id:
            msg["In-Reply-To"] = orig_id
            msg["References"] = f"{meta.get('references', '')} {orig_id}".strip()
        msg.set_content(content)
        try:
            if cfg.smtp_use_tls:
                srv = smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30)
                srv.starttls()
            else:
                srv = smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, context=ssl.create_default_context(), timeout=30)
            srv.login(cfg.smtp_username, cfg.smtp_password)
            srv.sendmail(from_addr, [to], msg.as_string())
            srv.quit()
        except Exception as e:
            logger.error(f"SMTP send failed: from={from_addr} to={to} error={e}")
            raise RuntimeError(f"SMTP: {e}")

    def _smtp_send_html(self, to: str, html_content: str, plain_content: str, meta: dict) -> None:
        """Send an email with both HTML and plain-text parts."""
        cfg = self.config
        from_addr = cfg.from_address or cfg.smtp_username
        logger.debug(f"SMTP HTML send: from={from_addr} to={to}")
        msg = MIMEMultipart("alternative")
        orig_subj = meta.get("subject", "")
        msg["Subject"] = f"{cfg.subject_prefix}{orig_subj}" if orig_subj and not orig_subj.lower().startswith("re:") else (orig_subj or "EvoScientist Reply")
        msg["From"] = from_addr
        msg["To"] = to
        orig_id = meta.get("original_message_id", "")
        if orig_id:
            msg["In-Reply-To"] = orig_id
            msg["References"] = f"{meta.get('references', '')} {orig_id}".strip()
        msg.attach(MIMEText(plain_content, "plain", "utf-8"))
        msg.attach(MIMEText(html_content, "html", "utf-8"))
        try:
            if cfg.smtp_use_tls:
                srv = smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30)
                srv.starttls()
            else:
                srv = smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, context=ssl.create_default_context(), timeout=30)
            srv.login(cfg.smtp_username, cfg.smtp_password)
            srv.sendmail(from_addr, [to], msg.as_string())
            srv.quit()
        except Exception as e:
            logger.error(f"SMTP HTML send failed: from={from_addr} to={to} error={e}")
            raise RuntimeError(f"SMTP HTML: {e}")

    # ── Markdown to HTML formatting ───────────────────────────────


    # ── Media send (email attachment) ─────────────────────────────

    async def _send_media_impl(
        self,
        recipient: str,
        file_path: str,
        caption: str = "",
        metadata: dict | None = None,
    ) -> bool:
        """Send a file as an email attachment via SMTP."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._smtp_send_attachment, recipient, file_path, caption, metadata or {},
        )
        return True

    def _smtp_send_attachment(self, to: str, file_path: str, caption: str, meta: dict) -> None:
        """Send an email with a file attachment."""
        cfg = self.config
        from_addr = cfg.from_address or cfg.smtp_username
        logger.debug(f"SMTP attachment send: from={from_addr} to={to} file={file_path}")
        msg = MIMEMultipart()
        orig_subj = meta.get("subject", "")
        msg["Subject"] = f"{cfg.subject_prefix}{orig_subj}" if orig_subj and not orig_subj.lower().startswith("re:") else (orig_subj or "EvoScientist Reply")
        msg["From"] = from_addr
        msg["To"] = to
        orig_id = meta.get("original_message_id", "")
        if orig_id:
            msg["In-Reply-To"] = orig_id
            msg["References"] = f"{meta.get('references', '')} {orig_id}".strip()

        # Text body
        if caption:
            msg.attach(MIMEText(caption, "plain", "utf-8"))

        # Attachment
        path = Path(file_path)
        part = MIMEBase("application", "octet-stream")
        part.set_payload(path.read_bytes())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={path.name}")
        msg.attach(part)

        try:
            if cfg.smtp_use_tls:
                srv = smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30)
                srv.starttls()
            else:
                srv = smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, context=ssl.create_default_context(), timeout=30)
            srv.login(cfg.smtp_username, cfg.smtp_password)
            srv.sendmail(from_addr, [to], msg.as_string())
            srv.quit()
        except Exception as e:
            logger.error(f"SMTP attachment send failed: from={from_addr} to={to} error={e}")
            raise RuntimeError(f"SMTP attachment: {e}")

    async def _cleanup(self) -> None:
        await self._stop_polling()
        if self._imap:
            try:
                self._imap.close(); self._imap.logout()
            except Exception:
                pass
            self._imap = None
        logger.info("Email channel stopped")
