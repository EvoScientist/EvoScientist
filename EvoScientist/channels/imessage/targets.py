"""iMessage target parsing and normalization.

Provides utilities for parsing iMessage targets and normalizing
phone numbers and email addresses, similar to OpenClaw's approach.
"""

import re
from dataclasses import dataclass
from enum import Enum


class IMessageService(Enum):
    """iMessage service type."""

    IMESSAGE = "imessage"
    SMS = "sms"
    AUTO = "auto"


@dataclass
class ChatIdTarget:
    """Target by chat ID."""

    kind: str = "chat_id"
    chat_id: int = 0


@dataclass
class ChatGuidTarget:
    """Target by chat GUID."""

    kind: str = "chat_guid"
    chat_guid: str = ""


@dataclass
class ChatIdentifierTarget:
    """Target by chat identifier."""

    kind: str = "chat_identifier"
    chat_identifier: str = ""


@dataclass
class HandleTarget:
    """Target by handle (phone/email)."""

    kind: str = "handle"
    to: str = ""
    service: IMessageService = IMessageService.AUTO


IMessageTarget = ChatIdTarget | ChatGuidTarget | ChatIdentifierTarget | HandleTarget


# Prefix constants
CHAT_ID_PREFIXES = ["chat_id:", "chatid:", "chat:"]
CHAT_GUID_PREFIXES = ["chat_guid:", "chatguid:", "guid:"]
CHAT_IDENTIFIER_PREFIXES = ["chat_identifier:", "chatidentifier:", "chatident:"]
SERVICE_PREFIXES = [
    ("imessage:", IMessageService.IMESSAGE),
    ("sms:", IMessageService.SMS),
    ("auto:", IMessageService.AUTO),
]


def normalize_e164(phone: str) -> str | None:
    """Normalize phone number to E.164 format.

    Args:
        phone: Raw phone number string

    Returns:
        Normalized E.164 format or None if invalid
    """
    # Remove all non-digit characters except leading +
    cleaned = re.sub(r"[^\d+]", "", phone)

    if not cleaned:
        return None

    # Already has + prefix
    if cleaned.startswith("+"):
        digits = cleaned[1:]
        if len(digits) >= 10 and len(digits) <= 15:
            return cleaned
        return None

    # US/Canada number without country code
    if len(cleaned) == 10:
        return f"+1{cleaned}"

    # Has country code
    if len(cleaned) >= 11 and len(cleaned) <= 15:
        return f"+{cleaned}"

    return None


def normalize_handle(raw: str) -> str:
    """Normalize an iMessage handle (phone or email).

    Args:
        raw: Raw handle string

    Returns:
        Normalized handle
    """
    trimmed = raw.strip()
    if not trimmed:
        return ""

    lowered = trimmed.lower()

    # Strip service prefixes
    for prefix, _ in SERVICE_PREFIXES:
        if lowered.startswith(prefix):
            return normalize_handle(trimmed[len(prefix) :])

    # Normalize chat_id/chat_guid/chat_identifier prefixes
    for prefix in CHAT_ID_PREFIXES:
        if lowered.startswith(prefix):
            value = trimmed[len(prefix) :].strip()
            return f"chat_id:{value}"

    for prefix in CHAT_GUID_PREFIXES:
        if lowered.startswith(prefix):
            value = trimmed[len(prefix) :].strip()
            return f"chat_guid:{value}"

    for prefix in CHAT_IDENTIFIER_PREFIXES:
        if lowered.startswith(prefix):
            value = trimmed[len(prefix) :].strip()
            return f"chat_identifier:{value}"

    # Email - lowercase
    if "@" in trimmed:
        return trimmed.lower()

    # Phone number - normalize to E.164
    normalized = normalize_e164(trimmed)
    if normalized:
        return normalized

    # Fallback: remove whitespace
    return re.sub(r"\s+", "", trimmed)


def parse_target(raw: str) -> IMessageTarget:
    """Parse an iMessage target string.

    Supports formats:
    - chat_id:123
    - chat_guid:abc-def
    - chat_identifier:iMessage;+;chat123
    - imessage:+1234567890
    - sms:+1234567890
    - +1234567890 (auto service)
    - email@example.com (auto service)

    Args:
        raw: Raw target string

    Returns:
        Parsed IMessageTarget

    Raises:
        ValueError: If target is invalid
    """
    trimmed = raw.strip()
    if not trimmed:
        raise ValueError("iMessage target is required")

    lower = trimmed.lower()

    # Check service prefixes first
    for prefix, service in SERVICE_PREFIXES:
        if lower.startswith(prefix):
            remainder = trimmed[len(prefix) :].strip()
            if not remainder:
                raise ValueError(f"{prefix} target is required")

            remainder_lower = remainder.lower()

            # Check if remainder is a chat target
            is_chat = any(
                remainder_lower.startswith(p)
                for p in CHAT_ID_PREFIXES
                + CHAT_GUID_PREFIXES
                + CHAT_IDENTIFIER_PREFIXES
            )
            if is_chat:
                return parse_target(remainder)

            return HandleTarget(to=remainder, service=service)

    # Check chat_id prefixes
    for prefix in CHAT_ID_PREFIXES:
        if lower.startswith(prefix):
            value = trimmed[len(prefix) :].strip()
            try:
                chat_id = int(value)
                return ChatIdTarget(chat_id=chat_id)
            except ValueError as e:
                raise ValueError(f"Invalid chat_id: {value}") from e

    # Check chat_guid prefixes
    for prefix in CHAT_GUID_PREFIXES:
        if lower.startswith(prefix):
            value = trimmed[len(prefix) :].strip()
            if not value:
                raise ValueError("chat_guid is required")
            return ChatGuidTarget(chat_guid=value)

    # Check chat_identifier prefixes
    for prefix in CHAT_IDENTIFIER_PREFIXES:
        if lower.startswith(prefix):
            value = trimmed[len(prefix) :].strip()
            if not value:
                raise ValueError("chat_identifier is required")
            return ChatIdentifierTarget(chat_identifier=value)

    # Default: handle with auto service
    return HandleTarget(to=trimmed, service=IMessageService.AUTO)


def format_chat_target(chat_id: int | None) -> str:
    """Format a chat ID as a target string.

    Args:
        chat_id: Chat ID number

    Returns:
        Formatted target string or empty string
    """
    if chat_id is None:
        return ""
    return f"chat_id:{chat_id}"
