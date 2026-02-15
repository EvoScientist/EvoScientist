"""Base configuration for all channel implementations.

Provides common fields shared across channels, reducing duplication.
Channel-specific configs inherit from BaseChannelConfig.

Also provides ready-made ConfigAdapter implementations for the two most
common account patterns:

- ``SingleAccountConfigAdapter`` — one account per channel (default).
- ``MultiAccountConfigAdapter`` — multiple accounts from a config dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BaseChannelConfig:
    """Common configuration fields for all channels.

    Subclass this for channel-specific configs. Only add fields
    here that are used by 3+ channels.
    """

    allowed_senders: set[str] | None = None
    allowed_channels: set[str] | None = None
    text_chunk_limit: int = 4096
    proxy: str | None = None
    include_attachments: bool = True
    accounts: dict | None = None  # multi-account config mapping


class SingleAccountConfigAdapter:
    """For channels that only ever have one account (most channels).

    Returns a single ``"default"`` account whose config is the entire
    channel config object.  This is the zero-change default: existing
    single-account channels get multi-account support for free.
    """

    def list_account_ids(self, config: Any) -> list[str]:
        return ["default"]

    def resolve_account(
        self, config: Any, account_id: str | None = None,
    ) -> Any:
        return config

    def is_enabled(self, account: Any, config: Any) -> bool:
        return True

    def is_configured(self, account: Any, config: Any) -> bool:
        """Check that the account has at least some non-None values."""
        if account is None:
            return False
        if isinstance(account, dict):
            return bool(account)
        # dataclass / object — check that at least one field is truthy
        if hasattr(account, "__dataclass_fields__"):
            return any(
                getattr(account, f, None)
                for f in account.__dataclass_fields__
            )
        return True


class MultiAccountConfigAdapter:
    """For channels that support multiple accounts.

    Expects the channel config to contain a mapping of accounts under
    a configurable key (default ``"accounts"``).  Each entry is keyed
    by account id and holds account-specific settings.

    Example config structure::

        {
            "accounts": {
                "bot1": {"token": "...", "enabled": true},
                "bot2": {"token": "...", "enabled": false},
            }
        }
    """

    def __init__(
        self,
        accounts_key: str = "accounts",
        required_fields: list[str] | None = None,
    ) -> None:
        self._accounts_key = accounts_key
        self._required_fields = required_fields or []

    def _get_accounts_map(self, config: Any) -> dict[str, Any]:
        """Extract the accounts mapping from config."""
        if isinstance(config, dict):
            return config.get(self._accounts_key, {})
        return getattr(config, self._accounts_key, None) or {}

    def list_account_ids(self, config: Any) -> list[str]:
        return list(self._get_accounts_map(config).keys())

    def resolve_account(
        self, config: Any, account_id: str | None = None,
    ) -> Any:
        accounts = self._get_accounts_map(config)
        if account_id is None:
            # Return the first account, or empty dict
            return next(iter(accounts.values()), {})
        return accounts.get(account_id, {})

    def is_enabled(self, account: Any, config: Any) -> bool:
        if isinstance(account, dict):
            return account.get("enabled", True)
        return getattr(account, "enabled", True)

    def is_configured(self, account: Any, config: Any) -> bool:
        if not account:
            return False
        for f in self._required_fields:
            if isinstance(account, dict):
                if not account.get(f):
                    return False
            elif not getattr(account, f, None):
                return False
        return True
