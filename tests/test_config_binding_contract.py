"""Binding contract between ``EvoScientistConfig`` fields and the served graphs.

Every config field must fall into exactly one of two binding regimes:

1. **fingerprint-covered (restart-to-apply)** — the field feeds
   ``_server_config_fingerprint`` in ``langgraph_dev/manager.py``. A keepalive
   server serves the construction-time value until restart; drift sets the
   restart hint. New dataclass fields land here automatically: the fingerprint
   iterates all fields, so the default failure mode is a spurious restart
   hint, never silent staleness.
2. **excluded with a named channel** — the field never reaches graph
   construction in the server process because it configures a client-side
   process or is delivered per call/per run by another channel. Every
   exclusion must name that channel in the tables below; an exclusion without
   a channel entry fails these tests.

The channel tables are the re-verification checklist: moving a field between
regimes (or adding a new exclusion) requires updating the table in the same
change, so the classification cannot drift silently.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields

from EvoScientist.config.settings import EvoScientistConfig
from EvoScientist.langgraph_dev.manager import (
    _FINGERPRINT_EXCLUDED_FIELDS,
    _FINGERPRINT_EXCLUDED_PREFIXES,
    _server_config_fingerprint,
)

# ── Channel tables ────────────────────────────────────────────────

#: Channel justification for each explicitly excluded field.
FIELD_CHANNELS: dict[str, str] = {
    "require_mention": "channel stack policy (CLI process)",
    "text_chunk_limit": "channel stack policy (CLI process)",
    "allowed_channels": "channel stack policy (CLI process)",
    "dm_policy": "channel stack policy (CLI process)",
    "shared_webhook_port": "channel stack policy (CLI process)",
    "show_thinking": "UI display flag (client-side rendering)",
    "ui_backend": "client UI selection",
    "log_level": "CLI-process logging",
    "default_mode": "client-side startup mode selection",
    "default_workdir": "client-side run-dir selection; workspace reaches the"
    " server via GraphTarget / EVOSCIENTIST_WORKSPACE_DIR",
    "webui_port": "WebUI client process",
    "webui_host": "WebUI client process",
    "langgraph_dev_keepalive": "server lifecycle flag consumed at spawn time,"
    " not by graph construction",
    "shell_allow_list": "per-call client-side approval policy, read fresh at"
    " each HITL interrupt (channels/interaction.py, stream/display.py)",
}

#: Channel justification for each prefix-based exclusion.
PREFIX_CHANNELS: dict[str, str] = {
    "channel_": "channel stack switches (CLI process)",
    "imessage_": "messaging channel adapter (CLI process)",
    "telegram_": "messaging channel adapter (CLI process)",
    "discord_": "messaging channel adapter (CLI process)",
    "slack_": "messaging channel adapter (CLI process)",
    "feishu_": "messaging channel adapter (CLI process)",
    "wechat_": "messaging channel adapter (CLI process)",
    "dingtalk_": "messaging channel adapter (CLI process)",
    "email_": "messaging channel adapter (CLI process)",
    "qq_": "messaging channel adapter (CLI process)",
    "signal_": "messaging channel adapter (CLI process)",
    "stt_": "speech-to-text (CLI process)",
}


# ── Helpers ───────────────────────────────────────────────────────


def _config_field_names() -> set[str]:
    return {f.name for f in dataclass_fields(EvoScientistConfig)}


def _excluded_field_names() -> set[str]:
    """Config fields excluded from the fingerprint (explicit + prefix)."""
    names = _config_field_names()
    excluded = {n for n in names if n in _FINGERPRINT_EXCLUDED_FIELDS}
    for name in names - excluded:
        if any(name.startswith(p) for p in _FINGERPRINT_EXCLUDED_PREFIXES):
            excluded.add(name)
    return excluded


def _covered_field_names() -> list[str]:
    return sorted(_config_field_names() - _excluded_field_names())


def _mutate(current: object) -> object:
    """Produce a value whose ``str()`` differs from ``current``.

    The fingerprint only hashes ``str(value)``, so any representable change
    suffices; keep the native type where cheap so nothing downstream chokes.
    """
    if isinstance(current, bool):
        return not current
    if isinstance(current, int):
        return current + 1
    if isinstance(current, float):
        return current + 0.5
    if isinstance(current, str):
        return current + "-binding-contract-mutation"
    return "binding-contract-mutation"


# ── Tests ─────────────────────────────────────────────────────────


class TestExclusionTable:
    """Every fingerprint exclusion names a channel; no stale entries."""

    def test_every_excluded_field_has_a_channel(self):
        missing = (
            _excluded_field_names()
            - set(FIELD_CHANNELS)
            - {
                n
                for n in _excluded_field_names()
                if any(n.startswith(p) for p in PREFIX_CHANNELS)
            }
        )
        assert not missing, (
            f"Fields excluded from the config fingerprint without a named"
            f" channel: {sorted(missing)}. Classify them (per-run channel,"
            f" client-side, or UI-only) or stop excluding them."
        )

    def test_field_channel_table_has_no_stale_entries(self):
        stale = set(FIELD_CHANNELS) - _config_field_names()
        assert not stale, (
            f"Channel-table entries that are no longer config fields: {sorted(stale)}"
        )
        reclassified = set(FIELD_CHANNELS) - _excluded_field_names()
        assert not reclassified, (
            f"Channel-table entries whose fields are fingerprint-covered"
            f" (not excluded) — remove the stale classification:"
            f" {sorted(reclassified)}"
        )

    def test_every_excluded_prefix_has_a_channel(self):
        missing = set(_FINGERPRINT_EXCLUDED_PREFIXES) - set(PREFIX_CHANNELS)
        assert not missing, (
            f"Excluded prefixes without a named channel: {sorted(missing)}"
        )

    def test_prefix_channel_table_has_no_stale_entries(self):
        stale = set(PREFIX_CHANNELS) - set(_FINGERPRINT_EXCLUDED_PREFIXES)
        assert not stale, f"Channel-table prefixes no longer excluded: {sorted(stale)}"


class TestFingerprintCoverage:
    """Fingerprint-covered fields drive the hash; excluded ones never do."""

    def test_baseline_fingerprint_is_deterministic(self):
        assert _server_config_fingerprint(
            EvoScientistConfig()
        ) == _server_config_fingerprint(EvoScientistConfig())

    def test_covered_fields_change_fingerprint(self):
        baseline = _server_config_fingerprint(EvoScientistConfig())
        offenders = []
        for field_name in _covered_field_names():
            cfg = EvoScientistConfig()
            setattr(cfg, field_name, _mutate(getattr(cfg, field_name)))
            if _server_config_fingerprint(cfg) == baseline:
                offenders.append(field_name)
        assert not offenders, (
            f"Fields supposed to be fingerprint-covered (restart-to-apply)"
            f" whose mutation leaves the fingerprint unchanged:"
            f" {sorted(offenders)}."
        )

    def test_excluded_fields_leave_fingerprint_unchanged(self):
        baseline = _server_config_fingerprint(EvoScientistConfig())
        offenders = []
        for field_name in sorted(_excluded_field_names()):
            cfg = EvoScientistConfig()
            setattr(cfg, field_name, _mutate(getattr(cfg, field_name)))
            if _server_config_fingerprint(cfg) != baseline:
                offenders.append(
                    (
                        field_name,
                        FIELD_CHANNELS.get(field_name, "prefix exclusion"),
                    )
                )
        assert not offenders, (
            f"Fields excluded from the fingerprint whose mutation changes"
            f" it - the exclusion list and the channel table disagree:"
            f" {offenders}."
        )
