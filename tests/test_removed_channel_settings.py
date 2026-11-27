"""Regression tests for removing dead top-level channel settings (#916)."""

from __future__ import annotations

from dataclasses import fields

from EvoScientist.channels.base import Channel
from EvoScientist.config.settings import EvoScientistConfig, config_from_mapping
from EvoScientist.langgraph_dev.settings_api import READ_ONLY_FIELDS


def test_dead_top_level_settings_are_gone() -> None:
    names = {field.name for field in fields(EvoScientistConfig)}
    assert "allowed_channels" not in names
    assert "text_chunk_limit" not in names


def test_old_config_keys_are_ignored() -> None:
    config = config_from_mapping(
        {"allowed_channels": "c1", "text_chunk_limit": 100, "model": "test-model"}
    )
    assert config.model == "test-model"
    assert not hasattr(config, "allowed_channels")
    assert not hasattr(config, "text_chunk_limit")


def test_dead_channel_mention_helper_is_gone() -> None:
    assert not hasattr(Channel, "_should_process")


def test_read_only_fields_have_no_removed_names() -> None:
    assert "allowed_channels" not in READ_ONLY_FIELDS
    assert "text_chunk_limit" not in READ_ONLY_FIELDS
