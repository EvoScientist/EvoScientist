"""Owned-runtime coverage for bounded onboarding async work."""

import asyncio
import threading

from EvoScientist.config import EvoScientistConfig
from EvoScientist.config.onboard.channels import _probe_channel
from EvoScientist.runtime import AsyncRuntime


def test_channel_probes_reuse_the_provided_runtime(monkeypatch):
    executions: list[tuple[str, asyncio.AbstractEventLoop]] = []

    async def validate_telegram(_token, _proxy):
        executions.append((threading.current_thread().name, asyncio.get_running_loop()))
        return True, "telegram ok"

    async def validate_discord(_token, _proxy):
        executions.append((threading.current_thread().name, asyncio.get_running_loop()))
        return True, "discord ok"

    monkeypatch.setattr(
        "EvoScientist.channels.telegram.probe.validate_telegram_token",
        validate_telegram,
    )
    monkeypatch.setattr(
        "EvoScientist.channels.discord.probe.validate_discord_token",
        validate_discord,
    )
    monkeypatch.setattr(
        "EvoScientist.config.onboard.channels.console.print", lambda *_a, **_k: None
    )

    config = EvoScientistConfig()
    updates = {
        "telegram_bot_token": "telegram-token",
        "discord_bot_token": "discord-token",
    }
    with AsyncRuntime(thread_name="test-onboard-runtime") as runtime:
        _probe_channel("telegram", config, updates, runtime=runtime)
        _probe_channel("discord", config, updates, runtime=runtime)

    assert [thread for thread, _loop in executions] == [
        "test-onboard-runtime",
        "test-onboard-runtime",
    ]
    assert executions[0][1] is executions[1][1]
