"""Tests for STT (speech-to-text) module and channel integration.

Unit tests mock the heavy ML backends so they run without GPU/models.
See bottom of file for manual integration test instructions.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from EvoScientist.stt import STT_MODELS, is_audio_file, transcribe_file
from tests.conftest import run_async


# ── is_audio_file ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path, expected",
    [
        ("voice.ogg", True),
        ("audio.mp3", True),
        ("audio.wav", True),
        ("audio.m4a", True),
        ("audio.opus", True),
        ("image.jpg", False),
        ("video.mp4", False),
        ("doc.pdf", False),
        ("noext", False),
    ],
)
def test_is_audio_file(path, expected):
    assert is_audio_file(path) is expected


# ── STT_MODELS mapping ────────────────────────────────────────────────


def test_stt_models_keys():
    assert "zh" in STT_MODELS
    assert "en" in STT_MODELS
    assert "auto" in STT_MODELS
    assert "faster-whisper" in STT_MODELS["zh"].lower()
    assert "faster-whisper" in STT_MODELS["en"].lower()
    assert "faster-whisper" in STT_MODELS["auto"].lower()


# ── transcribe_file: non-audio path ──────────────────────────────────


def test_transcribe_non_audio_returns_none():
    result = run_async(transcribe_file("photo.jpg", language="auto"))
    assert result is None


# ── transcribe_file: whisper backend (en / auto) ─────────────────────


def _make_whisper_mock(text: str):
    seg = MagicMock()
    seg.text = text
    seg.no_speech_prob = 0.0  # confident speech
    model = MagicMock()
    model.transcribe.return_value = ([seg], MagicMock())
    return model


def test_transcribe_en_uses_whisper():
    import EvoScientist.stt as stt_mod

    stt_mod._engines.clear()
    whisper_model = _make_whisper_mock("Hello world")
    faster_whisper_mock = MagicMock(
        WhisperModel=MagicMock(return_value=whisper_model)
    )
    with patch.dict("sys.modules", {"faster_whisper": faster_whisper_mock}):
        result = run_async(transcribe_file("voice.mp3", language="en"))
    stt_mod._engines.clear()
    assert result == "Hello world"


def test_transcribe_auto_uses_whisper():
    import EvoScientist.stt as stt_mod

    stt_mod._engines.clear()
    whisper_model = _make_whisper_mock("Bonjour monde")
    faster_whisper_mock = MagicMock(
        WhisperModel=MagicMock(return_value=whisper_model)
    )
    with patch.dict("sys.modules", {"faster_whisper": faster_whisper_mock}):
        result = run_async(transcribe_file("voice.ogg", language="auto"))
    stt_mod._engines.clear()
    assert result == "Bonjour monde"


# ── transcribe_file: SenseVoice backend (zh) ─────────────────────────


def test_transcribe_zh_uses_whisper():
    import EvoScientist.stt as stt_mod

    stt_mod._engines.clear()
    whisper_model = _make_whisper_mock("你好世界")
    faster_whisper_mock = MagicMock(
        WhisperModel=MagicMock(return_value=whisper_model)
    )
    with patch.dict("sys.modules", {"faster_whisper": faster_whisper_mock}):
        result = run_async(transcribe_file("voice.ogg", language="zh"))
    stt_mod._engines.clear()
    assert result == "你好世界"


# ── transcribe_file: missing dependency ──────────────────────────────


def test_transcribe_missing_dep_returns_none():
    import EvoScientist.stt as stt_mod
    import sys

    stt_mod._engines.clear()
    saved = sys.modules.pop("faster_whisper", None)
    try:
        with patch.dict("sys.modules", {"faster_whisper": None}):
            result = run_async(transcribe_file("voice.mp3", language="auto"))
    finally:
        if saved is not None:
            sys.modules["faster_whisper"] = saved
        stt_mod._engines.clear()

    assert result is None


# ── Channel _enqueue_raw STT integration ─────────────────────────────


def test_enqueue_raw_stt_prepends_transcript():
    """_enqueue_raw prepends STT transcript to raw.text when stt_enabled."""
    from EvoScientist.channels.base import RawIncoming
    from EvoScientist.channels.telegram.channel import TelegramChannel, TelegramConfig

    cfg = TelegramConfig(bot_token="dummy")
    ch = TelegramChannel(cfg)
    captured: list = []

    async def _fake_build(raw):
        captured.append(raw)
        return None

    ch._build_inbound_async = _fake_build  # type: ignore[method-assign]

    raw = RawIncoming(
        sender_id="123",
        chat_id="456",
        text="",
        media_files=["voice.ogg"],
        timestamp=datetime.now(),
    )

    mock_config = MagicMock()
    mock_config.stt_enabled = True
    mock_config.stt_language = "zh"

    async def _run():
        with (
            patch("EvoScientist.config.settings.load_config", return_value=mock_config),
            patch("EvoScientist.stt.transcribe_file", new=AsyncMock(return_value="你好")),
            patch("EvoScientist.stt.is_audio_file", return_value=True),
        ):
            await ch._enqueue_raw(raw)

    run_async(_run())
    assert len(captured) == 1
    assert captured[0].text == "你好"


def test_enqueue_raw_stt_disabled_skips_transcription():
    """When stt_enabled=False, transcription is not called."""
    from EvoScientist.channels.base import RawIncoming
    from EvoScientist.channels.telegram.channel import TelegramChannel, TelegramConfig

    cfg = TelegramConfig(bot_token="dummy")
    ch = TelegramChannel(cfg)
    captured: list = []

    async def _fake_build(raw):
        captured.append(raw)
        return None

    ch._build_inbound_async = _fake_build  # type: ignore[method-assign]

    raw = RawIncoming(
        sender_id="123",
        chat_id="456",
        text="",
        media_files=["voice.ogg"],
        timestamp=datetime.now(),
    )

    mock_config = MagicMock()
    mock_config.stt_enabled = False
    mock_transcribe = AsyncMock()

    async def _run():
        with (
            patch("EvoScientist.config.settings.load_config", return_value=mock_config),
            patch("EvoScientist.stt.transcribe_file", mock_transcribe),
        ):
            await ch._enqueue_raw(raw)

    run_async(_run())
    mock_transcribe.assert_not_called()
    assert captured[0].text == ""


def test_enqueue_raw_stt_appends_to_existing_text():
    """When raw.text already has content, transcript is prepended with newline."""
    from EvoScientist.channels.base import RawIncoming
    from EvoScientist.channels.telegram.channel import TelegramChannel, TelegramConfig

    cfg = TelegramConfig(bot_token="dummy")
    ch = TelegramChannel(cfg)
    captured: list = []

    async def _fake_build(raw):
        captured.append(raw)
        return None

    ch._build_inbound_async = _fake_build  # type: ignore[method-assign]

    raw = RawIncoming(
        sender_id="123",
        chat_id="456",
        text="caption text",
        media_files=["voice.ogg"],
        timestamp=datetime.now(),
    )

    mock_config = MagicMock()
    mock_config.stt_enabled = True
    mock_config.stt_language = "auto"

    async def _run():
        with (
            patch("EvoScientist.config.settings.load_config", return_value=mock_config),
            patch("EvoScientist.stt.transcribe_file", new=AsyncMock(return_value="hello world")),
            patch("EvoScientist.stt.is_audio_file", return_value=True),
        ):
            await ch._enqueue_raw(raw)

    run_async(_run())
    assert captured[0].text == "hello world\ncaption text"


# ── Manual integration test (run by hand) ────────────────────────────
#
# 1. Install deps:
#      uv pip install 'EvoScientist[stt]'
#
# 2. Enable STT:
#      uv run EvoSci config set stt_enabled true
#      uv run EvoSci config set stt_language zh   # zh / en / auto
#
# 3. Transcribe a local audio file directly:
#      python -c "
#      import asyncio
#      from EvoScientist.stt import transcribe_file
#      print(run_async(transcribe_file('sample.ogg', language='zh')))
#      "
#
# 4. End-to-end via Telegram:
#      uv run EvoSci serve --channel telegram
#      → send a voice message → bot receives transcribed text
