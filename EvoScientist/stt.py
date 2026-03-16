"""Speech-to-text transcription.

Language → model mapping:
  "zh"   → Systran/faster-whisper-small  (language=zh, ~250MB)
  "en"   → Systran/faster-whisper-small.en (~250MB, en-only)
  "auto" → Systran/faster-whisper-small   (~250MB, multilingual auto-detect)

Install:
  pip install 'EvoScientist[stt]'   # faster-whisper only

Note: FunASR/SenseVoiceSmall backend planned for zh but currently skipped
due to llvmlite build failures on macOS Python 3.12.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_AUDIO_EXTS = frozenset({
    ".ogg", ".mp3", ".m4a", ".wav", ".flac", ".opus", ".weba", ".webm",
})

# Language → HuggingFace model id (all via faster-whisper)
STT_MODELS: dict[str, str] = {
    "zh":   "Systran/faster-whisper-small",
    "en":   "Systran/faster-whisper-small.en",
    "auto": "Systran/faster-whisper-small",
}

# Cached engines keyed by language
_engines: dict[str, "_BaseEngine"] = {}


# ── Engine base + implementations ────────────────────────────────────


class _BaseEngine:
    def transcribe(self, file_path: str) -> str:
        raise NotImplementedError


class _WhisperEngine(_BaseEngine):
    """faster-whisper backend for 'en' and 'auto' languages."""

    def __init__(self, model_id: str, language: str) -> None:
        from faster_whisper import WhisperModel  # type: ignore[import]

        self._model = WhisperModel(model_id, device="cpu", compute_type="int8")
        self._language: str | None = None if language == "auto" else language

    def transcribe(self, file_path: str) -> str:
        segments, _ = self._model.transcribe(
            file_path,
            language=self._language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        # Skip segments where the model is not confident there is real speech
        parts = [s.text.strip() for s in segments if s.no_speech_prob < 0.6]
        return " ".join(parts).strip()



def _get_engine(language: str) -> _BaseEngine:
    if language not in _engines:
        model_id = STT_MODELS.get(language, STT_MODELS["auto"])
        logger.info(f"[STT] Loading model for language='{language}': {model_id}")
        _engines[language] = _WhisperEngine(model_id, language)
    return _engines[language]


# ── Public API ────────────────────────────────────────────────────────


def is_audio_file(file_path: str) -> bool:
    """Return True if *file_path* has an audio extension."""
    return Path(file_path).suffix.lower() in _AUDIO_EXTS


async def transcribe_file(file_path: str, language: str = "auto") -> str | None:
    """Transcribe an audio file asynchronously.

    Selects the backend automatically based on *language*:
      - ``"zh"`` → FunASR SenseVoiceSmall
      - ``"en"`` → faster-whisper-small.en
      - ``"auto"`` (default) → faster-whisper-small (multilingual)

    Returns the transcript string, or ``None`` on error / silence.
    """
    if not is_audio_file(file_path):
        return None
    try:
        engine = _get_engine(language)
        loop = asyncio.get_event_loop()
        result: str = await loop.run_in_executor(None, engine.transcribe, file_path)
        return result or None
    except ImportError as e:
        logger.warning(
            f"[STT] Missing dependency: {e}. "
            "Install with: pip install 'EvoScientist[stt]'"
        )
        return None
    except Exception as e:
        logger.error(f"[STT] Transcription failed for {file_path}: {e}")
        return None
