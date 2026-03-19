"""Speech-to-text transcription.

Default language → model mapping (all via faster-whisper):
  "zh"   → Systran/faster-whisper-small  (language=zh, ~250MB)
  "en"   → Systran/faster-whisper-small.en (~250MB, en-only)
  "auto" → Systran/faster-whisper-small   (~250MB, multilingual auto-detect)

All three parameters can be overridden via config:
  stt_model        — any HuggingFace faster-whisper model id
  stt_device       — "cpu" (default) or "cuda"
  stt_compute_type — "int8" (default), "float16", "float32", etc.

Install:
  pip install 'EvoScientist[stt]'
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_AUDIO_EXTS = frozenset({
    ".ogg", ".mp3", ".m4a", ".wav", ".flac", ".opus", ".weba", ".webm",
})

# Default language → HuggingFace model id
STT_MODELS: dict[str, str] = {
    "zh":   "Systran/faster-whisper-small",
    "en":   "Systran/faster-whisper-small.en",
    "auto": "Systran/faster-whisper-small",
}

# Single cached engine — re-created only when settings change
_engine: "_WhisperEngine | None" = None
_engine_key: tuple[str, str, str] | None = None  # (model_id, device, compute_type)


# ── Engine ────────────────────────────────────────────────────────────


class _WhisperEngine:
    """faster-whisper transcription engine."""

    def __init__(self, model_id: str, device: str, compute_type: str) -> None:
        from faster_whisper import WhisperModel  # type: ignore[import]

        self._model = WhisperModel(model_id, device=device, compute_type=compute_type)

    def transcribe(self, file_path: str, language: str | None) -> str:
        segments, _ = self._model.transcribe(
            file_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        # Skip segments where the model is not confident there is real speech
        parts = [s.text.strip() for s in segments if s.no_speech_prob < 0.6]
        return " ".join(parts).strip()


def _get_engine(
    model_id: str,
    device: str = "cpu",
    compute_type: str = "int8",
) -> _WhisperEngine:
    global _engine, _engine_key
    key = (model_id, device, compute_type)
    if _engine is None or _engine_key != key:
        logger.info(f"[STT] Loading model '{model_id}' device={device} compute={compute_type}")
        _engine = _WhisperEngine(model_id, device, compute_type)
        _engine_key = key
    return _engine


# ── Public API ────────────────────────────────────────────────────────


def is_audio_file(file_path: str) -> bool:
    """Return True if *file_path* has an audio extension."""
    return Path(file_path).suffix.lower() in _AUDIO_EXTS


async def transcribe_file(
    file_path: str,
    language: str = "auto",
    model: str = "",
    device: str = "cpu",
    compute_type: str = "int8",
) -> str | None:
    """Transcribe an audio file asynchronously.

    Args:
        file_path:    Path to the audio file.
        language:     Language hint — ``"zh"``, ``"en"``, or ``"auto"`` (default).
        model:        Override the HuggingFace model id. Empty = use STT_MODELS mapping.
        device:       Inference device — ``"cpu"`` (default) or ``"cuda"``.
        compute_type: Quantisation — ``"int8"`` (default), ``"float16"``, etc.

    Returns the transcript string, or ``None`` on error / silence.
    """
    if not is_audio_file(file_path):
        return None
    try:
        model_id = model or STT_MODELS.get(language, STT_MODELS["auto"])
        lang: str | None = None if language == "auto" else language
        engine = _get_engine(model_id, device, compute_type)
        loop = asyncio.get_running_loop()
        result: str = await loop.run_in_executor(
            None, engine.transcribe, file_path, lang
        )
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
