"""Piper synthesis engine wrapper (M2).

Lazy-loads the Czech voice once and streams int16 PCM chunks as Piper produces
them (one chunk per sentence) — callers can start playback before the whole
utterance is synthesized.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator, Optional

from common.logging import get_logger
from config.settings import settings

log = get_logger("tts.engine")

# Voices are large binaries kept out of git; default lookup dir is voices/.
VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"


def resolve_voice_path(voice: Optional[str] = None) -> Path:
    """Locate the .onnx voice file.

    Priority: explicit PIPER_VOICE_PATH -> voices/<PIPER_VOICE>.onnx.
    """
    if settings.piper_voice_path:
        return Path(settings.piper_voice_path)
    name = voice or settings.piper_voice
    return VOICES_DIR / f"{name}.onnx"


class TTSEngine:
    """Thread-safe, lazily-initialized Piper voice."""

    def __init__(self, voice: Optional[str] = None) -> None:
        self._voice_name = voice or settings.piper_voice
        self._voice = None  # piper.PiperVoice, loaded on first use
        self._lock = threading.Lock()
        self._sample_rate = settings.tts_sample_rate

    def _ensure_loaded(self) -> None:
        if self._voice is not None:
            return
        with self._lock:
            if self._voice is not None:
                return
            from piper import PiperVoice  # imported lazily so tests can skip

            path = resolve_voice_path(self._voice_name)
            if not path.exists():
                raise FileNotFoundError(
                    f"Voice model not found: {path}. Run scripts/download_voice.py "
                    f"or set PIPER_VOICE_PATH."
                )
            log.info("loading Piper voice: %s", path)
            self._voice = PiperVoice.load(str(path))
            self._sample_rate = self._voice.config.sample_rate

    @property
    def sample_rate(self) -> int:
        self._ensure_loaded()
        return self._sample_rate

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Yield int16 little-endian mono PCM, one chunk per sentence."""
        self._ensure_loaded()
        assert self._voice is not None
        text = (text or "").strip()
        if not text:
            return
        for chunk in self._voice.synthesize(text):
            yield chunk.audio_int16_bytes

    def synthesize_wav_bytes(self, text: str) -> bytes:
        """Synthesize the whole utterance into a single in-memory WAV."""
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            for pcm in self.synthesize_stream(text):
                w.writeframes(pcm)
        return buf.getvalue()


# Process-wide singleton (the model is heavy — load it once).
_engine: Optional[TTSEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> TTSEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = TTSEngine()
    return _engine
