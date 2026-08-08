"""faster-whisper transcription engine (M3).

Lazy-loads the Whisper model (size/compute/device from env) and transcribes
audio forced to Czech, with Whisper's built-in Silero VAD filter trimming
silence before decoding.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from common.logging import get_logger
from config.settings import settings

log = get_logger("stt.engine")


@dataclass
class Transcript:
    text: str
    language: str
    duration_s: float
    latency_ms: float


class STTEngine:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            # Make bundled CUDA DLLs loadable on Windows before touching CUDA.
            if settings.whisper_device in ("cuda", "auto"):
                from common.cuda import ensure_cuda_dlls

                ensure_cuda_dlls()
            from faster_whisper import WhisperModel

            log.info(
                "loading Whisper model=%s compute=%s device=%s",
                settings.whisper_model, settings.whisper_compute, settings.whisper_device,
            )
            self._model = WhisperModel(
                settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute,
            )

    def transcribe(self, audio: Union[str, np.ndarray, "BinaryIO"]) -> Transcript:  # noqa: F821
        """Transcribe audio (path, float32 mono 16 kHz array, or WAV file-like)."""
        self._ensure_loaded()
        assert self._model is not None
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            audio,
            language=settings.whisper_language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": settings.vad_silence_ms},
            beam_size=5,
        )
        # segments is a generator; consuming it runs the decode.
        text = "".join(seg.text for seg in segments).strip()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return Transcript(
            text=text,
            language=getattr(info, "language", settings.whisper_language),
            duration_s=float(getattr(info, "duration", 0.0)),
            latency_ms=latency_ms,
        )


_engine: Optional[STTEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> STTEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = STTEngine()
    return _engine
