"""Silero VAD helpers (M3).

Two uses:
  * `iter_utterance()` — streaming end-of-utterance detection for the mic loop
    (scripts/listen.py): collect audio once speech starts, stop after
    VAD_SILENCE_MS of trailing silence.
  * faster-whisper additionally applies its own built-in Silero VAD filter to
    trim silence right before transcription (see stt/engine.py).
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from common.logging import get_logger
from config.settings import settings

log = get_logger("stt.vad")

# Silero operates on fixed 512-sample frames at 16 kHz (32 ms).
SILERO_SR = 16000
FRAME_SAMPLES = 512

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from silero_vad import load_silero_vad

                log.info("loading Silero VAD (onnx)")
                _model = load_silero_vad(onnx=True)
    return _model


class UtteranceCollector:
    """Feed 16 kHz float32 frames; emits a full utterance when speech ends.

    Usage:
        col = UtteranceCollector()
        for frame in mic_frames_512():   # np.float32 in [-1, 1]
            audio = col.push(frame)
            if audio is not None:
                # 'audio' is the complete utterance (np.float32)
                break
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        silence_ms: Optional[int] = None,
        min_speech_ms: int = 250,
    ) -> None:
        from silero_vad import VADIterator

        self.threshold = threshold if threshold is not None else settings.vad_threshold
        self.silence_ms = silence_ms if silence_ms is not None else settings.vad_silence_ms
        self.min_speech_frames = int(min_speech_ms / 1000 * SILERO_SR / FRAME_SAMPLES)
        self._it = VADIterator(
            _get_model(),
            threshold=self.threshold,
            sampling_rate=SILERO_SR,
            min_silence_duration_ms=self.silence_ms,
        )
        self._buf: list[np.ndarray] = []
        self._in_speech = False
        self._speech_frames = 0

    def push(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return the collected utterance (float32) when it ends, else None."""
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        event = self._it(frame, return_seconds=False)
        if self._in_speech:
            self._buf.append(frame)
            self._speech_frames += 1
        if event:
            if "start" in event:
                self._in_speech = True
                self._buf = [frame]
                self._speech_frames = 1
            elif "end" in event and self._in_speech:
                # Ignore blips shorter than min_speech.
                if self._speech_frames >= self.min_speech_frames:
                    audio = np.concatenate(self._buf) if self._buf else np.zeros(0, np.float32)
                    self._reset()
                    return audio
                self._reset()
        return None

    def _reset(self) -> None:
        self._it.reset_states()
        self._buf = []
        self._in_speech = False
        self._speech_frames = 0
