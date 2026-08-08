"""openWakeWord detection (M4).

Loads the trained `viky.onnx` when configured; otherwise falls back to a
pre-trained model (default "hey_jarvis") so development works before the custom
model exists. Feed 16 kHz int16 mono frames of FRAME_SAMPLES (80 ms).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from common.logging import get_logger
from config.settings import settings

log = get_logger("wakeword.detector")

# openWakeWord expects 16 kHz; 1280 samples = 80 ms is the recommended chunk.
WAKE_SR = 16000
FRAME_SAMPLES = 1280


def _resolve_model() -> tuple[str, str]:
    """Return (model_ref, label) — a custom .onnx path or a builtin name."""
    path = settings.wakeword_model_path
    if path and Path(path).exists() and not settings.wakeword_use_fallback:
        return path, Path(path).stem
    if path and Path(path).exists() and settings.wakeword_use_fallback:
        # Custom model present but fallback explicitly on — prefer the real one.
        log.info("viky model present; using it despite fallback flag")
        return path, Path(path).stem
    log.info("using fallback wake word model: %s", settings.wakeword_fallback_name)
    return settings.wakeword_fallback_name, settings.wakeword_fallback_name


class WakeWordDetector:
    def __init__(self, threshold: Optional[float] = None, cooldown_s: float = 2.0) -> None:
        from openwakeword.model import Model

        self.threshold = threshold if threshold is not None else settings.wakeword_threshold
        self.cooldown_s = cooldown_s
        model_ref, self.label = _resolve_model()
        self._model = Model(wakeword_models=[model_ref], inference_framework="onnx")
        self.model_keys = list(self._model.models.keys())
        self._last_fire = 0.0
        log.info("wake word ready: keys=%s threshold=%.2f", self.model_keys, self.threshold)

    def score(self, frame: np.ndarray) -> float:
        """Feed one int16 frame; return the current max wake-word score."""
        frame = np.asarray(frame, dtype=np.int16).reshape(-1)
        scores = self._model.predict(frame)
        return max(float(scores.get(k, 0.0)) for k in self.model_keys)

    def detect(self, frame: np.ndarray) -> bool:
        """True when the wake word fires (respecting the cooldown)."""
        s = self.score(frame)
        now = time.monotonic()
        if s >= self.threshold and (now - self._last_fire) >= self.cooldown_s:
            self._last_fire = now
            self._model.reset()  # clear internal buffers so we don't re-fire
            return True
        return False

    def reset(self) -> None:
        self._model.reset()
        self._last_fire = 0.0
