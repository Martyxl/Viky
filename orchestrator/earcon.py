"""Earcons — short audio cues on wake detection and timeout (M4).

Generates two small WAVs into orchestrator/sounds/ (wake = rising two-tone,
timeout = falling tone) and plays them via sounddevice. Playback failures are
non-fatal (e.g. headless / no output device).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from common.logging import get_logger
from config.settings import settings

log = get_logger("orchestrator.earcon")

SOUNDS_DIR = Path(__file__).resolve().parent / "sounds"
SR = 22050


def _tone(freq: float, ms: int, volume: float = 0.35) -> np.ndarray:
    n = int(SR * ms / 1000)
    t = np.linspace(0, ms / 1000, n, endpoint=False)
    wave_ = np.sin(2 * np.pi * freq * t)
    # short fade in/out to avoid clicks
    fade = max(1, int(0.005 * SR))
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (wave_ * env * volume).astype(np.float32)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def ensure_earcons() -> dict[str, Path]:
    """Create the earcon WAVs if missing; return their paths."""
    SOUNDS_DIR.mkdir(parents=True, exist_ok=True)
    wake = SOUNDS_DIR / "wake.wav"
    timeout = SOUNDS_DIR / "timeout.wav"
    gap = np.zeros(int(SR * 0.03), dtype=np.float32)
    if not wake.exists():
        _write_wav(wake, np.concatenate([_tone(880, 90), gap, _tone(1320, 110)]))
    if not timeout.exists():
        _write_wav(timeout, np.concatenate([_tone(660, 90), gap, _tone(440, 130)]))
    return {"wake": wake, "timeout": timeout}


def play(name: str) -> None:
    """Play an earcon by name ('wake' | 'timeout'). Never raises."""
    paths = ensure_earcons()
    path = paths.get(name)
    if not path or not path.exists():
        return
    try:
        import sounddevice as sd

        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        device = settings.audio_output_device or None
        dev = int(device) if (device and str(device).isdigit()) else device
        sd.play(data, samplerate=sr, device=dev)
        sd.wait()
    except Exception as exc:  # noqa: BLE001
        log.debug("earcon playback skipped: %s", exc)
