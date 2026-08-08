"""Robust microphone capture (mic-fix).

Recording directly at 16 kHz on a device whose native rate is 44100/48000 Hz
produced garbled audio (Whisper heard nonsense). Instead we open the input
stream at the device's NATIVE rate and resample to 16 kHz in software, which is
far more reliable across devices. Also provides peak-normalization to lift quiet
mics before STT.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


def peak_normalize(audio: np.ndarray, target_peak: float = 0.9, min_peak: float = 0.02) -> np.ndarray:
    """Scale so the loudest sample hits target_peak. No-op on near-silence."""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < min_peak:
        return audio  # too quiet to be real speech; don't amplify noise
    return (audio * (target_peak / peak)).astype(np.float32)


class ResamplingMicReader:
    """Open the mic at its native rate; yield fixed-size 16 kHz float32 frames.

    Use as a context manager and call read() to get one `out_block`-sample frame
    of resampled float32 audio in [-1, 1].
    """

    def __init__(
        self,
        device: Optional[object] = None,
        out_sr: int = 16000,
        out_block: int = 512,
        chunk_ms: int = 40,
    ) -> None:
        import sounddevice as sd

        self._sd = sd
        idx = device
        query_idx = idx if idx is not None else sd.default.device[0]
        info = sd.query_devices(query_idx)
        self.native_sr = int(info["default_samplerate"])
        self.out_sr = out_sr
        self.out_block = out_block
        self._read_n = max(1, int(self.native_sr * chunk_ms / 1000))
        self._buf = np.zeros(0, dtype=np.float32)
        self._stream = sd.InputStream(
            samplerate=self.native_sr, channels=1, dtype="float32",
            blocksize=self._read_n, device=idx,
        )

    def __enter__(self) -> "ResamplingMicReader":
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        try:
            self._stream.stop()
        finally:
            self._stream.close()

    def read(self) -> np.ndarray:
        """Return one out_block-sample frame of 16 kHz float32 audio."""
        while len(self._buf) < self.out_block:
            block, _ = self._stream.read(self._read_n)
            res = resample_linear(block.reshape(-1), self.native_sr, self.out_sr)
            self._buf = np.concatenate([self._buf, res])
        frame = self._buf[: self.out_block]
        self._buf = self._buf[self.out_block :]
        return frame

    def read_int16(self) -> np.ndarray:
        """Same as read() but int16 PCM (for openWakeWord)."""
        f = np.clip(self.read(), -1.0, 1.0)
        return (f * 32767.0).astype(np.int16)
