"""Diagnose the microphone: record at the device's NATIVE rate, report level,
resample to 16 kHz, and transcribe. Isolates audio-quality issues from the
model. Uses CPU Whisper so it never fights the GPU/LM Studio for VRAM.

    python scripts/mic_check.py                 # default input device
    python scripts/mic_check.py --device 9      # try a specific device index
    python scripts/mic_check.py --seconds 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402

from config.settings import settings  # noqa: E402


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Microphone diagnostic (M-fix).")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--save", type=Path, default=None, help="save recorded WAV here")
    args = ap.parse_args()

    dev = args.device
    info = sd.query_devices(dev if dev is not None else sd.default.device[0])
    native_sr = int(info["default_samplerate"])
    name = info["name"]
    print(f"Zařízení: [{dev if dev is not None else 'default'}] {name}  @ {native_sr} Hz")
    print(f"Nahrávám {args.seconds:.0f} s v nativní frekvenci — MLUV TEĎ (např. 'Kolik je hodin?')...")

    rec = sd.rec(int(args.seconds * native_sr), samplerate=native_sr, channels=1,
                 dtype="float32", device=dev)
    sd.wait()
    audio = rec.reshape(-1)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    print(f"Úroveň zvuku: peak={peak:.3f}  rms={rms:.4f}", end="  ")
    if peak < 0.02:
        print("→ ⚠️ skoro TICHO (špatný mikrofon / ztlumeno / mluvíš jinam)")
    elif peak > 0.99:
        print("→ ⚠️ PŘEBUZENO (ořezává se)")
    else:
        print("→ ✅ ok")

    from common.audio import peak_normalize

    audio16 = peak_normalize(_resample(audio, native_sr, 16000))

    if args.save:
        import wave
        pcm = (np.clip(audio16, -1, 1) * 32767).astype("<i2")
        with wave.open(str(args.save), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(pcm.tobytes())
        print(f"WAV uložen: {args.save}")

    print("Přepisuji (Whisper na CPU)...")
    from faster_whisper import WhisperModel

    model = WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(audio16, language="cs", beam_size=5)
    text = "".join(s.text for s in segs).strip()
    print(f"\n>>> PŘEPIS: {text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
