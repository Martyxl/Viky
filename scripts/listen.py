"""M3 CLI test — record from mic, transcribe Czech.

    python scripts/listen.py                       # record until you stop talking
    python scripts/listen.py --local               # transcribe in-process (no server)
    python scripts/listen.py --file recording.wav  # transcribe a WAV (no mic)

Records from the default input device, uses Silero VAD to detect the end of
your utterance (VAD_SILENCE_MS of trailing silence), then sends the audio to
the STT service /transcribe (or transcribes locally) and prints the transcript.
"""

from __future__ import annotations

import argparse
import io
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from config.settings import settings  # noqa: E402
from stt.vad import FRAME_SAMPLES, SILERO_SR  # noqa: E402


def _record_utterance(timeout_s: float) -> np.ndarray:
    """Block until one utterance is captured (or timeout with no speech)."""
    import sounddevice as sd

    from stt.vad import UtteranceCollector

    collector = UtteranceCollector()
    device = settings.audio_input_device or None
    dev = int(device) if (device and str(device).isdigit()) else device

    print("[listen] mluv... (poslouchám)")
    captured: dict = {"audio": None}
    frames_seen = 0
    max_frames = int(timeout_s * SILERO_SR / FRAME_SAMPLES)

    with sd.InputStream(
        samplerate=SILERO_SR, channels=1, dtype="float32", blocksize=FRAME_SAMPLES, device=dev
    ) as stream:
        while frames_seen < max_frames or collector._in_speech:
            block, _ = stream.read(FRAME_SAMPLES)
            frame = block.reshape(-1)
            audio = collector.push(frame)
            frames_seen += 1
            if audio is not None:
                captured["audio"] = audio
                break

    if captured["audio"] is None:
        print("[listen] (ticho — nic nezachyceno)")
        return np.zeros(0, dtype=np.float32)
    return captured["audio"]


def _float_to_wav_bytes(audio: np.ndarray, sr: int = SILERO_SR) -> bytes:
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _transcribe_via_service(wav_bytes: bytes, base_url: str) -> dict:
    import httpx

    r = httpx.post(
        f"{base_url}/transcribe",
        content=wav_bytes,
        headers={"Content-Type": "audio/wav"},
        timeout=120.0,
    )
    r.raise_for_status()
    return r.json()


def _transcribe_local(wav_bytes: bytes) -> dict:
    from stt.engine import get_engine

    res = get_engine().transcribe(io.BytesIO(wav_bytes))
    return {
        "text": res.text,
        "language": res.language,
        "duration_s": round(res.duration_s, 3),
        "latency_ms": round(res.latency_ms, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Record mic and transcribe Czech (M3 test).")
    ap.add_argument("--local", action="store_true", help="transcribe in-process (no server)")
    ap.add_argument("--file", type=Path, help="transcribe this WAV instead of recording")
    ap.add_argument("--url", default=settings.stt_url, help="STT service base URL")
    args = ap.parse_args()

    if args.file:
        wav_bytes = args.file.read_bytes()
    else:
        audio = _record_utterance(settings.listen_timeout_s)
        if audio.size == 0:
            return 1
        wav_bytes = _float_to_wav_bytes(audio)

    if args.local:
        result = _transcribe_local(wav_bytes)
    else:
        try:
            result = _transcribe_via_service(wav_bytes, args.url)
        except Exception as exc:  # noqa: BLE001
            print(f"[listen] service unavailable ({exc}); falling back to --local", file=sys.stderr)
            result = _transcribe_local(wav_bytes)

    print(f"\n>>> {result['text']!r}")
    print(f"    ({result.get('duration_s')}s audio, {result.get('latency_ms')} ms, "
          f"lang={result.get('language')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
