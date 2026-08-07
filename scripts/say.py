"""M2 CLI test — make Viky speak Czech.

    python scripts/say.py "Ahoj, jsem Viky."
    python scripts/say.py "Pošli mi večerní report." --out reply.wav
    python scripts/say.py "Jaká je dnešní statistika na MNQ?" --local

Modes:
  (default)  POST to the running TTS service /speak and play audio as it
             streams in — proves the end-to-end streaming path.
  --local    Bypass the service; synthesize in-process (no server needed).
  --out F    Save a WAV to F instead of playing (works without speakers).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402


def _play_pcm_stream(pcm_chunks, sample_rate: int) -> None:
    """Play int16 PCM chunks through the default output device as they arrive."""
    import numpy as np
    import sounddevice as sd

    device = settings.audio_output_device or None
    stream = sd.RawOutputStream(
        samplerate=sample_rate, channels=1, dtype="int16",
        device=int(device) if (device and str(device).isdigit()) else device,
    )
    stream.start()
    try:
        for chunk in pcm_chunks:
            if chunk:
                stream.write(chunk)
    finally:
        stream.stop()
        stream.close()


def _save_wav(pcm_chunks, sample_rate: int, out: Path) -> None:
    import wave

    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for chunk in pcm_chunks:
            if chunk:
                w.writeframes(chunk)
    print(f"Saved {out}")


def _stream_from_service(text: str, base_url: str):
    """Yield (chunks_iterator, sample_rate) from the TTS service /speak."""
    import httpx

    client = httpx.Client(timeout=60.0)
    cm = client.stream("POST", f"{base_url}/speak", json={"text": text})
    resp = cm.__enter__()  # the context manager, not the response, owns teardown
    if resp.status_code != 200:
        resp.read()
        body = resp.text
        cm.__exit__(None, None, None)
        client.close()
        raise RuntimeError(f"TTS service error {resp.status_code}: {body}")
    sample_rate = int(resp.headers.get("X-Sample-Rate", settings.tts_sample_rate))

    def gen():
        try:
            for chunk in resp.iter_bytes():
                yield chunk
        finally:
            cm.__exit__(None, None, None)
            client.close()

    return gen(), sample_rate


def _stream_local(text: str):
    from tts.engine import get_engine

    engine = get_engine()
    return engine.synthesize_stream(text), engine.sample_rate


def main() -> int:
    ap = argparse.ArgumentParser(description="Make Viky speak Czech (M2 test).")
    ap.add_argument("text", help="Czech text to synthesize")
    ap.add_argument("--local", action="store_true", help="synthesize in-process (no server)")
    ap.add_argument("--out", type=Path, help="save WAV instead of playing")
    ap.add_argument("--url", default=settings.tts_url, help="TTS service base URL")
    args = ap.parse_args()

    if args.local:
        chunks, sr = _stream_local(args.text)
    else:
        try:
            chunks, sr = _stream_from_service(args.text, args.url)
        except Exception as exc:  # noqa: BLE001
            print(f"[say] service unavailable ({exc}); falling back to --local", file=sys.stderr)
            chunks, sr = _stream_local(args.text)

    if args.out:
        _save_wav(chunks, sr, args.out)
        return 0

    try:
        _play_pcm_stream(chunks, sr)
    except Exception as exc:  # noqa: BLE001 — no audio device? save instead
        fallback = ROOT / "say_output.wav"
        print(f"[say] playback failed ({exc}); saving to {fallback}", file=sys.stderr)
        # chunks may be exhausted; re-synthesize locally to the file
        chunks2, sr2 = _stream_local(args.text)
        _save_wav(chunks2, sr2, fallback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
