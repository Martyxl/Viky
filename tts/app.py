"""TTS service (Piper) — M2.

Streams Czech speech from the female voice (cs_CZ-kasandra-medium by default).

Endpoints:
  GET  /health      — service health (from common factory)
  GET  /            — voice info
  POST /speak       — stream raw int16 mono PCM as it is synthesized
                      (header X-Sample-Rate gives the rate). True streaming:
                      the first sentence starts flowing before the rest is done.
  POST /speak.wav   — synthesize the whole utterance into one WAV (convenience)
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from common.logging import get_logger
from common.service import create_service
from config.settings import settings
from tts.engine import get_engine

log = get_logger("tts.app")
app: FastAPI = create_service("tts")


class SpeakRequest(BaseModel):
    text: str


@app.get("/")
def root() -> dict:
    return {"service": "tts", "voice": settings.piper_voice, "milestone": "M2"}


@app.post("/speak")
def speak(req: SpeakRequest):
    """Stream raw int16 LE mono PCM chunks as Piper produces them."""
    engine = get_engine()
    try:
        sample_rate = engine.sample_rate
    except FileNotFoundError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    def pcm_iter():
        for chunk in engine.synthesize_stream(req.text):
            yield chunk

    headers = {
        "X-Sample-Rate": str(sample_rate),
        "X-Audio-Format": "pcm_s16le_mono",
        "X-Voice": settings.piper_voice,
    }
    return StreamingResponse(pcm_iter(), media_type="application/octet-stream", headers=headers)


@app.post("/speak.wav")
def speak_wav(req: SpeakRequest):
    """Return the full utterance as a single WAV (easy to curl / save)."""
    engine = get_engine()
    try:
        wav = engine.synthesize_wav_bytes(req.text)
    except FileNotFoundError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    return Response(content=wav, media_type="audio/wav")
