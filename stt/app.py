"""STT service (faster-whisper + silero-VAD) — M3.

Endpoints:
  GET  /health       — service health (from common factory)
  GET  /             — model info
  POST /transcribe   — upload a WAV file (field `file`) OR a raw WAV body;
                       returns {text, language, duration_s, latency_ms}.
"""

from __future__ import annotations

import io

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse

from common.logging import get_logger
from common.service import create_service
from config.settings import settings
from stt.engine import get_engine

log = get_logger("stt.app")
app: FastAPI = create_service("stt")


@app.get("/")
def root() -> dict:
    return {
        "service": "stt",
        "model": settings.whisper_model,
        "compute": settings.whisper_compute,
        "device": settings.whisper_device,
        "language": settings.whisper_language,
        "milestone": "M3",
    }


async def _read_audio(request: Request, file: UploadFile | None) -> bytes:
    if file is not None:
        return await file.read()
    return await request.body()


@app.post("/transcribe")
async def transcribe(request: Request, file: UploadFile | None = None) -> JSONResponse:
    audio_bytes = await _read_audio(request, file)
    if not audio_bytes:
        return JSONResponse(status_code=422, content={"detail": "no audio provided"})

    engine = get_engine()
    try:
        result = engine.transcribe(io.BytesIO(audio_bytes))
    except Exception as exc:  # noqa: BLE001
        log.exception("transcription failed")
        return JSONResponse(status_code=500, content={"detail": f"transcription failed: {exc}"})

    return JSONResponse(
        content={
            "text": result.text,
            "language": result.language,
            "duration_s": round(result.duration_s, 3),
            "latency_ms": round(result.latency_ms, 1),
        }
    )
