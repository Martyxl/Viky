"""STT service (faster-whisper + silero-VAD) — M1 skeleton.

M1: exposes /health and a stubbed /transcribe.
M3: /transcribe will accept a WAV upload and return a Czech transcript.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from common.service import create_service
from config.settings import settings

app: FastAPI = create_service("stt")


@app.get("/")
def root() -> dict:
    return {
        "service": "stt",
        "model": settings.whisper_model,
        "compute": settings.whisper_compute,
        "language": settings.whisper_language,
        "milestone": "M1",
    }


@app.post("/transcribe")
def transcribe() -> JSONResponse:
    """Stubbed until M3."""
    return JSONResponse(
        status_code=501,
        content={
            "detail": "Transcription not implemented until M3",
            "model": settings.whisper_model,
        },
    )
