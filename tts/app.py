"""TTS service (Piper) — M1 skeleton.

M1: exposes /health and a stubbed /speak.
M2: /speak will stream Czech audio chunks synthesized by Piper.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from common.service import create_service
from config.settings import settings

app: FastAPI = create_service("tts")


@app.get("/")
def root() -> dict:
    return {"service": "tts", "voice": settings.piper_voice, "milestone": "M1"}


@app.post("/speak")
def speak(payload: dict) -> JSONResponse:
    """Stubbed until M2. Echoes back what it *would* synthesize."""
    text = (payload or {}).get("text", "")
    return JSONResponse(
        status_code=501,
        content={
            "detail": "TTS synthesis not implemented until M2",
            "would_speak": text,
            "voice": settings.piper_voice,
        },
    )
