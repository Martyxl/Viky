"""Orchestrator service — M1 skeleton.

M1: /health plus /status which aggregates the health of the stt & tts
services so `docker compose up` can be verified from one endpoint.

Later milestones add the wake -> record -> STT -> LLM -> tools -> TTS state
machine (M6) and hosts wake-word detection + audio capture (needs host audio
device access; see README).
"""

from __future__ import annotations

import httpx
from fastapi import FastAPI

from common.service import create_service
from config.settings import settings

app: FastAPI = create_service("orchestrator")


@app.get("/")
def root() -> dict:
    return {"service": "orchestrator", "milestone": "M1"}


async def _probe(client: httpx.AsyncClient, name: str, base: str) -> dict:
    try:
        resp = await client.get(f"{base}/health", timeout=2.0)
        resp.raise_for_status()
        return {"name": name, "reachable": True, "health": resp.json()}
    except Exception as exc:  # noqa: BLE001 - report any failure uniformly
        return {"name": name, "reachable": False, "error": str(exc)}


@app.get("/status")
async def status() -> dict:
    """Aggregate downstream service health for a one-call system check."""
    async with httpx.AsyncClient() as client:
        stt = await _probe(client, "stt", settings.stt_url)
        tts = await _probe(client, "tts", settings.tts_url)
    all_ok = stt["reachable"] and tts["reachable"]
    return {
        "orchestrator": "ok",
        "all_services_ok": all_ok,
        "services": [stt, tts],
    }
