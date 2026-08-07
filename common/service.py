"""FastAPI app factory shared by the stt / tts services.

Gives every service an identical `/health` contract so the orchestrator (and
docker-compose healthchecks) can probe them the same way.
"""

from __future__ import annotations

from fastapi import FastAPI

from common.logging import setup_logging
from config.settings import settings


def create_service(name: str, version: str = "0.1.0") -> FastAPI:
    setup_logging(settings.log_level)
    app = FastAPI(title=f"Viky · {name}", version=version)

    @app.get("/health")
    def health() -> dict:
        return {
            "service": name,
            "status": "ok",
            "version": version,
            "env": settings.env,
            "dry_run": settings.dry_run,
        }

    return app
