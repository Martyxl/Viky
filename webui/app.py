"""Viky web UI — serves the orb interface and streams live state over WebSocket.

The voice orchestrator runs in a background thread; its events (state /
transcript / reply / level) are pushed to all connected browsers. Launch with:

    uvicorn webui.app:app --host 127.0.0.1 --port 8080
    # then open http://127.0.0.1:8080  (or use scripts/viky_ui.py)
"""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from common.logging import get_logger, setup_logging
from config.settings import settings

log = get_logger("webui")
STATIC = Path(__file__).resolve().parent / "static"


class EventHub:
    """Bridges the orchestrator worker thread to async WebSocket clients."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        # Last known snapshot so a new client renders immediately.
        self.snapshot: dict = {"state": "IDLE", "transcript": "", "reply": ""}

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue = asyncio.Queue()

    def publish(self, event: dict) -> None:
        """Thread-safe: called from the orchestrator worker thread."""
        for k in ("state", "transcript", "reply"):
            if k in event:
                self.snapshot[k] = event[k]
        if self._loop and self._queue:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    async def broadcaster(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            for ws in list(self.clients):
                try:
                    await ws.send_json(event)
                except Exception:  # noqa: BLE001
                    self.clients.discard(ws)


hub = EventHub()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    hub.bind(asyncio.get_running_loop())
    asyncio.create_task(hub.broadcaster())
    # VIKY_UI_HEADLESS lets tests / CI serve the UI without audio + models.
    if not os.environ.get("VIKY_UI_HEADLESS"):
        threading.Thread(target=_run_orchestrator, daemon=True).start()
    yield
    audio = getattr(app.state, "audio", None)
    if audio is not None:
        audio.request_stop()


app = FastAPI(title="Viky UI", lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/health")
def health() -> dict:
    return {"service": "webui", "status": "ok", "clients": len(hub.clients)}


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    hub.clients.add(websocket)
    try:
        await websocket.send_json(hub.snapshot)
        while True:
            await websocket.receive_text()  # keep the connection open
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(websocket)


def _run_orchestrator() -> None:
    """Blocking voice loop, run in a background thread."""
    from orchestrator.earcon import ensure_earcons, play
    from orchestrator.runtime import build_orchestrator

    ensure_earcons()
    log.info("loading + warming up models...")
    hub.publish({"reply": "Probouzím se…"})
    orch, audio = build_orchestrator(on_event=hub.publish, play_earcon=play)
    app.state.audio = audio
    label = audio.detector.label
    log.info("Viky ready. wake word: %s", label)
    hub.publish({"reply": f"Připravena. Řekni „{label}“." , "state": "IDLE"})
    try:
        orch.run()
    except Exception as exc:  # noqa: BLE001
        log.exception("orchestrator stopped: %s", exc)
