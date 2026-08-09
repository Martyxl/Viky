"""Web UI layer: static serving, health, and WebSocket snapshot/broadcast.

Runs headless (VIKY_UI_HEADLESS) so no audio hardware or models are touched.
"""

import os

import pytest
from fastapi.testclient import TestClient

os.environ["VIKY_UI_HEADLESS"] = "1"


@pytest.fixture()
def client():
    from webui.app import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "webui"


def test_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "VIKY" in r.text or "Viky" in r.text


def test_ws_snapshot_and_broadcast(client):
    from webui.app import hub

    with client.websocket_connect("/ws") as ws:
        snap = ws.receive_json()  # initial snapshot
        assert "state" in snap
        # a published event reaches the client
        hub.publish({"state": "LISTENING", "transcript": "ahoj"})
        msg = ws.receive_json()
        assert msg.get("state") == "LISTENING"


def test_hub_snapshot_updates():
    from webui.app import EventHub

    h = EventHub()
    h.publish({"state": "SPEAKING", "reply": "ahoj"})
    assert h.snapshot["state"] == "SPEAKING"
    assert h.snapshot["reply"] == "ahoj"
