"""M1 — every service answers /health with status ok."""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    "module_path, service_name",
    [
        ("stt.app", "stt"),
        ("tts.app", "tts"),
        ("orchestrator.app", "orchestrator"),
    ],
)
def test_health_ok(module_path, service_name):
    import importlib

    mod = importlib.import_module(module_path)
    client = TestClient(mod.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == service_name
    assert body["status"] == "ok"
