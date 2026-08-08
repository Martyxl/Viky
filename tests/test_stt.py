"""M3 — STT endpoint tests (no 1.5 GB model needed: the engine is stubbed)."""

import io
import wave

import pytest
from fastapi.testclient import TestClient


def _tiny_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)  # 0.1 s silence
    return buf.getvalue()


def test_transcribe_returns_text(monkeypatch):
    import stt.app as app_mod
    from stt.engine import Transcript

    class FakeEngine:
        def transcribe(self, audio):
            return Transcript(text="ahoj světe", language="cs", duration_s=0.1, latency_ms=12.3)

    monkeypatch.setattr(app_mod, "get_engine", lambda: FakeEngine())
    client = TestClient(app_mod.app)
    resp = client.post("/transcribe", content=_tiny_wav(), headers={"Content-Type": "audio/wav"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "ahoj světe"
    assert body["language"] == "cs"


def test_transcribe_rejects_empty_body():
    import stt.app as app_mod

    client = TestClient(app_mod.app)
    resp = client.post("/transcribe", content=b"", headers={"Content-Type": "audio/wav"})
    assert resp.status_code == 422
