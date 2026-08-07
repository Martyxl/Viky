"""M2 — TTS engine + /speak endpoint tests.

These need the Piper voice model (a large binary kept out of git), so they
skip cleanly when it is absent — config/health tests still cover CI.
"""

import wave
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from tts.engine import resolve_voice_path

_VOICE_PRESENT = resolve_voice_path().exists()
requires_voice = pytest.mark.skipif(
    not _VOICE_PRESENT,
    reason="Piper voice model not downloaded (run scripts/download_voice.py)",
)


@requires_voice
def test_engine_streams_pcm():
    from tts.engine import get_engine

    engine = get_engine()
    chunks = list(engine.synthesize_stream("Ahoj, jsem Viky."))
    assert chunks, "expected at least one PCM chunk"
    assert all(isinstance(c, (bytes, bytearray)) for c in chunks)
    assert engine.sample_rate == 22050


@requires_voice
def test_engine_wav_is_valid():
    from tts.engine import get_engine

    wav = get_engine().synthesize_wav_bytes("Krátká věta.")
    with wave.open(BytesIO(wav), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() > 0


@requires_voice
def test_speak_endpoint_streams():
    from tts.app import app

    client = TestClient(app)
    resp = client.post("/speak", json={"text": "Ahoj."})
    assert resp.status_code == 200
    assert resp.headers["x-sample-rate"] == "22050"
    assert len(resp.content) > 0


def test_speak_rejects_missing_text():
    from tts.app import app

    client = TestClient(app)
    resp = client.post("/speak", json={})
    assert resp.status_code == 422  # pydantic validation, no model load needed
