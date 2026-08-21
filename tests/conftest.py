"""Shared test fixtures.

The live .env may select the XTTS engine (which needs the WSL server), but the
TTS tests exercise the local Piper path — force it so tests don't depend on an
external server.
"""

import pytest

from config.settings import settings


@pytest.fixture(autouse=True)
def _force_piper_engine(monkeypatch):
    monkeypatch.setattr(settings, "tts_engine", "piper")
