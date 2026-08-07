"""M1 — config loading tests (no audio hardware required)."""

import importlib

import pytest


def test_defaults_load():
    from config.settings import Settings

    s = Settings()
    assert s.env in ("dev", "prod")
    assert s.dry_run is True  # safe default on a fresh checkout
    assert s.whisper_language == "cs"
    assert s.piper_voice.startswith("cs_CZ")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "large-v3")
    monkeypatch.setenv("VIKY_DRY_RUN", "false")
    monkeypatch.setenv("LISTEN_TIMEOUT_S", "12.5")

    from config.settings import Settings

    s = Settings()
    assert s.whisper_model == "large-v3"
    assert s.dry_run is False
    assert s.listen_timeout_s == 12.5


def test_derived_urls(monkeypatch):
    monkeypatch.setenv("STT_HOST", "127.0.0.1")
    monkeypatch.setenv("STT_PORT", "9001")
    monkeypatch.delenv("STT_BASE_URL", raising=False)

    from config.settings import Settings

    s = Settings()
    assert s.stt_url == "http://127.0.0.1:9001"


def test_base_url_overrides_host_port(monkeypatch):
    monkeypatch.setenv("TTS_BASE_URL", "http://tts:8002")
    from config.settings import Settings

    s = Settings()
    assert s.tts_url == "http://tts:8002"
