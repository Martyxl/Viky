"""Central configuration for Viky.

Every tunable — model names, endpoints, ports, device ids, feature flags — is
loaded from the environment (or a local `.env`). Nothing about switching
DEV -> PROD requires touching code: edit `.env` only.

Usage:
    from config.settings import settings
    print(settings.whisper_model)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ #
    # General
    # ------------------------------------------------------------------ #
    env: Literal["dev", "prod"] = Field(default="dev", alias="VIKY_ENV")
    log_level: str = Field(default="INFO", alias="VIKY_LOG_LEVEL")
    # When true, tools log what they would do instead of doing it. Default
    # true so a fresh checkout never fires a real webhook/email by accident.
    dry_run: bool = Field(default=True, alias="VIKY_DRY_RUN")

    # ------------------------------------------------------------------ #
    # Service networking (orchestrator -> stt/tts over HTTP on localhost)
    # ------------------------------------------------------------------ #
    stt_host: str = Field(default="127.0.0.1", alias="STT_HOST")
    stt_port: int = Field(default=8001, alias="STT_PORT")
    tts_host: str = Field(default="127.0.0.1", alias="TTS_HOST")
    tts_port: int = Field(default=8002, alias="TTS_PORT")
    orchestrator_host: str = Field(default="127.0.0.1", alias="ORCHESTRATOR_HOST")
    orchestrator_port: int = Field(default=8000, alias="ORCHESTRATOR_PORT")

    # URLs the orchestrator uses to reach the sibling services. Under
    # docker-compose these are overridden to service names (http://stt:8001).
    stt_base_url: str = Field(default="", alias="STT_BASE_URL")
    tts_base_url: str = Field(default="", alias="TTS_BASE_URL")

    # ------------------------------------------------------------------ #
    # STT — faster-whisper (M3)
    # ------------------------------------------------------------------ #
    whisper_model: str = Field(default="medium", alias="WHISPER_MODEL")
    whisper_compute: str = Field(default="int8", alias="WHISPER_COMPUTE")
    whisper_device: str = Field(default="auto", alias="WHISPER_DEVICE")
    whisper_language: str = Field(default="cs", alias="WHISPER_LANGUAGE")

    # VAD — silero (M3)
    vad_silence_ms: int = Field(default=800, alias="VAD_SILENCE_MS")
    vad_threshold: float = Field(default=0.5, alias="VAD_THRESHOLD")

    # ------------------------------------------------------------------ #
    # TTS — Piper (M2)
    # ------------------------------------------------------------------ #
    piper_voice: str = Field(default="cs_CZ-jirka-medium", alias="PIPER_VOICE")
    piper_voice_path: str = Field(default="", alias="PIPER_VOICE_PATH")
    piper_binary: str = Field(default="piper", alias="PIPER_BINARY")
    tts_sample_rate: int = Field(default=22050, alias="TTS_SAMPLE_RATE")

    # ------------------------------------------------------------------ #
    # Wake word — openWakeWord (M4)
    # ------------------------------------------------------------------ #
    # Set VIKY_WAKEWORD_MODEL to the trained viky.onnx once it exists.
    # Until then use_fallback keeps a pre-trained model ("hey jarvis") active.
    wakeword_model_path: str = Field(default="", alias="VIKY_WAKEWORD_MODEL")
    wakeword_use_fallback: bool = Field(default=True, alias="VIKY_WAKEWORD_FALLBACK")
    wakeword_fallback_name: str = Field(default="hey_jarvis", alias="VIKY_WAKEWORD_FALLBACK_NAME")
    wakeword_threshold: float = Field(default=0.5, alias="VIKY_WAKEWORD_THRESHOLD")

    # ------------------------------------------------------------------ #
    # LLM — everything through LiteLLM, OpenAI-compatible (M5)
    # ------------------------------------------------------------------ #
    llm_model: str = Field(default="claude-sonnet-4-5", alias="LLM_MODEL")
    llm_api_base: str = Field(default="", alias="LLM_API_BASE")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_temperature: float = Field(default=0.4, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")

    # ------------------------------------------------------------------ #
    # Audio I/O — sounddevice (M2/M3)
    # ------------------------------------------------------------------ #
    # Leave empty to use the system default device; set to an index or name.
    audio_input_device: str = Field(default="", alias="AUDIO_INPUT_DEVICE")
    audio_output_device: str = Field(default="", alias="AUDIO_OUTPUT_DEVICE")
    audio_sample_rate: int = Field(default=16000, alias="AUDIO_SAMPLE_RATE")

    # ------------------------------------------------------------------ #
    # Orchestrator state machine (M6/M7)
    # ------------------------------------------------------------------ #
    listen_timeout_s: float = Field(default=8.0, alias="LISTEN_TIMEOUT_S")
    followup_window_s: float = Field(default=5.0, alias="FOLLOWUP_WINDOW_S")
    logs_dir: str = Field(default="logs", alias="VIKY_LOGS_DIR")

    # ------------------------------------------------------------------ #
    # Tool endpoints (M5)
    # ------------------------------------------------------------------ #
    n8n_webhook_base: str = Field(default="http://localhost:5678/webhook", alias="N8N_WEBHOOK_BASE")
    stats_api_base: str = Field(default="http://localhost:8010", alias="STATS_API_BASE")

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #
    @property
    def stt_url(self) -> str:
        return self.stt_base_url or f"http://{self.stt_host}:{self.stt_port}"

    @property
    def tts_url(self) -> str:
        return self.tts_base_url or f"http://{self.tts_host}:{self.tts_port}"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so every import shares one parsed instance."""
    return Settings()


# Module-level singleton for convenient `from config.settings import settings`.
settings = get_settings()
