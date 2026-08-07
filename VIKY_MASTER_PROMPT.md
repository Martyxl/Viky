# MASTER PROMPT — Project "Viky" (Czech Voice AI Assistant)

You are building **Viky**, a local, Czech-speaking voice assistant (Jarvis-style) that will orchestrate AI agents and trading-statistics workflows. Read this entire document before writing any code. Work milestone by milestone, commit after each one.

---

## 1. Project Overview

Viky is a hands-free voice assistant that:
- Listens continuously for the wake word **"Viky"**
- Understands and speaks **Czech** (STT + TTS fully local)
- Uses an LLM "brain" that is **swappable via environment variable** (Claude API during development → local Qwen3 on production hardware later)
- Executes actions via **tool calling**: triggering n8n workflows, querying trading statistics (tradezer.app backend), sending emails, managing agents
- Runs 24/7, fully containerized, portable between machines with zero code changes

## 2. Hardware Context (design for both)

| | DEV (now) | PROD (later) |
|---|---|---|
| Machine | Desktop, RTX 3060 Ti (8 GB VRAM), Windows 11 | GMKtec EVO-X2, AMD Ryzen AI Max+ 395, 96 GB VRAM, Windows 11 Pro / Linux |
| STT | faster-whisper `medium` int8 (~1.5 GB) | faster-whisper `large-v3` int8 |
| LLM | Claude API (via LiteLLM) — do NOT run local LLM alongside Whisper on 8 GB | Local Qwen3 30B-A3B via LM Studio/Ollama (via LiteLLM) |
| TTS | Piper (CPU) | Piper (CPU) |

**Hard constraint:** All model choices, endpoints, and sizes must be configurable via `.env` — never hardcoded. Switching DEV→PROD must require only editing `.env`.

## 3. Repository Structure

```
viky/
├── wakeword/          # openWakeWord detection loop
├── stt/               # faster-whisper server (FastAPI, /transcribe)
├── tts/               # Piper wrapper (FastAPI, /speak, streaming audio)
├── brain/             # LiteLLM routing, system prompt, tool definitions
├── orchestrator/      # main state machine: wake → record → STT → LLM → tools → TTS
├── tools/             # tool implementations (n8n trigger, stats query, email)
├── config/            # settings.py (pydantic-settings), all from env
├── tests/             # unit + integration tests
├── docker-compose.yml
├── .env.example
└── README.md
```

## 4. Technology Stack (mandatory)

- **Python 3.11+**, `pydantic-settings` for config, `uv` or `pip-tools` for deps
- **Wake word:** `openwakeword` — load custom model file `viky.onnx` (path from env; user trains it separately in Colab). Fallback for dev: use a pre-trained model (e.g. "hey jarvis") behind a config flag until `viky.onnx` exists.
- **VAD:** `silero-vad` — trim silence before STT, detect end of utterance (configurable silence threshold, default 800 ms)
- **STT:** `faster-whisper`, model name + compute type from env (`WHISPER_MODEL=medium`, `WHISPER_COMPUTE=int8`), language forced to `cs`
- **TTS:** Piper with a Czech voice (`cs_CZ-jirka-medium` default, voice path from env), stream audio chunks to speaker as they are generated (do not wait for full synthesis)
- **LLM:** all calls go through **LiteLLM** (`LLM_MODEL`, `LLM_API_BASE`, `LLM_API_KEY` from env). OpenAI-compatible interface. Must support tool calling.
- **Audio I/O:** `sounddevice` (input + output device selectable via env)
- **Services:** each of stt/tts runs as its own FastAPI service; orchestrator talks to them over HTTP (localhost). This allows independent scaling/replacement on PROD.
- **Docker:** docker-compose with services `stt`, `tts`, `orchestrator`. Wake word + audio capture run in orchestrator (needs host audio device access — document Windows vs Linux differences in README).

## 5. Orchestrator State Machine

States: `IDLE → LISTENING → TRANSCRIBING → THINKING → SPEAKING → IDLE`

Requirements:
- **Barge-in:** if wake word is detected while SPEAKING, stop TTS playback immediately and go to LISTENING
- **Timeout:** LISTENING auto-returns to IDLE after `LISTEN_TIMEOUT_S` (default 8 s) of no speech
- **Follow-up mode:** after Viky answers, stay in LISTENING for `FOLLOWUP_WINDOW_S` (default 5 s) without requiring the wake word again
- **Audio feedback:** short earcon (beep) on wake detection and on timeout — files in `orchestrator/sounds/`
- Log every turn as structured JSON (timestamp, transcript, LLM latency, tool calls, response) to `logs/` — this is the future statistics source

## 6. Brain: System Prompt & Tools

System prompt (in `brain/system_prompt.md`, Czech): Viky is a concise, slightly witty Czech assistant for a professional trader and engineer named Marty. She answers in spoken-style Czech (short sentences, no markdown, no lists — output goes to TTS). She uses tools when asked about statistics, workflows, or emails, and confirms destructive actions verbally before executing.

Tools (define with JSON schema in `brain/tools.py`, implement in `tools/`):
1. `trigger_n8n_workflow(workflow_id, payload)` — POST to `N8N_WEBHOOK_BASE/{workflow_id}`
2. `get_trading_stats(instrument, period)` — GET to `STATS_API_BASE` (tradezer.app backend; mock server included in `tools/mock_stats.py` for dev)
3. `send_email(to, subject, body)` — via n8n email workflow (never SMTP directly)
4. `list_agents()` / `run_agent(agent_id, task)` — stubs for now, wired to n8n
5. `get_time()` — local time (no LLM guessing)

All external URLs from env. Every tool must have a `--dry-run` mode controlled by `VIKY_DRY_RUN=true` (default true in dev): log the call instead of executing.

## 7. Milestones (commit after each, in order)

1. **M1 — Skeleton:** repo structure, config loading, `.env.example`, docker-compose stubs, README. All services start and respond to `/health`.
2. **M2 — TTS:** Piper service streams Czech audio; CLI test script `scripts/say.py "Ahoj, jsem Viky"`.
3. **M3 — STT:** faster-whisper service + VAD; CLI test `scripts/listen.py` records mic → prints Czech transcript.
4. **M4 — Wake word:** detection loop with fallback model + earcon; config flag to swap in `viky.onnx`.
5. **M5 — Brain:** LiteLLM integration, system prompt, tool schemas with dry-run implementations; CLI chat test `scripts/chat.py` (text in → text out, tools logged).
6. **M6 — Orchestrator:** full state machine wiring everything; end-to-end voice conversation works.
7. **M7 — Hardening:** barge-in, follow-up mode, structured logging, graceful shutdown, integration tests, final README with DEV→PROD migration checklist.

## 8. Testing & Definition of Done

- Unit tests for: config loading, VAD segmentation, tool schema validation, state transitions (pytest, no audio hardware required — mock audio streams)
- Integration test: WAV file in → transcript → mocked LLM response → WAV out
- DoD per milestone: tests pass, `docker compose up` works, README section updated
- Target end-to-end latency (dev): wake → start of spoken reply **< 3 s** for a short question without tools. Measure and print per-stage timings in logs.

## 9. Rules

- Czech language quality matters: test STT/TTS with real Czech sentences including diacritics ("Jaká je dnešní statistika na MNQ?", "Pošli mi večerní report emailem.")
- No cloud STT/TTS ever — voice pipeline is 100% local (privacy requirement)
- LLM layer must remain provider-agnostic (LiteLLM only, no `anthropic` SDK imports in business logic)
- Windows-first for dev (document any Linux-only steps separately)
- Keep dependencies minimal; justify anything beyond the stack above in README
- Ask before adding: GUI, web dashboard, database — out of scope for v1

Start with Milestone 1 now.
