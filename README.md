# Viky — Czech Voice AI Assistant

A local, Czech-speaking voice assistant (Jarvis-style) that listens for the
wake word **"Viky"**, understands and speaks Czech with a **100% local** voice
pipeline, and acts through **tool calling** (n8n workflows, trading stats,
email, agents). The LLM "brain" is **swappable via `.env`** — Claude API in
dev, a local Qwen3 on production hardware later — with no code changes.

> Status: **Milestone 1 (Skeleton)** complete. See [Milestones](#milestones).

## Architecture

```
                        ┌──────────────────────────────────────────┐
   mic ──> wakeword ──> │ orchestrator  (state machine + audio I/O) │
                        │  IDLE→LISTENING→TRANSCRIBING→THINKING→...  │
                        └───────┬───────────────┬──────────────┬────┘
                                │ HTTP          │ HTTP         │ LiteLLM
                          ┌─────▼─────┐   ┌──────▼─────┐   ┌────▼─────┐
                          │    stt    │   │    tts     │   │  brain   │
                          │ f-whisper │   │   Piper    │   │  + tools │
                          │  + VAD    │   │  (stream)  │   │ (n8n/…)  │
                          └───────────┘   └────────────┘   └──────────┘
```

Each of `stt` and `tts` is an independent FastAPI service the orchestrator
reaches over localhost HTTP, so either can be scaled or replaced on PROD
without touching the others. Wake-word detection and audio capture live in the
orchestrator because they need direct host audio-device access.

## Repository layout

| Path             | Purpose                                             | Milestone |
|------------------|-----------------------------------------------------|-----------|
| `config/`        | `settings.py` — all config from env (pydantic)      | M1        |
| `common/`        | Shared logging + FastAPI service factory            | M1        |
| `stt/`           | faster-whisper + silero-VAD service (`/transcribe`) | M3        |
| `tts/`           | Piper service, streaming Czech audio (`/speak`)     | M2        |
| `wakeword/`      | openWakeWord detection loop                         | M4        |
| `brain/`         | LiteLLM routing, system prompt, tool schemas        | M5        |
| `tools/`         | Tool implementations (+ `mock_stats.py`)            | M5        |
| `orchestrator/`  | State machine wiring everything together            | M6        |
| `scripts/`       | `run_local.py`, `healthcheck.py`, CLI test tools    | —         |
| `tests/`         | pytest unit + integration tests (mock audio)        | all       |

## Quick start (dev, Windows)

Python 3.11+ is required. On this machine Python is `py` (3.13).

```bash
# 1. Config
cp .env.example .env          # then edit as needed

# 2. Virtual env + deps
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. Run all three services locally (no Docker)
python scripts/run_local.py

# 4. In another shell — verify everything is up
python scripts/healthcheck.py
# or hit the aggregate endpoint:
#   curl http://localhost:8000/status
```

### With Docker

```bash
cp .env.example .env
docker compose up --build
# once healthy:
curl http://localhost:8000/status
```

All three services expose `GET /health`; the orchestrator's `GET /status`
aggregates the health of `stt` and `tts` in one call.

## Make Viky speak (M2)

Viky uses a **female Czech voice** (`cs_CZ-kasandra-medium`, Piper, 100% local).

```bash
# 1. Download the voice model into voices/ (~60 MB, gitignored)
python scripts/download_voice.py            # uses PIPER_VOICE from .env
#   or a specific one:  python scripts/download_voice.py cs_CZ-jirka-medium

# 2a. Speak through the running TTS service (streams audio as it synthesizes)
python scripts/say.py "Ahoj, jsem Viky. Jaká je dnešní statistika na MNQ?"

# 2b. Or without a server (in-process):
python scripts/say.py "Pošli mi večerní report emailem." --local

# 2c. Save a WAV instead of playing (no speakers needed):
python scripts/say.py "Krátká věta." --out reply.wav
```

The TTS service exposes:
- `POST /speak` → streams raw **int16 LE mono PCM** as Piper produces it (header
  `X-Sample-Rate`); playback can start before the whole sentence is synthesized.
- `POST /speak.wav` → the full utterance as one WAV (handy for `curl`).

> **Windows + diacritics:** set `PYTHONUTF8=1` before running CLI scripts so
> Czech characters in command-line arguments are decoded as UTF-8. (The HTTP
> path is UTF-8 already — this only affects `argv`.)

## Configuration

Everything is driven by `.env` (see `.env.example` for the full annotated
list). Nothing is hardcoded — **switching DEV → PROD requires editing `.env`
only**. Key groups: service ports/URLs, Whisper model + compute, VAD, Piper
voice, wake-word model + fallback, LLM (LiteLLM) model/base/key, audio devices,
state-machine timeouts, and tool endpoints.

## DEV → PROD migration checklist

| Setting             | DEV (RTX 3060 Ti, 8 GB)     | PROD (EVO-X2, 96 GB)                        |
|---------------------|-----------------------------|---------------------------------------------|
| `WHISPER_MODEL`     | `medium`                    | `large-v3`                                  |
| `WHISPER_COMPUTE`   | `int8`                      | `int8`                                      |
| `LLM_MODEL`         | `claude-sonnet-4-5`         | `openai/qwen3-30b-a3b` (local)              |
| `LLM_API_BASE`      | *(empty — hosted Claude)*   | `http://localhost:1234/v1` (LM Studio)      |
| `LLM_API_KEY`       | Claude API key              | `not-needed`                                |
| `VIKY_WAKEWORD_MODEL` | *(empty → fallback)*      | `path/to/viky.onnx`                         |
| `VIKY_WAKEWORD_FALLBACK` | `true`                 | `false`                                     |

> On 8 GB VRAM do **not** run a local LLM alongside Whisper — use the Claude
> API in dev. The full checklist is finalized in M7.

## Audio: Windows vs Linux

- **Windows (dev):** run natively (`scripts/run_local.py`) so `sounddevice`
  reaches the host mic/speakers directly. Docker Desktop on Windows does not
  pass through host audio to Linux containers, so wake-word/audio runs on the
  host while `stt`/`tts` can still run in containers.
- **Linux (prod):** the orchestrator container needs the host audio device
  (e.g. `--device /dev/snd` and the PulseAudio/PipeWire socket mounted). Device
  passthrough wiring is added in a later milestone.

## Testing

```bash
pip install -r requirements.txt
pytest
```

M1 covers config loading and `/health` for all services (no audio hardware
required). Later milestones add VAD segmentation, tool-schema validation,
state-machine transitions, and a WAV-in → transcript → mock-LLM → WAV-out
integration test.

## Milestones

- [x] **M1 — Skeleton:** structure, config, `.env.example`, docker-compose, README; all services answer `/health`.
- [x] **M2 — TTS:** Piper streams Czech audio from the female voice; `scripts/say.py` + `scripts/download_voice.py`.
- [ ] **M3 — STT:** faster-whisper + VAD; `scripts/listen.py`.
- [ ] **M4 — Wake word:** detection loop + earcon; swap in `viky.onnx`.
- [ ] **M5 — Brain:** LiteLLM + system prompt + tool schemas (dry-run); `scripts/chat.py`.
- [ ] **M6 — Orchestrator:** full state machine; end-to-end voice.
- [ ] **M7 — Hardening:** barge-in, follow-up, structured logging, tests, migration checklist.

## Design rules

- Voice pipeline is **100% local** (privacy) — no cloud STT/TTS ever.
- LLM layer stays **provider-agnostic** — LiteLLM only, no `anthropic` SDK in
  business logic.
- Tools default to **dry-run** (`VIKY_DRY_RUN=true`) in dev.
- Windows-first for dev; Linux-only steps documented separately.
