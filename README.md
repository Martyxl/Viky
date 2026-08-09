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

**English pronunciation:** known English words (trading/tech terms) are read the
English way instead of being spelled out Czech-style. The Czech voice knows all
the English phonemes, so each English word is phonemized with `en-us` and the
rest stays Czech. Edit the word list in
[tts/english_terms.txt](tts/english_terms.txt) (one word per line); toggle with
`TTS_ENGLISH_PRONUNCIATION` in `.env`. Words with Czech diacritics are always
read Czech.

> **Windows + diacritics:** set `PYTHONUTF8=1` before running CLI scripts so
> Czech characters in command-line arguments are decoded as UTF-8. (The HTTP
> path is UTF-8 already — this only affects `argv`.)

## Make Viky listen (M3)

STT uses **faster-whisper** (`medium`/`int8` in dev) forced to Czech, with
**Silero VAD** detecting the end of an utterance.

```bash
# First run downloads the Whisper model (~1.5 GB, cached in ~/.cache/huggingface).
# Speak into the mic; recording stops after ~VAD_SILENCE_MS of silence:
python scripts/listen.py

# Transcribe a WAV without a mic:
python scripts/listen.py --file recording.wav

# Transcribe in-process (no server):
python scripts/listen.py --file recording.wav --local
```

The STT service exposes `POST /transcribe` — send a WAV as the raw body or as a
`file` upload; it returns `{text, language, duration_s, latency_ms}`.

> **GPU:** set `WHISPER_DEVICE=cuda` in `.env` for a big speed-up (RTX 3060 Ti:
> `medium` ~0.9 s vs ~5 s on CPU). Needs the NVIDIA CUDA 12 runtime — install
> `nvidia-cublas-cu12 nvidia-cudnn-cu12` (in `requirements-stt.txt`); on Windows
> `common/cuda.py` auto-registers their DLLs so no PATH tweaking is needed. The
> **model itself is downloaded automatically** by faster-whisper (CTranslate2
> format) — do **not** download GGUF Whisper models, those are a different
> runtime (whisper.cpp). Do not run a local LLM alongside Whisper on 8 GB VRAM.

## Wake word (M4)

Viky listens for a wake word before recording. Until you train `viky.onnx`, a
pre-trained **fallback** model is used (`hey_jarvis` by default).

```bash
# Continuously listen; beeps (earcon) on each detection. Say "Hey Jarvis":
python scripts/wakeword_listen.py
python scripts/wakeword_listen.py --debug     # print live scores while tuning
```

openWakeWord models download automatically on first use. To switch to your own
model, set in `.env`:

```bash
VIKY_WAKEWORD_MODEL=path/to/viky.onnx
VIKY_WAKEWORD_FALLBACK=false
```

> **Training `viky.onnx`:** use the openWakeWord training notebook in Google
> Colab (record samples of the word "Viky", ~15–30 min). Drop the resulting
> `.onnx` in and point `VIKY_WAKEWORD_MODEL` at it — no code change.

Earcons live in `orchestrator/sounds/` (`wake.wav`, `timeout.wav`); they are
generated on first run if missing.

## Brain & tools (M5)

Viky's "brain" routes through **LiteLLM** (provider-agnostic — no `anthropic`
SDK in the code) and calls tools. The persona lives in
[brain/system_prompt.md](brain/system_prompt.md) (spoken-style Czech).

Tools (JSON schemas in [brain/tools.py](brain/tools.py), implementations in
`tools/`): `get_time`, `get_trading_stats`, `trigger_n8n_workflow`,
`send_email`, `list_agents`, `run_agent`. Side-effecting tools honour
`VIKY_DRY_RUN=true` (default) — they log the intended call and return a
simulated result instead of doing it.

```bash
# Optional: run the mock stats backend so get_trading_stats has real data
uvicorn tools.mock_stats:app --port 8010 &

# Chat with Viky in text (needs LLM_* in .env, e.g. Claude key):
python scripts/chat.py "Kolik je hodin?"
python scripts/chat.py "Jaká je dnešní statistika na MNQ?"
python scripts/chat.py                      # interactive REPL
```

> `LLM_MODEL` must be a LiteLLM id with a provider prefix
> (`anthropic/claude-sonnet-4-5`, `openai/qwen3-30b-a3b`, …). Set `LLM_API_KEY`
> (or `ANTHROPIC_API_KEY`). Swapping Claude ↔ local Qwen3 is a `.env` change.

## Graphical app (web UI)

A futuristic orb interface (glowing teal sphere that reacts to Viky's state,
with live transcript + reply). The voice orchestrator runs inside the web
server and streams state/transcript/reply/audio-level to the browser over
WebSocket.

```bash
python scripts/viky_ui.py     # starts the server + opens the orb UI in a window
```

Or **double-click `Viky.bat`**. The UI opens in Edge/Chrome "app mode" (no
browser chrome) so it looks like a native app. Server code: [webui/app.py](webui/app.py);
front-end: [webui/static/index.html](webui/static/index.html). Configure the
port with `VIKY_UI_PORT` (default 8080). Relaunch is safe — if Viky is already
running it just opens the window; a stale process holding the port is freed.

### Talk from a phone (push-to-talk)

Tap the 🎙️ button and speak — the browser records, sends the audio to
`POST /api/utterance` (STT → brain → TTS) and plays Viky's spoken reply back;
the orb ripples with the real voice. Works on the PC and on a phone on the LAN.

Browsers only allow microphone access over **HTTPS** (or localhost), so for the
phone you must serve over https:

```bash
python scripts/make_cert.py     # one-time self-signed cert for your LAN IP
```

Then set `VIKY_UI_HOST=0.0.0.0` in `.env`, launch, and open
`https://<PC-IP>:<port>` on the phone (same Wi-Fi). Accept the one-time "not
trusted" warning (self-signed). Allow inbound on the port through the firewall
(run as Admin): `New-NetFirewallRule -DisplayName "Viky UI" -Direction Inbound
-LocalPort 8080 -Protocol TCP -Action Allow -Profile Any`, and make sure the
network profile is **Private** (`Set-NetConnectionProfile -InterfaceAlias
Ethernet -NetworkCategory Private`).

> The phone is a full client (talk + listen). Remote access from outside the
> home needs a secure tunnel (e.g. Tailscale) — that's on the roadmap.

## End-to-end voice (M6)

Everything wired together — say the wake word, ask in Czech, hear the answer:

```bash
python scripts/viky.py
```

State machine: `IDLE → LISTENING → TRANSCRIBING → THINKING → SPEAKING → …`
- **Wake** → earcon → record until you stop talking (Silero VAD).
- **Transcribe** (Czech) → **brain** (LiteLLM + tools) → **speak** (streamed).
- **Follow-up:** after a reply Viky keeps listening for `FOLLOWUP_WINDOW_S`
  without the wake word.
- **Barge-in:** say the wake word while Viky is speaking to cut her off and ask
  again immediately.
- **Timeout:** silence in LISTENING returns to IDLE (timeout earcon).
- Every turn is logged as JSON to `logs/turns-YYYYMMDD.jsonl` (future stats).

> Needs a mic, speakers, and `LLM_*` set in `.env`. The state-machine logic is
> unit-tested with fakes (`tests/test_orchestrator.py`); the live audio runner
> (`scripts/viky.py`) needs hardware. A single input stream is opened at a time
> to avoid device contention (wake, record, and barge-in never overlap).

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
> API in dev.

**Migration procedure (DEV → PROD):**

1. Copy the repo + your `.env` to the PROD machine; `pip install -r requirements.txt` and the per-service `requirements-*.txt`.
2. Edit `.env` only — apply the table above (`WHISPER_MODEL=large-v3`, `WHISPER_DEVICE=cuda`, local `LLM_MODEL`/`LLM_API_BASE`, `VIKY_WAKEWORD_MODEL=…/viky.onnx`, `VIKY_WAKEWORD_FALLBACK=false`).
3. Start the local LLM (LM Studio / Ollama) and point `LLM_API_BASE` at it; confirm `python scripts/chat.py "Ahoj"` answers.
4. `python scripts/download_voice.py` (voice) — Whisper downloads on first run.
5. Flip `VIKY_DRY_RUN=false` only once n8n webhooks and the stats backend are wired and verified.
6. `pytest` should stay green (config/health/tools/state-machine run without hardware). Then `python scripts/viky.py` for the live loop.

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
- [x] **M3 — STT:** faster-whisper (forced Czech) + Silero VAD end-of-utterance; `scripts/listen.py`.
- [x] **M4 — Wake word:** openWakeWord loop + earcon, `hey_jarvis` fallback, config flag to swap in `viky.onnx`; `scripts/wakeword_listen.py`.
- [x] **M5 — Brain:** LiteLLM tool-calling + Czech system prompt + 6 tools (dry-run) + mock stats server; `scripts/chat.py`.
- [x] **M6 — Orchestrator:** state machine (barge-in, follow-up, timeout, JSON turn logs); `scripts/viky.py` end-to-end voice.
- [x] **M7 — Hardening:** graceful shutdown (SIGINT/SIGTERM), WAV→transcript→mock-LLM→WAV integration test, finalized DEV→PROD migration procedure.

**All milestones complete.** 33 tests passing (config, health, TTS, STT, VAD,
wake word, tools, brain, orchestrator, integration).

## Design rules

- Voice pipeline is **100% local** (privacy) — no cloud STT/TTS ever.
- LLM layer stays **provider-agnostic** — LiteLLM only, no `anthropic` SDK in
  business logic.
- Tools default to **dry-run** (`VIKY_DRY_RUN=true`) in dev.
- Windows-first for dev; Linux-only steps documented separately.
