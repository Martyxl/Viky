# XTTS voice (optional, nicer TTS)

A higher-quality neural voice (XTTS v2, speaker **Alison Dietlinde**) that also
pronounces English words natively — an alternative to the fast Piper voice.

It runs as a small GPU server **inside WSL2** (Coqui XTTS needs CUDA + audio
libs that are painful on Windows/Python 3.13), and Viky (on Windows) calls it
over `localhost` (WSL2 forwards the port). Enable with `TTS_ENGINE=xtts` in
`.env`; `scripts/viky_ui.py` auto-starts the server. Falls back to Piper if the
server is unreachable.

## One-time WSL setup (Ubuntu, with an NVIDIA GPU)

```bash
# in WSL Ubuntu (root), with miniconda installed:
conda create -y -n xtts -c conda-forge --override-channels python=3.11
conda activate xtts
sudo apt-get install -y ffmpeg
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install coqui-tts "transformers>=4.57,<5" torchcodec fastapi uvicorn pydantic
# copy xtts_server.py and start_xtts.sh to /root/
```

## Run

`scripts/viky_ui.py` starts it automatically when `TTS_ENGINE=xtts`. Manually:

```bash
wsl -d Ubuntu -u root -- bash -lc "cd /root && bash start_xtts.sh"
```

Server: `http://127.0.0.1:8020` — `GET /health`, `POST /speak {text}` streams
int16 mono PCM @ 24 kHz (sentence by sentence). Speaker via `XTTS_SPEAKER`.

## VRAM note

XTTS is only ~2 GB, but with a large local LLM (Qwen ~19 GB) + Whisper on the
same 24 GB GPU it gets tight — drop Whisper to `medium` or lower Qwen if you hit
out-of-memory.

> License: XTTS v2 is CPML (non-commercial) — fine for a personal assistant.
