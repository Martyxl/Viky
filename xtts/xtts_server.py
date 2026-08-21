"""XTTS v2 streaming TTS server (runs in WSL, GPU). Viky calls it over localhost.

POST /speak {text} -> streams int16 mono PCM @ 24 kHz, sentence by sentence
(so playback starts on the first sentence while the rest synthesizes).
"""
import os, re, time
os.environ["COQUI_TOS_AGREED"] = "1"
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

SPEAKER = os.environ.get("XTTS_SPEAKER", "Alison Dietlinde")
LANG = os.environ.get("XTTS_LANGUAGE", "cs")
SR = 24000

app = FastAPI(title="Viky XTTS")
_model = None
_latent = None
_spk_emb = None

_SENT = re.compile(r"[^.!?]+[.!?]?", re.UNICODE)


@app.on_event("startup")
def _load():
    global _model, _latent, _spk_emb
    from TTS.api import TTS
    t0 = time.time()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to("cuda")
    _model = tts.synthesizer.tts_model
    sp = _model.speaker_manager.speakers[SPEAKER]
    _latent = sp["gpt_cond_latent"]
    _spk_emb = sp["speaker_embedding"]
    print(f"[xtts] loaded, speaker={SPEAKER}, {time.time()-t0:.1f}s", flush=True)


@app.get("/health")
def health():
    return {"service": "xtts", "status": "ok" if _model else "loading",
            "speaker": SPEAKER, "sample_rate": SR}


class SpeakReq(BaseModel):
    text: str


@app.post("/speak")
def speak(req: SpeakReq):
    if _model is None:
        return JSONResponse(status_code=503, content={"detail": "still loading"})
    sentences = [s.strip() for s in _SENT.findall(req.text or "") if s.strip()]

    def gen():
        for s in sentences:
            out = _model.inference(s, LANG, _latent, _spk_emb, temperature=0.65)
            wav = np.asarray(out["wav"], dtype=np.float32)
            m = float(np.max(np.abs(wav))) if wav.size else 0.0
            if m > 1e-8:
                wav = wav / m * 0.95
            yield (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()

    return StreamingResponse(gen(), media_type="application/octet-stream",
                             headers={"X-Sample-Rate": str(SR)})
