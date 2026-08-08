"""M3 — VAD segmentation tests.

The silence test is cheap and offline (Silero's ONNX model ships with the
package). The speech-segmentation test needs the Piper voice to synthesize a
real utterance, so it skips when the voice is absent.
"""

import numpy as np
import pytest

from stt.vad import FRAME_SAMPLES, SILERO_SR, UtteranceCollector
from tts.engine import resolve_voice_path

requires_voice = pytest.mark.skipif(
    not resolve_voice_path().exists(),
    reason="Piper voice model not downloaded (needed to synthesize test speech)",
)


def _frames(audio: np.ndarray):
    for i in range(0, len(audio) - FRAME_SAMPLES, FRAME_SAMPLES):
        yield audio[i : i + FRAME_SAMPLES]


def test_pure_silence_yields_no_utterance():
    col = UtteranceCollector()
    silence = np.zeros(SILERO_SR, dtype=np.float32)  # 1 s
    results = [col.push(f) for f in _frames(silence)]
    assert all(r is None for r in results)


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


@requires_voice
def test_speech_then_silence_segments_one_utterance():
    from tts.engine import get_engine

    # Synthesize a short Czech utterance, resample to 16 kHz for Silero.
    engine = get_engine()
    pcm = b"".join(engine.synthesize_stream("Dobrý den, tady Viky."))
    audio22k = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    speech = _resample_linear(audio22k, engine.sample_rate, SILERO_SR)
    # Pad with trailing silence so the VAD fires an end-of-utterance event.
    padded = np.concatenate([speech, np.zeros(SILERO_SR, dtype=np.float32)])

    col = UtteranceCollector()
    utterances = [r for r in (col.push(f) for f in _frames(padded)) if r is not None]
    assert len(utterances) == 1
    # Captured utterance should be a meaningful fraction of the speech length.
    assert utterances[0].size > 0.3 * speech.size
