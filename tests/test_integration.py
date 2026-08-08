"""M7 — integration test: WAV in → transcript → mocked LLM → WAV out.

Exercises the real STT and TTS engines with a mocked brain, matching the DoD
pipeline (master prompt §8). Skips when the models aren't present so CI without
the large binaries stays green.
"""

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from tts.engine import resolve_voice_path


def _whisper_cached() -> bool:
    from config.settings import settings

    cache = Path.home() / ".cache" / "huggingface" / "hub"
    return any(cache.glob(f"models--*faster-whisper-{settings.whisper_model}*")) if cache.exists() else False


requires_models = pytest.mark.skipif(
    not (resolve_voice_path().exists() and _whisper_cached()),
    reason="Whisper and/or Piper models not present",
)


@requires_models
def test_full_pipeline_wav_to_wav():
    from stt.engine import get_engine as stt_engine
    from tts.engine import get_engine as tts_engine

    tts = tts_engine()

    # 1. WAV in — synthesize a known Czech question.
    wav_in = io.BytesIO()
    with wave.open(wav_in, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(tts.sample_rate)
        for pcm in tts.synthesize_stream("Kolik je hodin?"):
            w.writeframes(pcm)
    wav_in.seek(0)

    # 2. Transcribe (real Whisper, forced Czech).
    transcript = stt_engine().transcribe(wav_in).text
    assert transcript.strip(), "expected a non-empty transcript"

    # 3. Mocked LLM response (no API key needed).
    from types import SimpleNamespace

    class MockBrain:
        def chat(self, text, history=None):
            assert text  # the transcript reached the brain
            return SimpleNamespace(reply="Jsou dvě hodiny odpoledne.", tool_calls=[])

    reply = MockBrain().chat(transcript).reply

    # 4. WAV out — synthesize the reply.
    wav_out = tts.synthesize_wav_bytes(reply)
    with wave.open(io.BytesIO(wav_out), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == tts.sample_rate
        assert w.getnframes() > 0


@requires_models
def test_orchestrator_end_to_end_with_real_engines():
    """Drive the state machine with real STT/TTS engines and a mock brain,
    using a synthesized clip as the recorded utterance and a null speaker."""
    from types import SimpleNamespace

    from orchestrator.state_machine import State, VikyOrchestrator
    from stt.engine import get_engine as stt_engine
    from tts.engine import get_engine as tts_engine

    tts = tts_engine()
    pcm = b"".join(tts.synthesize_stream("Ahoj Viky."))
    clip = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    # resample 22050 -> 16000 for Whisper
    n_out = int(round(len(clip) * 16000 / tts.sample_rate))
    clip16 = np.interp(
        np.linspace(0, 1, n_out, endpoint=False),
        np.linspace(0, 1, len(clip), endpoint=False),
        clip,
    ).astype(np.float32)

    class OneShotWake:
        def __init__(self):
            self.n = 1

        def wait_for_wake(self):
            if self.n > 0:
                self.n -= 1
                return True
            return False

        def arm(self):
            pass

        def barge_in_detected(self):
            return False

    class OneClipRecorder:
        def __init__(self):
            self.done = False

        def record_utterance(self, timeout_s):
            if self.done:
                return None
            self.done = True
            return clip16

    class RealSTTAdapter:
        def transcribe(self, audio):
            return stt_engine().transcribe(np.asarray(audio, dtype=np.float32)).text

    class MockBrain:
        def chat(self, text, history=None):
            return SimpleNamespace(reply="Ahoj Marty.", tool_calls=[])

    class NullSpeaker:
        def __init__(self):
            self.said = []

        def speak(self, text, interrupt_check):
            self.said.append(text)
            return False

    speaker = NullSpeaker()
    orch = VikyOrchestrator(
        OneShotWake(), OneClipRecorder(), RealSTTAdapter(), MockBrain(), speaker,
    )
    results = orch.run(max_turns=1)
    assert len(results) == 1
    assert results[0].transcript.strip()
    assert speaker.said == ["Ahoj Marty."]
    assert State.SPEAKING in orch.states_visited
