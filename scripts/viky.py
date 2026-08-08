"""M6 — run Viky end-to-end (voice). Needs a mic, speakers, and LLM_* in .env.

    python scripts/viky.py

Flow: wait for wake word → record until you stop talking → transcribe (Czech)
→ brain (LiteLLM + tools) → speak the reply. After each reply Viky keeps
listening for FOLLOWUP_WINDOW_S without the wake word; say the wake word during
a reply to barge in. Ctrl-C to quit.

This wires the real components into orchestrator.state_machine.VikyOrchestrator.
It cannot be unit-tested (audio hardware) — the state logic is covered by
tests/test_orchestrator.py.
"""

from __future__ import annotations

import signal
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.logging import get_logger, setup_logging  # noqa: E402
from config.settings import settings  # noqa: E402

log = get_logger("viky")


# --------------------------------------------------------------------------- #
# RealAudio implements BOTH WakeSource and Recorder so mic access stays
# serialized (only one input stream open at a time), plus a background barge-in
# listener during SPEAKING.
# --------------------------------------------------------------------------- #
class RealAudio:
    def __init__(self) -> None:
        import sounddevice as sd

        from wakeword.detector import FRAME_SAMPLES as WW_FRAME
        from wakeword.detector import WAKE_SR, WakeWordDetector

        self._sd = sd
        self._ww_frame = WW_FRAME
        self._ww_sr = WAKE_SR
        self.detector = WakeWordDetector()

        device = settings.audio_input_device or None
        self._dev = int(device) if (device and str(device).isdigit()) else device

        self._stop = False
        self._barge_flag = threading.Event()
        self._barge_stop = threading.Event()
        self._barge_thread: Optional[threading.Thread] = None

    def request_stop(self) -> None:
        self._stop = True

    # -- WakeSource -------------------------------------------------------- #
    def wait_for_wake(self) -> bool:
        from common.audio import ResamplingMicReader

        self._stop_barge()
        self.detector.reset()
        with ResamplingMicReader(self._dev, out_sr=self._ww_sr, out_block=self._ww_frame) as mic:
            while not self._stop:
                if self.detector.detect(mic.read_int16()):
                    return True
        return False

    def arm(self) -> None:
        self._stop_barge()
        self._barge_flag.clear()
        self._barge_stop.clear()
        self._barge_thread = threading.Thread(target=self._barge_loop, daemon=True)
        self._barge_thread.start()

    def barge_in_detected(self) -> bool:
        return self._barge_flag.is_set()

    def _barge_loop(self) -> None:
        from common.audio import ResamplingMicReader

        try:
            self.detector.reset()
            with ResamplingMicReader(self._dev, out_sr=self._ww_sr, out_block=self._ww_frame) as mic:
                while not self._barge_stop.is_set():
                    if self.detector.detect(mic.read_int16()):
                        self._barge_flag.set()
                        return
        except Exception as exc:  # noqa: BLE001
            log.debug("barge-in listener stopped: %s", exc)

    def _stop_barge(self) -> None:
        if self._barge_thread and self._barge_thread.is_alive():
            self._barge_stop.set()
            self._barge_thread.join(timeout=1.0)
        self._barge_thread = None

    # -- Recorder ---------------------------------------------------------- #
    def record_utterance(self, timeout_s: float) -> Optional[np.ndarray]:
        self._stop_barge()
        from common.audio import ResamplingMicReader, peak_normalize
        from stt.vad import FRAME_SAMPLES, SILERO_SR, UtteranceCollector

        collector = UtteranceCollector()
        frames_seen = 0
        max_frames = int(timeout_s * SILERO_SR / FRAME_SAMPLES)
        with ResamplingMicReader(self._dev, out_sr=SILERO_SR, out_block=FRAME_SAMPLES) as mic:
            while frames_seen < max_frames or collector._in_speech:
                if self._stop:
                    return None
                audio = collector.push(mic.read())
                frames_seen += 1
                if audio is not None:
                    return peak_normalize(audio)  # lift quiet mics before STT
        return None


class RealTranscriber:
    def __init__(self) -> None:
        from stt.engine import get_engine

        self._engine = get_engine()

    def transcribe(self, audio: np.ndarray) -> str:
        return self._engine.transcribe(np.asarray(audio, dtype=np.float32)).text


class RealSpeaker:
    """Streams TTS PCM to the speaker, checking for barge-in between chunks."""

    def __init__(self) -> None:
        import sounddevice as sd

        from tts.engine import get_engine

        self._sd = sd
        self._engine = get_engine()
        device = settings.audio_output_device or None
        self._dev = int(device) if (device and str(device).isdigit()) else device

    def speak(self, text: str, interrupt_check: Callable[[], bool]) -> bool:
        if not text.strip():
            return False
        sr = self._engine.sample_rate
        stream = self._sd.RawOutputStream(samplerate=sr, channels=1, dtype="int16", device=self._dev)
        stream.start()
        interrupted = False
        try:
            for chunk in self._engine.synthesize_stream(text):
                if interrupt_check():
                    interrupted = True
                    break
                stream.write(chunk)
        finally:
            stream.stop()
            stream.close()
        return interrupted


def main() -> int:
    setup_logging(settings.log_level)
    from orchestrator.earcon import ensure_earcons, play
    from orchestrator.state_machine import State, VikyOrchestrator
    from brain.llm import Brain

    ensure_earcons()
    print("Viky se probouzí... (načítám modely)")
    audio = RealAudio()
    transcriber = RealTranscriber()
    speaker = RealSpeaker()
    orch = VikyOrchestrator(
        wake=audio,
        recorder=audio,
        transcriber=transcriber,
        brain=Brain(),
        speaker=speaker,
        on_state=lambda s: print(f"  [{s.value}]"),
        play_earcon=play,
    )

    # Warm up everything NOW so the first turn doesn't stall mid-conversation
    # (previously Silero VAD + Whisper loaded lazily on the first utterance,
    # which swallowed the user's first question).
    print("Zahřívám modely (VAD, Whisper, hlas)...")
    from stt.vad import UtteranceCollector

    UtteranceCollector()  # load Silero VAD
    transcriber.transcribe(np.zeros(8000, dtype=np.float32))  # load Whisper
    _ = speaker._engine.sample_rate  # load Piper voice

    label = audio.detector.label
    print(f"Připravena. Řekni '{label}' pro probuzení. Ctrl-C ukončí.")
    if label != "viky":
        print("(fallback wake word — po natrénování nastav VIKY_WAKEWORD_MODEL na viky.onnx)")

    # Graceful shutdown: on SIGINT/SIGTERM stop the loop and the mic threads.
    def _shutdown(signum, _frame):
        log.info("signal %s — ukončuji", signum)
        orch.stop()
        audio.request_stop()

    signal.signal(signal.SIGINT, _shutdown)
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (AttributeError, ValueError):
        pass  # SIGTERM not settable on some Windows setups

    try:
        orch.run()
    except KeyboardInterrupt:
        orch.stop()
        audio.request_stop()
    finally:
        audio._stop_barge()
        print("\nViky: Tak zatím, Marty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
