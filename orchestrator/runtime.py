"""Real audio/LLM components + orchestrator wiring (shared by CLI and web UI).

RealAudio implements both WakeSource and Recorder so mic access stays serialized
(one input stream at a time) with a background barge-in listener during speech.
Capture is at the device's native rate and resampled to 16 kHz (see
common/audio.py). The speaker reports a live RMS level so the UI can react.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np

from common.logging import get_logger
from config.settings import settings

log = get_logger("runtime")


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
        self._stop_barge()

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
                    return peak_normalize(audio)
        return None


class RealTranscriber:
    def __init__(self) -> None:
        from stt.engine import get_engine

        self._engine = get_engine()

    def transcribe(self, audio: np.ndarray) -> str:
        return self._engine.transcribe(np.asarray(audio, dtype=np.float32)).text


class RealSpeaker:
    """Streams TTS PCM to the speaker, checking barge-in, reporting RMS level."""

    def __init__(self, on_level: Optional[Callable[[float], None]] = None) -> None:
        import sounddevice as sd

        from tts.engine import get_engine

        self._sd = sd
        self._engine = get_engine()
        self._on_level = on_level
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
                if self._on_level:
                    samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
                    if samples.size:
                        self._on_level(float(np.sqrt(np.mean(samples ** 2))) * 3.0)
                stream.write(chunk)
        finally:
            stream.stop()
            stream.close()
            if self._on_level:
                self._on_level(0.0)
        return interrupted


def warmup(transcriber: RealTranscriber, speaker: RealSpeaker) -> None:
    """Load VAD / Whisper / Piper now so the first turn doesn't stall."""
    from stt.vad import UtteranceCollector

    UtteranceCollector()
    transcriber.transcribe(np.zeros(8000, dtype=np.float32))
    _ = speaker._engine.sample_rate


def build_orchestrator(
    on_event: Optional[Callable[[dict], None]] = None,
    on_state=None,
    play_earcon=None,
    do_warmup: bool = True,
):
    """Wire real components into a VikyOrchestrator. Returns (orch, audio)."""
    from brain.llm import Brain
    from orchestrator.state_machine import VikyOrchestrator

    on_level = None
    if on_event is not None:
        on_level = lambda lvl: on_event({"level": lvl})  # noqa: E731

    audio = RealAudio()
    transcriber = RealTranscriber()
    speaker = RealSpeaker(on_level=on_level)
    if do_warmup:
        warmup(transcriber, speaker)

    orch = VikyOrchestrator(
        wake=audio,
        recorder=audio,
        transcriber=transcriber,
        brain=Brain(),
        speaker=speaker,
        on_state=on_state,
        on_event=on_event,
        play_earcon=play_earcon,
    )
    return orch, audio
