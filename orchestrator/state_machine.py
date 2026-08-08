"""Viky orchestrator state machine (M6).

    IDLE → LISTENING → TRANSCRIBING → THINKING → SPEAKING → (LISTENING | IDLE)

Components are injected via Protocols so the transitions can be unit-tested
with fakes (no audio hardware). The real audio wiring lives in scripts/viky.py.

Features (master prompt §5):
  * Barge-in — wake word during SPEAKING stops TTS and returns to LISTENING.
  * Follow-up — after a reply, keep LISTENING for FOLLOWUP_WINDOW_S without the
    wake word.
  * Timeout — LISTENING with no speech returns to IDLE (timeout earcon).
  * Structured JSON turn logs (the future statistics source).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Protocol

import numpy as np

from common.logging import get_logger, log_turn
from config.settings import settings

log = get_logger("orchestrator")


class State(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


# --------------------------------------------------------------------------- #
# Injected component interfaces
# --------------------------------------------------------------------------- #
class WakeSource(Protocol):
    def wait_for_wake(self) -> bool:
        """Block until the wake word is detected; return True (False = stop)."""

    def barge_in_detected(self) -> bool:
        """Non-blocking: True if the wake word fired since the last arm()."""

    def arm(self) -> None:
        """Reset barge-in state at the start of SPEAKING."""


class Recorder(Protocol):
    def record_utterance(self, timeout_s: float) -> Optional[np.ndarray]:
        """Record one utterance; return audio, or None on silence/timeout."""


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class BrainLike(Protocol):
    def chat(self, text: str, history: Optional[list[dict]] = None): ...


class Speaker(Protocol):
    def speak(self, text: str, interrupt_check: Callable[[], bool]) -> bool:
        """Speak text; return True if interrupted (barge-in) before finishing."""


@dataclass
class TurnResult:
    transcript: str = ""
    reply: str = ""
    tool_calls: list = field(default_factory=list)
    interrupted: bool = False
    empty: bool = False
    latency_ms: dict = field(default_factory=dict)


def _has_audio(audio: Optional[np.ndarray]) -> bool:
    return audio is not None and getattr(audio, "size", len(audio) if audio is not None else 0) > 0


class VikyOrchestrator:
    def __init__(
        self,
        wake: WakeSource,
        recorder: Recorder,
        transcriber: Transcriber,
        brain: BrainLike,
        speaker: Speaker,
        on_state: Optional[Callable[[State], None]] = None,
        play_earcon: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.wake = wake
        self.recorder = recorder
        self.transcriber = transcriber
        self.brain = brain
        self.speaker = speaker
        self._on_state = on_state
        self._play_earcon = play_earcon or (lambda name: None)

        self.state = State.IDLE
        self.states_visited: list[State] = []
        self.history: list[dict] = []
        self._stop = False

    # -- helpers ----------------------------------------------------------- #
    def _set_state(self, s: State) -> None:
        self.state = s
        self.states_visited.append(s)
        log.debug("state -> %s", s.value)
        if self._on_state:
            self._on_state(s)

    def stop(self) -> None:
        self._stop = True

    # -- one interaction --------------------------------------------------- #
    def _process(self, audio: np.ndarray) -> TurnResult:
        lat: dict = {}

        self._set_state(State.TRANSCRIBING)
        t0 = time.perf_counter()
        transcript = (self.transcriber.transcribe(audio) or "").strip()
        lat["stt_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if not transcript:
            self._set_state(State.IDLE)
            return TurnResult(empty=True, latency_ms=lat)

        self._set_state(State.THINKING)
        t0 = time.perf_counter()
        try:
            reply = self.brain.chat(transcript, history=self.history[-8:])
            reply_text = (reply.reply or "").strip()
            tool_calls = getattr(reply, "tool_calls", [])
        except Exception as exc:  # noqa: BLE001 — an LLM/API error must not kill Viky
            log.exception("brain failed")
            reply_text = "Promiň, něco se mi teď nepovedlo. Zkus to prosím znovu."
            tool_calls = []
        if not reply_text:
            # Never speak or store an empty turn — an empty assistant message
            # makes the next LLM call fail.
            reply_text = "Promiň, tomu jsem nerozuměla."
        lat["llm_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        self.history.append({"role": "user", "content": transcript})
        self.history.append({"role": "assistant", "content": reply_text})

        self._set_state(State.SPEAKING)
        self.wake.arm()
        t0 = time.perf_counter()
        try:
            interrupted = self.speaker.speak(reply_text, interrupt_check=self.wake.barge_in_detected)
        except Exception as exc:  # noqa: BLE001 — audio errors must not kill Viky
            log.exception("speak failed")
            interrupted = False
        lat["tts_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        result = TurnResult(
            transcript=transcript,
            reply=reply_text,
            tool_calls=tool_calls,
            interrupted=interrupted,
            latency_ms=lat,
        )
        self._log_turn(result)
        return result

    def _log_turn(self, r: TurnResult) -> None:
        try:
            log_turn(settings.logs_dir, {
                "transcript": r.transcript,
                "reply": r.reply,
                "tool_calls": r.tool_calls,
                "interrupted": r.interrupted,
                "latency_ms": r.latency_ms,
            })
        except Exception as exc:  # noqa: BLE001 — logging must never break the loop
            log.warning("turn log failed: %s", exc)

    # -- main loop --------------------------------------------------------- #
    def run(self, max_turns: Optional[int] = None) -> list[TurnResult]:
        """Run the conversation loop. `max_turns` bounds it for tests."""
        results: list[TurnResult] = []
        awaiting_wake = True
        listen_timeout = settings.listen_timeout_s

        while not self._stop:
            if awaiting_wake:
                self._set_state(State.IDLE)
                if not self.wake.wait_for_wake():
                    break  # stop signalled
                self._play_earcon("wake")

            self._set_state(State.LISTENING)
            audio = self.recorder.record_utterance(listen_timeout)

            if not _has_audio(audio):
                # Silence: end of a wake turn or of the follow-up window.
                self._play_earcon("timeout")
                awaiting_wake = True
                listen_timeout = settings.listen_timeout_s
                if max_turns is not None and len(results) >= max_turns:
                    break
                continue

            result = self._process(audio)
            results.append(result)

            if max_turns is not None and len(results) >= max_turns:
                break

            if result.interrupted:
                # Barge-in: listen again immediately, no wake needed.
                awaiting_wake = False
                listen_timeout = settings.listen_timeout_s
            elif result.empty:
                awaiting_wake = True
                listen_timeout = settings.listen_timeout_s
            else:
                # Follow-up window: keep listening without the wake word.
                awaiting_wake = False
                listen_timeout = settings.followup_window_s

        self._set_state(State.IDLE)
        return results
