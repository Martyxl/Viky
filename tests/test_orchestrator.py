"""M6 — orchestrator state-machine transitions (fakes, no audio hardware)."""

from types import SimpleNamespace

import numpy as np
import pytest

from orchestrator.state_machine import State, VikyOrchestrator


class FakeWake:
    def __init__(self, wakes: int = 1, barge: list[bool] | None = None):
        self._remaining = wakes
        self.wait_calls = 0
        self._barge = list(barge or [])
        self.armed = 0

    def wait_for_wake(self) -> bool:
        self.wait_calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            return True
        return False

    def arm(self) -> None:
        self.armed += 1

    def barge_in_detected(self) -> bool:
        return self._barge.pop(0) if self._barge else False


class FakeRecorder:
    def __init__(self, clips):
        self._clips = list(clips)

    def record_utterance(self, timeout_s):
        return self._clips.pop(0) if self._clips else None


class FakeTranscriber:
    def __init__(self, texts):
        self._texts = list(texts)

    def transcribe(self, audio):
        return self._texts.pop(0) if self._texts else ""


class FakeBrain:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def chat(self, text, history=None):
        self.calls.append(text)
        r = self._replies.pop(0) if self._replies else "…"
        return SimpleNamespace(reply=r, tool_calls=[])


class FakeSpeaker:
    def __init__(self, interrupts=None):
        self.spoken = []
        self._interrupts = list(interrupts or [])

    def speak(self, text, interrupt_check):
        self.spoken.append(text)
        return self._interrupts.pop(0) if self._interrupts else False


def _clip():
    return np.ones(1600, dtype=np.float32)


def _run(wake, recorder, transcriber, brain, speaker, max_turns=1):
    earcons = []
    orch = VikyOrchestrator(
        wake, recorder, transcriber, brain, speaker,
        play_earcon=earcons.append,
    )
    results = orch.run(max_turns=max_turns)
    return orch, results, earcons


def test_normal_turn_state_sequence():
    orch, results, earcons = _run(
        FakeWake(wakes=1), FakeRecorder([_clip()]),
        FakeTranscriber(["kolik je hodin"]), FakeBrain(["je devět hodin"]),
        FakeSpeaker(), max_turns=1,
    )
    assert len(results) == 1
    assert results[0].transcript == "kolik je hodin"
    assert results[0].reply == "je devět hodin"
    # full pipeline visited in order
    seq = orch.states_visited
    assert seq[:5] == [
        State.IDLE, State.LISTENING, State.TRANSCRIBING, State.THINKING, State.SPEAKING,
    ]
    assert seq[-1] == State.IDLE
    assert "wake" in earcons


def test_timeout_returns_to_idle():
    wake = FakeWake(wakes=1)
    orch, results, earcons = _run(
        wake, FakeRecorder([None]),  # no speech after wake
        FakeTranscriber([]), FakeBrain([]), FakeSpeaker(), max_turns=1,
    )
    assert results == []
    assert "timeout" in earcons
    assert State.THINKING not in orch.states_visited


def test_followup_skips_wake_word():
    wake = FakeWake(wakes=1)
    orch, results, earcons = _run(
        wake, FakeRecorder([_clip(), _clip()]),
        FakeTranscriber(["ahoj", "díky"]), FakeBrain(["čau", "není zač"]),
        FakeSpeaker(), max_turns=2,
    )
    assert len(results) == 2
    # wake word waited for only once; the second turn used the follow-up window
    assert wake.wait_calls == 1
    assert earcons.count("wake") == 1


def test_barge_in_reenters_listening_without_wake():
    wake = FakeWake(wakes=1)
    speaker = FakeSpeaker(interrupts=[True, False])  # first reply interrupted
    orch, results, earcons = _run(
        wake, FakeRecorder([_clip(), _clip()]),
        FakeTranscriber(["dlouhá otázka", "nová otázka"]),
        FakeBrain(["dlouhá odpověď", "krátká"]), speaker, max_turns=2,
    )
    assert results[0].interrupted is True
    assert len(results) == 2
    assert wake.wait_calls == 1  # barge-in did not require a new wake word


def test_empty_transcript_skips_thinking():
    orch, results, earcons = _run(
        FakeWake(wakes=1), FakeRecorder([_clip()]),
        FakeTranscriber([""]), FakeBrain([]), FakeSpeaker(), max_turns=1,
    )
    assert len(results) == 1
    assert results[0].empty is True
    assert State.THINKING not in orch.states_visited
    assert State.SPEAKING not in orch.states_visited
