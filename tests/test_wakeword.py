"""M4 — wake word + earcon tests.

Earcon generation is offline and always runs. The detector test needs the
openWakeWord models (downloaded on first use) and skips if unavailable.
"""

import wave

import numpy as np
import pytest


def test_earcons_are_generated(tmp_path, monkeypatch):
    import orchestrator.earcon as earcon

    monkeypatch.setattr(earcon, "SOUNDS_DIR", tmp_path)
    paths = earcon.ensure_earcons()
    for name in ("wake", "timeout"):
        p = paths[name]
        assert p.exists()
        with wave.open(str(p), "rb") as w:
            assert w.getnchannels() == 1
            assert w.getframerate() == earcon.SR
            assert w.getnframes() > 0


def _make_detector():
    try:
        from wakeword.detector import WakeWordDetector

        return WakeWordDetector()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"openWakeWord unavailable: {exc}")


def test_silence_does_not_trigger():
    detector = _make_detector()
    from wakeword.detector import FRAME_SAMPLES

    fired = False
    max_score = 0.0
    for _ in range(25):  # ~2 s of silence
        frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        max_score = max(max_score, detector.score(frame))
        if detector.detect(frame):
            fired = True
    assert not fired
    assert max_score < detector.threshold
