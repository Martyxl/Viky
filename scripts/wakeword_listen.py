"""M4 CLI test — continuously listen for the wake word.

    python scripts/wakeword_listen.py            # mic loop; beeps on detection
    python scripts/wakeword_listen.py --debug    # print live scores

Uses the trained viky.onnx if VIKY_WAKEWORD_MODEL points to it; otherwise the
fallback model (VIKY_WAKEWORD_FALLBACK_NAME, default "hey_jarvis"). Plays the
wake earcon on each detection. Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from wakeword.detector import FRAME_SAMPLES, WAKE_SR, WakeWordDetector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Listen for the wake word (M4 test).")
    ap.add_argument("--debug", action="store_true", help="print live scores")
    args = ap.parse_args()

    import numpy as np
    import sounddevice as sd

    from orchestrator.earcon import ensure_earcons, play

    ensure_earcons()
    detector = WakeWordDetector()
    device = settings.audio_input_device or None
    dev = int(device) if (device and str(device).isdigit()) else device

    label = detector.label
    print(f"[wakeword] poslouchám wake word '{label}' (práh {detector.threshold:.2f}). Ctrl-C ukončí.")
    if label != "viky":
        print("[wakeword] pozn.: používá se fallback model — řekni např. 'Hey Jarvis'. "
              "Po natrénování nastav VIKY_WAKEWORD_MODEL na viky.onnx.")

    count = 0
    try:
        with sd.InputStream(
            samplerate=WAKE_SR, channels=1, dtype="int16",
            blocksize=FRAME_SAMPLES, device=dev,
        ) as stream:
            while True:
                block, _ = stream.read(FRAME_SAMPLES)
                frame = block.reshape(-1)
                if args.debug:
                    s = detector.score(frame)
                    if s > 0.05:
                        print(f"  score={s:.3f}")
                    fired = s >= detector.threshold and (
                        time.monotonic() - detector._last_fire
                    ) >= detector.cooldown_s
                    if fired:
                        detector._last_fire = time.monotonic()
                        detector._model.reset()
                else:
                    fired = detector.detect(frame)
                if fired:
                    count += 1
                    print(f"\n🔔 [wakeword] DETEKOVÁNO '{label}'! (#{count})")
                    play("wake")
    except KeyboardInterrupt:
        print(f"\n[wakeword] konec. Celkem detekcí: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
