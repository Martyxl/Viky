"""Run Viky end-to-end (voice) from the terminal. Needs mic, speakers, LLM_*.

    python scripts/viky.py

Real audio wiring lives in orchestrator/runtime.py; state logic in
orchestrator/state_machine.py (unit-tested). For the graphical app see
scripts/viky_ui.py.
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.logging import get_logger, setup_logging  # noqa: E402
from config.settings import settings  # noqa: E402

log = get_logger("viky")


def main() -> int:
    setup_logging(settings.log_level)
    from orchestrator.earcon import ensure_earcons, play
    from orchestrator.runtime import build_orchestrator

    ensure_earcons()
    print("Viky se probouzí... (načítám a zahřívám modely)")
    orch, audio = build_orchestrator(
        on_state=lambda s: print(f"  [{s.value}]"),
        play_earcon=play,
    )

    label = audio.detector.label
    print(f"Připravena. Řekni '{label}' pro probuzení. Ctrl-C ukončí.")
    if label != "viky":
        print("(fallback wake word — po natrénování nastav VIKY_WAKEWORD_MODEL na viky.onnx)")

    def _shutdown(signum, _frame):
        log.info("signal %s — ukončuji", signum)
        orch.stop()
        audio.request_stop()

    signal.signal(signal.SIGINT, _shutdown)
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (AttributeError, ValueError):
        pass

    try:
        orch.run()
    except KeyboardInterrupt:
        orch.stop()
        audio.request_stop()
    finally:
        audio.request_stop()
        print("\nViky: Tak zatím, Marty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
