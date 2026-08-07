"""Start all Viky services locally (without Docker), for dev on Windows.

Launches stt, tts, and orchestrator with uvicorn as child processes and
streams their logs. Ctrl-C stops all of them.

    python scripts/run_local.py
"""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SERVICES = [
    ("stt", "stt.app:app", 8001),
    ("tts", "tts.app:app", 8002),
    ("orchestrator", "orchestrator.app:app", 8000),
]


def main() -> int:
    procs: list[subprocess.Popen] = []
    try:
        for name, target, port in SERVICES:
            print(f"[run_local] starting {name} on :{port}")
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        target,
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                    ],
                    cwd=str(ROOT),
                )
            )
        print("[run_local] all services up. Ctrl-C to stop.")
        signal.pause() if hasattr(signal, "pause") else _wait(procs)
    except KeyboardInterrupt:
        print("\n[run_local] shutting down...")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


def _wait(procs: list[subprocess.Popen]) -> None:
    # Windows has no signal.pause(); block on the first process instead.
    if procs:
        procs[0].wait()


if __name__ == "__main__":
    raise SystemExit(main())
