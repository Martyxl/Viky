"""Ping every service's /health endpoint. Exit non-zero if any is down.

    python scripts/healthcheck.py
"""

from __future__ import annotations

import sys

import httpx

from config.settings import settings

TARGETS = {
    "orchestrator": f"http://{settings.orchestrator_host}:{settings.orchestrator_port}",
    "stt": settings.stt_url,
    "tts": settings.tts_url,
}


def main() -> int:
    ok = True
    for name, base in TARGETS.items():
        try:
            r = httpx.get(f"{base}/health", timeout=3.0)
            r.raise_for_status()
            print(f"  OK   {name:<13} {base}  -> {r.json().get('status')}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  FAIL {name:<13} {base}  -> {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
