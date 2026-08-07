"""Shared logging setup.

Human-readable console logs for services, plus a helper to emit structured
JSON turn-logs (used by the orchestrator from M6/M7 onward).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_turn(logs_dir: str | Path, record: dict[str, Any]) -> Path:
    """Append one conversation turn as a JSON line to logs/turns-YYYYMMDD.jsonl.

    This file is the future statistics source (see master prompt §5).
    """
    directory = Path(logs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = directory / f"turns-{day}.jsonl"
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
