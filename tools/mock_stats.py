"""Mock tradezer.app stats backend for dev (M5).

Run it so get_trading_stats has something to hit:
    uvicorn tools.mock_stats:app --port 8010
(matches STATS_API_BASE=http://localhost:8010 in .env.example)
"""

from __future__ import annotations

import hashlib

from fastapi import FastAPI

app = FastAPI(title="Viky · mock stats")


def _deterministic(instrument: str, period: str) -> dict:
    # Stable pseudo-values so repeated calls look consistent.
    h = int(hashlib.sha256(f"{instrument}:{period}".encode()).hexdigest(), 16)
    trades = 5 + h % 30
    wins = h % (trades + 1)
    win_rate = round(wins / trades, 2) if trades else 0.0
    pnl = round(((h % 4000) - 1500) / 10.0, 1)
    return {
        "instrument": instrument.upper(),
        "period": period,
        "source": "mock-stats-server",
        "trades": trades,
        "wins": wins,
        "win_rate": win_rate,
        "pnl": pnl,
    }


@app.get("/health")
def health() -> dict:
    return {"service": "mock-stats", "status": "ok"}


@app.get("/stats")
def stats(instrument: str, period: str = "day") -> dict:
    return _deterministic(instrument, period)
