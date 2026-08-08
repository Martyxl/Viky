"""get_trading_stats — query the tradezer.app stats backend (M5).

Read-only, so it runs even under dry-run. In dev it points at the local mock
server (tools/mock_stats.py, STATS_API_BASE); if that is unreachable it returns
a clearly-labelled inline mock so chat.py works with no services running.
"""

from __future__ import annotations

import httpx

from common.logging import get_logger
from config.settings import settings
from tools.base import ToolSpec, register

log = get_logger("tools.stats")


def get_trading_stats(instrument: str, period: str = "day") -> dict:
    url = f"{settings.stats_api_base}/stats"
    try:
        r = httpx.get(url, params={"instrument": instrument, "period": period}, timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001 — degrade gracefully to inline mock
        log.warning("stats backend unreachable (%s); returning inline mock", exc)
        return {
            "instrument": instrument,
            "period": period,
            "source": "inline-mock",
            "trades": 12,
            "win_rate": 0.58,
            "pnl": 415.0,
            "note": "Mock data — stats backend nedostupný.",
        }


register(
    ToolSpec(
        name="get_trading_stats",
        description=(
            "Vrátí obchodní statistiku pro daný instrument a období "
            "(např. instrument 'MNQ', period 'day'|'week'|'month')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "instrument": {"type": "string", "description": "Ticker, např. MNQ, ES, NQ."},
                "period": {
                    "type": "string",
                    "enum": ["day", "week", "month"],
                    "description": "Období statistiky.",
                },
            },
            "required": ["instrument"],
        },
        handler=get_trading_stats,
        side_effect=False,
    )
)
