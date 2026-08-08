"""get_time — real local time so the LLM never guesses (M5)."""

from __future__ import annotations

from datetime import datetime

from tools.base import ToolSpec, register

# Czech day names for spoken-style answers.
_DAYS = ["pondělí", "úterý", "středa", "čtvrtek", "pátek", "sobota", "neděle"]


def get_time() -> dict:
    now = datetime.now().astimezone()
    return {
        "iso": now.isoformat(),
        "date": now.strftime("%d.%m.%Y"),
        "time": now.strftime("%H:%M"),
        "weekday_cs": _DAYS[now.weekday()],
        "tz": now.tzname(),
    }


register(
    ToolSpec(
        name="get_time",
        description="Vrátí aktuální místní datum a čas. Použij vždy, když se uživatel ptá na čas nebo datum.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=get_time,
        side_effect=False,
    )
)
