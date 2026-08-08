"""Tool framework (M5): registry + dry-run plumbing.

Each tool is a `ToolSpec` with a JSON-schema parameter definition and a handler.
The same registry drives both the LLM tool schemas (brain/tools.py) and dispatch
(brain/llm.py), so the two can never drift apart.

Side-effecting tools honour `VIKY_DRY_RUN` (default true in dev): they log the
intended call and return a simulated result instead of doing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from common.logging import get_logger
from config.settings import settings

log = get_logger("tools")

Handler = Callable[..., dict]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON Schema (object)
    handler: Handler
    # side_effect tools are gated by dry-run; read-only tools (get_time,
    # get_trading_stats) always execute.
    side_effect: bool = False


_REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"duplicate tool: {spec.name}")
    _REGISTRY[spec.name] = spec
    return spec


def all_tools() -> list[ToolSpec]:
    return list(_REGISTRY.values())


def get_tool(name: str) -> ToolSpec | None:
    return _REGISTRY.get(name)


def dry_run_result(tool: str, action: str, **detail: Any) -> dict:
    """Uniform simulated result for a side-effecting tool under dry-run."""
    log.info("[DRY-RUN] %s: %s %s", tool, action, detail)
    return {"dry_run": True, "tool": tool, "action": action, **detail}


def dispatch(name: str, arguments: dict) -> dict:
    """Execute a tool by name with parsed arguments; never raises."""
    spec = get_tool(name)
    if spec is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return spec.handler(**(arguments or {}))
    except TypeError as exc:  # bad/missing args from the model
        return {"error": f"invalid arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001
        log.exception("tool %s failed", name)
        return {"error": f"{name} failed: {exc}"}


def is_dry_run() -> bool:
    return settings.dry_run
