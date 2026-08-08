"""n8n-backed tools (M5): workflow trigger, email, agents.

All side-effecting → gated by VIKY_DRY_RUN. Email goes through an n8n workflow,
never SMTP directly (master prompt §6).
"""

from __future__ import annotations

import httpx

from common.logging import get_logger
from config.settings import settings
from tools.base import ToolSpec, dry_run_result, is_dry_run, register

log = get_logger("tools.n8n")


def _post_webhook(workflow_id: str, payload: dict) -> dict:
    url = f"{settings.n8n_webhook_base}/{workflow_id}"
    r = httpx.post(url, json=payload or {}, timeout=10.0)
    r.raise_for_status()
    try:
        return {"status": "ok", "workflow_id": workflow_id, "response": r.json()}
    except Exception:  # noqa: BLE001 — non-JSON response
        return {"status": "ok", "workflow_id": workflow_id, "response": r.text}


def trigger_n8n_workflow(workflow_id: str, payload: dict | None = None) -> dict:
    payload = payload or {}
    if is_dry_run():
        return dry_run_result("trigger_n8n_workflow", "POST webhook",
                              workflow_id=workflow_id, payload=payload)
    return _post_webhook(workflow_id, payload)


def send_email(to: str, subject: str, body: str) -> dict:
    if is_dry_run():
        return dry_run_result("send_email", "send via n8n", to=to, subject=subject, body=body)
    # Real send routes through a dedicated n8n email workflow.
    return _post_webhook("send-email", {"to": to, "subject": subject, "body": body})


def list_agents() -> dict:
    # Stub for now (wired to n8n later). Read-only listing.
    return {
        "agents": [
            {"id": "news-impact", "name": "News Impact Trading Agent"},
            {"id": "report", "name": "Evening Report Agent"},
        ],
        "note": "Stub — agents will be wired to n8n.",
    }


def run_agent(agent_id: str, task: str) -> dict:
    if is_dry_run():
        return dry_run_result("run_agent", "start agent", agent_id=agent_id, task=task)
    return _post_webhook(f"agent/{agent_id}", {"task": task})


register(ToolSpec(
    name="trigger_n8n_workflow",
    description="Spustí n8n workflow podle ID s volitelným payloadem.",
    parameters={
        "type": "object",
        "properties": {
            "workflow_id": {"type": "string", "description": "ID/slug workflow v n8n."},
            "payload": {"type": "object", "description": "Data pro workflow.", "additionalProperties": True},
        },
        "required": ["workflow_id"],
    },
    handler=trigger_n8n_workflow,
    side_effect=True,
))

register(ToolSpec(
    name="send_email",
    description="Pošle e-mail přes n8n workflow (nikdy ne přímo SMTP). Před odesláním potvrď s uživatelem.",
    parameters={
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Adresa příjemce."},
            "subject": {"type": "string", "description": "Předmět."},
            "body": {"type": "string", "description": "Tělo zprávy."},
        },
        "required": ["to", "subject", "body"],
    },
    handler=send_email,
    side_effect=True,
))

register(ToolSpec(
    name="list_agents",
    description="Vrátí seznam dostupných agentů.",
    parameters={"type": "object", "properties": {}, "required": []},
    handler=list_agents,
    side_effect=False,
))

register(ToolSpec(
    name="run_agent",
    description="Spustí agenta s daným úkolem.",
    parameters={
        "type": "object",
        "properties": {
            "agent_id": {"type": "string", "description": "ID agenta (viz list_agents)."},
            "task": {"type": "string", "description": "Zadání pro agenta."},
        },
        "required": ["agent_id", "task"],
    },
    handler=run_agent,
    side_effect=True,
))
