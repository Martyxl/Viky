"""Deploy an n8n workflow via the public REST API (Viky-builds-workflows).

Viky's brain (Claude) designs a workflow from a spoken idea and calls
`deploy_n8n_workflow` with the node graph. The workflow is created **inactive**
so Marty can review it, attach credentials, and activate it in the n8n editor.

Needs N8N_API_KEY (create one in n8n: Settings -> n8n API -> Create).
"""

from __future__ import annotations

import httpx

from common.logging import get_logger
from config.settings import settings
from tools.base import ToolSpec, register

log = get_logger("tools.n8n_deploy")


def _editor_url(workflow_id: str) -> str:
    base = settings.n8n_api_base.split("/api/")[0]
    return f"{base}/workflow/{workflow_id}"


def deploy_n8n_workflow(name: str, nodes: list, connections: dict | None = None) -> dict:
    """Create an n8n workflow (inactive) from a node graph Viky designed."""
    if not settings.n8n_api_key:
        return {"error": "N8N_API_KEY není nastavený v .env (n8n → Settings → n8n API → Create)."}

    body = {
        "name": name,
        "nodes": nodes or [],
        "connections": connections or {},
        "settings": {"executionOrder": "v1"},
    }

    # Not gated by VIKY_DRY_RUN: workflows are created INACTIVE, so nothing runs
    # until Marty reviews and activates — safe even in dry-run mode.
    try:
        r = httpx.post(
            f"{settings.n8n_api_base}/workflows",
            headers={"X-N8N-API-KEY": settings.n8n_api_key, "Content-Type": "application/json"},
            json=body,
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
        wf_id = data.get("id", "")
        log.info("deployed workflow %s (%s)", name, wf_id)
        return {
            "status": "created",
            "id": wf_id,
            "name": name,
            "active": False,
            "editor_url": _editor_url(wf_id),
            "note": "Vytvořeno jako NEAKTIVNÍ — otevři v n8n, přicvakni credentials a aktivuj.",
        }
    except httpx.HTTPStatusError as exc:
        # Feed the API error back so the LLM can fix the workflow JSON and retry.
        detail = exc.response.text[:500] if exc.response is not None else str(exc)
        return {"error": f"n8n odmítl workflow ({exc.response.status_code}): {detail}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"deploy selhal: {exc}"}


register(ToolSpec(
    name="deploy_n8n_workflow",
    description=(
        "Nahraje do n8n workflow, který jsi navrhla, přes REST API (vytvoří ho "
        "jako NEAKTIVNÍ, aby si Marty mohl přicvaknout přihlašovací údaje a "
        "aktivovat). Použij, když Marty popíše nápad na automatizaci. "
        "Formát n8n workflow: 'nodes' je pole uzlů, každý má "
        "{name, type (např. 'n8n-nodes-base.webhook', 'n8n-nodes-base.set', "
        "'n8n-nodes-base.httpRequest', 'n8n-nodes-base.if', 'n8n-nodes-base.code', "
        "'n8n-nodes-base.emailSend'), typeVersion (číslo), position [x,y], "
        "parameters {...}}. 'connections' propojuje uzly podle jmen: "
        "{'JménoUzlu': {'main': [[{'node':'DalšíUzel','type':'main','index':0}]]}}. "
        "Data z webhooku bývají pod $json.body. Po nahrání řekni Martymu, ať to "
        "otevře v n8n, přidá credentials a aktivuje."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Název workflow."},
            "nodes": {
                "type": "array",
                "description": "Pole n8n uzlů (viz popis).",
                "items": {"type": "object", "additionalProperties": True},
            },
            "connections": {
                "type": "object",
                "description": "Propojení uzlů podle jmen.",
                "additionalProperties": True,
            },
        },
        "required": ["name", "nodes"],
    },
    handler=deploy_n8n_workflow,
    side_effect=False,  # safe: creates inactive workflows, not gated by dry-run
))
