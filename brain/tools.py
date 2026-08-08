"""LLM-facing tool schemas (M5).

Converts the tool registry into the OpenAI-compatible `tools` format that
LiteLLM passes to any provider. Keeping this derived from the registry means
schemas and implementations never drift.
"""

from __future__ import annotations

from tools.registry import all_tools


def openai_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in all_tools()
    ]


def tool_names() -> list[str]:
    return [spec.name for spec in all_tools()]
