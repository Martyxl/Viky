"""Brain — LiteLLM routing with tool calling (M5).

Provider-agnostic: everything goes through LiteLLM's OpenAI-compatible
`completion()`. Swap Claude ↔ local Qwen3 by editing LLM_* in .env only; no
`anthropic`/provider SDK is imported here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from brain.tools import openai_tools
from common.logging import get_logger
from config.settings import settings
from tools.registry import dispatch

log = get_logger("brain.llm")
PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.md"


def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


@dataclass
class BrainReply:
    reply: str
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    messages: list[dict] = field(default_factory=list)


class Brain:
    def __init__(self, max_tool_iters: int = 5) -> None:
        self.system_prompt = load_system_prompt()
        self.max_tool_iters = max_tool_iters

    def _completion(self, messages: list[dict]):
        import litellm  # lazy import; only needed when actually calling the LLM

        kwargs: dict = {
            "model": settings.llm_model,
            "messages": messages,
            "tools": openai_tools(),
            "tool_choice": "auto",
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }
        if settings.llm_api_base:
            kwargs["api_base"] = settings.llm_api_base
        if settings.llm_api_key:
            kwargs["api_key"] = settings.llm_api_key
        return litellm.completion(**kwargs)

    @staticmethod
    def _assistant_toolcall_msg(msg, tool_calls) -> dict:
        """Rebuild the assistant turn (with tool_calls) as a plain dict."""
        return {
            "role": "assistant",
            "content": getattr(msg, "content", None) or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ],
        }

    def chat(self, user_text: str, history: Optional[list[dict]] = None) -> BrainReply:
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        tool_log: list[dict] = []
        t0 = time.perf_counter()

        for _ in range(self.max_tool_iters + 1):
            resp = self._completion(messages)
            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                reply = (getattr(msg, "content", None) or "").strip()
                return BrainReply(
                    reply=reply,
                    tool_calls=tool_log,
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                    messages=messages,
                )

            messages.append(self._assistant_toolcall_msg(msg, tool_calls))
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = dispatch(name, args)
                tool_log.append({"name": name, "arguments": args, "result": result})
                log.info("tool %s(%s) -> %s", name, args, result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # Tool loop exhausted — return best-effort content.
        return BrainReply(
            reply="Promiň, zamotala jsem se v krocích. Zkus to prosím znovu.",
            tool_calls=tool_log,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            messages=messages,
        )
