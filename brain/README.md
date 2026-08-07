# brain/

LiteLLM routing + system prompt + tool definitions (**M5**).

- `system_prompt.md` — Viky's Czech persona (spoken-style, concise).
- `tools.py` — JSON-schema tool definitions; implementations live in `tools/`.
- All LLM calls go through LiteLLM (`LLM_MODEL`, `LLM_API_BASE`, `LLM_API_KEY`).
  No `anthropic` SDK imports in business logic — provider stays swappable.
