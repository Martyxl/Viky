"""M5 CLI test — text chat with Viky's brain (tools logged).

    python scripts/chat.py                       # interactive REPL
    python scripts/chat.py "Kolik je hodin?"     # one-shot

Text in → text out. Tool calls are printed so you can see the dry-run behaviour.
Requires LLM_* configured in .env (e.g. Claude API key). Keeps short history in
interactive mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.llm import Brain  # noqa: E402
from config.settings import settings  # noqa: E402


def _print_tools(tool_calls: list[dict]) -> None:
    for tc in tool_calls:
        dry = " [DRY-RUN]" if tc["result"].get("dry_run") else ""
        print(f"   🔧 {tc['name']}({tc['arguments']}){dry} -> {tc['result']}")


def _one(brain: Brain, text: str, history: list[dict]) -> None:
    try:
        rep = brain.chat(text, history=history)
    except Exception as exc:  # noqa: BLE001
        print(f"[chat] LLM chyba: {exc}", file=sys.stderr)
        print("[chat] Zkontroluj LLM_MODEL / LLM_API_KEY v .env "
              "(model musí být ve formátu pro LiteLLM, např. 'anthropic/claude-...').",
              file=sys.stderr)
        return
    if rep.tool_calls:
        _print_tools(rep.tool_calls)
    print(f"\nViky: {rep.reply}")
    print(f"   ({rep.latency_ms} ms, dry_run={settings.dry_run})")
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": rep.reply})


def main() -> int:
    brain = Brain()
    history: list[dict] = []

    if len(sys.argv) > 1:
        _one(brain, " ".join(sys.argv[1:]), history)
        return 0

    print("Viky chat (M5). Napiš zprávu, 'konec' ukončí.")
    while True:
        try:
            text = input("Ty: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"konec", "exit", "quit"}:
            break
        _one(brain, text, history[-8:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
