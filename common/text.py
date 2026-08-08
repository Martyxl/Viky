"""Text sanitization for anything sent to the LLM API.

Whisper can emit invalid Unicode (lone surrogates) on noisy audio, which makes
the JSON request body invalid and the Anthropic API returns 400
("invalid high surrogate in string"). Strip those before they reach the API.
"""

from __future__ import annotations


def sanitize_text(text: str) -> str:
    if not text:
        return ""
    # Remove unpaired surrogate code points.
    cleaned = "".join(c for c in text if not 0xD800 <= ord(c) <= 0xDFFF)
    # Belt-and-braces: round-trip through UTF-8, dropping anything invalid.
    cleaned = cleaned.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return cleaned.strip()
