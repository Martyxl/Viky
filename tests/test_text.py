"""Text sanitization — the fix for the 'invalid high surrogate' API crash."""

import json

from common.text import sanitize_text


def test_removes_lone_surrogate():
    bad = "Kolik je hodin\ud83c?"  # lone high surrogate mid-string
    clean = sanitize_text(bad)
    assert "\ud83c" not in clean
    assert clean == "Kolik je hodin?"


def test_clean_text_is_json_serializable():
    bad = "ahoj \ud800 svete \udfff"
    clean = sanitize_text(bad)
    # Must survive JSON encoding to UTF-8 bytes (what the HTTP client does).
    json.dumps({"content": clean}, ensure_ascii=False).encode("utf-8")


def test_keeps_czech_diacritics():
    assert sanitize_text("Příliš žluťoučký kůň") == "Příliš žluťoučký kůň"


def test_empty_and_whitespace():
    assert sanitize_text("") == ""
    assert sanitize_text("   ") == ""
