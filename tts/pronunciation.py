"""Mixed Czech/English phonemization for Piper (pronunciation fix).

The Czech voice (Kasandra) knows every English phoneme, so we phonemize known
English words with the en-us espeak voice and the rest with Czech, then feed the
combined phonemes to the voice. English terms then sound English (with a light
Czech accent) instead of being spelled out Czech-style.

Which words are English is decided by an editable list (tts/english_terms.txt);
words carrying Czech diacritics are always Czech.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Optional

from common.logging import get_logger

log = get_logger("tts.pronunciation")

CZ_DIACRITICS = set("áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ")
TERMS_FILE = Path(__file__).resolve().parent / "english_terms.txt"

# Split into word / whitespace / other, keeping everything (Unicode letters).
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+|\s+|[^\w\s]", re.UNICODE)

_phonemizer = None
_phon_lock = threading.Lock()
_terms: Optional[set[str]] = None


def _load_terms() -> set[str]:
    global _terms
    if _terms is not None:
        return _terms
    terms: set[str] = set()
    try:
        for line in TERMS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                terms.add(line.lower())
    except FileNotFoundError:
        log.warning("english_terms.txt not found; English pronunciation disabled")
    _terms = terms
    return terms


def _get_phonemizer():
    global _phonemizer
    if _phonemizer is None:
        with _phon_lock:
            if _phonemizer is None:
                from piper.phonemize_espeak import EspeakPhonemizer

                _phonemizer = EspeakPhonemizer()
    return _phonemizer


def _classify(word: str) -> str:
    core = "".join(c for c in word if c.isalpha())
    if not core:
        return "cs"
    if any(c in CZ_DIACRITICS for c in word):
        return "cs"
    return "en-us" if core.lower() in _load_terms() else "cs"


def has_english_terms(text: str) -> bool:
    return any(
        _classify(tok) == "en-us"
        for tok in _TOKEN_RE.findall(text)
        if any(ch.isalpha() for ch in tok)
    )


def language_runs(text: str) -> list[tuple[str, str]]:
    """Group text into maximal contiguous (language, segment) runs.

    Whitespace/punctuation/digits attach to the current run, so Czech words stay
    together and get phonemized as whole phrases (proper prosody). Only genuinely
    English words split a run off.
    """
    runs: list[tuple[str, str]] = []
    cur_lang: Optional[str] = None
    cur: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        lang = _classify(tok) if any(ch.isalpha() for ch in tok) else None
        if lang is None:
            cur.append(tok)  # neutral: stays in the current run
            continue
        if cur_lang is None:
            cur_lang = lang
        if lang != cur_lang:
            runs.append((cur_lang, "".join(cur)))
            cur = []
            cur_lang = lang
        cur.append(tok)
    if cur:
        runs.append((cur_lang or "cs", "".join(cur)))
    return runs


def mixed_phonemes(text: str) -> list[str]:
    """Phonemize `text`, switching espeak voice per contiguous language run."""
    ph = _get_phonemizer()
    out: list[str] = []
    for lang, seg in language_runs(text):
        if not seg.strip():
            out.extend(list(seg))
            continue
        for sentence in ph.phonemize(lang, seg):
            out.extend(sentence)
        if out and out[-1] != " ":
            out.append(" ")
    # drop trailing filler space
    while out and out[-1] == " ":
        out.pop()
    return out


# Split text into sentences so synthesis can still stream chunk-by-chunk.
_SENT_RE = re.compile(r"[^.!?]+[.!?]?", re.UNICODE)


def split_sentences(text: str) -> list[str]:
    parts = [m.group(0).strip() for m in _SENT_RE.finditer(text)]
    return [p for p in parts if p]
