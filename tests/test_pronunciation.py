"""Mixed CZ/EN pronunciation: word classification + detection."""

import pytest

from tts.pronunciation import _classify, has_english_terms, split_sentences


def test_english_terms_classified_en():
    for word in ("report", "email", "workflow", "trading", "Report", "EMAIL"):
        assert _classify(word) == "en-us", word


def test_czech_words_classified_cs():
    for word in ("ahoj", "hodin", "dnes", "spustím", "příliš", "žluťoučký"):
        assert _classify(word) == "cs", word


def test_diacritic_word_never_english():
    # even if some ASCII prefix matched, diacritics force Czech
    assert _classify("reportů") == "cs"


def test_has_english_terms():
    assert has_english_terms("Pošli mi report na email.")
    assert not has_english_terms("Kolik je dnes hodin?")


def test_split_sentences():
    s = split_sentences("Ahoj. Jak se máš? Dobře!")
    assert s == ["Ahoj.", "Jak se máš?", "Dobře!"]


@pytest.mark.skipif(
    __import__("tts.engine", fromlist=["resolve_voice_path"]).resolve_voice_path().exists() is False,
    reason="Piper voice not downloaded",
)
def test_mixed_engine_produces_audio():
    from tts.engine import get_engine

    pcm = b"".join(get_engine().synthesize_stream("Pošli report na email."))
    assert len(pcm) > 0
