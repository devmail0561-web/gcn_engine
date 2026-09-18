"""Détection de langue légère — FR vs EN — sans dépendance externe."""
from __future__ import annotations

_FR_MARKERS = frozenset({
    "le", "la", "les", "de", "du", "des", "et", "est", "une", "un",
    "en", "à", "que", "qui", "dans", "sur", "par", "avec", "pour",
    "il", "elle", "ils", "elles", "nous", "vous", "ce", "se", "si",
})
_EN_MARKERS = frozenset({
    "the", "is", "are", "was", "were", "and", "of", "to", "in",
    "that", "it", "for", "on", "with", "as", "at", "by", "an",
    "be", "have", "this", "from", "or", "but", "not", "we", "you",
})


def detect_lang(text: str) -> str:
    """Détecte 'fr' ou 'en' depuis les stopwords. Retourne 'fr' si ex aequo."""
    words = set(text.lower().split())
    fr_score = len(words & _FR_MARKERS)
    en_score = len(words & _EN_MARKERS)
    return "fr" if fr_score >= en_score else "en"
