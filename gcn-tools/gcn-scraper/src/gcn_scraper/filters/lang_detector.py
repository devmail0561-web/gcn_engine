"""Détection de langue via langdetect (55+ langues, aucun stopword hardcodé)."""
from __future__ import annotations

try:
    from langdetect import detect, LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


def detect_lang(text: str, default: str = "en") -> str:
    """
    Détecte la langue d'un texte. Supporte 55+ langues via langdetect.

    Retourne le code ISO 639-1 (ex: 'fr', 'en', 'es', 'de', 'zh', 'ar'...).
    Retourne `default` si la détection échoue ou si langdetect n'est pas installé.

    Pour les textes très courts (< 3 mots), retourne `default`.
    """
    if not _LANGDETECT_AVAILABLE or not text or len(text.split()) < 3:
        return default
    try:
        return detect(text)
    except LangDetectException:
        return default
