"""Splitter de texte en phrases."""

import re


def split_sentences(text: str, min_length: int = 10, max_length: int = 500) -> list[str]:
    """Decoupe un texte en phrases.

    Pas de filtre - le LLM determine la causalite.
    Filtrage uniquement par longueur.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if min_length <= len(s.strip()) <= max_length]
