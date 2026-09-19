"""Découpage de texte en phrases — FR+EN avec protection des abréviations."""
from __future__ import annotations
import re

# Coupure après ponctuation de fin suivie d'un espace
_SENT_END = re.compile(r'(?<=[.!?])\s+')

# Abréviations courantes FR+EN à ne pas couper
_ABBREV = re.compile(
    r'\b(M|Mme|Mme\.|Dr|Prof|art|al|etc|cf|ex|vs|fig|vol|p|pp|no|approx|env|nb|'
    r'Mr|Ms|Mrs|Jr|Sr|Fig|Vol|No|Approx|Dept|St|Ave)\.\s',
    re.IGNORECASE,
)

_SPACE_NORM = re.compile(r'\s+')
_MARKER = "\x00ABBR\x00"  # null bytes : ne peut pas apparaître dans du texte scraped


def split_sentences(text: str, min_length: int = 30, max_length: int = 600) -> list[str]:
    """
    Découpe un texte en phrases.

    Protège les abréviations connues pour éviter les coupures fausses.
    Filtre par longueur (min_length / max_length en caractères).
    """
    # Normalise les sauts de ligne
    text = text.replace("\r\n", " ").replace("\n", " ")
    # Protège les abréviations
    protected = _ABBREV.sub(lambda m: m.group().replace(". ", _MARKER), text)
    raw = _SENT_END.split(protected)
    result = []
    for s in raw:
        s = s.replace(_MARKER, ". ").strip()
        s = _SPACE_NORM.sub(" ", s)
        if min_length <= len(s) <= max_length:
            result.append(s)
    return result
