# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer L0 — split fixe + ids stables (socle refacto, étape 1).

Références : ``docs/refactory/plan-remodelage-tout-appris.md`` ([T6] moteur
agnostique langue, [R5] hash + ids-stables, [R1] ids→offsets),
``plan-remodelage-inference.md`` §[7] (un seul ``tokenizer.split``).

Cadre : 1 seul moteur, 0 frontend par langue. L0 = 1 tokenizer + profils par
FAMILLE DE SCRIPT (latin / CJK-sans-espaces / arabe RTL / autres, regex stdlib
uniquement) + normalisation par script. Ajouter une langue = ajouter des
DONNÉES (D2), pas du code.

Sortie ``split(text)`` : ``(ids, offsets, tokens)`` alignés 1:1 —
``ids`` = entiers 64 bits STABLES (sha256, sans vocab — [R5] « ids-stables
AVANT vocab », indépendants de ``PYTHONHASHSEED``) ; ``offsets`` = spans
``(debut, fin)`` dans le texte ORIGINAL ; ``tokens`` = vues normalisées.
Le vocabulaire tok + ``tokenizer_hash`` arrivent en s1 ([R4]) — ici seul le
mécanisme de hash existe (vocab vide admis).

Contrat inviolable : ``split`` est pur, déterministe, stdlib uniquement
(``re`` + ``unicodedata``), jamais de ``hash()`` ni d'itération d'ensemble
dans les chemins de sortie.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable

#: Version des règles : bumper à tout changement de segmentation/normalisation
#: (les ids changent avec — gate ids-stables : train/val raise si >5 %, R5).
RULES_VERSION = "l0-v1"

#: Familles de script (T6).
LATIN = "latin"
CJK = "cjk"
ARABIC = "arabic"
OTHER = "other"


def _in_ranges(code: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= code <= hi for lo, hi in ranges)


#: CJK + syllabaires et scripts sans espaces (découpage au caractère).
_CJK_RANGES = (
    (0x3040, 0x30FF),  # hiragana + katakana
    (0x3400, 0x4DBF),  # CJK ext. A
    (0x4E00, 0x9FFF),  # CJK unifié
    (0xF900, 0xFAFF),  # compatibilité
    (0x20000, 0x2A6DF),  # ext. B
    (0x0E00, 0x0E7F),  # thaï (sans espaces)
    (0x0E80, 0x0EFF),  # lao (sans espaces)
    (0x1000, 0x109F),  # myanmar (sans espaces)
    (0x1780, 0x17FF),  # khmer (sans espaces)
)

#: Arabe RTL (+ formes de présentation).
_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def char_family(ch: str) -> str:
    """Famille de script d'un caractère (déterminisme total)."""
    code = ord(ch)
    if _in_ranges(code, _CJK_RANGES):
        return CJK
    if _in_ranges(code, _ARABIC_RANGES):
        return ARABIC
    cat = unicodedata.category(ch)
    if cat.startswith("L"):
        return LATIN
    return OTHER


def normalize_token(token: str, family: str) -> str:
    """Normalisation par script (T6) : latin = minuscule + sans-accent ;
    autres = NFC (+ règles propres : arabe garde ses diacritiques).
    """
    if family == LATIN:
        decomposed = unicodedata.normalize("NFKD", token)
        stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
        return stripped.lower()
    return unicodedata.normalize("NFC", token)


def stable_id(normalized: str) -> int:
    """Id stable 64 bits d'un token normalisé (sha256, [R5] avant vocab)."""
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big")


def tokenizer_hash(rules_version: str, vocab: Iterable[str] = ()) -> str:
    """Hash socle ([R5]) : ``sha256(règles + vocab trié)`` (hex, 64 chars).

    ``vocab`` : itérable de tokens (vide admis en s0 — le vocab arrive en s1).
    Toute collision de vocab multilingue se règle par vocab séparé par script,
    jamais par code (T6).
    """
    h = hashlib.sha256()
    h.update(rules_version.encode("utf-8"))
    h.update(b"\n")
    for tok in sorted(vocab):
        h.update(tok.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


_APOSTROPHES = {"'", "’", "‘", "`"}


def _is_word_char(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith(("L", "M")) or cat == "Nd"


def split(text: str) -> tuple[list[int], list[tuple[int, int]], list[str]]:
    """Split fixe : ``(ids, offsets, tokens)`` alignés (1j, étape L0).

    - CJK/sans-espaces : 1 caractère = 1 token ;
    - mots : lettres/chiffres (+ apostrophe interne : ``l'homme`` = 1 token) ;
    - ponctuation/symboles : 1 token chacun (``?`` compte pour les actes) ;
    - blancs : séparateurs jetés ; marques combinantes : jamais coupées.
    ``offsets`` indexe le texte ORIGINAL ; ``tokens`` sont normalisés par
    famille du token (arabe si ≥1 lettre arabe, CJK déjà découpé, sinon latin).
    Texte vide/blanc → triplets vides. Lève ``TypeError`` si non-``str``.
    """
    if not isinstance(text, str):
        raise TypeError(f"text doit être str, reçu {type(text).__name__}")
    ids: list[int] = []
    offsets: list[tuple[int, int]] = []
    tokens: list[str] = []

    def flush_word(start: int, end: int) -> None:
        brut = text[start:end]
        family = token_family(brut)
        norm = normalize_token(brut, family)
        if not norm:
            return
        ids.append(stable_id(norm))
        offsets.append((start, end))
        tokens.append(norm)

    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if char_family(ch) == CJK:
            norm = normalize_token(ch, CJK)
            ids.append(stable_id(norm))
            offsets.append((i, i + 1))
            tokens.append(norm)
            i += 1
            continue
        if _is_word_char(ch):
            start = i
            i += 1
            while i < n:
                nxt = text[i]
                if char_family(nxt) == CJK:
                    break
                if nxt in _APOSTROPHES and i + 1 < n and _is_word_char(text[i + 1]):
                    i += 1
                    continue
                if not _is_word_char(nxt):
                    break
                i += 1
            flush_word(start, i)
            continue
        cat = unicodedata.category(ch)
        if cat.startswith(("P", "S")):
            norm = normalize_token(ch, OTHER)
            ids.append(stable_id(norm))
            offsets.append((i, i + 1))
            tokens.append(norm)
        i += 1
    return ids, offsets, tokens


def token_family(token: str) -> str:
    """Famille d'un token : arabe si ≥1 lettre arabe, sinon latin (CJK déjà
    découpé en amont — un token mixte résiduel tombe en latin par défaut)."""
    for ch in token:
        if char_family(ch) == ARABIC:
            return ARABIC
    return LATIN


__all__ = [
    "ARABIC",
    "CJK",
    "LATIN",
    "OTHER",
    "RULES_VERSION",
    "char_family",
    "normalize_token",
    "split",
    "stable_id",
    "token_family",
    "tokenizer_hash",
]
