# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Segmentation socle + association apprise texte → CIR (pas un frontend).

T6 : 0 frontend — ce module n'en est pas un : il ne contient AUCUNE règle
lexicale par langue (vérifiable par grep : zéro lemme, zéro table, zéro
mot-clé). Il segmente via le socle B (clauses + connecteurs + source,
même rôle que dans ``nlu/infer.py``) et délègue l'association texte → CIR
au runtime pur (``pipeline/forward_ids.py`` : E_tok + têtes apprises).
Ajouter une langue = ajouter des DONNÉES (D2), jamais du code ici.

Seul endroit avec ``nlu/infer.py`` autorisé à appeler le socle B : la garde
du runtime interdit ``pipeline/``, ``training/`` et ``engine.py`` de le faire.
"""

from __future__ import annotations

from ..layer0 import engine_b
from ..layer0.tokenizer import split as tok_split


def parse_text(text: str, lang: str = "fr") -> dict:
    """Texte brut → segments tok prêts pour le runtime.

    Retourne ``{"ids", "offsets", "clause_spans", "clause_textes",
    "connecteurs", "source"}`` — ``source`` = ``silver-B|fallback`` (B).
    Texte vide/blanc → segments vides (jamais d'exception — miroir du
    legacy ``parse_lattice`` qui ne retourne jamais d'erreur).
    ``lang`` hors budget → ``ValueError`` de B (pas de silencieux).
    """
    if not isinstance(text, str):
        raise TypeError(f"text doit être str, reçu {type(text).__name__}")
    if not text.strip():
        return {
            "ids": [],
            "offsets": [],
            "clause_spans": [],
            "clause_textes": [],
            "connecteurs": [],
            "source": "vide",
        }
    sortie_b = engine_b.analyze(text, lang=lang)
    ids, offsets, _tokens = tok_split(text)
    return {
        "ids": ids,
        "offsets": offsets,
        "clause_spans": [list(c["span"]) for c in sortie_b.clauses],
        "clause_textes": [(c.get("texte") or "").strip() for c in sortie_b.clauses],
        "connecteurs": list(sortie_b.connecteurs or []),
        "source": sortie_b.source,
    }


def analyze_tok_text(tok_engine, text: str, lang: str = "fr") -> dict:
    """Texte brut → CIR dict (segmentation B + runtime tok pur).

    ``tok_engine`` : ``TokForwardEngine`` avec vocabulaire construit
    (gate capacitaire dans ``forward_from_spans``). Texte vide →
    CIR vide (pas d'appel B inutile).
    """
    segments = parse_text(text, lang=lang)
    if not segments["clause_spans"]:
        from ..pipeline.ir_emitter import emit

        return emit(text, [], [], [], [], [])
    return tok_engine.forward_from_spans(
        text,
        segments["ids"],
        segments["offsets"],
        segments["clause_spans"],
        segments["clause_textes"],
        connecteurs=segments["connecteurs"],
        source_b=segments["source"],
    )
