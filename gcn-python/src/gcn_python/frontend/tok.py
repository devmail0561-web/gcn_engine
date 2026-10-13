# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Segmentation tok in-process (B + split) — frontend, pas runtime appris.

Seul endroit avec ``nlu/`` autorisé à appeler le moteur B : la garde
``test_runtime_appris_n_importe_jamais_moteur_b`` interdit
``pipeline/``, ``training/`` et ``engine.py`` d'importer ``engine_b`` /
``fallback_regex``. Ce module segmente (B : clauses + connecteurs + source)
et délègue le forward au runtime pur (``pipeline/forward_ids.py``) —
même découpe frontend→pipeline que le bridge legacy.
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
