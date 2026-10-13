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
from ..layer2.act import detect_act


def parse_text(text: str, lang: str = "fr") -> dict:
    """Texte brut → segments tok prêts pour le runtime.

    Retourne ``{"ids", "offsets", "clause_spans", "clause_textes",
    "connecteurs", "source", "intent"}`` — ``source`` = ``silver-B|fallback`` (B) ;
    ``intent`` = règles v0 ``detect_act`` (message-level, T4 — la tête apprise
    prendra le relais ; supervision intent absente du gold à ce pas).
    Texte vide/blanc → segments vides + intent par défaut (jamais d'exception —
    miroir du legacy ``parse_lattice`` qui ne retourne jamais d'erreur).
    ``lang`` hors budget → ``ValueError`` de B (pas de silencieux).
    """
    if not isinstance(text, str):
        raise TypeError(f"text doit être str, reçu {type(text).__name__}")
    intent = detect_act(text)
    if not text.strip():
        return {
            "ids": [],
            "offsets": [],
            "clause_spans": [],
            "clause_textes": [],
            "connecteurs": [],
            "source": "vide",
            "intent": intent,
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
        "intent": intent,
    }


def analyze_tok_text(tok_engine, text: str, lang: str = "fr") -> dict:
    """Texte brut → CIR dict (segmentation socle + runtime tok pur + cache).

    ``tok_engine`` : ``TokForwardEngine`` avec vocabulaire construit
    (gate capacitaire dans ``forward_from_spans``). Texte vide →
    CIR vide (pas d'appel socle inutile). Répétition → hit cache M1a
    (zéro recalcul : ni socle, ni forward — copies indépendantes).
    Compromis assumé : les hits ne rejouent pas le socle, donc les
    compteurs B (fallback_stats) ne voient que les textes uniques —
    comme le cache M1a legacy qui saute le parse.
    """
    import copy as _copy

    from ..param_epoch import current as _epoch
    from ..pipeline.forward_ids import TOK_CACHE_MAX

    if not isinstance(text, str):
        raise TypeError(f"text doit être str, reçu {type(text).__name__}")
    if not text.strip():
        from ..pipeline.ir_emitter import emit

        return emit(text, [], [], [], [], [], intent=detect_act(text))
    cle = (_epoch(), text, lang)
    present = tok_engine._tok_cache.get(cle)
    if present is not None:
        tok_engine._tok_hits += 1
        return _copy.deepcopy(present)
    segments = parse_text(text, lang=lang)
    if not segments["clause_spans"]:
        from ..pipeline.ir_emitter import emit

        return emit(text, [], [], [], [], [], intent=segments["intent"])
    cir = tok_engine.forward_from_spans(
        text,
        segments["ids"],
        segments["offsets"],
        segments["clause_spans"],
        segments["clause_textes"],
        connecteurs=segments["connecteurs"],
        source_b=segments["source"],
        intent=segments["intent"],
    )
    tok_engine._tok_misses += 1
    if len(tok_engine._tok_cache) >= TOK_CACHE_MAX:
        tok_engine._tok_cache.pop(next(iter(tok_engine._tok_cache)))
    tok_engine._tok_cache[cle] = _copy.deepcopy(cir)
    return cir
