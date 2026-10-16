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

import re as _re

from ..layer0 import engine_b
from ..layer0.tokenizer import split as tok_split
from ..layer2.act import detect_act

#: Clause dégénérée (gate d'entrée Ph5, 2026-10-16) : ≤2 tokens ou ponctuation
#: pure (aucun alphanumérique). Forme seule, jamais de lexique : une clause
#: ≤2 tokens ne porte pas de proposition donc aucune arête la touchant ne peut
#: être un gold (vérifié : 71/71 abstentions non-gold sur Ph5, 0 gold perdu,
#: 13/14 surconfiants interceptés). Le train (gold) n'est jamais concerné —
#: seule l'entrée inférence est filtrée.
_RE_ALNUM = _re.compile(r"[A-Za-zÀ-ÿ0-9]")


def clause_degeneree(texte_clause: str) -> str | None:
    """Motif de dégénérescence d'une clause, ou None si valide."""
    t = (texte_clause or "").strip()
    if not t:
        return "vide"
    _ids, _offs, _toks = tok_split(t)
    if len(_toks) <= 2:
        return f"court({len(_toks)}tok)"
    if not _RE_ALNUM.search(t):
        return "ponctuation-pure"
    return None


def parse_text(text: str, lang: str = "fr") -> dict:
    """Texte brut → segments tok prêts pour le runtime.

    Retourne ``{"ids", "offsets", "clause_spans", "clause_textes",
    "connecteurs", "source", "intent", "clauses_rejetees"}`` — ``source`` =
    ``silver-B|fallback`` (B) ; ``intent`` = règles v0 ``detect_act``
    (message-level, T4 — la tête apprise prendra le relais ; supervision
    intent absente du gold à ce pas) ; ``clauses_rejetees`` = [(texte, motif)]
    dégénérées écartées AVANT forward (gate Ph5 — abstention, pas silence :
    le rejet est tracé). Texte vide/blanc → segments vides + intent par défaut
    (jamais d'exception — miroir du legacy ``parse_lattice`` qui ne retourne
    jamais d'erreur). ``lang`` hors budget → ``ValueError`` de B (pas de silencieux).
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
            "clauses_rejetees": [],
        }
    sortie_b = engine_b.analyze(text, lang=lang)
    ids, offsets, _tokens = tok_split(text)
    spans, textes, rejetees = [], [], []
    for c in sortie_b.clauses:
        txt = (c.get("texte") or "").strip()
        motif = clause_degeneree(txt)
        if motif is not None:
            rejetees.append((txt, motif))
        else:
            spans.append(list(c["span"]))
            textes.append(txt)
    return {
        "ids": ids,
        "offsets": offsets,
        "clause_spans": spans,
        "clause_textes": textes,
        "connecteurs": list(sortie_b.connecteurs or []),
        "source": sortie_b.source,
        "intent": intent,
        "clauses_rejetees": rejetees,
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
