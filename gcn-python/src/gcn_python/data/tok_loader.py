# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Chargeur générique de paires gold tok (train, éval, audit — même contrat).

Extrait de ``training/train_tok.py:charger_split`` (délégation, parité
testée) : UNE seule implémentation de l'alignement gold → tok, réutilisée
par l'entraînement, la future éval découplée et les audits. Règles
inchangées : rejet explicite (``SpanError`` compté), paires hors fenêtre
ou arrières exclues mais comptées, types/relations inconnus refusés.
"""

from __future__ import annotations

import json

from ..constants import NODE_TYPES, RELATION_TYPES
from ..layer0.tokenizer import split as tok_split
from ..pipeline.forward_ids import WINDOW_K
from .gold_spans import SpanError, reconstruire_offsets, span_clause


def iter_phrases(fichier: str):
    """Générateur ``(ident, phrase)`` depuis un split JSON gold (IO seule)."""
    with open(fichier, encoding="utf-8") as fh:
        doc = json.load(fh)
    for k, s in enumerate(doc["document"]["sentences"], 1):
        yield s.get("id", f"#{k}"), s


def convertir_phrase(s: dict, *, window_k: int = WINDOW_K) -> dict:
    """Phrase gold → sample tok (lève ``SpanError`` : rejet explicite).

    Sample : ``{id, text, ids, offsets, clause_spans, clause_textes,
    node_gold: [idx], edge_gold: {(i,j): rel_idx}, hors_fenetre, arrieres}``.
    """
    if window_k < 1:
        raise ValueError(f"window_k doit être ≥ 1 (reçu {window_k!r}).")
    offs = reconstruire_offsets(s["text"], s["tokens"])
    ntoks = len(s["tokens"])
    rangs = {n["id"]: k for k, n in enumerate(s["cir"]["nodes"])}
    cspans, ctextes, ngold = [], [], []
    for n in s["cir"]["nodes"]:
        a, b = n["token_span"]
        cspans.append(list(span_clause(offs, a, b, ntoks)))
        ctextes.append(s["text"][cspans[-1][0] : cspans[-1][1]])
        try:
            ngold.append(NODE_TYPES.index(n["type"]))
        except ValueError:
            raise SpanError(f"type de nœud inconnu : {n['type']!r}.") from None
    egold, hors, arrieres = {}, 0, 0
    for e in s["cir"]["edges"]:
        srcs = e.get("sources") or ([e["source"]] if "source" in e else [])
        for src in srcs:
            i, j = rangs[src], rangs[e["target"]]
            try:
                r = RELATION_TYPES.index(e["relation"])
            except ValueError:
                raise SpanError(f"relation inconnue : {e['relation']!r}.") from None
            if i >= j:
                arrieres += 1
                continue
            if j - i > window_k:
                hors += 1
                continue
            egold[(i, j)] = r
    ids, offsets, _ = tok_split(s["text"])
    return {
        "id": s.get("id", "?"),
        "text": s["text"],
        "ids": ids,
        "offsets": offsets,
        "clause_spans": cspans,
        "clause_textes": ctextes,
        "node_gold": ngold,
        "edge_gold": egold,
        "hors_fenetre": hors,
        "arrieres": arrieres,
    }


def charger_split(fichier: str, *, window_k: int = WINDOW_K) -> tuple[list, dict]:
    """Split JSON gold → ``(samples, stats)`` (rejets + exclusions comptés)."""
    samples = []
    stats: dict = {"phrases": 0, "rejetees": [], "hors_fenetre": 0, "arrieres": 0}
    for ident, s in iter_phrases(fichier):
        stats["phrases"] += 1
        try:
            echantillon = convertir_phrase(s, window_k=window_k)
        except SpanError as exc:
            stats["rejetees"].append((ident, str(exc)))
            continue
        stats["hors_fenetre"] += echantillon.pop("hors_fenetre")
        stats["arrieres"] += echantillon.pop("arrieres")
        samples.append(echantillon)
    return samples, stats


def couverture_vocab(est_dans_vocab, samples: list) -> tuple[int, int]:
    """Comptage OOV réutilisable : ``(oov, total)`` sur les ids des samples.

    ``est_dans_vocab`` : ``id -> bool`` (ex. ``lambda i: emb.row_index(i) is not None``).
    Reporté, jamais masqué (R5 : le taux est une mesure).
    """
    oov = sum(1 for s in samples for i in s["ids"] if not est_dans_vocab(i))
    total = sum(len(s["ids"]) for s in samples)
    return oov, total
