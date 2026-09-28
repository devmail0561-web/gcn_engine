# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Éq.6 ETUDE — NLU routing : question → commande DSL Pearl.

Langue-agnostique : aucun lemme, aucune config externe.
Priorité 1 : tête d'intention apprise (_cached_intent_logits du pipeline).
Priorité 2 : heuristique surface (SentenceProfile couche 1 ou ?, 2 concepts).
"""
from __future__ import annotations

import numpy as np


def _is_interrogative_surface(text: str) -> bool:
    return text.strip().endswith("?")


def _content_words(text: str) -> list[str]:
    return [
        w.strip("?.,!()«»:;\"'").lower()
        for w in text.split()
        if len(w.strip("?.,!()«»:;\"'")) > 2
    ]


_ONE_CONCEPT_INTENTS = frozenset({
    "explain", "effects", "abduct", "counterfactual",
    "spof", "centrality", "summarize",
    "density", "coverage", "reliability", "verbalize",
    "zoom_in", "zoom_out", "aggregate",
})
_TWO_CONCEPT_INTENTS = frozenset({
    "chain", "chain_t", "before", "delay", "diff", "analogy",
})


def _build_dsl(intent: str, concepts: list[str]) -> str | None:
    if not concepts:
        return None
    if intent in _TWO_CONCEPT_INTENTS and len(concepts) >= 2:
        return f"{intent}: {concepts[0]} {concepts[-1]}"
    return f"{intent}: {concepts[-1]}"


def nlu_route(
    text: str,
    tokens: list[dict] | None = None,
    pipeline=None,
) -> str | None:
    """Éq.6 — mappe une question vers une commande DSL Pearl.

    Priorité 1 — tête intention apprise :
        Utilise _cached_intent_logits si le pipeline a déjà exécuté forward().
    Priorité 2 — heuristique surface :
        SentenceProfile couche 1 UD (si tokens fournis) ou détection par "?".
    Returns None pour les phrases déclaratives.
    """
    # Priorité 1 — tête apprise
    if pipeline is not None:
        cached = getattr(pipeline, '_cached_intent_logits', None)
        if cached is not None and len(cached) > 0:
            from ..constants import INTENT_TYPES
            idx = int(np.argmax(cached[0]))
            intent = INTENT_TYPES[idx]
            if intent != "none":
                return _build_dsl(intent, _content_words(text))
            return None

    # Priorité 2 — heuristique surface
    is_question = _is_interrogative_surface(text)

    if tokens is not None:
        try:
            from ..layer1.sentence_type import classify, SentenceType
            profile = classify(tokens)
            is_question = profile.sentence_type == SentenceType.INTERROGATIVE
        except Exception:
            pass

    if not is_question:
        return None

    words = _content_words(text)
    if not words:
        return None

    if len(words) >= 2 and words[0] != words[-1]:
        return f"chain: {words[0]} {words[-1]}"
    return f"explain: {words[-1]}"


def semantic_resolve_concept(
    query_vecs: "np.ndarray",
    graph_index: "list[tuple[np.ndarray, str]]",
    threshold: float = 0.3,
) -> str | None:
    """Résolution sémantique : vecteur(s) de la requête → label du nœud le plus proche.

    query_vecs : enriched_vecs de la requête (N_clauses × d_eff), produits par forward().
    graph_index : [(enriched_vec, node_label), ...] depuis load_graph_vecs_index().
    threshold   : score cosine minimum pour accepter un match.

    Retourne le node_label du meilleur nœud ou None si sous le seuil.
    Dans le même espace vectoriel (même modèle) — pas d'embedding externe requis.
    """
    if query_vecs is None or len(query_vecs) == 0 or not graph_index:
        return None

    best_label: str | None = None
    best_score: float = threshold

    for q_vec in query_vecs:
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm == 0.0:
            continue
        for g_vec, label in graph_index:
            g_norm = float(np.linalg.norm(g_vec))
            if g_norm == 0.0:
                continue
            score = float(np.dot(q_vec, g_vec) / (q_norm * g_norm))
            if score > best_score:
                best_score = score
                best_label = label

    return best_label
