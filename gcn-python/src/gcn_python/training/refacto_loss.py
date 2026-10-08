# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Loss refacto — CE nœuds/arêtes/actes + masques + silver + fallback (étape 5).

Références : ``plan-remodelage-tout-appris.md`` (étape 5, [R8] masques N≥30,
[R9] silver 0.0 par défaut → 0.3 si prouvé, [T1] loss masquée en fallback).

- Classes rares (< 30 exemples, [R8]) : masquées à l'entraînement
  (``build_class_mask``) ; procédure de calibration gelée en dessous
  (``require_min_counts`` → ``GelError``).
- Silver (``source=silver-B``) : poids 0.0 par défaut (n'apprend rien tant que
  sa qualité n'est pas prouvée, [R9]) ; 0.3 seulement sur preuve (erreur ≤ 20 %).
- Fallback (``source=fallback``, [T1]) : loss masquée (contribution nulle).
- Legacy (BCE opt) exclu du périmètre (choix acté : sans BCE).
"""

from __future__ import annotations

import numpy as np

#: Poids silver par défaut : n'apprend rien ([R9]).
SILVER_WEIGHT_DEFAULT = 0.0
#: Poids silver après preuve (erreur mesurée ≤ 20 %, [R9]).
SILVER_WEIGHT_PROVEN = 0.3
#: Seuil d'effectif par classe ([R8]/[R10]).
MIN_CLASS_COUNT = 30


class GelError(ValueError):
    """Gate N≥30/classe non franchie : procédure gelée (pas de seuils,
    pas de calibration — [R8]/[R10]). Données, pas code : gold-3000 requis."""


def build_class_mask(counts, min_n: int = MIN_CLASS_COUNT) -> np.ndarray:
    """Masque booléen des classes gardées (effectif ≥ ``min_n``)."""
    arr = np.asarray(counts, dtype=np.int64).ravel()
    if (arr < 0).any():
        raise ValueError(f"effectifs négatifs : {arr}.")
    return arr >= min_n


def require_min_counts(counts, min_n: int = MIN_CLASS_COUNT, nom: str = "classes") -> None:
    """Gate ([R8]/[R10]) : lève ``GelError`` si une classe a < ``min_n`` exemples."""
    arr = np.asarray(counts, dtype=np.int64).ravel()
    if (arr < min_n).any():
        rares = int((arr < min_n).sum())
        raise GelError(
            f"{nom} : {rares} classe(s) sous {min_n} exemples "
            f"(min={int(arr.min())}) — procédure gelée."
        )


def ce_loss(
    logits: np.ndarray,
    gold: np.ndarray,
    *,
    actives: np.ndarray | None = None,
    class_mask: np.ndarray | None = None,
    weight: float = 1.0,
) -> float:
    """Cross-entropie stable moyennée sur les lignes gardées (0.0 si aucune).

    ``actives`` : bool par ligne (False = fallback/exclu) ; ``class_mask`` :
    bool par classe (False = rare masquée) ; ``weight`` : poids (silver).
    """
    mat = np.asarray(logits, dtype=np.float64)
    refs = np.asarray(gold, dtype=np.int64).ravel()
    if mat.ndim != 2 or mat.shape[0] != refs.shape[0]:
        raise ValueError(f"logits {mat.shape} vs gold {refs.shape}.")
    n_classes = mat.shape[1]
    if ((refs < 0) | (refs >= n_classes)).any():
        raise ValueError("labels gold hors vocabulaire.")
    garde = np.ones(mat.shape[0], dtype=bool)
    if actives is not None:
        masque = np.asarray(actives, dtype=bool).ravel()
        if masque.shape[0] != garde.shape[0]:
            raise ValueError("actives de mauvaise longueur.")
        garde &= masque
    if class_mask is not None:
        masque = np.asarray(class_mask, dtype=bool).ravel()
        if masque.shape[0] != n_classes:
            raise ValueError("class_mask de mauvaise taille.")
        garde &= masque[refs]
    if not garde.any():
        return 0.0
    m = mat[garde] - mat[garde].max(axis=1, keepdims=True)
    log_prob = m - np.log(np.exp(m).sum(axis=1, keepdims=True))
    return float(-log_prob[np.arange(m.shape[0]), refs[garde]].mean() * weight)


def refacto_loss(
    node_logits: np.ndarray,
    node_gold: np.ndarray,
    edge_logits: np.ndarray,
    edge_gold: np.ndarray,
    act_logits: np.ndarray | None = None,
    act_gold: np.ndarray | None = None,
    *,
    node_class_mask=None,
    edge_class_mask=None,
    act_class_mask=None,
    sample_weight: float = 1.0,
    is_fallback: bool = False,
) -> dict:
    """Loss totale refacto : ``node + edge (+ act)`` avec masques [R8]/[R9]/[T1].

    Appelée par phrase (comme le legacy) : ``sample_weight`` 1.0 (gold) ;
    silver non prouvé → ``SILVER_WEIGHT_DEFAULT`` (0.0, [R9]) ;
    ``is_fallback=True`` (``source=fallback``) → loss nulle (masquée, [T1]). Les trois têtes partagent les mêmes gardes.
    Retourne ``{"total", "node", "edge", "act"}`` (``act`` = 0.0 si absent).
    """
    if (act_logits is None) != (act_gold is None):
        raise ValueError("act_logits et act_gold vont par paire.")
    if np.asarray(edge_gold).ravel().shape[0] != np.asarray(edge_logits).shape[0]:
        raise ValueError("logits/gold arêtes désalignés.")
    if is_fallback:
        return {"total": 0.0, "node": 0.0, "edge": 0.0, "act": 0.0}
    node = ce_loss(node_logits, node_gold,
                   class_mask=node_class_mask, weight=sample_weight)
    edge = ce_loss(edge_logits, edge_gold,
                   class_mask=edge_class_mask, weight=sample_weight)
    act = 0.0
    if act_logits is not None and act_gold is not None:
        act = ce_loss(act_logits, act_gold,
                      class_mask=act_class_mask, weight=sample_weight)
    return {"total": node + edge + act, "node": node, "edge": edge, "act": act}


__all__ = [
    "MIN_CLASS_COUNT",
    "SILVER_WEIGHT_DEFAULT",
    "SILVER_WEIGHT_PROVEN",
    "GelError",
    "build_class_mask",
    "ce_loss",
    "refacto_loss",
    "require_min_counts",
]
