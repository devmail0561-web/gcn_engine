# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Grid F1 des seuils — pas ECE ([R10], étape 5).

Références : ``plan-remodelage-tout-appris.md`` ([R8] gate N≥30, [R10] grid F1,
``assert != (0.0, 0.65, 1.0)``, gel si <30/classe), ``plan-remodelage-inference.md``
(seuils legacy → grid F1).

Paramètres : ``edge_threshold`` (confiance min pour garder une arête),
``theta_ambiguity`` (marge top-1 − top-2 sous laquelle on s'abstient —
définition opérationnelle v0 de l'ambiguïté), ``temperature`` (mise à
l'échelle des logits AVANT seuils). Référence legacy interdite :
``REF_THRESHOLDS = (0.0, 0.65, 1.0)`` — si le grid la rend, c'est un
tripwire (grille dégénérée ou scores plats) : ``ValueError``.

Gate ([R8]/[R10]) : effectifs < 30/classe → ``GelError`` (pas de seuils,
pas de calibration) — l'appelant gèle AVANT le grid.
Métrique : micro-F1 sur relations (abstention = erreur).
"""

from __future__ import annotations

import itertools

import numpy as np

from ..training.refacto_loss import require_min_counts

#: Triple legacy interdit (edge_threshold, theta_ambiguity, temperature).
REF_THRESHOLDS = (0.0, 0.65, 1.0)


def predict_with_thresholds(
    edge_logits: np.ndarray,
    edge_threshold: float,
    theta_ambiguity: float,
    temperature: float,
) -> np.ndarray:
    """Prédit par arête : argmax si ``max_prob ≥ edge_threshold`` ET marge
    top-1 − top-2 ≥ ``(1 - theta_ambiguity)``, sinon ``-1`` (abstention)."""
    mat = np.asarray(edge_logits, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"logits (E, R) attendus, reçus {mat.shape}.")
    if temperature <= 0:
        raise ValueError(f"temperature > 0 (reçue {temperature!r}).")
    m = mat / temperature
    m = m - m.max(axis=1, keepdims=True)
    e = np.exp(m)
    probs = e / e.sum(axis=1, keepdims=True)
    tri = np.sort(probs, axis=1)
    meilleur = tri[:, -1]
    marge = tri[:, -1] - (tri[:, -2] if mat.shape[1] > 1 else 0.0)
    garde = (meilleur >= edge_threshold) & (marge >= (1.0 - theta_ambiguity))
    return np.where(garde, np.argmax(probs, axis=1), -1)


def f1_micro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Micro-F1 multi-classe (abstention ``-1`` = erreur systématique)."""
    vrais = np.asarray(y_true, dtype=np.int64).ravel()
    predits = np.asarray(y_pred, dtype=np.int64).ravel()
    if vrais.shape != predits.shape:
        raise ValueError("tailles désalignées.")
    if vrais.size == 0:
        return 0.0
    return float((vrais == predits).mean())


def run_grid(
    edge_logits: np.ndarray,
    edge_gold: np.ndarray,
    *,
    edge_thresholds=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    thetas=(0.5, 0.6, 0.65, 0.7, 0.8),
    temperatures=(0.5, 0.75, 1.0, 1.5, 2.0),
) -> dict:
    """Balaye le produit cartésien, maximise le micro-F1, vérifie ≠ référence.

    Retourne ``{"edge_threshold", "theta_ambiguity", "temperature", "f1",
    "n_tries"}``. Lève ``ValueError`` si le meilleur = ``REF_THRESHOLDS``
    (tripwire : grille dégénérée).
    """
    refs = np.asarray(edge_gold, dtype=np.int64).ravel()
    meilleur = {"edge_threshold": None, "theta_ambiguity": None,
                "temperature": None, "f1": -1.0, "n_tries": 0}
    for et, th, tp in itertools.product(edge_thresholds, thetas, temperatures):
        predits = predict_with_thresholds(edge_logits, float(et), float(th), float(tp))
        score = f1_micro(refs, predits)
        meilleur["n_tries"] += 1
        if score > meilleur["f1"]:
            meilleur.update({"edge_threshold": float(et), "theta_ambiguity": float(th),
                             "temperature": float(tp), "f1": float(score)})
    triple = (meilleur["edge_threshold"], meilleur["theta_ambiguity"], meilleur["temperature"])
    if triple == REF_THRESHOLDS:
        raise ValueError(
            f"grid F1 retombe sur la référence legacy {REF_THRESHOLDS} — "
            "grille dégénérée ou scores plats (revoir la grille, pas les seuils)."
        )
    return meilleur


def check_gel_or_grid(counts, min_n: int = 30) -> None:
    """Gate ([R8]/[R10]) : lève ``GelError`` si une classe < ``min_n``."""
    require_min_counts(counts, min_n=min_n, nom="gold")


__all__ = [
    "REF_THRESHOLDS",
    "check_gel_or_grid",
    "f1_micro",
    "predict_with_thresholds",
    "run_grid",
]
