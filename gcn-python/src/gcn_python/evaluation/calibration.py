# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Calibration production — T5-min (temperature) + helpers ECE.

Référence plan : PHASE D §D.5 (T5-min par défaut, T5-isotonie
conditionnelle si ECE-temperature > 0.15 sur N_val >= 100).

Contenu :
- `softmax`, `ece_score` : briques pures NumPy (aucune dépendance sklearn).
- `optimize_temperature` : grid search sur ECE (T5-min).
- `apply_isotonic_params` : application des params nus `calibrator_x/y`
  via `np.interp`, avec fallback identité si clés absentes
  (checkpoint v3 pré-calibration ou v2 — jamais de KeyError).

Ordre pipeline (plan §C.5) : brut ML → D6 (rung×f×g) → isotonie (T5)
→ confidence_cal. T5 calibre les sorties D6, pas les logits bruts.
"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax stable ligne par ligne."""
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def ece_score(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error sur les confidences max.

    Args:
        probs: (N, C) probabilités post-softmax.
        labels: (N,) indices gold.
        n_bins: nombre de bins uniformes sur [0, 1].
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if probs.ndim != 2 or len(probs) != len(labels) or len(probs) == 0:
        raise ValueError(
            "ece_score : probs (N, C) et labels (N,) non vides requis, "
            f"reçu probs.shape={probs.shape}, labels.shape={labels.shape}."
        )
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    ece_val = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = (confs >= lo) & (confs < hi)
        if np.any(mask):
            ece_val += (mask.sum() / len(confs)) * abs(
                float(confs[mask].mean()) - float(correct[mask].mean())
            )
    return float(ece_val)


def optimize_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    t_min: float = 0.1,
    t_max: float = 5.0,
    n_steps: int = 50,
) -> tuple[float, float]:
    """Grid search de la température minimisant l'ECE (T5-min).

    Returns:
        (best_temperature, ece_at_best). best_temperature == 1.0 signifie
        « le modèle est déjà calibré, T5-isotonie inutile ».
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.ndim != 2 or len(logits) != len(labels) or len(logits) < 10:
        raise ValueError(
            "optimize_temperature : logits (N, C) et labels (N,) avec N >= 10 requis."
        )
    if not (t_min > 0 and t_max > t_min and n_steps >= 2):
        raise ValueError("optimize_temperature : grille invalide (t_min > 0, t_max > t_min).")
    best_t, best_ece = 1.0, float("inf")
    for t in np.linspace(t_min, t_max, n_steps):
        e = ece_score(softmax(logits / t), labels)
        if e < best_ece:
            best_ece = e
            best_t = float(t)
    return best_t, float(best_ece)


def apply_isotonic_params(
    confidence_raw: np.ndarray,
    checkpoint_data: dict,
) -> np.ndarray:
    """Applique la calibration isotonique stockée en params nus.

    `checkpoint_data` doit contenir `calibrator_x` / `calibrator_y`
    (seuils sklearn sérialisés en arrays NumPy — jamais de pickle,
    gate anti-RCE `guarded_np_load` préservé).
    Clés absentes → identité (pas de KeyError, cf. plan §D.5).
    """
    raw = np.asarray(confidence_raw, dtype=np.float64)
    if "calibrator_x" in checkpoint_data and "calibrator_y" in checkpoint_data:
        return np.interp(
            raw,
            np.asarray(checkpoint_data["calibrator_x"], dtype=np.float64),
            np.asarray(checkpoint_data["calibrator_y"], dtype=np.float64),
        ).astype(np.float64)
    return raw
