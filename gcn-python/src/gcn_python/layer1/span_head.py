# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Têtes ancres/bornes (R3) : segmentation apprise au niveau token.

Remplace à terme la segmentation B (phrases par langue) par des bornes
prédites depuis les représentations token : chaque noeud = span [s,e)
arbitraire, pas forcément une clause B. Zéro lexique, zéro règle langue.

v0 : têtes linéaires start/end + décodage glouton non-recouvrant + F1 spans.
Supervision : clause_spans gold (même signal que les noeuds actuels — la
granularité sub-clause viendra du gold-3000, pas d'une règle).
"""
from __future__ import annotations

import numpy as np


class SpanHead:
    """2 têtes linéaires (start, end) sur reps token (NumPy, CPU)."""

    def __init__(self, d_model: int, seed: int = 42) -> None:
        if d_model < 1:
            raise ValueError(f"d_model doit être ≥ 1 (reçu {d_model!r}).")
        rng = np.random.default_rng(int(seed))
        s = float(np.sqrt(1.0 / d_model))
        self.w_s = rng.normal(0, s, (d_model,)).astype(np.float32)
        self.b_s = np.float32(0.0)
        self.w_e = rng.normal(0, s, (d_model,)).astype(np.float32)
        self.b_e = np.float32(0.0)

    def forward(self, reps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,d) → (scores_start(N,), scores_end(N,))."""
        R = np.asarray(reps, dtype=np.float32)
        if R.ndim != 2:
            raise ValueError(f"reps (N,d) requises, reçu {R.shape}.")
        return R @ self.w_s + self.b_s, R @ self.w_e + self.b_e

    def loss_and_update(
        self,
        reps: np.ndarray,
        start_gold: np.ndarray,
        end_gold: np.ndarray,
        lr: float,
        pos_weight: float | None = None,
    ) -> float:
        """Un pas SGD sigmoïde binaire pondéré (start + end). Retourne la loss.

        Sans ``pos_weight`` : ratio négatifs/positifs du batch (bornes
        déséquilibrées ~2/20 — sans pondération la tête s'effondre à zéro).
        Pur comptage, zéro lexique.
        """
        if not (np.isfinite(lr) and lr > 0):
            raise ValueError(f"lr doit être > 0 et fini (reçu {lr!r}).")
        R = np.asarray(reps, dtype=np.float32)
        ys = np.asarray(start_gold, dtype=np.float32).ravel()
        ye = np.asarray(end_gold, dtype=np.float32).ravel()
        n = R.shape[0]
        if n == 0 or ys.shape[0] != n or ye.shape[0] != n:
            return 0.0
        total = 0.0
        for w, b_attr, y in ((self.w_s, "b_s", ys), (self.w_e, "b_e", ye)):
            logits = R @ w + getattr(self, b_attr)
            probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
            n_pos = float(y.sum())
            pw = float(pos_weight) if pos_weight else (n - n_pos) / max(1.0, n_pos)
            eps = 1e-12
            total += float(
                -(pw * y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)).mean()
            )
            err = (probs - y).astype(np.float32)
            err = np.where(y > 0.5, err * np.float32(pw), err) / n
            grad_w = R.T @ err
            grad_b = np.float32(err.sum())
            if np.all(np.isfinite(grad_w)) and np.isfinite(grad_b):
                w -= np.float32(lr) * grad_w
                setattr(self, b_attr, np.float32(getattr(self, b_attr) - lr * grad_b))
        return float(total)

    def parameters(self) -> list[np.ndarray]:
        return [self.w_s, np.asarray(self.b_s), self.w_e, np.asarray(self.b_e)]

    def grad_input(
        self,
        reps: np.ndarray,
        start_gold: np.ndarray,
        end_gold: np.ndarray,
        pos_weight: float | None = None,
    ) -> np.ndarray:
        """dL/dReps (N,d) sans mise à jour — reflux vers l'encodeur (O1)."""
        R = np.asarray(reps, dtype=np.float32)
        ys = np.asarray(start_gold, dtype=np.float32).ravel()
        ye = np.asarray(end_gold, dtype=np.float32).ravel()
        n = R.shape[0]
        dR = np.zeros_like(R)
        for w, y in ((self.w_s, ys), (self.w_e, ye)):
            logits = R @ w
            probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
            n_pos = float(y.sum())
            pw = float(pos_weight) if pos_weight else (n - n_pos) / max(1.0, n_pos)
            err = (probs - y).astype(np.float32)
            err = np.where(y > 0.5, err * np.float32(pw), err) / n
            dR += np.outer(err, w).astype(np.float32)
        return dR


def decode_spans(
    start_probs: np.ndarray,
    end_probs: np.ndarray,
    offsets: list,
    threshold: float = 0.5,
) -> list[list[int]]:
    """Glouton non-recouvrant : ouvre aux starts ≥ seuil, ferme au 1er end.

    Retourne des spans chars [s,e). Zéro seuil caché : threshold en param.
    """
    sp = np.asarray(start_probs, dtype=float).ravel()
    ep = np.asarray(end_probs, dtype=float).ravel()
    n = len(offsets)
    if sp.shape[0] != n or ep.shape[0] != n or n == 0:
        return []
    spans: list[list[int]] = []
    ouvert: int | None = None
    for i in range(n):
        if ouvert is None:
            if sp[i] >= threshold:
                ouvert = i
        elif ep[i] >= threshold or i == n - 1:
            s = int(offsets[ouvert][0])
            e = int(offsets[i][1])
            if e > s:
                spans.append([s, e])
            ouvert = None
    return spans


def span_f1(pred: list[list[int]], gold: list[list[int]]) -> float:
    """F1 exacte sur spans [s,e) (égalité stricte, pas de recouvrement partiel)."""
    set_p = {(int(a), int(b)) for a, b in pred}
    set_g = {(int(a), int(b)) for a, b in gold}
    if not set_p and not set_g:
        return 1.0
    if not set_p or not set_g:
        return 0.0
    inter = len(set_p & set_g)
    return 2 * inter / (len(set_p) + len(set_g))


__all__ = ["SpanHead", "decode_spans", "span_f1"]
