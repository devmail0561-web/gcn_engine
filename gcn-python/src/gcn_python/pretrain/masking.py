# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Pré-entraînement brut sans labels ni lexique (O1, CPU-only).

Zéro liste de langue, même au train : seuls la ponctuation, l'ordre et le
masquage aléatoire fournissent le signal. Torch éventuel reste cantonné au
train offline, jamais au runtime (guard CI comme subprocess==0).
"""
from __future__ import annotations

import numpy as np


def span_mask(
    ids: list[int],
    mask_ratio: float = 0.15,
    span_len: int = 3,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Masque des spans contigus pour span-MLM.

    Retourne (ids_masques, positions_masquées). Le token masque = -1
    (hors vocab ids stables u64, jamais collision). Pur NumPy/std, CPU.
    """
    if not 0.0 < float(mask_ratio) < 1.0:
        raise ValueError(f"mask_ratio doit être dans ]0,1[ (reçu {mask_ratio!r}).")
    if int(span_len) < 1:
        raise ValueError(f"span_len doit être ≥ 1 (reçu {span_len!r}).")
    rng = np.random.default_rng(int(seed))
    n = len(ids)
    if n == 0:
        return [], []
    n_mask = max(1, round(n * float(mask_ratio)))
    starts = rng.choice(n, size=min(n_mask, n), replace=False)
    masked = set()
    for s in sorted(int(x) for x in starts):
        for k in range(int(span_len)):
            if len(masked) >= n_mask:
                break
            masked.add(min(n - 1, s + k))
        if len(masked) >= n_mask:
            break
    out = [(-1 if i in masked else v) for i, v in enumerate(ids)]
    return out, sorted(masked)


def permute_clauses(
    clause_spans: list[list[int]],
    seed: int = 42,
) -> tuple[list[list[int]], bool]:
    """Détection de permutation (O3) : mélange l'ordre des clauses.

    Retourne (spans_mélangés, est_permRompu). Signal de direction pour
    Sequence, sans lexique. Les cycles ne s'apprennent pas (Tarjan).
    """
    rng = np.random.default_rng(int(seed))
    order = list(range(len(clause_spans)))
    rng.shuffle(order)
    shuffled = [clause_spans[i] for i in order]
    return shuffled, order != list(range(len(clause_spans)))
