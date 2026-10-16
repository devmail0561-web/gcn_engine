# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests loss refacto — CE + masques + silver + fallback + gel ([R8]/[R9]/[T1])."""

from __future__ import annotations

import numpy as np
import pytest

from gcn_python.training.refacto_loss import (
    MIN_CLASS_COUNT,
    SILVER_WEIGHT_DEFAULT,
    SILVER_WEIGHT_PROVEN,
    GelError,
    build_class_mask,
    ce_loss,
    refacto_loss,
    require_min_counts,
)


def logits_parfaits(n=6, c=4, fort=10.0, seed=0):
    rng = np.random.default_rng(seed)
    gold = np.arange(n) % c
    mat = rng.normal(0, 0.1, (n, c)).astype(np.float64)
    mat[np.arange(n), gold] = fort
    return mat, gold


def test_parfaits_presque_zero():
    mat, gold = logits_parfaits()
    assert ce_loss(mat, gold) < 1e-3


def test_actives_exclut_fallback():
    mat, gold = logits_parfaits(n=4, c=3)
    plein = ce_loss(mat, gold)
    partiel = ce_loss(mat, gold, actives=np.array([True, True, False, False]))
    assert 0.0 <= partiel <= plein + 1e-9
    assert ce_loss(mat, gold, actives=np.zeros(4, dtype=bool)) == 0.0


def test_class_mask_masque_rares():
    mat, gold = logits_parfaits(n=6, c=3)  # classes 0,1,2 vues
    masque = np.array([True, True, False])
    assert ce_loss(mat, gold, class_mask=masque) >= 0.0
    with pytest.raises(ValueError, match="class_mask"):
        ce_loss(mat, gold, class_mask=np.array([True, True]))


def test_weight_silver():
    mat, gold = logits_parfaits()
    plein = ce_loss(mat, gold, weight=1.0)
    assert ce_loss(mat, gold, weight=SILVER_WEIGHT_DEFAULT) == 0.0
    assert ce_loss(mat, gold, weight=SILVER_WEIGHT_PROVEN) == pytest.approx(plein * 0.3)
    with pytest.raises(ValueError, match=r"désalignés|hors vocabulaire|longueur|taille"):
        ce_loss(mat, np.array([0, 99, 1, 2, 3, 0]))


def test_masque_classes_et_gate():
    assert MIN_CLASS_COUNT == 30
    assert build_class_mask([30, 29, 100]).tolist() == [True, False, True]
    require_min_counts([30, 45, 100])
    with pytest.raises(GelError, match="gelée"):
        require_min_counts([30, 12, 100])
    with pytest.raises(ValueError, match="négatifs"):
        build_class_mask([3, -1])


def test_refacto_loss_total_et_fallback():
    mat_n, gold_n = logits_parfaits(n=4, c=8)
    mat_e, gold_e = logits_parfaits(n=3, c=19)
    mat_a, gold_a = logits_parfaits(n=2, c=3)
    r = refacto_loss(mat_n, gold_n, mat_e, gold_e, mat_a, gold_a)
    assert set(r) == {"total", "node", "edge", "act"}
    assert r["total"] == pytest.approx(r["node"] + r["edge"] + r["act"])
    assert r["act"] >= 0.0
    nul = refacto_loss(mat_n, gold_n, mat_e, gold_e, is_fallback=True)
    assert nul == {"total": 0.0, "node": 0.0, "edge": 0.0, "act": 0.0}
    sans_act = refacto_loss(mat_n, gold_n, mat_e, gold_e)
    assert sans_act["act"] == 0.0
    with pytest.raises(ValueError, match="paire"):
        refacto_loss(mat_n, gold_n, mat_e, gold_e, mat_a, None)
    with pytest.raises(ValueError, match="désalignés"):
        refacto_loss(mat_n, gold_n, mat_e, np.array([0, 1]))
