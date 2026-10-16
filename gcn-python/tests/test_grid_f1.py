# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests grid F1 — balayage + tripwire référence + gel ([R8]/[R10])."""

from __future__ import annotations

import numpy as np
import pytest

from gcn_python.evaluation.grid_f1 import (
    REF_THRESHOLDS,
    check_gel_or_grid,
    f1_micro,
    predict_with_thresholds,
    run_grid,
)
from gcn_python.training.refacto_loss import GelError


def logits_nets(n=40, c=4, fort=6.0, seed=1):
    rng = np.random.default_rng(seed)
    gold = np.arange(n) % c
    mat = rng.normal(0, 0.5, (n, c))
    mat[np.arange(n), gold] = fort
    return mat, gold


def test_predict_seuils_et_abstention():
    mat, _ = logits_nets()
    predits = predict_with_thresholds(mat, 0.0, 0.5, 1.0)
    assert predits.shape == (40,) and (predits >= 0).all()
    stricts = predict_with_thresholds(mat, 0.9999, 0.99, 1.0)
    assert (stricts == -1).any(), "seuils durs → abstentions"
    with pytest.raises(ValueError, match="temperature"):
        predict_with_thresholds(mat, 0.0, 0.5, 0.0)
    with pytest.raises(ValueError, match=r"\(E, R\)"):
        predict_with_thresholds(np.zeros(5), 0.0, 0.5, 1.0)


def test_f1_micro():
    assert f1_micro([0, 1, 2], [0, 1, 2]) == 1.0
    assert f1_micro([0, 1, 2], [0, 0, 0]) == pytest.approx(1 / 3)
    assert f1_micro([0, 1], [0, -1]) == 0.5  # abstention = erreur
    assert f1_micro([], []) == 0.0
    with pytest.raises(ValueError, match="désalignées"):
        f1_micro([0], [0, 1])


def test_grid_trouve_et_evite_reference():
    mat, gold = logits_nets()
    best = run_grid(mat, gold)
    assert best["n_tries"] == 10 * 5 * 5
    assert 0.0 <= best["f1"] <= 1.0 and best["f1"] > 0.9
    assert (best["edge_threshold"], best["theta_ambiguity"], best["temperature"]) != REF_THRESHOLDS


def test_grid_tripwire_reference():
    # Grille réduite à la seule référence → tripwire (grille dégénérée).
    mat, gold = logits_nets()
    with pytest.raises(ValueError, match="legacy"):
        run_grid(mat, gold, edge_thresholds=(0.0,), thetas=(0.65,), temperatures=(1.0,))


def test_gate_gel():
    assert REF_THRESHOLDS == (0.0, 0.65, 1.0)
    check_gel_or_grid([30, 45, 100])
    with pytest.raises(GelError, match="gelée"):
        check_gel_or_grid([30, 12])
