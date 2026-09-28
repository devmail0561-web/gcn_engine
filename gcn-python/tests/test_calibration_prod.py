# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Phase D — T5-min production : optimize_temperature (plan §D.5).

Le helper de grid search existait en copie locale dans test_calibration.py ;
ce fichier verrouille la version production
(gcn_python.evaluation.calibration) : réduction ECE + gate ECE < 0.15,
fallback identité des params nus.
"""
from __future__ import annotations

import numpy as np

from gcn_python.evaluation.calibration import (
    apply_isotonic_params,
    ece_score,
    optimize_temperature,
    softmax,
)


def test_optimize_temperature_returns_valid_range():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 1, (120, 4))
    labels = rng.integers(0, 4, 120)
    best_t, best_ece = optimize_temperature(logits, labels)
    assert 0.1 <= best_t <= 5.0
    assert 0.0 <= best_ece <= 1.0


def test_optimize_temperature_reduces_ece_overconfident():
    rng = np.random.default_rng(1)
    N, C = 200, 5
    labels = rng.integers(0, C, N)
    logits = np.zeros((N, C))
    logits[np.arange(N), labels] = 5.0  # sur-confiant
    ece_before = ece_score(softmax(logits), labels)
    best_t, ece_after = optimize_temperature(logits, labels)
    assert ece_after <= ece_before


def test_t5_gate_ece_below_015_well_separated():
    rng = np.random.default_rng(42)
    N, C = 150, 3
    labels = rng.integers(0, C, N)
    logits = np.zeros((N, C))
    logits[np.arange(N), labels] = 2.0
    _, ece = optimize_temperature(logits, labels)
    assert ece < 0.15


def test_apply_isotonic_params_identity_without_keys():
    raw = np.array([0.4, 0.7, 0.9])
    out = apply_isotonic_params(raw, {"encoder_0": np.zeros(3)})
    np.testing.assert_array_equal(out, raw)


def test_apply_isotonic_params_interp_with_keys():
    data = {"calibrator_x": np.array([0.0, 0.5, 1.0]),
            "calibrator_y": np.array([0.0, 0.6, 1.0])}
    out = apply_isotonic_params(np.array([0.25]), data)
    assert abs(float(out[0]) - 0.3) < 1e-9


def test_optimize_temperature_rejects_small_n():
    with np.errstate(all="ignore"):
        try:
            optimize_temperature(np.zeros((5, 3)), np.zeros(5, dtype=int))
        except ValueError:
            return
    raise AssertionError("N < 10 devrait lever ValueError")
