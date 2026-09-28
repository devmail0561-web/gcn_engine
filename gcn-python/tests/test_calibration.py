# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests calibration — T5-min + ECE (PHASE D §D.5)."""
import numpy as np
import pytest

from gcn_python.evaluation.calibration import (
    apply_isotonic_params,
    ece_score,
    optimize_temperature,
    softmax,
)


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

def test_softmax_sums_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-6)


def test_softmax_stable_large_values():
    logits = np.array([[1000.0, 1001.0, 999.0]])
    probs = softmax(logits)
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# ece_score
# ---------------------------------------------------------------------------

def test_ece_perfect_model_is_zero():
    # Modèle parfait : confiance 1.0, toujours correct
    probs = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    labels = np.array([1, 1, 0])
    assert ece_score(probs, labels) == pytest.approx(0.0, abs=1e-6)


def test_ece_always_wrong_is_high():
    # Confiance max sur mauvaise classe
    probs = np.array([[0.9, 0.1]] * 20)
    labels = np.array([1] * 20)
    ece = ece_score(probs, labels)
    assert ece > 0.5


def test_ece_rejects_bad_shapes():
    with pytest.raises(ValueError):
        ece_score(np.ones((3,)), np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        ece_score(np.ones((3, 2)), np.array([0, 1]))
    with pytest.raises(ValueError):
        ece_score(np.ones((0, 2)), np.array([]))


# ---------------------------------------------------------------------------
# optimize_temperature
# ---------------------------------------------------------------------------

def _make_logits_labels(n: int = 30, n_classes: int = 3, seed: int = 42):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, n_classes))
    labels = rng.integers(0, n_classes, size=n)
    return logits, labels


def test_optimize_temperature_returns_valid_range():
    logits, labels = _make_logits_labels()
    t, ece = optimize_temperature(logits, labels)
    assert 0.1 <= t <= 5.0
    assert 0.0 <= ece <= 1.0


def test_optimize_temperature_rejects_small_n():
    logits = np.ones((5, 2))
    labels = np.array([0, 1, 0, 1, 0])
    with pytest.raises(ValueError):
        optimize_temperature(logits, labels)


def test_optimize_temperature_rejects_invalid_grid():
    logits, labels = _make_logits_labels()
    with pytest.raises(ValueError):
        optimize_temperature(logits, labels, t_min=0.0, t_max=5.0)
    with pytest.raises(ValueError):
        optimize_temperature(logits, labels, t_min=3.0, t_max=1.0)


# ---------------------------------------------------------------------------
# apply_isotonic_params
# ---------------------------------------------------------------------------

def test_apply_isotonic_params_identity_when_keys_absent():
    raw = np.array([0.3, 0.7, 0.9])
    result = apply_isotonic_params(raw, {})
    np.testing.assert_array_equal(result, raw)


def test_apply_isotonic_params_identity_partial_keys():
    raw = np.array([0.3, 0.7])
    result = apply_isotonic_params(raw, {"calibrator_x": [0.0, 1.0]})
    np.testing.assert_array_equal(result, raw)


def test_apply_isotonic_params_interpolates():
    raw = np.array([0.0, 0.5, 1.0])
    ckpt = {"calibrator_x": [0.0, 1.0], "calibrator_y": [0.1, 0.9]}
    result = apply_isotonic_params(raw, ckpt)
    np.testing.assert_allclose(result, [0.1, 0.5, 0.9], atol=1e-6)


def test_apply_isotonic_params_clips_outside_range():
    raw = np.array([-0.5, 1.5])
    ckpt = {"calibrator_x": [0.0, 1.0], "calibrator_y": [0.2, 0.8]}
    result = apply_isotonic_params(raw, ckpt)
    np.testing.assert_allclose(result, [0.2, 0.8], atol=1e-6)
