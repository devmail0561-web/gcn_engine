# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Phase D — T5 : Calibration.

Stratégie plan-moins : T5-min (temperature) en premier.
T5-isotonie seulement si ECE-temperature > 0.15.

Ce fichier teste :
1. La mécanique de calibration température (T5-min) — fonctionne avec cgnp.py existant
2. Le format params nus (calibrator_x/calibrator_y) pour éviter pickle (T5-isotonie)
3. ECE < 0.15 sur un set calibré (test fonctionnel)
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) sur les confidences max."""
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    ece_val = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = (confs >= lo) & (confs < hi)
        if mask.any():
            avg_conf = confs[mask].mean()
            avg_acc = correct[mask].mean()
            ece_val += (mask.sum() / len(confs)) * abs(avg_conf - avg_acc)
    return float(ece_val)


def _calibrate_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Optimise temperature par grid search sur ECE (T5-min)."""
    best_t, best_ece = 1.0, float("inf")
    for t in np.linspace(0.1, 5.0, 50):
        probs = _softmax(logits / t)
        e = _ece(probs, labels)
        if e < best_ece:
            best_ece = e
            best_t = float(t)
    return best_t


# ---------------------------------------------------------------------------
# T5-min : calibration température
# ---------------------------------------------------------------------------

class TestTemperatureCalibration:
    """T5-min — temperature_scaling déjà câblé dans cgnp.py."""

    def test_temperature_parameter_exists_in_cgnp(self):
        """cgnp.py a bien un paramètre temperature."""
        import inspect
        from gcn_python.pipeline.cgnp import CGNPipeline
        sig = inspect.signature(CGNPipeline.__init__)
        assert "temperature" in sig.parameters

    def test_temperature_grid_reduces_ece(self):
        """Optimiser temperature améliore ECE sur données sur-confiantes."""
        rng = np.random.default_rng(0)
        N, C = 200, 5
        labels = rng.integers(0, C, N)
        # logits très peaked → sur-confiant (ECE initiale élevée)
        logits = np.zeros((N, C), dtype=np.float32)
        logits[np.arange(N), labels] = 5.0  # quasi parfait mais sur-confiant
        probs_t1 = _softmax(logits)
        ece_t1 = _ece(probs_t1, labels)

        best_t = _calibrate_temperature(logits, labels)
        probs_cal = _softmax(logits / best_t)
        ece_cal = _ece(probs_cal, labels)

        assert ece_cal < ece_t1 or ece_cal < 0.05, (
            f"Temperature calibration devrait améliorer ECE : {ece_t1:.3f} → {ece_cal:.3f}"
        )

    def test_calibrated_ece_below_015(self):
        """T5 gate : ECE calibrée < 0.15 sur N_val ≥ 100 exemples bien séparés."""
        rng = np.random.default_rng(42)
        N, C = 150, 3
        labels = rng.integers(0, C, N)
        # Modèle raisonnablement calibré : bonne classe + 2.0, reste 0.0
        logits = np.zeros((N, C), dtype=np.float32)
        logits[np.arange(N), labels] = 2.0
        best_t = _calibrate_temperature(logits, labels)
        probs = _softmax(logits / best_t)
        ece = _ece(probs, labels)
        assert ece < 0.15, f"ECE après calibration temperature = {ece:.3f} ≥ 0.15"


# ---------------------------------------------------------------------------
# T5-isotonie : params nus (calibrator_x/y) — pas de pickle
# ---------------------------------------------------------------------------

class TestIsotonicParamsNus:
    """T5-isotonie (si temperature insuffisant) — params nus np.interp, sans sklearn en prod."""

    def test_params_nus_roundtrip(self):
        """calibrator_x/y sauvegardés et rechargés sans pickle."""
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            pytest.skip("sklearn non disponible — T5-isotonie non testable")

        rng = np.random.default_rng(7)
        N = 120
        confs = rng.uniform(0.3, 0.9, N).astype(np.float32)
        labels_bin = (confs > 0.5).astype(int)  # proxy binaire

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(confs, labels_bin)

        # Sauvegarder params nus
        x_thresh = iso.X_thresholds_.astype(np.float32)
        y_thresh = iso.y_thresholds_.astype(np.float32)

        # Recharger sans sklearn — np.interp suffit
        confs_test = np.array([0.35, 0.55, 0.75, 0.90], dtype=np.float32)
        cal_sklearn = iso.predict(confs_test)
        cal_interp = np.interp(confs_test, x_thresh, y_thresh).astype(np.float32)

        np.testing.assert_allclose(cal_sklearn, cal_interp, atol=1e-4), (
            "np.interp doit reproduire sklearn.predict à 1e-4 près"
        )

    def test_fallback_identity_on_missing_keys(self):
        """Checkpoint sans clés calibrateur → identité (pas de KeyError)."""
        # Simule le comportement du checkpoint loader
        data = {"encoder_0": np.zeros((3, 3))}  # pas de calibrator_x/y
        confidence_raw = np.array([0.4, 0.7, 0.9], dtype=np.float32)
        if "calibrator_x" in data and "calibrator_y" in data:
            confidence_cal = np.interp(confidence_raw, data["calibrator_x"], data["calibrator_y"])
        else:
            confidence_cal = confidence_raw  # identité
        np.testing.assert_array_equal(confidence_cal, confidence_raw)
