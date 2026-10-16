# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Étage-2 : dims, déterminisme, checkpoint + GARDE ANTI-LEXIQUE.

Doctrine (moteur agnostique) : aucun littéral FR/EN ni import
`gcn_annotate` dans `gcn_python`. La langue vit dans `gcn-tools`
(étage-1) et les données — jamais dans le moteur.
"""
import numpy as np
import pytest

from gcn_python.layer2.stage2 import (
    N_CLASSES,
    N_FEATURES,
    REJECT_LABEL,
    Stage2Classifier,
)

# Sondes lexicales FR/EN : toute présence dans le source moteur = fuite.
_LEX_PROBES = ["parce que", "because", "malgré", "although", "en raison",
               "in order to", "tandis que", "whereas", "néanmoins",
               "nevertheless", "grâce à", "depends on"]


def test_dims_and_schema():
    assert N_FEATURES == 36
    assert N_CLASSES == 12
    assert REJECT_LABEL == 11
    clf = Stage2Classifier()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, N_FEATURES)).astype(np.float32)
    assert clf.forward(X).shape == (5, 12)
    assert clf.predict(X).shape == (5,)


def test_forward_rejects_bad_dims():
    clf = Stage2Classifier()
    with pytest.raises(ValueError):
        clf.forward(np.zeros((3, 35), dtype=np.float32))


def test_softmax_and_determinism():
    a = Stage2Classifier(seed=7)
    b = Stage2Classifier(seed=7)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, N_FEATURES)).astype(np.float32)
    np.testing.assert_allclose(a.forward(X), b.forward(X))
    assert set(a.predict(X).tolist()) <= set(range(12))


def test_toy_overfit_proves_reject_learnable():
    """12 classes séparables par le prior one-hot : doit surapprendre."""
    rng = np.random.default_rng(3)
    X = np.zeros((24, N_FEATURES), dtype=np.float32)
    y = np.arange(24) % 12
    for i, c in enumerate(y):
        if c < 11:
            X[i, c] = 1.0
        else:
            X[i, 11] = 0.0  # REJECT : prior nul + confiance basse
            X[i, 13] = 3.0  # n_markers élevé, marqueur ambigu
    X += rng.normal(0, 0.01, X.shape).astype(np.float32)
    clf = Stage2Classifier(seed=11)
    _hist = clf.fit(X, y, val=(X, y), epochs=500, lr=0.1,
                   patience=60)
    acc = float((clf.predict(X) == y).mean())
    assert acc == 1.0, f"MLP incapable de mémoriser 24 pts (acc={acc})"


def test_checkpoint_roundtrip(tmp_path):
    clf = Stage2Classifier(seed=5)
    p = str(tmp_path / "s2.npz")
    clf.save(p, provenance={"seed": 5})
    back, prov = Stage2Classifier.load(p)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3, N_FEATURES)).astype(np.float32)
    np.testing.assert_allclose(back.forward(X), clf.forward(X))
    assert prov["seed"] == 5


def test_scaler_roundtrip_and_identity(tmp_path):
    clf = Stage2Classifier(seed=5)
    rng = np.random.default_rng(4)
    X = (rng.normal(size=(10, N_FEATURES)) * 50).astype(np.float32)
    clf.fit_scaler(X)
    Xt = clf.transform(X)
    np.testing.assert_allclose(Xt.mean(axis=0),
                               np.zeros(N_FEATURES), atol=1e-5)
    np.testing.assert_allclose(Xt.std(axis=0),
                               np.ones(N_FEATURES), atol=1e-5)
    p = str(tmp_path / "s2s.npz")
    clf.save(p)
    back, _ = Stage2Classifier.load(p)
    np.testing.assert_allclose(back.mu, clf.mu)
    np.testing.assert_allclose(back.sigma, clf.sigma)
    np.testing.assert_allclose(back.transform(X), Xt)


def test_no_lexicon_in_engine():
    import pathlib
    src = pathlib.Path(__file__).resolve().parent.parent / "src" / \
        "gcn_python" / "layer2" / "stage2.py"
    text = src.read_text(encoding="utf-8").lower()
    # Garde réelle : aucun import outil + aucune sonde lexicale.
    assert "gcn_annotate" not in text
    for probe in _LEX_PROBES:
        assert probe not in text, f"fuite lexicale : {probe!r}"
