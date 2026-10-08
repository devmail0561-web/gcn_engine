# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests tête act — MLP (1,3) + format s3 + smoke d'apprentissage."""

from __future__ import annotations

import numpy as np
import pytest

from gcn_python.layer2.act import ACT_TYPES
from gcn_python.layer2.act_head import ActHead


def test_forward_distribution_et_predict():
    h = ActHead(d_model=8, hidden=16, seed=1)
    x = np.random.default_rng(0).normal(0, 1, 8).astype(np.float32)
    p = h.forward(x)
    assert p.shape == (3,) and abs(float(p.sum()) - 1.0) < 1e-5 and (p >= 0).all()
    assert h.predict(x) in ACT_TYPES
    with pytest.raises(ValueError, match="d_model"):
        h.forward(np.zeros(7, dtype=np.float32))
    with pytest.raises(ValueError):
        ActHead(d_model=0)


def test_smoke_apprend_trois_actes():
    # 6 reps fixes, 2 par acte : la tête doit les mémoriser (signal, pas généralisation).
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (6, 12)).astype(np.float32)
    y = [0, 0, 1, 1, 2, 2]
    h = ActHead(d_model=12, hidden=32, seed=3)
    def cout():
        total = 0.0
        for xi, yi in zip(X, y, strict=True):
            c, _ = h.loss_and_grad(xi, yi)
            total += c
        return total / len(y)
    avant = cout()
    for _ in range(300):
        for xi, yi in zip(X, y, strict=True):
            _, g = h.loss_and_grad(xi, yi)
            h.apply_grads(g, 0.1)
    assert cout() < avant / 4
    assert [h.predict(xi) for xi in X] == ["assert", "assert", "question", "question", "request", "request"]
    with pytest.raises(ValueError, match="gold_idx"):
        h.loss_and_grad(X[0], 5)


def test_format_s3_roundtrip_et_rejets(tmp_path):
    h = ActHead(d_model=8, hidden=16, seed=4)
    r = ActHead.from_json(h.to_json())
    assert r.act_types == ["assert", "question", "request"]
    assert np.array_equal(r.W2, h.W2)
    d = h.to_dict()
    assert d["format_refacto"] == 20261007 and d["d_model"] == 8
    with pytest.raises(ValueError, match="format"):
        ActHead.from_dict({**d, "format_refacto": 1})
    with pytest.raises(ValueError, match="act_types"):
        ActHead.from_dict({**d, "act_types": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="dims"):
        ActHead.from_dict({**d, "d_model": 9})
