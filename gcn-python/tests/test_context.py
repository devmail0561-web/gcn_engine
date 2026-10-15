# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests encodeur contextuel : formes, résidu, budget params, gradient."""
import numpy as np

from gcn_python.layer1.context import ContextEncoder, DilatedConvBlock


def test_block_shapes():
    blk = DilatedConvBlock(d_model=8, dilation=2)
    out = blk.forward(np.zeros((5, 8), dtype=np.float32))
    assert out.shape == (5, 8)


def test_block_rejects_bad_input():
    blk = DilatedConvBlock(d_model=8)
    try:
        blk.forward(np.zeros((5,), dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("aurait dû lever")


def test_block_rejects_bad_args():
    for kw in ({"d_model": 0}, {"d_model": 8, "dilation": 0}):
        try:
            DilatedConvBlock(**kw)
        except ValueError:
            continue
        raise AssertionError(f"aurait dû lever pour {kw}")


def test_encoder_budget():
    assert ContextEncoder(d_model=128).n_params < 1_000_000


def test_encoder_shapes():
    enc = ContextEncoder(d_model=8)
    out = enc.forward(np.zeros((5, 8), dtype=np.float32))
    assert out.shape == (5, 8)
    d_in, grads = enc.backward(np.ones((5, 8), dtype=np.float32))
    assert d_in.shape == (5, 8)
    assert len(grads) == 4


def test_block_grad_finite_diff():
    rng = np.random.default_rng(0)
    blk = DilatedConvBlock(6, dilation=2, seed=1)
    X = rng.normal(size=(7, 6)).astype(np.float32)
    d = rng.normal(size=(7, 6)).astype(np.float32)
    W0 = blk.W.copy()
    blk.forward(X)
    _, (dW, _db) = blk.backward(d)
    errs = []
    for k, i, j in [(0, 0, 0), (1, 2, 3), (2, 5, 1)]:
        Wp = W0.copy()
        Wp[k, i, j] += 1e-3
        blk.W = Wp
        lp = float((blk.forward(X) * d).sum())
        Wm = W0.copy()
        Wm[k, i, j] -= 1e-3
        blk.W = Wm
        lm = float((blk.forward(X) * d).sum())
        errs.append(abs(dW[k, i, j] - (lp - lm) / 2e-3))
    blk.W = W0
    assert max(errs) < 1e-2
