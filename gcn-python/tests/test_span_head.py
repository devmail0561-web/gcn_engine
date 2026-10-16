# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests têtes ancres/bornes : formes, décodage, F1, pondération."""
import numpy as np

from gcn_python.layer1.span_head import SpanHead, decode_spans, span_f1


def test_forward_shapes():
    head = SpanHead(d_model=8)
    s, e = head.forward(np.zeros((5, 8), dtype=np.float32))
    assert s.shape == (5,) and e.shape == (5,)


def test_forward_rejects_bad_shape():
    head = SpanHead(d_model=8)
    try:
        head.forward(np.zeros((5,), dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("aurait dû lever")


def test_decode_greedy_no_overlap():
    offs = [(0, 5), (6, 10), (11, 15), (16, 20)]
    sp = np.array([0.9, 0.1, 0.8, 0.1])
    ep = np.array([0.1, 0.9, 0.1, 0.9])
    assert decode_spans(sp, ep, offs) == [[0, 10], [11, 20]]
    assert decode_spans(np.zeros(4), np.zeros(4), offs) == []
    assert decode_spans(sp, ep, []) == []


def test_span_f1_exact():
    assert span_f1([[0, 5]], [[0, 5]]) == 1.0
    assert span_f1([], []) == 1.0
    assert span_f1([[0, 5]], []) == 0.0
    assert span_f1([[0, 5]], [[0, 6]]) == 0.0
    assert span_f1([[0, 5], [6, 9]], [[0, 5]]) == 2 / 3


def test_loss_weighted_learns():
    rng = np.random.default_rng(7)
    head = SpanHead(d_model=4, seed=7)
    R = rng.normal(size=(20, 4)).astype(np.float32)
    ys = np.zeros(20)
    ys[0] = 1.0
    ye = np.zeros(20)
    ye[5] = 1.0
    avant = head.loss_and_update(R, ys, ye, 0.5)
    for _ in range(20):
        head.loss_and_update(R, ys, ye, 0.5)
    apres = head.loss_and_update(R, ys, ye, 0.5)
    assert apres < avant
