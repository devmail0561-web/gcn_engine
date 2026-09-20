# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer1.features import FeatureVocabulary


def test_forward_node_shape():
    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    x = np.random.randn(vocab.d_clause).astype(np.float32)
    out = enc.forward_node(x)
    assert out.shape == (7,), f"Expected (7,), got {out.shape}"


def test_forward_edge_shape():
    vocab = FeatureVocabulary()
    d_cl = vocab.d_clause
    d_edge_cl = vocab.d_edge_closed_loop(d_cl, 7)
    enc = MLPEncoder(d_clause=d_cl, d_edge=d_edge_cl)
    x = np.random.randn(d_edge_cl).astype(np.float32)
    out = enc.forward_edge(x)
    assert out.shape == (11,), f"Expected (11,), got {out.shape}"


def test_parameters_list():
    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    params = enc.parameters()
    assert len(params) > 0
    assert all(isinstance(p, np.ndarray) for p in params)
