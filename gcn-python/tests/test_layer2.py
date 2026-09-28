# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from gcn_python.constants import NODE_TYPES, RELATION_TYPES
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer2.reference import MLPEncoder


def test_forward_node_shape():
    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, len(NODE_TYPES)))
    x = np.random.randn(vocab.d_clause).astype(np.float32)
    out = enc.forward_node(x)
    assert out.shape == (len(NODE_TYPES),), f"Expected (len(NODE_TYPES),), got {out.shape}"


def test_forward_edge_shape():
    vocab = FeatureVocabulary()
    d_cl = vocab.d_clause
    d_edge_cl = vocab.d_edge_closed_loop(d_cl, len(NODE_TYPES))
    enc = MLPEncoder(d_clause=d_cl, d_edge=d_edge_cl)
    x = np.random.randn(d_edge_cl).astype(np.float32)
    out = enc.forward_edge(x)
    assert out.shape == (len(RELATION_TYPES),), f"Expected (len(RELATION_TYPES),), got {out.shape}"


def test_parameters_list():
    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, len(NODE_TYPES)))
    params = enc.parameters()
    assert len(params) > 0
    assert all(isinstance(p, np.ndarray) for p in params)
