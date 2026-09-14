import numpy as np
from gcn_python.layer3.reference import RGCNLayer


def test_message_pass_shape():
    layer = RGCNLayer(d_in=16, d_out=16, n_relations=11)
    node_feats = np.random.randn(4, 16).astype(np.float32)
    edge_index = np.array([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=np.int64)
    edge_types = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    out = layer.message_pass(node_feats, edge_index, edge_types)
    assert out.shape == (4, 16)


def test_no_edges():
    layer = RGCNLayer(d_in=8, d_out=8, n_relations=3)
    node_feats = np.random.randn(3, 8).astype(np.float32)
    edge_index = np.zeros((2, 0), dtype=np.int64)
    edge_types = np.zeros(0, dtype=np.int64)
    out = layer.message_pass(node_feats, edge_index, edge_types)
    assert out.shape == (3, 8)


def test_cycle_support():
    """R-GCN handles cycles without error or NaN."""
    layer = RGCNLayer(d_in=4, d_out=4, n_relations=2)
    node_feats = np.random.randn(3, 4).astype(np.float32)
    edge_index = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    edge_types = np.array([0, 0, 0], dtype=np.int64)
    out = layer.message_pass(node_feats, edge_index, edge_types)
    assert out.shape == (3, 4)
    assert not np.any(np.isnan(out))
