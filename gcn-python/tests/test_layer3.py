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


def test_rgcn_gradient_finite_differences():
    """Vérification numérique des gradients dW_r et dW_0 par différences finies."""
    rng = np.random.default_rng(0)
    D, N, n_rel = 8, 4, 3
    eps = 1e-4

    layer = RGCNLayer(d_in=D, d_out=D, n_relations=n_rel, seed=7)
    node_feats = rng.random((N, D)).astype(np.float64)
    layer.W_r = layer.W_r.astype(np.float64)
    layer.W_0 = layer.W_0.astype(np.float64)

    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    edge_types = np.array([0, 1, 0], dtype=np.int64)

    def scalar_loss(W_r, W_0):
        l = RGCNLayer(d_in=D, d_out=D, n_relations=n_rel)
        l.W_r, l.W_0 = W_r, W_0
        out = l.message_pass(node_feats, edge_index, edge_types)
        return float(out.sum())

    # Backward analytique
    layer.message_pass(node_feats, edge_index, edge_types)
    d_out = np.ones((N, D), dtype=np.float64)
    _, grads = layer.backward_message_pass(d_out)
    dW_r_analytic, dW_0_analytic = grads[0].astype(np.float64), grads[1].astype(np.float64)

    # Gradient numérique W_r (un élément représentatif)
    for idx in [(0, 0, 0), (1, 2, 3), (2, 0, 1)]:
        r, i, j = idx
        W_r_plus = layer.W_r.copy(); W_r_plus[r, i, j] += eps
        W_r_minus = layer.W_r.copy(); W_r_minus[r, i, j] -= eps
        fd = (scalar_loss(W_r_plus, layer.W_0) - scalar_loss(W_r_minus, layer.W_0)) / (2 * eps)
        assert abs(dW_r_analytic[r, i, j] - fd) < 1e-3, (
            f"dW_r[{r},{i},{j}] analytique={dW_r_analytic[r,i,j]:.6f} ≠ fd={fd:.6f}"
        )

    # Gradient numérique W_0 (un élément représentatif)
    for idx in [(0, 0), (2, 3), (4, 1)]:
        i, j = idx
        W_0_plus = layer.W_0.copy(); W_0_plus[i, j] += eps
        W_0_minus = layer.W_0.copy(); W_0_minus[i, j] -= eps
        fd = (scalar_loss(layer.W_r, W_0_plus) - scalar_loss(layer.W_r, W_0_minus)) / (2 * eps)
        assert abs(dW_0_analytic[i, j] - fd) < 1e-3, (
            f"dW_0[{i},{j}] analytique={dW_0_analytic[i,j]:.6f} ≠ fd={fd:.6f}"
        )
