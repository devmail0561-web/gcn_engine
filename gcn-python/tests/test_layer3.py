# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
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


# ─── Phase 2 : activation configurable ──────────────────────────────────────

def test_rgcn_relu_activation_output_range():
    """Avec output_activation='relu', la sortie est >= 0 (pas squashée [0,1])."""
    layer = RGCNLayer(d_in=8, d_out=8, n_relations=3, output_activation="relu")
    node_feats = np.random.default_rng(0).normal(0, 1, (4, 8)).astype(np.float32)
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    edge_types = np.array([0, 1], dtype=np.int64)
    out = layer.message_pass(node_feats, edge_index, edge_types)
    assert np.all(out >= 0), "ReLU output must be >= 0"
    assert out.max() > 1.0 or out.max() == 0.0, "ReLU should not squash to [0,1]"


def test_rgcn_none_activation_passes_through():
    """Avec output_activation='none', la sortie est le pré-activation brut."""
    layer = RGCNLayer(d_in=8, d_out=8, n_relations=3, output_activation="none")
    node_feats = np.random.default_rng(1).normal(0, 1, (3, 8)).astype(np.float32)
    edge_index = np.array([[0], [1]], dtype=np.int64)
    edge_types = np.array([0], dtype=np.int64)
    out = layer.message_pass(node_feats, edge_index, edge_types)
    assert out.min() < 0, "Sans activation, des valeurs négatives sont possibles"


def test_rgcn_sigmoid_default_backward_compat():
    """Le défaut est toujours sigmoid (rétrocompatibilité)."""
    layer = RGCNLayer(d_in=8, d_out=8)
    assert layer.output_activation == "sigmoid"


def test_rgcn_gradient_finite_differences_relu():
    """Gradient check par différences finies avec activation relu."""
    rng = np.random.default_rng(42)
    D, N, n_rel = 8, 4, 3
    eps = 1e-4

    layer = RGCNLayer(d_in=D, d_out=D, n_relations=n_rel, seed=7,
                      output_activation="relu")
    node_feats = rng.random((N, D)).astype(np.float64)
    layer.W_r = layer.W_r.astype(np.float64)
    layer.W_0 = layer.W_0.astype(np.float64)

    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    edge_types = np.array([0, 1, 0], dtype=np.int64)

    def scalar_loss(W_r, W_0):
        l = RGCNLayer(d_in=D, d_out=D, n_relations=n_rel, output_activation="relu")
        l.W_r, l.W_0 = W_r, W_0
        out = l.message_pass(node_feats, edge_index, edge_types)
        return float(out.sum())

    layer.message_pass(node_feats, edge_index, edge_types)
    d_out = np.ones((N, D), dtype=np.float64)
    _, grads = layer.backward_message_pass(d_out)
    dW_r_analytic = grads[0].astype(np.float64)
    dW_0_analytic = grads[1].astype(np.float64)

    for idx in [(0, 0, 0), (1, 2, 3), (2, 0, 1)]:
        r, i, j = idx
        W_r_plus = layer.W_r.copy(); W_r_plus[r, i, j] += eps
        W_r_minus = layer.W_r.copy(); W_r_minus[r, i, j] -= eps
        fd = (scalar_loss(W_r_plus, layer.W_0) - scalar_loss(W_r_minus, layer.W_0)) / (2 * eps)
        assert abs(dW_r_analytic[r, i, j] - fd) < 1e-3, (
            f"dW_r[{r},{i},{j}] relu analytique={dW_r_analytic[r,i,j]:.6f} ≠ fd={fd:.6f}"
        )

    for idx in [(0, 0), (2, 3), (4, 1)]:
        i, j = idx
        W_0_plus = layer.W_0.copy(); W_0_plus[i, j] += eps
        W_0_minus = layer.W_0.copy(); W_0_minus[i, j] -= eps
        fd = (scalar_loss(layer.W_r, W_0_plus) - scalar_loss(layer.W_r, W_0_minus)) / (2 * eps)
        assert abs(dW_0_analytic[i, j] - fd) < 1e-3, (
            f"dW_0[{i},{j}] relu analytique={dW_0_analytic[i,j]:.6f} ≠ fd={fd:.6f}"
        )


# ─── Phase 3 : LayerNorm après R-GCN ────────────────────────────────────────

from gcn_python.layer3.reference import LayerNormNumPy


def test_layernorm_forward_normalizes():
    """LayerNorm forward : mean ≈ 0, var ≈ 1 sur la dernière dimension."""
    ln = LayerNormNumPy(d=16)
    x = np.random.default_rng(0).normal(5.0, 3.0, (4, 16)).astype(np.float32)
    y = ln.forward(x)
    assert y.shape == x.shape
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(y.var(axis=-1), 1.0, atol=1e-2)


def test_layernorm_backward_finite_differences():
    """Gradient check par différences finies pour LayerNormNumPy."""
    rng = np.random.default_rng(42)
    D, N = 8, 4
    eps = 1e-4
    ln = LayerNormNumPy(d=D)
    x = rng.normal(0, 1, (N, D)).astype(np.float64)
    ln.gamma = ln.gamma.astype(np.float64)
    ln.beta = ln.beta.astype(np.float64)

    y = ln.forward(x)
    d_out = np.ones_like(y)
    dx_ana, dg_ana, db_ana = ln.backward(d_out)

    dx_fd = np.zeros_like(x)
    for i in range(N):
        for j in range(D):
            x_p = x.copy(); x_p[i, j] += eps
            x_m = x.copy(); x_m[i, j] -= eps
            ln2 = LayerNormNumPy(d=D)
            ln2.gamma = ln.gamma.copy(); ln2.beta = ln.beta.copy()
            y_p = ln2.forward(x_p).sum()
            ln2.gamma = ln.gamma.copy(); ln2.beta = ln.beta.copy()
            y_m = ln2.forward(x_m).sum()
            dx_fd[i, j] = (y_p - y_m) / (2 * eps)
    assert np.allclose(dx_ana, dx_fd, atol=1e-3), f"dx max err={np.abs(dx_ana - dx_fd).max()}"

    dg_fd = np.zeros(D, dtype=np.float64)
    for j in range(D):
        ln3 = LayerNormNumPy(d=D)
        ln3.gamma = ln.gamma.copy(); ln3.gamma[j] += eps; ln3.beta = ln.beta.copy()
        y_p = ln3.forward(x).sum()
        ln3.gamma = ln.gamma.copy(); ln3.gamma[j] -= eps; ln3.beta = ln.beta.copy()
        y_m = ln3.forward(x).sum()
        dg_fd[j] = (y_p - y_m) / (2 * eps)
    assert np.allclose(dg_ana, dg_fd, atol=1e-3), f"dgamma max err={np.abs(dg_ana - dg_fd).max()}"


def test_rgcn_layernorm_output_normalized():
    """RGCNLayer avec use_layernorm=True produit des sorties normalisées."""
    layer = RGCNLayer(d_in=16, d_out=16, n_relations=3,
                      output_activation="relu", use_layernorm=True)
    node_feats = np.random.default_rng(0).normal(0, 1, (4, 16)).astype(np.float32)
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    edge_types = np.array([0, 1], dtype=np.int64)
    out = layer.message_pass(node_feats, edge_index, edge_types)
    assert out.shape == (4, 16)
    assert np.allclose(out.mean(axis=-1), 0.0, atol=0.1)
    assert not np.any(np.isnan(out))


def test_rgcn_layernorm_backward_grads():
    """backward_message_pass avec LayerNorm retourne 4 gradients [W_r, W_0, gamma, beta]."""
    layer = RGCNLayer(d_in=8, d_out=8, n_relations=3,
                      output_activation="relu", use_layernorm=True)
    node_feats = np.random.default_rng(0).normal(0, 1, (4, 8)).astype(np.float32)
    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    edge_types = np.array([0, 1, 0], dtype=np.int64)
    out = layer.message_pass(node_feats, edge_index, edge_types)
    d_out = np.ones_like(out)
    d_input, grads = layer.backward_message_pass(d_out)
    assert len(grads) == 4, f"Expected 4 grads with layernorm, got {len(grads)}"
    assert grads[2].shape == (8,), f"d_gamma shape {grads[2].shape}"
    assert grads[3].shape == (8,), f"d_beta shape {grads[3].shape}"
    assert np.all(np.isfinite(d_input))
    for g in grads:
        assert np.all(np.isfinite(g))


def test_rgcn_layernorm_parameters_count():
    """parameters() retourne 4 arrays avec LayerNorm, 2 sans."""
    layer_ln = RGCNLayer(d_in=8, d_out=8, n_relations=3, use_layernorm=True)
    layer_no = RGCNLayer(d_in=8, d_out=8, n_relations=3, use_layernorm=False)
    assert len(layer_ln.parameters()) == 4
    assert len(layer_no.parameters()) == 2


def test_rgcn_layernorm_update():
    """update() avec LayerNorm modifie gamma et beta."""
    layer = RGCNLayer(d_in=8, d_out=8, n_relations=3, use_layernorm=True)
    gamma_before = layer.norm.gamma.copy()
    beta_before = layer.norm.beta.copy()
    fake_grads = [np.ones_like(layer.W_r), np.ones_like(layer.W_0),
                  np.ones(8, dtype=np.float32), np.ones(8, dtype=np.float32)]
    layer.update(fake_grads, lr=0.01)
    assert not np.array_equal(layer.norm.gamma, gamma_before)
    assert not np.array_equal(layer.norm.beta, beta_before)


def test_rgcn_layernorm_checkpoint_roundtrip(tmp_path):
    """Checkpoint save → load préserve gamma/beta de LayerNorm."""
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint

    vocab = FeatureVocabulary()
    d_edge = vocab.d_edge_closed_loop(vocab.d_clause, 7)
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=d_edge)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause, n_relations=3,
                      output_activation="relu", use_layernorm=True)
    graph.norm.gamma[:] = np.array([0.5] * vocab.d_clause, dtype=np.float32)
    graph.norm.beta[:] = np.array([0.1] * vocab.d_clause, dtype=np.float32)
    pipe = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)
    ckpt = tmp_path / "ln.npz"
    save_checkpoint(pipe, ckpt)

    graph2 = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause, n_relations=3,
                       output_activation="relu", use_layernorm=True)
    pipe2 = CGNPipeline(encoder=MLPEncoder(d_clause=vocab.d_clause, d_edge=d_edge),
                        graph=graph2, vocabulary=vocab)
    load_checkpoint(pipe2, ckpt, trusted=True)
    np.testing.assert_allclose(pipe2._graph_layers[0].norm.gamma, 0.5, atol=1e-6)
    np.testing.assert_allclose(pipe2._graph_layers[0].norm.beta, 0.1, atol=1e-6)


def test_rgcn_layernorm_extra_layers_inherit():
    """Avec n_rgcn_layers > 1, les extra layers héritent de use_layernorm."""
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.pipeline.cgnp import CGNPipeline

    vocab = FeatureVocabulary()
    d_edge = vocab.d_edge_closed_loop(vocab.d_clause, 7)
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=d_edge)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause, n_relations=3,
                      use_layernorm=True)
    pipe = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                       n_rgcn_layers=2)
    assert len(pipe._graph_layers) == 2
    for i, layer in enumerate(pipe._graph_layers):
        assert layer.norm is not None, f"Layer {i} manque LayerNorm"
        assert layer.use_layernorm is True, f"Layer {i} use_layernorm=False"
