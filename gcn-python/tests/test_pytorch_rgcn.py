# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests pour RGCNLayerPT — skip automatique si PyTorch n'est pas installé."""
import pytest
import numpy as np

torch = pytest.importorskip("torch", reason="PyTorch non installé — test ignoré")

from gcn_python.layer3.pytorch_rgcn import RGCNLayerPT
from gcn_python.layer3.interface import CausalGraph


def make_graph(n: int = 4, e: int = 4, n_rel: int = 3, seed: int = 0) -> tuple:
    """Génère un graphe synthétique (features, edge_index, edge_types)."""
    rng = np.random.default_rng(seed)
    features = rng.normal(0, 1, (n, 8)).astype(np.float32)
    src = rng.integers(0, n, size=e)
    dst = rng.integers(0, n, size=e)
    edge_index = np.stack([src, dst])
    edge_types = rng.integers(0, n_rel, size=e)
    return features, edge_index, edge_types


# ─── Protocol compliance ─────────────────────────────────────────────────────

def test_implements_causal_graph_protocol():
    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu")
    assert isinstance(layer, CausalGraph), "RGCNLayerPT doit satisfaire le Protocol CausalGraph"


# ─── message_pass output shape ───────────────────────────────────────────────

def test_output_shape():
    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu")
    features, edge_index, edge_types = make_graph(n=5, e=6, n_rel=3)
    out = layer.message_pass(features, edge_index, edge_types)
    assert out.shape == (5, 16), f"attendu (5, 16), obtenu {out.shape}"


def test_output_is_numpy():
    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu")
    features, edge_index, edge_types = make_graph()
    out = layer.message_pass(features, edge_index, edge_types)
    assert isinstance(out, np.ndarray)


# ─── sigmoid activation (values ∈ (0, 1)) ───────────────────────────────────

def test_output_in_sigmoid_range():
    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu")
    features, edge_index, edge_types = make_graph()
    out = layer.message_pass(features, edge_index, edge_types)
    assert np.all(out > 0) and np.all(out < 1), "sigmoid → valeurs dans (0, 1)"


# ─── Empty edge case ─────────────────────────────────────────────────────────

def test_empty_edges():
    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu")
    features = np.random.randn(3, 8).astype(np.float32)
    edge_index = np.empty((2, 0), dtype=np.int64)
    edge_types = np.empty((0,), dtype=np.int64)
    out = layer.message_pass(features, edge_index, edge_types)
    assert out.shape == (3, 16)


# ─── parameters() retourne des tableaux NumPy ────────────────────────────────

def test_parameters_returns_numpy():
    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu")
    params = layer.parameters()
    assert len(params) == 2
    assert all(isinstance(p, np.ndarray) for p in params)
    assert params[0].shape == (3, 16, 8)  # W_r: (R, D_out, D_in)
    assert params[1].shape == (16, 8)     # W_0: (D_out, D_in)


# ─── update() change les poids ───────────────────────────────────────────────

def test_update_changes_weights():
    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu")
    W_r_before = layer.parameters()[0].copy()
    grads = [np.ones_like(layer.parameters()[0]), np.ones_like(layer.parameters()[1])]
    layer.update(grads, lr=0.1)
    W_r_after = layer.parameters()[0]
    assert not np.allclose(W_r_before, W_r_after), "update() doit modifier les poids"


# ─── torch_parameters() pour optimizer PyTorch ───────────────────────────────

def test_torch_parameters_for_optimizer():
    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu")
    params = layer.torch_parameters()
    assert len(params) == 2
    assert all(isinstance(p, torch.nn.Parameter) for p in params)
    assert params[0].requires_grad
    assert params[1].requires_grad


# ─── Gradient flow via forward_torch ─────────────────────────────────────────

def test_gradient_flows():
    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu")
    features_np, edge_index, edge_types = make_graph(n=4, e=4, n_rel=3)
    H = torch.tensor(features_np, requires_grad=False)

    out = layer.forward_torch(H, edge_index, edge_types)
    loss = out.sum()
    loss.backward()

    assert layer.W_r.grad is not None, "W_r doit avoir des gradients"
    assert layer.W_0.grad is not None, "W_0 doit avoir des gradients"


# ─── Conformité sortie vs NumPy référence (approximative) ────────────────────

def test_output_not_constant():
    """Sortie varie selon l'entrée (pas un module dégénéré)."""
    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu")
    f1, ei, et = make_graph(seed=0)
    f2, _, _ = make_graph(seed=1)
    out1 = layer.message_pass(f1, ei, et)
    out2 = layer.message_pass(f2, ei, et)
    assert not np.allclose(out1, out2), "Des entrées différentes doivent donner des sorties différentes"


# ─── repr lisible ────────────────────────────────────────────────────────────

def test_repr():
    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu")
    r = repr(layer)
    assert "RGCNLayerPT" in r
    assert "d_in=8" in r
    assert "d_out=16" in r
