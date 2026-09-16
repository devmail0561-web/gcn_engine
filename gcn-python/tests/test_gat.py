"""Tests pour RGCNLayerGAT — skip automatique si PyTorch n'est pas installé."""
import pytest
import numpy as np

torch = pytest.importorskip("torch", reason="PyTorch non installé — test ignoré")

from gcn_python.layer3.gat import RGCNLayerGAT
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
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=3, device="cpu")
    assert isinstance(layer, CausalGraph), "RGCNLayerGAT doit satisfaire le Protocol CausalGraph"


# ─── message_pass output shape ───────────────────────────────────────────────

def test_output_shape():
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=3, device="cpu")
    features, edge_index, edge_types = make_graph(n=5, e=6, n_rel=3)
    out = layer.message_pass(features, edge_index, edge_types)
    assert out.shape == (5, 16), f"attendu (5, 16), obtenu {out.shape}"


def test_output_is_numpy():
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=3, device="cpu")
    features, edge_index, edge_types = make_graph()
    out = layer.message_pass(features, edge_index, edge_types)
    assert isinstance(out, np.ndarray)


# ─── sigmoid activation (values ∈ (0, 1)) ───────────────────────────────────

def test_output_in_sigmoid_range():
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=3, device="cpu")
    features, edge_index, edge_types = make_graph()
    out = layer.message_pass(features, edge_index, edge_types)
    assert np.all(out > 0) and np.all(out < 1), "sigmoid → valeurs dans (0, 1)"


# ─── Empty edge case ─────────────────────────────────────────────────────────

def test_empty_edges():
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=3, device="cpu")
    features = np.random.randn(3, 8).astype(np.float32)
    edge_index = np.empty((2, 0), dtype=np.int64)
    edge_types = np.empty((0,), dtype=np.int64)
    out = layer.message_pass(features, edge_index, edge_types)
    assert out.shape == (3, 16)


# ─── parameters() retourne 3 tableaux NumPy (W_r, W_0, a_r) ────────────────

def test_parameters_returns_numpy():
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=3, device="cpu")
    params = layer.parameters()
    assert len(params) == 3
    assert all(isinstance(p, np.ndarray) for p in params)
    assert params[0].shape == (3, 16, 8)   # W_r: (R, D_out, D_in)
    assert params[1].shape == (16, 8)       # W_0: (D_out, D_in)
    assert params[2].shape == (3, 32)       # a_r: (R, 2 * D_out)


# ─── Shapes avec n_relations=22 (bidirectionnel) ────────────────────────────

def test_shapes_n_relations_22():
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=22, device="cpu")
    params = layer.parameters()
    assert params[0].shape == (22, 16, 8), f"W_r attendu (22,16,8), obtenu {params[0].shape}"
    assert params[2].shape == (22, 32), f"a_r attendu (22,32), obtenu {params[2].shape}"


# ─── update() change les poids ───────────────────────────────────────────────

def test_update_changes_weights():
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=3, device="cpu")
    W_r_before = layer.parameters()[0].copy()
    grads = [np.ones_like(p) for p in layer.parameters()]
    layer.update(grads, lr=0.1)
    assert not np.allclose(W_r_before, layer.parameters()[0]), "update() doit modifier les poids"


# ─── Gradient a_r non-nul (l'attention apprend) ─────────────────────────────

def test_attention_gradient_nonzero():
    """Vérifie que le gradient de a_r est non-nul — l'attention apprend réellement."""
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=3, device="cpu")
    rng = np.random.default_rng(42)
    features = rng.normal(0, 1, (4, 8)).astype(np.float32)
    # Graphe avec edges convergent (2 arêtes vers même destination)
    edge_index = np.array([[0, 1, 2], [2, 2, 3]], dtype=np.int64)
    edge_types = np.array([0, 0, 0], dtype=np.int64)

    out = layer.message_pass(features, edge_index, edge_types)
    d_output = rng.normal(0, 1, out.shape).astype(np.float32)
    _d_input, grads = layer.backward_message_pass(d_output)

    assert np.any(grads[2] != 0), "Gradient a_r doit être non-nul (l'attention apprend)"
    assert np.any(grads[0] != 0), "Gradient W_r doit être non-nul"


# ─── load_state ValueError si shape incompatible ────────────────────────────

def test_load_state_value_error_on_shape_mismatch():
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=22, device="cpu")
    bad_W_r = np.zeros((11, 16, 8), dtype=np.float32)  # mauvais n_relations
    bad_W_0 = np.zeros((16, 8), dtype=np.float32)
    bad_a_r = np.zeros((11, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="incompatible"):
        layer.load_state([bad_W_r, bad_W_0, bad_a_r])


def test_load_state_value_error_pytorch_rgcn():
    """Vérifie que RGCNLayerPT vérifie aussi la shape."""
    from gcn_python.layer3.pytorch_rgcn import RGCNLayerPT
    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=22, device="cpu")
    bad_W_r = np.zeros((11, 16, 8), dtype=np.float32)
    bad_W_0 = np.zeros((16, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="incompatible"):
        layer.load_state([bad_W_r, bad_W_0])


# ─── Checkpoint round-trip avec n_relations=22 ──────────────────────────────

def test_checkpoint_roundtrip_n22():
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=22, device="cpu")
    params_before = [p.copy() for p in layer.parameters()]

    layer.load_state(params_before)
    params_after = layer.parameters()

    for p_before, p_after in zip(params_before, params_after):
        np.testing.assert_array_equal(p_before, p_after)


# ─── Bidirectional: représentations différentes de forward seul ─────────────

def test_bidirectional_repr_differs_from_forward():
    """Un même graphe produit des représentations différentes avec bidirectional."""
    features, edge_index, edge_types = make_graph(n=5, e=6, n_rel=3, seed=42)

    layer_fwd = RGCNLayerGAT(d_in=8, d_out=16, n_relations=3, device="cpu")
    out_fwd = layer_fwd.message_pass(features, edge_index, edge_types)

    # Bidirectional = 22 relations, arêtes inverses ajoutées
    layer_bidi = RGCNLayerGAT(d_in=8, d_out=16, n_relations=22, device="cpu")
    rev_index = edge_index[[1, 0], :]
    rev_types = edge_types + 3  # indices 3-5 pour les inverses
    edge_index_mp = np.concatenate([edge_index, rev_index], axis=1)
    edge_types_mp = np.concatenate([edge_types, rev_types])
    out_bidi = layer_bidi.message_pass(features, edge_index_mp, edge_types_mp)

    assert not np.allclose(out_fwd, out_bidi), \
        "bidirectional=True doit produire des représentations différentes"


# ─── GAT produit des sorties différentes de RGCNLayerPT ─────────────────────

def test_gat_differs_from_rgcn_pt():
    from gcn_python.layer3.pytorch_rgcn import RGCNLayerPT
    rng = np.random.default_rng(42)
    features = rng.normal(0, 1, (5, 8)).astype(np.float32)
    # Graphe avec edges convergentes (2 arêtes vers même nœud) — GAT ≠ RGCN
    edge_index = np.array([[0, 1, 2], [2, 2, 3]], dtype=np.int64)
    edge_types = np.array([0, 0, 0], dtype=np.int64)

    gat = RGCNLayerGAT(d_in=8, d_out=16, n_relations=3, device="cpu")
    pt = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu")

    out_gat = gat.message_pass(features, edge_index, edge_types)
    out_pt = pt.message_pass(features, edge_index, edge_types)
    assert not np.allclose(out_gat, out_pt), "GAT et RGCN doivent produire des sorties différentes"


# ─── repr lisible ────────────────────────────────────────────────────────────

def test_repr():
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=3, device="cpu")
    r = repr(layer)
    assert "RGCNLayerGAT" in r
    assert "d_in=8" in r
    assert "d_out=16" in r
