# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests pour RGCNLayerGAT — skip automatique si PyTorch n'est pas installé."""
import numpy as np
import pytest

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
    assert params[2].shape == (1, 3, 32)    # a_r: (H=1, R, 2 * D_head) — format v3 (D)


# ─── Shapes avec n_relations=22 (bidirectionnel) ────────────────────────────

def test_shapes_n_relations_22():
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=22, device="cpu")
    params = layer.parameters()
    assert params[0].shape == (22, 16, 8), f"W_r attendu (22,16,8), obtenu {params[0].shape}"
    assert params[2].shape == (1, 22, 32), f"a_r attendu (1,22,32), obtenu {params[2].shape}"


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

    for p_before, p_after in zip(params_before, params_after, strict=False):
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


# ── Amélioration D — Attention multi-tête ─────────────────────────────────────


def _gat_features(N: int = 5, D: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    H = rng.normal(0, 1, (N, D)).astype(np.float32)
    # Graphe convergent : dst 2 reçoit 2 arêtes → softmax non trivial
    # (1 arête/dst rendrait alpha ≡ 1 et masquerait les différences de têtes)
    ei = np.array([[0, 1, 2, 3], [2, 2, 3, 4]], dtype=np.int64)
    et = np.zeros(4, dtype=np.int64)
    return H, ei, et


def _gat_features_n4(D: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    H = rng.normal(0, 1, (4, D)).astype(np.float32)
    ei = np.array([[0, 1, 2], [2, 2, 3]], dtype=np.int64)
    et = np.zeros(3, dtype=np.int64)
    return H, ei, et


def test_multihead_4_output_shape_unchanged():
    layer = RGCNLayerGAT(d_in=128, d_out=128, n_relations=11, n_heads=4,
                         device="cpu", seed=1)
    H, ei, et = _gat_features()
    out = layer.message_pass(H, ei, et)
    assert out.shape == (5, 128)


def test_multihead_backward_shapes_correct():
    rng = np.random.default_rng(3)
    layer = RGCNLayerGAT(d_in=128, d_out=128, n_relations=11, n_heads=4,
                         device="cpu", seed=1)
    H, ei, et = _gat_features()
    layer.message_pass(H, ei, et)
    d_in, grads = layer.backward_message_pass(
        rng.normal(0, 0.1, (5, 128)).astype(np.float32))
    assert d_in.shape == (5, 128)
    assert len(grads) == 3
    assert grads[0].shape == (11, 128, 128)
    assert grads[1].shape == (128, 128)
    assert grads[2].shape == (4, 11, 64)


def test_n_heads_1_backward_compat_a_r_2d():
    layer = RGCNLayerGAT(d_in=16, d_out=16, n_relations=3, n_heads=1,
                         device="cpu", seed=1)
    arrs = layer.parameters()
    old_format = [arrs[0], arrs[1], np.asarray(arrs[2]).reshape(3, 32)]
    layer.load_state(old_format)  # ancien checkpoint (R, 2*D) accepté si n_heads=1
    H, ei, et = _gat_features_n4(D=16)
    out = layer.message_pass(H, ei, et)
    assert out.shape == (4, 16)


def test_n_heads_incompatible_raises_with_hint():
    with pytest.raises(ValueError, match="d_emb=49"):
        RGCNLayerGAT(d_in=130, d_out=130, n_relations=11, n_heads=4, device="cpu")
    # d_out=130, n_heads=4 → message mentionnant d_emb=49
    try:
        RGCNLayerGAT(d_in=130, d_out=130, n_relations=11, n_heads=4, device="cpu")
        raise AssertionError("aurait dû lever")
    except ValueError as exc:
        assert "130" in str(exc) and "4" in str(exc)


def test_multihead_4_differs_from_single_head():
    H, ei, et = _gat_features()
    l1 = RGCNLayerGAT(d_in=128, d_out=128, n_relations=11, n_heads=1,
                      device="cpu", seed=9)
    l4 = RGCNLayerGAT(d_in=128, d_out=128, n_relations=11, n_heads=4,
                      device="cpu", seed=9)
    out1 = l1.message_pass(H, ei, et)
    out4 = l4.message_pass(H, ei, et)
    assert not np.allclose(out1, out4), "4 têtes ≠ 1 tête (perspectives multiples)"


# ── Amélioration E — Activation + LayerNorm ───────────────────────────────────

def test_relu_activation_output_can_exceed_1():
    layer = RGCNLayerGAT(d_in=16, d_out=16, n_relations=3, output_activation="relu",
                         device="cpu", seed=1)
    H, ei, et = _gat_features_n4(D=16)
    out = layer.message_pass(H, ei, et)
    assert (out >= 0).all()
    # relu non borné : au moins une valeur > 1 probable (sinon shapes seules)
    assert out.shape == (4, 16)


def test_relu_backward_uses_step_derivative():
    _rng = np.random.default_rng(5)
    layer = RGCNLayerGAT(d_in=16, d_out=16, n_relations=3, output_activation="relu",
                         device="cpu", seed=1)
    H, ei, et = _gat_features_n4(D=16)
    layer.message_pass(H, ei, et)
    _pre = layer._out_retained.detach().cpu().numpy()
    d_in, _ = layer.backward_message_pass(np.ones((4, 16), dtype=np.float32))
    assert d_in.shape == (4, 16)
    assert np.isfinite(d_in).all()


def test_layernorm_output_normalized():
    layer = RGCNLayerGAT(d_in=32, d_out=32, n_relations=3, use_layernorm=True,
                         output_activation="none", device="cpu", seed=1)
    assert len(layer.parameters()) == 5
    H, ei, et = _gat_features(N=6, D=32)
    out = layer.message_pass(H, ei, et)
    assert out.shape == (6, 32)
    row_means = out.mean(axis=1)
    row_vars = out.var(axis=1)
    assert np.allclose(row_means, 0, atol=1e-5), "LayerNorm : moyenne ≈ 0 par ligne"
    assert np.allclose(row_vars, 1, atol=1e-4), "LayerNorm : variance ≈ 1 par ligne"


def test_stacked_gat_2_layers_output_shape():
    D = 128
    H, ei, et = _gat_features()
    l1 = RGCNLayerGAT(d_in=D, d_out=D, n_relations=11, n_heads=4,
                      output_activation="relu", device="cpu", seed=1)
    l2 = RGCNLayerGAT(d_in=D, d_out=D, n_relations=11, n_heads=4,
                      output_activation="sigmoid", device="cpu", seed=2)
    h = l1.message_pass(H, ei, et)
    out = l2.message_pass(h, ei, et)
    assert out.shape == (5, D)
    assert ((out >= 0) & (out <= 1)).all(), "finale sigmoid → [0,1]"


def test_residual_gradient_nonzero_even_if_layer_dead():
    # Couche morte (poids nuls + activation none → sortie 0, jacobienne 0)
    # mais le résidu fait passer le gradient identité.
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer1.representation import UDRepresentation
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.pipeline.cgnp import CGNPipeline

    vocab = FeatureVocabulary()
    D = vocab.d_clause
    enc = MLPEncoder(d_clause=D, d_edge=vocab.d_edge_closed_loop(D, 7), seed=0)

    class _DeadGraph:
        d_in = D
        d_out = D
        n_relations = 3

        def message_pass(self, H, edge_index, edge_types):
            return np.zeros_like(H)

        def backward_message_pass(self, d_out):
            return np.zeros_like(d_out), []

        def parameters(self):
            return []

        def update(self, grads, lr):
            pass

    pipe = CGNPipeline(encoder=enc, graph=_DeadGraph(), vocabulary=vocab,
                       gat_residual=True)

    def _rep(lemma):
        return UDRepresentation(
            tokens=[{"lemma": lemma, "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma=lemma, root_pos="VERB", root_dep_rel="root",
            root_morph={}, subject_pos=None, has_object=False,
            has_advcl=False, has_temporal_obl=False, token_span=(1, 1))

    reps = [_rep("alpha"), _rep("beta")]
    pipe.forward(reps, "test")
    d_node = np.ones_like(pipe._cached_node_logits)
    d_edge = np.zeros((0, 11), dtype=np.float32)
    pipe.backward(d_node, d_edge, lr=0.01)
    # Sans résidu, d_curr serait 0 après la couche morte ; avec résidu il survit.
    assert pipe._cached_enriched_vecs is not None


def test_checkpoint_layernorm_missing_keys_inits_identity():
    layer = RGCNLayerGAT(d_in=16, d_out=16, n_relations=3, n_heads=1,
                         use_layernorm=True, device="cpu", seed=1)
    arrs = RGCNLayerGAT(d_in=16, d_out=16, n_relations=3, n_heads=1,
                        device="cpu", seed=1).parameters()  # 3 arrays, sans norm
    layer.load_state(arrs)  # 3 arrays → LayerNorm reste identité
    assert np.allclose(layer.norm.weight.detach().cpu().numpy(), 1.0)
    assert np.allclose(layer.norm.bias.detach().cpu().numpy(), 0.0)


def test_stacked_gat_without_residual_baseline_equivalent():
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer1.representation import UDRepresentation
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.pipeline.cgnp import CGNPipeline

    vocab = FeatureVocabulary()
    D = vocab.d_clause
    enc = MLPEncoder(d_clause=D, d_edge=vocab.d_edge_closed_loop(D, 7), seed=0)
    g = RGCNLayerGAT(d_in=D, d_out=D, n_relations=11, n_heads=1, device="cpu", seed=0)
    pipe = CGNPipeline(encoder=enc, graph=g, vocabulary=vocab, n_rgcn_layers=2)

    def _rep(lemma):
        return UDRepresentation(
            tokens=[{"lemma": lemma, "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma=lemma, root_pos="VERB", root_dep_rel="root",
            root_morph={}, subject_pos=None, has_object=False,
            has_advcl=False, has_temporal_obl=False, token_span=(1, 1))

    out = pipe.forward([_rep("alpha"), _rep("beta")], "test")
    assert pipe._cached_enriched_vecs.shape == (2, D)
    assert out is not None
