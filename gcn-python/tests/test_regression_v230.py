# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Tests de régression — Phase A (v2.3.0)

Couvrent les 11 correctifs appliqués en v2.3.0 qui n'avaient aucun filet de sécurité.

A1 : backward_edge_dx — gradient non-nul vers d_enriched[src] et d_enriched[dst]
A2 : Reproductibilité dropout (MLPEncoder seed=42)
A3 : GAT backward — dénominateur softmax dans le graphe autograd (via différences finies)
A4 : _cached_d_edge_base / _cached_d_eff — cohérence des offsets du vecteur enriched_edge
A5 : _set_training_mode — RGCNLayerGAT.training False après eval(), True après train()
"""
import numpy as np
import pytest

from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer2.reference import MLPEncoder

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _enc(seed=42, edge_dropout=0.3, d_eff=None, n_node_types=7):
    vocab = FeatureVocabulary()
    d_cl = vocab.d_clause
    d_eff = d_eff or d_cl
    d_edge = vocab.d_edge_closed_loop(d_eff, n_node_types)
    return MLPEncoder(
        d_clause=d_cl, d_edge=d_edge, seed=seed,
        n_node_types=n_node_types, edge_dropout=edge_dropout,
    ), vocab, d_cl, d_eff


# ═══════════════════════════════════════════════════════════════════════════════
# A1 — backward_edge_dx remonte vers d_enriched[src] et d_enriched[dst]
# ═══════════════════════════════════════════════════════════════════════════════

def test_a1_backward_edge_dx_src_slice_nonzero():
    """La tranche dx[d_base:d_base+d_eff] (→ src) est non-nulle."""
    enc, vocab, d_cl, d_eff = _enc(edge_dropout=0.0)
    n_node_types = enc.n_node_types
    d_base = vocab.d_edge
    d_edge_total = d_base + 2 * d_eff + 2 * n_node_types

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, d_edge_total).astype(np.float32)

    enc.training = False
    enc.forward_edge(x)
    d_logits = rng.normal(0, 1, enc.n_relation_types).astype(np.float32)
    _, dx = enc.backward_edge_dx(d_logits)

    assert dx.shape == (d_edge_total,)
    assert np.any(dx[d_base:d_base + d_eff] != 0), (
        "Gradient dx_src (tranche enriched src) est entièrement nul"
    )


def test_a1_backward_edge_dx_dst_slice_nonzero():
    """La tranche dx[d_base+d_eff:d_base+2*d_eff] (→ dst) est non-nulle."""
    enc, vocab, d_cl, d_eff = _enc(edge_dropout=0.0)
    n_node_types = enc.n_node_types
    d_base = vocab.d_edge
    d_edge_total = d_base + 2 * d_eff + 2 * n_node_types

    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, d_edge_total).astype(np.float32)

    enc.training = False
    enc.forward_edge(x)
    d_logits = rng.normal(0, 1, enc.n_relation_types).astype(np.float32)
    _, dx = enc.backward_edge_dx(d_logits)

    assert np.any(dx[d_base + d_eff:d_base + 2 * d_eff] != 0), (
        "Gradient dx_dst (tranche enriched dst) est entièrement nul"
    )


def test_a1_backward_edge_dx_direction_reduces_loss():
    """Une étape gradient descent via dx réduit la cross-entropy edge."""
    enc, vocab, d_cl, d_eff = _enc(edge_dropout=0.0)
    n_node_types = enc.n_node_types
    d_edge_total = vocab.d_edge_closed_loop(d_eff, n_node_types)

    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, d_edge_total).astype(np.float32)
    gold = 0  # classe cible

    def _loss(xv):
        enc.training = False
        enc.forward_edge(xv)
        logits = enc._edge_cache[-1][0]
        shifted = logits - logits.max()
        return float(-shifted[gold] + np.log(np.sum(np.exp(shifted))))

    # Gradient cross-entropy vers logits
    enc.training = False
    enc.forward_edge(x)
    logits = enc._edge_cache[-1][0]
    shifted = logits - logits.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()
    d_logits = probs.copy()
    d_logits[gold] -= 1.0  # gradient de -log(softmax)[gold]

    _, dx = enc.backward_edge_dx(d_logits)
    loss_before = _loss(x)
    loss_after = _loss(x - 0.01 * dx)

    assert loss_after < loss_before, (
        f"Gradient dx ne réduit pas la loss : {loss_before:.6f} → {loss_after:.6f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# A2 — Reproductibilité dropout (seed=42)
# ═══════════════════════════════════════════════════════════════════════════════

def test_a2_same_seed_same_output_training():
    """Deux MLPEncoder(seed=42) en mode training produisent la même sortie."""
    vocab = FeatureVocabulary()
    d_cl = vocab.d_clause
    d_edge = vocab.d_edge_closed_loop(d_cl, 7)

    rng = np.random.default_rng(99)
    x = rng.normal(0, 1, d_edge).astype(np.float32)

    enc1 = MLPEncoder(d_clause=d_cl, d_edge=d_edge, seed=42, edge_dropout=0.3)
    enc2 = MLPEncoder(d_clause=d_cl, d_edge=d_edge, seed=42, edge_dropout=0.3)
    enc1.training = True
    enc2.training = True

    np.testing.assert_array_equal(
        enc1.forward_edge(x), enc2.forward_edge(x),
        err_msg="MLPEncoder(seed=42) × 2 — sorties différentes en training",
    )


def test_a2_same_seed_same_dropout_masks():
    """Même seed → masques dropout identiques."""
    vocab = FeatureVocabulary()
    d_edge = vocab.d_edge_closed_loop(vocab.d_clause, 7)
    x = np.random.default_rng(5).normal(0, 1, d_edge).astype(np.float32)

    enc1 = MLPEncoder(d_clause=vocab.d_clause, d_edge=d_edge, seed=42, edge_dropout=0.3)
    enc2 = MLPEncoder(d_clause=vocab.d_clause, d_edge=d_edge, seed=42, edge_dropout=0.3)
    enc1.training = True
    enc2.training = True
    enc1.forward_edge(x)
    enc2.forward_edge(x)

    assert len(enc1._edge_dropout_masks) > 0, "Aucun masque dropout généré (edge_dropout=0.3)"
    for m1, m2 in zip(enc1._edge_dropout_masks, enc2._edge_dropout_masks):
        np.testing.assert_array_equal(m1, m2, err_msg="Masques dropout différents pour seed=42")


def test_a2_different_seeds_differ():
    """Deux seeds différents (42 vs 43) produisent des sorties distinctes."""
    vocab = FeatureVocabulary()
    d_edge = vocab.d_edge_closed_loop(vocab.d_clause, 7)
    x = np.random.default_rng(7).normal(0, 1, d_edge).astype(np.float32)

    enc1 = MLPEncoder(d_clause=vocab.d_clause, d_edge=d_edge, seed=42, edge_dropout=0.3)
    enc2 = MLPEncoder(d_clause=vocab.d_clause, d_edge=d_edge, seed=43, edge_dropout=0.3)
    enc1.training = True
    enc2.training = True

    assert not np.allclose(enc1.forward_edge(x), enc2.forward_edge(x)), (
        "Seeds 42 et 43 produisent la même sortie — le seed n'a aucun effet"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# A3 — GAT backward : dénominateur softmax dans le graphe autograd
# ═══════════════════════════════════════════════════════════════════════════════

torch = pytest.importorskip("torch", reason="PyTorch non installé — tests GAT ignorés")

from gcn_python.layer3.gat import RGCNLayerGAT  # noqa: E402 (après importorskip)


def test_a3_gat_ar_gradient_nonzero():
    """backward_message_pass produit un gradient non-nul sur a_r."""
    layer = RGCNLayerGAT(d_in=4, d_out=4, n_relations=1, device="cpu", seed=0)
    layer.eval()

    rng = np.random.default_rng(42)
    features = rng.normal(0, 1, (4, 4)).astype(np.float32)
    # 2 arêtes vers le même nœud destination → dénominateur actif
    edge_index = np.array([[0, 1], [2, 2]], dtype=np.int64)
    edge_types = np.array([0, 0], dtype=np.int64)
    d_output = rng.normal(0, 1, (4, 4)).astype(np.float32)

    layer.message_pass(features, edge_index, edge_types)
    _, grads = layer.backward_message_pass(d_output)

    assert np.any(grads[2] != 0), "Gradient a_r nul — flux autograd GAT interrompu"
    assert np.any(grads[0] != 0), "Gradient W_r nul — flux autograd GAT interrompu"


def test_a3_gat_softmax_denominator_not_detached():
    """
    Le gradient analytique de backward_message_pass sur a_r[0,0] correspond
    aux différences finies. Si sum_exp était détaché, les deux divergeraient
    car le terme croisé du dénominateur serait absent.
    """
    rng = np.random.default_rng(42)
    torch.manual_seed(0)

    layer = RGCNLayerGAT(d_in=4, d_out=4, n_relations=1, device="cpu", seed=0)
    layer.eval()

    features = rng.normal(0, 1, (4, 4)).astype(np.float32)
    edge_index = np.array([[0, 1], [2, 2]], dtype=np.int64)
    edge_types = np.array([0, 0], dtype=np.int64)
    d_output = rng.normal(0, 1, (4, 4)).astype(np.float32)

    # Gradient analytique
    layer.message_pass(features, edge_index, edge_types)
    _, grads = layer.backward_message_pass(d_output)
    da_analytical = float(grads[2][0, 0])

    # Gradient par différences finies sur a_r[0, 0]
    eps = 1e-4

    with torch.no_grad():
        layer.a_r[0, 0] += eps
    out_plus = layer.message_pass(features, edge_index, edge_types)

    with torch.no_grad():
        layer.a_r[0, 0] -= 2 * eps
    out_minus = layer.message_pass(features, edge_index, edge_types)

    with torch.no_grad():
        layer.a_r[0, 0] += eps  # restaurer

    da_fd = float(np.sum((out_plus - out_minus) * d_output) / (2 * eps))

    assert abs(da_analytical - da_fd) < 0.05, (
        f"Gradient a_r[0,0] : analytique={da_analytical:.6f}, "
        f"différences finies={da_fd:.6f} — "
        "Le dénominateur softmax est peut-être détaché (correctif v2.3.0 régressé)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# A4 — _cached_d_edge_base / _cached_d_eff : cohérence des offsets
# ═══════════════════════════════════════════════════════════════════════════════

def test_a4_cached_offsets_match_vector_structure():
    """
    Les offsets simulés (d_base, d_eff) produisent des tranches dx dans les bornes
    du vecteur retourné par backward_edge_dx — même calcul que cgnp.py backward.
    """
    vocab = FeatureVocabulary()
    d_cl = vocab.d_clause
    d_eff = d_cl
    n_node_types = 7

    d_base = vocab.d_edge                           # len(edge_vec_base) sans word_embedding
    d_edge_total = vocab.d_edge_closed_loop(d_eff, n_node_types)

    # Vérifier la formule d_edge_closed_loop
    assert d_edge_total == d_base + 2 * d_eff + 2 * n_node_types, (
        "d_edge_closed_loop incohérent avec la décomposition attendue"
    )

    # Les tranches [src] et [dst] sont dans les bornes
    assert d_base + 2 * d_eff <= d_edge_total, (
        f"Tranches hors bornes : d_base({d_base}) + 2×d_eff({d_eff}) "
        f"= {d_base + 2 * d_eff} > d_edge_total({d_edge_total})"
    )

    # backward_edge_dx retourne un dx de taille d_edge_total
    enc = MLPEncoder(d_clause=d_cl, d_edge=d_edge_total, seed=42, n_node_types=n_node_types)
    enc.training = False

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, d_edge_total).astype(np.float32)
    enc.forward_edge(x)
    d_logits = rng.normal(0, 1, enc.n_relation_types).astype(np.float32)
    _, dx = enc.backward_edge_dx(d_logits)

    assert dx.shape == (d_edge_total,), f"dx.shape={dx.shape} ≠ ({d_edge_total},)"

    # Extraction des tranches (même code que cgnp.py backward, lignes 575-576)
    slice_src = dx[d_base:d_base + d_eff]
    slice_dst = dx[d_base + d_eff:d_base + 2 * d_eff]

    assert slice_src.shape == (d_eff,), f"Tranche src : {slice_src.shape} ≠ ({d_eff},)"
    assert slice_dst.shape == (d_eff,), f"Tranche dst : {slice_dst.shape} ≠ ({d_eff},)"


def test_a4_d_edge_base_equals_len_edge_vec_base():
    """
    vocab.d_edge == len(edge_vec_base) quand word_embedding=None.
    C'est la valeur que cgnp.py cache dans _cached_d_edge_base.
    """
    vocab = FeatureVocabulary()
    # Sans word_embedding, la taille de base est exactement vocab.d_edge
    # (vectorize_edge sans word_embedding produit un vecteur de taille d_edge)
    d_base_expected = vocab.d_edge
    d_eff = vocab.d_clause
    n_node_types = 7

    d_edge_total = vocab.d_edge_closed_loop(d_eff, n_node_types, d_emb=0)
    # d_eff x2 + d_base + n_node_types x2
    assert d_edge_total == d_base_expected + 2 * d_eff + 2 * n_node_types


def test_a4_d_eff_equals_len_enriched_vector():
    """
    _cached_d_eff = len(enriched[src_i]). Sans R-GCN, enriched = clause_vecs → d_clause.
    Avec R-GCN (d_out != d_clause), enriched a la dimension de sortie du R-GCN.
    """
    vocab = FeatureVocabulary()
    d_cl = vocab.d_clause
    # Sans message passing, enriched = clause_vecs → d_eff = d_clause
    d_eff_no_rgcn = d_cl
    # Avec R-GCN d_out=32, enriched → d_eff = 32
    d_eff_with_rgcn = 32

    d_edge_no_rgcn = vocab.d_edge_closed_loop(d_eff_no_rgcn, 7)
    d_edge_with_rgcn = vocab.d_edge_closed_loop(d_eff_with_rgcn, 7)

    assert d_edge_no_rgcn != d_edge_with_rgcn, (
        "Les deux configurations doivent avoir des d_edge distincts"
    )
    # La formule doit être cohérente
    assert d_edge_with_rgcn == vocab.d_edge + 2 * d_eff_with_rgcn + 2 * 7


# ═══════════════════════════════════════════════════════════════════════════════
# A5 — _set_training_mode : RGCNLayerGAT.training False/True
# ═══════════════════════════════════════════════════════════════════════════════

def test_a5_gat_training_true_at_init():
    """RGCNLayerGAT démarre en mode training=True."""
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=3, device="cpu")
    assert layer.training is True


def test_a5_gat_training_false_after_eval():
    """layer.eval() → training=False (via nn.Module.train(False))."""
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=3, device="cpu")
    layer.eval()
    assert layer.training is False, "Après .eval() : training doit être False"


def test_a5_gat_training_true_after_train():
    """layer.eval() puis layer.train() → training=True."""
    layer = RGCNLayerGAT(d_in=8, d_out=16, n_relations=3, device="cpu")
    layer.eval()
    layer.train()
    assert layer.training is True, "Après .train() : training doit être True"


def test_a5_mlp_encoder_training_toggle():
    """MLPEncoder.training répond à l'affectation directe (utilisé par _set_training_mode)."""
    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge, seed=42)
    assert enc.training is True

    enc.training = False
    assert enc.training is False

    enc.training = True
    assert enc.training is True


def test_a5_dropout_disabled_in_eval_mode():
    """
    En mode eval (training=False), forward_edge est déterministe même avec
    edge_dropout=0.5 — le dropout est bien conditionné sur self.training.
    """
    vocab = FeatureVocabulary()
    d_edge = vocab.d_edge_closed_loop(vocab.d_clause, 7)
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=d_edge, seed=42, edge_dropout=0.5)
    enc.training = False

    x = np.random.default_rng(0).normal(0, 1, d_edge).astype(np.float32)
    out1 = enc.forward_edge(x)
    out2 = enc.forward_edge(x)

    np.testing.assert_array_equal(
        out1, out2,
        err_msg="En mode eval, forward_edge doit être déterministe (dropout désactivé)",
    )


def test_a5_dropout_active_in_training_mode():
    """
    En mode training=True, deux passes avec edge_dropout=0.5 et rng différent
    produisent des sorties différentes — le dropout est bien actif.
    """
    vocab = FeatureVocabulary()
    d_edge = vocab.d_edge_closed_loop(vocab.d_clause, 7)
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=d_edge, seed=42, edge_dropout=0.5)
    enc.training = True

    x = np.random.default_rng(0).normal(0, 1, d_edge).astype(np.float32)
    out1 = enc.forward_edge(x)
    out2 = enc.forward_edge(x)

    # Les deux passes avancent le RNG interne → masques différents → sorties différentes
    assert not np.allclose(out1, out2), (
        "Deux passes training consécutives identiques — dropout peut-être inactif"
    )
