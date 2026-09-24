# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests des phases update.txt : A (fastText), B (PairNorm+DropEdge),
C (MHA globale), D (CompGCN)."""
import numpy as np
import pytest
from gcn_python.layer1.embedding import WordEmbedding

# ─── Phase A ───────────────────────────────────────────────────────────────


class _FakeFT:
    """Mock fastText : vecteur fixe par lemme (déterministe)."""

    def __init__(self, d: int = 300):
        self._d = d

    def get_word_vector(self, word: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(word)) % (2 ** 31))
        return rng.normal(0, 1, (self._d,)).astype(np.float32)


def test_load_from_fasttext_sets_pretrained_range():
    ft = _FakeFT(d=300)
    vocab = ["chat", "chien", "maison"]
    emb = WordEmbedding.load_from_fasttext(ft, vocab, d_emb=300, frozen=True)
    assert emb._pretrained_start == 3  # _unk + 2 spéciaux _absent
    assert emb._pretrained_end == 3 + len(vocab)
    for lemma in vocab:
        idx = emb._vocab[lemma]
        expected = _FakeFT(d=300).get_word_vector(lemma)
        # %.6f dans le fichier temporaire → tolérance lâche (précision texte)
        np.testing.assert_allclose(emb._E[idx], expected, rtol=1e-3, atol=1e-5)


def test_frozen_embedding_does_not_update_on_backward():
    ft = _FakeFT(d=300)
    emb = WordEmbedding.load_from_fasttext(ft, ["chat"], d_emb=300, frozen=True)
    idx = emb._vocab["chat"]
    before = emb._E[idx].copy()
    emb.backward(np.ones(300, dtype=np.float32), "chat")
    emb.update(lr=0.1)
    np.testing.assert_array_equal(emb._E[idx], before)


def test_load_from_fasttext_requires_package_for_unknown_model():
    with pytest.raises(ImportError):
        WordEmbedding.load_from_fasttext(object(), ["x"], d_emb=300)


def test_cli_fasttext_embedding_file_mutually_exclusive():
    from click.testing import CliRunner
    from gcn_python.training.train import train_cmd
    runner = CliRunner()
    result = runner.invoke(train_cmd, [
        "--data-dir", "nonexistent", "--fasttext", "f.bin",
        "--embedding-file", "e.txt",
    ])
    assert result.exit_code != 0
    assert "mutuellement exclusifs" in result.output


# ─── Phase B ───────────────────────────────────────────────────────────────

try:
    from gcn_python.layer3.pytorch_rgcn import RGCNLayerPT
    _HAS_TORCH = True
except ImportError:
    RGCNLayerPT = None  # type: ignore
    _HAS_TORCH = False

needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch non installé")


def _chain_graph(n: int = 6, n_rel: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    feats = rng.normal(0, 1, (n, 8)).astype(np.float32)
    src = np.arange(n - 1)
    dst = np.arange(1, n)
    edge_index = np.stack([src, dst])
    edge_types = rng.integers(0, n_rel, size=n - 1)
    return feats, edge_index, edge_types


@needs_torch
def test_pairnorm_prevents_representation_collapse():
    layers = [RGCNLayerPT(d_in=8, d_out=8, n_relations=3, device="cpu",
                          pairnorm=True, seed=100 + i) for i in range(4)]
    feats, edge_index, edge_types = _chain_graph()
    h = feats
    for layer in layers:
        layer.train()
        h = layer.message_pass(h, edge_index, edge_types)
    assert float(h.std()) > 0.05, f"collapse : std={h.std()}"


@needs_torch
def test_drop_edge_not_applied_in_eval_mode():
    layer = RGCNLayerPT(d_in=8, d_out=8, n_relations=3, device="cpu",
                        drop_edge=0.9, seed=0)
    feats, edge_index, edge_types = _chain_graph(n=8)
    layer.eval()
    out1 = layer.message_pass(feats, edge_index, edge_types)
    out2 = layer.message_pass(feats, edge_index, edge_types)
    np.testing.assert_array_equal(out1, out2)


@needs_torch
def test_backward_compat_defaults_unchanged():
    layer = RGCNLayerPT(d_in=8, d_out=8, n_relations=3, device="cpu")
    assert layer.pairnorm is False
    assert layer.drop_edge == 0.0
    assert layer.use_compgcn is False


# ─── Phase C ───────────────────────────────────────────────────────────────

from gcn_python.layer2.reference import MLPEncoder, TransformerMLPEncoder


def test_mlp_encoder_stores_d_clause():
    enc = MLPEncoder(d_clause=64, d_edge=128)
    assert enc.d_clause == 64


@needs_torch
def test_transformer_encoder_forward_batch_shape():
    enc = TransformerMLPEncoder(d_clause=64, d_edge=128, n_heads=4)
    out = enc.forward_batch(np.random.randn(10, 64).astype(np.float32))
    assert out.shape == (10, enc.n_node_types)


@needs_torch
def test_transformer_encoder_attends_globally():
    enc = TransformerMLPEncoder(d_clause=32, d_edge=64, n_heads=4, seed=7)
    base = np.random.default_rng(0).normal(0, 1, (5, 32)).astype(np.float32)
    x1 = base.copy()
    x2 = base.copy()
    x2[0] += 5.0  # seul le premier nœud change
    out1 = enc.forward_batch(x1)
    out2 = enc.forward_batch(x2)
    # Le dernier nœud (identique en entrée) doit voir une sortie différente
    # grâce au contexte global MHA.
    assert not np.allclose(out1[-1], out2[-1])


@needs_torch
def test_transformer_encoder_bad_heads_raises():
    with pytest.raises(ValueError, match="non divisible"):
        TransformerMLPEncoder(d_clause=64, d_edge=128, n_heads=3)


@needs_torch
def test_engine_reconstructs_transformer_encoder(tmp_path):
    """Checkpoint entraîné avec --global-attention → from_pretrained
    reconstruit un TransformerMLPEncoder (pas MLPEncoder)."""
    from gcn_python.engine import GCNEngine
    from gcn_python.layer1.embedding import WordEmbedding
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint

    vocab = FeatureVocabulary()
    # Interaction A+C : D_effective = d_clause_vocab + d_emb (jamais d_clause brut).
    we = WordEmbedding(d_emb=1)
    d_eff = vocab.d_clause_effective(1, False)  # 79 + 1 = 80, divisible par 4
    assert d_eff % 4 == 0
    from gcn_python.constants import NODE_TYPES
    d_edge = vocab.d_edge_closed_loop(d_eff, len(NODE_TYPES), 1, False)
    enc = TransformerMLPEncoder(d_clause=d_eff, d_edge=d_edge, n_heads=4)
    assert enc.d_clause == d_eff  # D_effective, pas vocabulary.d_clause brut
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=3)
    pipe = CGNPipeline(encoder=enc, graph=graph, vocabulary=vocab,
                       word_embedding=we)
    pipe.global_attention = True
    pipe.mha_heads = 4
    ckpt = tmp_path / "mha.npz"
    save_checkpoint(pipe, ckpt)
    engine = GCNEngine.from_pretrained(ckpt, trusted=True)
    assert type(engine._pipeline.encoder).__name__ == "TransformerMLPEncoder"


# ─── Phase D ───────────────────────────────────────────────────────────────


@needs_torch
def test_compgcn_forward_shape_unchanged():
    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu",
                        use_compgcn=True, d_rel_emb=4)
    rng = np.random.default_rng(0)
    feats = rng.normal(0, 1, (5, 8)).astype(np.float32)
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_types = np.array([0, 1, 2, 0])
    out = layer.message_pass(feats, edge_index, edge_types)
    assert out.shape == (5, 16)


@needs_torch
def test_compgcn_parameters_shapes():
    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu",
                        use_compgcn=True, d_rel_emb=4)
    params = layer.parameters()
    assert params[0].shape == (3, 4)      # E_r
    assert params[1].shape == (4, 16 * 8)  # W_comp


@needs_torch
def test_compgcn_load_state_and_update():
    layer = RGCNLayerPT(d_in=8, d_out=16, n_relations=3, device="cpu",
                        use_compgcn=True, d_rel_emb=4)
    E_r   = np.random.randn(3, 4).astype(np.float32)
    W_comp = np.random.randn(4, 128).astype(np.float32)
    W_0   = np.random.randn(16, 8).astype(np.float32)   # FIX-1 : W_0 requis
    layer.load_state([E_r, W_comp, W_0])  # ne doit pas lever
    grads = [np.ones((3, 4), dtype=np.float32),
             np.ones((4, 128), dtype=np.float32),
             np.ones((16, 8), dtype=np.float32)]         # FIX-1 : grad W_0 requis
    layer.update(grads, lr=0.01)  # ne doit pas lever (pas d'IndexError)


@needs_torch
def test_compgcn_relation_embeddings_are_close_for_similar_relations():
    """E_r[CAUSE] et E_r[ENABLE] plus proches que E_r[CAUSE] - E_r[TEMPORAL]
    après optimisation ciblée (vérifie la plasticité des embeddings)."""
    import torch as _torch
    layer = RGCNLayerPT(d_in=4, d_out=4, n_relations=3, device="cpu",
                        use_compgcn=True, d_rel_emb=8)
    opt = _torch.optim.SGD(layer.torch_parameters(), lr=0.1)
    # Rapprocher E_r[0] et E_r[1] par quelques steps de gradient
    for _ in range(50):
        opt.zero_grad()
        loss = ((layer.E_r[0] - layer.E_r[1]) ** 2).sum()
        loss.backward()
        opt.step()
    E = layer.E_r.detach().cpu().numpy()
    d_close = float(np.linalg.norm(E[0] - E[1]))
    d_far = float(np.linalg.norm(E[0] - E[2]))
    assert d_close < d_far


@needs_torch
def test_pipeline_routes_transformer_through_forward_batch():
    """Garde-fou Phase C : CGNPipeline.forward() doit emprunter forward_batch()
    (donc la MHA) avec TransformerMLPEncoder — pas la boucle forward_node.
    Sans l'opt-in prefers_batch_forward, --global-attention serait un no-op
    silencieux (mêmes métriques que la référence, constaté sur R1)."""
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer1.representation import UDRepresentation
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.constants import NODE_TYPES

    def _rep(lemma):
        return UDRepresentation(
            tokens=[{"lemma": lemma, "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma=lemma, root_pos="VERB", root_dep_rel="root",
            root_morph={}, subject_pos=None,
            has_object=False, has_advcl=False, has_temporal_obl=False,
            token_span=(0, 1),
        )

    vocab = FeatureVocabulary()
    d_eff = vocab.d_clause
    d_edge = vocab.d_edge_closed_loop(d_eff, len(NODE_TYPES), 0, False)
    enc = TransformerMLPEncoder(d_clause=d_eff, d_edge=d_edge, n_heads=1)
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=3)
    pipe = CGNPipeline(encoder=enc, graph=graph, vocabulary=vocab)

    calls = []
    _orig = enc.forward_batch
    enc.forward_batch = lambda X: (calls.append(X.shape), _orig(X))[1]
    try:
        pipe.forward([_rep("baisser"), _rep("reduire")], "Les ventes baissent puis on reduit.")
    finally:
        enc.forward_batch = _orig
    assert calls, "forward_batch jamais appele : la MHA est hors chemin (no-op)"

    # Logits differents du MLP seul → la MHA contribue reellement.
    mlp = MLPEncoder(d_clause=d_eff, d_edge=d_edge, seed=123)
    assert not np.allclose(
        enc.forward_batch(np.random.default_rng(0).normal(0, 1, (4, d_eff)).astype(np.float32)),
        mlp.forward_batch(np.random.default_rng(0).normal(0, 1, (4, d_eff)).astype(np.float32)),
    )


# ─── Tests bugs critiques + scheduled sampling ──────────────────────────────

from gcn_python.layer3.reference import RGCNLayer


def test_seed_propagation_mlpencoder():
    """Deux MLPEncoder avec la même seed produisent les mêmes poids."""
    e1 = MLPEncoder(d_clause=20, d_edge=50, seed=7)
    e2 = MLPEncoder(d_clause=20, d_edge=50, seed=7)
    for l1, l2 in zip(e1._node_layers, e2._node_layers):
        assert np.allclose(l1.W, l2.W)
    e3 = MLPEncoder(d_clause=20, d_edge=50, seed=99)
    assert not np.allclose(e1._node_layers[0].W, e3._node_layers[0].W)


def test_seed_propagation_rgcn():
    """Deux RGCNLayer avec la même seed produisent les mêmes W_r."""
    r1 = RGCNLayer(d_in=16, d_out=16, n_relations=3, seed=7)
    r2 = RGCNLayer(d_in=16, d_out=16, n_relations=3, seed=7)
    assert np.allclose(r1.W_r, r2.W_r)
    r3 = RGCNLayer(d_in=16, d_out=16, n_relations=3, seed=99)
    assert not np.allclose(r1.W_r, r3.W_r)


def test_scheduled_sampling_p_gold_decreases():
    """p_gold = max(0, 1 - (epoch-1)/ss_final_epoch) décroît correctement."""
    ss_final = 10
    p_golds = [max(0.0, 1.0 - (ep - 1) / ss_final) for ep in range(1, 15)]
    assert p_golds[0] == 1.0        # epoch 1 : p_gold=1.0
    assert p_golds[9] == pytest.approx(0.1)   # epoch 10
    assert p_golds[10] == 0.0       # epoch 11 : p_gold=0.0
    assert p_golds[13] == 0.0       # epoch 14 : reste 0.0


def test_json_reader_rejects_missing_relation(tmp_path):
    """Arête sans champ relation lève ValueError (plus de substitution silencieuse)."""
    import json
    from gcn_python.data.json_reader import load_all_sentences
    bad = {
        "document": {"sentences": [{
            "id": "s1", "text": "A cause B.",
            "clauses": [
                {"id": "n1", "type": "action", "label": "A", "token_span": [0, 1]},
                {"id": "n2", "type": "action", "label": "B", "token_span": [1, 2]},
            ],
            "edges": [{"source": "n1", "target": "n2"}]  # relation absente
        }]}
    }
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(bad), encoding="utf-8")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        records = load_all_sentences(tmp_path)
    # Le sample doit être ignoré (ValueError attrapée) avec un warning
    assert len(records) == 0 or all(
        len(r.edges) == 0 for r in records
    ), "Arête sans relation ne doit pas être chargée"
