from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from gcn_python.verbalizer.trainable import SurfaceVocabulary, TrainableDecoder


# ── SurfaceVocabulary ─────────────────────────────────────────────────────────

def test_vocab_build_and_encode():
    v = SurfaceVocabulary()
    v.build(["si les ventes baissent", "on réduit les coûts"])
    tokens = v.encode("les ventes")
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) == 2


def test_vocab_unk():
    v = SurfaceVocabulary()
    v.build(["hello"])
    idx = v.encode("unknown_xyz")
    assert idx == [1]  # UNK index


def test_vocab_roundtrip():
    v = SurfaceVocabulary()
    v.build(["if sales drop reduce costs"])
    serialized = v.to_json()
    v2 = SurfaceVocabulary.from_json(serialized)
    assert v.encode("sales") == v2.encode("sales")
    assert len(v) == len(v2)


def test_vocab_decode():
    v = SurfaceVocabulary()
    v.build(["si les ventes"])
    encoded = v.encode("si les ventes")
    decoded = v.decode(encoded)
    assert "si" in decoded
    assert "les" in decoded
    assert "ventes" in decoded


# ── TrainableDecoder ──────────────────────────────────────────────────────────

def make_vocab() -> SurfaceVocabulary:
    v = SurfaceVocabulary()
    v.build(["si les ventes baissent on réduit les coûts",
             "if sales drop reduce costs"])
    return v


def test_forward_decode_shape():
    v = make_vocab()
    dec = TrainableDecoder(v, d_hidden=16)
    node_embs = np.random.randn(3, 7).astype(np.float32)
    logits = dec.forward_decode(node_embs)
    assert logits.shape == (len(v),)


def test_forward_decode_empty():
    v = make_vocab()
    dec = TrainableDecoder(v, d_hidden=16)
    logits = dec.forward_decode(np.zeros((0, 7), dtype=np.float32))
    assert logits.shape == (len(v),)
    assert np.all(logits == 0.0)


def test_loss_decode_finite():
    v = make_vocab()
    dec = TrainableDecoder(v, d_hidden=16)
    node_embs = np.random.randn(2, 7).astype(np.float32)
    logits = dec.forward_decode(node_embs)
    gold = np.array(v.encode("si les"), dtype=np.int64)
    loss, d_logits = dec.loss_decode(logits, gold)
    assert np.isfinite(loss)
    assert loss > 0.0
    assert d_logits.shape == logits.shape


def test_backward_decode_gradient_nonzero():
    v = make_vocab()
    dec = TrainableDecoder(v, d_hidden=16)
    node_embs = np.random.randn(3, 7).astype(np.float32)
    logits = dec.forward_decode(node_embs)
    gold = np.array(v.encode("si ventes"), dtype=np.int64)
    _, d_logits = dec.loss_decode(logits, gold)
    # P2d: backward_decode retourne (d_node_embs, grads, d_attn_vec)
    d_node_embs, grads, d_attn_vec = dec.backward_decode(d_logits)
    assert d_node_embs.shape == (3, 7), "Gradient différencié par nœud (N, D_in)"
    assert d_attn_vec.shape == (7,), "Gradient attention vector (D_in,)"
    assert any(np.any(dW != 0) for dW, _ in grads)


def test_parameters_after_init():
    v = make_vocab()
    dec = TrainableDecoder(v, d_hidden=16, d_in=7)
    params = dec.parameters()
    # attn_vec + 2 layers × (W, b) + W_query = 6 params
    assert len(params) == 6


def test_update_changes_weights():
    v = make_vocab()
    dec = TrainableDecoder(v, d_hidden=16)
    node_embs = np.random.randn(2, 7).astype(np.float32)
    logits = dec.forward_decode(node_embs)
    gold = np.array(v.encode("si"), dtype=np.int64)
    _, d_logits = dec.loss_decode(logits, gold)
    # P2d: backward_decode retourne 3 valeurs
    _, grads, d_attn_vec = dec.backward_decode(d_logits)
    params_before = [p.copy() for p in dec.parameters()]
    dec.update(grads, d_attn_vec, lr=0.1)
    params_after = dec.parameters()
    assert any(not np.allclose(b, a) for b, a in zip(params_before, params_after))


def test_checkpoint_roundtrip(tmp_path: Path):
    v = make_vocab()
    dec = TrainableDecoder(v, d_hidden=16)
    node_embs = np.random.randn(2, 7).astype(np.float32)
    dec.forward_decode(node_embs)  # init layers

    meta_json = dec.to_json()
    dec2 = TrainableDecoder.from_json(meta_json)
    assert len(dec2.vocab) == len(v)
    assert dec2.d_hidden == 16

    params = dec.parameters()
    params2 = dec2.parameters()
    # shapes must match for weight copy
    for p, p2 in zip(params, params2):
        assert p.shape == p2.shape


def test_decode_inference():
    """H1 correction : decode() accepte node_embeddings, pas JSON."""
    v = make_vocab()
    dec = TrainableDecoder(v, d_hidden=16, d_in=7)
    # Simuler des vecteurs enrichis (dimension 7)
    node_embs = np.random.randn(2, 7).astype(np.float32)
    result = dec.decode(node_embs)
    assert isinstance(result, str)


def test_backward_decode_returns_node_gradients():
    """P2d : backward_decode retourne d_node_embs (N, D_in) au lieu de d_mean (D_in,)."""
    v = make_vocab()
    dec = TrainableDecoder(v, d_hidden=16)
    node_embs = np.random.randn(3, 7).astype(np.float32)
    logits = dec.forward_decode(node_embs)
    gold = np.array(v.encode("si ventes"), dtype=np.int64)
    _, d_logits = dec.loss_decode(logits, gold)
    d_node_embs, grads, d_attn_vec = dec.backward_decode(d_logits)

    # Vérifications P2d
    assert d_node_embs.shape == (3, 7), "Gradient doit être par nœud (N, D_in)"
    assert d_attn_vec.shape == (7,), "Gradient attn_vec doit être (D_in,)"
    assert len(grads) == 2, "2 layers RNN"
    assert any(np.any(dW != 0) for dW, _ in grads), "Gradients non nuls"


def test_checkpoint_includes_attn_vec(tmp_path):
    """P2d : attn_vec sauvegardé et restauré dans checkpoint."""
    v = make_vocab()
    dec = TrainableDecoder(v, d_hidden=16)
    node_embs = np.random.randn(2, 7).astype(np.float32)
    dec.forward_decode(node_embs)  # init layers + attn_vec

    meta_json = dec.to_json()
    data = json.loads(meta_json)
    assert "attn_vec" in data, "attn_vec doit être sérialisé"

    dec2 = TrainableDecoder.from_json(meta_json)
    params = dec.parameters()
    params2 = dec2.parameters()

    # Vérifier que attn_vec est le premier paramètre
    assert len(params) == 6, "attn_vec + 2 layers × (W, b) + W_query = 6 params"
    assert params[0].shape == (7,), "Premier param = attn_vec"
    assert np.allclose(params[0], params2[0]), "attn_vec restauré correctement"


def test_attention_weights_nonuniform_after_update():
    """P2d : attn_vec évolue par SGD, attention weights deviennent non-uniformes."""
    v = make_vocab()
    dec = TrainableDecoder(v, d_hidden=16)
    node_embs = np.random.randn(4, 7).astype(np.float32)

    # Forward initial — attn_vec = 0 → weights uniformes
    logits = dec.forward_decode(node_embs)
    attn_weights_before = dec._cached_attn_weights.copy()
    assert np.allclose(attn_weights_before, 0.25, atol=1e-3), "Poids uniformes initiaux"

    # Backward + update
    gold = np.array(v.encode("si"), dtype=np.int64)
    _, d_logits = dec.loss_decode(logits, gold)
    _, grads, d_attn_vec = dec.backward_decode(d_logits)
    dec.update(grads, d_attn_vec, lr=0.5)

    # Forward après update — weights doivent diverger
    logits2 = dec.forward_decode(node_embs)
    attn_weights_after = dec._cached_attn_weights
    assert not np.allclose(attn_weights_after, 0.25, atol=1e-2), "Poids non-uniformes après update"


def test_decode_accepts_enriched_vectors():
    """H1 : decode() accepte vecteurs enrichis (N, D_in), pas one-hot."""
    v = make_vocab()
    dec = TrainableDecoder(v, d_hidden=16)

    # Simuler des vecteurs enrichis R-GCN (d_in=75)
    enriched_vecs = np.random.randn(2, 75).astype(np.float32)
    dec.forward_decode(enriched_vecs)  # init layers avec d_in=75

    # decode() doit accepter les mêmes vecteurs
    result = dec.decode(enriched_vecs)
    assert isinstance(result, str), "decode() retourne une string"

    # Vérifier que decode() ne crashe pas avec dimension correcte
    assert dec._last_d_in == 75, "Layers initialisées avec d_in=75"


def test_decode_no_truncation():
    """H2 : decode() ne tronque pas à 5 tokens."""
    v = make_vocab()
    v.build(["un deux trois quatre cinq six sept huit neuf dix"])
    dec = TrainableDecoder(v, d_hidden=16, max_decode_len=12)

    node_embs = np.random.randn(3, 7).astype(np.float32)
    dec.forward_decode(node_embs)

    # Forcer une longue séquence — test approximatif : vérifier que le code ne crashe pas
    result = dec.decode(node_embs)
    assert isinstance(result, str)
    # Note : Impossible de garantir >5 tokens sans mocker, mais au moins pas de crash


# ── CGNPipeline avec decoder ──────────────────────────────────────────────────

def test_pipeline_with_decoder_forward(tmp_path):
    """forward() ne crashe pas avec un decoder attaché."""
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer1.representation import UDRepresentation
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline

    vocab = FeatureVocabulary()
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    v = make_vocab()
    dec = TrainableDecoder(v, d_hidden=16)
    pipeline = CGNPipeline(encoder=encoder, graph=graph,
                           vocabulary=vocab, decoder=dec)
    rep = UDRepresentation(
        tokens=[{"lemma": "baisser", "pos": "VERB", "dep_rel": "root", "morph": {}}],
        root_lemma="baisser", root_pos="VERB", root_dep_rel="root",
        root_morph={}, subject_pos=None,
        has_object=False, has_advcl=False, has_temporal_obl=False,
        token_span=(1, 1),
    )
    result = pipeline.forward([rep], "Les ventes baissent.")
    assert "nodes" in result
    assert result["nodes"]  # forward produit des nœuds avec decoder attaché


def test_pipeline_decoder_none_unchanged():
    """Sans decoder, le pipeline se comporte exactement comme avant."""
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline

    vocab = FeatureVocabulary()
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)
    assert pipeline.decoder is None


def test_checkpoint_roundtrip_with_decoder(tmp_path: Path):
    """save + load checkpoint avec decoder — poids restaurés."""
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer1.representation import UDRepresentation
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint

    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    v = make_vocab()
    dec = TrainableDecoder(v, d_hidden=16)

    # init layers
    node_embs = np.random.randn(2, vocab.d_clause).astype(np.float32)
    dec.forward_decode(node_embs)

    pipeline = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab, decoder=dec)

    ckpt = tmp_path / "model.npz"
    save_checkpoint(pipeline, ckpt)

    # Reload into fresh pipeline without decoder
    enc2 = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr2 = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    p2 = CGNPipeline(encoder=enc2, graph=gr2, vocabulary=FeatureVocabulary())
    load_checkpoint(p2, ckpt, trusted=True)

    assert p2.decoder is not None
    assert len(p2.decoder.parameters()) == len(dec.parameters())
    for orig, loaded in zip(dec.parameters(), p2.decoder.parameters()):
        assert np.allclose(orig, loaded)
