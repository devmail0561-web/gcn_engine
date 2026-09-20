# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests e2e verbalizer — enriched_vecs, engine.verbalize(), gradient cohérence, B3."""
from __future__ import annotations
import numpy as np
import pytest
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline
from gcn_python.verbalizer.trainable import SurfaceVocabulary, TrainableDecoder


def _make_pipeline_with_decoder():
    vocab = FeatureVocabulary()
    D = vocab.d_clause
    encoder = MLPEncoder(d_clause=D, d_edge=vocab.d_edge_closed_loop(D, 7), seed=0)
    graph = RGCNLayer(d_in=D, d_out=D, seed=0)
    sv = SurfaceVocabulary()
    sv.build(["parce que les prix baissent", "donc l effet augmente"])
    decoder = TrainableDecoder(sv, d_hidden=16, d_in=D, seed=0)
    return CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab, decoder=decoder)


def _make_rep(lemma="baisser"):
    from gcn_python.layer1.representation import UDRepresentation
    return UDRepresentation(
        tokens=[{"lemma": lemma, "pos": "VERB", "dep_rel": "root", "morph": {}}],
        root_lemma=lemma, root_pos="VERB", root_dep_rel="root",
        root_morph={}, subject_pos=None,
        has_object=False, has_advcl=False, has_temporal_obl=False,
        token_span=(0, 1),
    )


# --- Test 1 : node_labels dans le loader ---

def test_node_labels_in_loader():
    """VerbalizeSample.node_labels est list[str] de longueur N."""
    from pathlib import Path
    data_dir = Path(__file__).parent.parent.parent / "gcn-datasets" / "real" / "verbalize"
    if not data_dir.exists():
        pytest.skip("gcn-datasets/real/verbalize absent")
    from gcn_python.data.verbalize_loader import VerbalizerDataLoader
    loader = VerbalizerDataLoader(data_dir)
    samples = list(loader)
    assert len(samples) > 0
    for s in samples[:5]:
        assert isinstance(s.node_labels, list)
        assert all(isinstance(x, str) for x in s.node_labels)
        assert len(s.node_labels) == s.node_type_embeddings.shape[0]


# --- Test 2 : Path B utilise enriched_vecs (shape correcte) ---

def test_path_b_uses_enriched_vecs():
    """`_minimal_reps_from_labels` + pipeline.forward → enriched_vecs shape (N, d_clause)."""
    from gcn_python.training.train import _minimal_reps_from_labels
    pipeline = _make_pipeline_with_decoder()
    node_labels = ["les prix baissent", "la demande augmente"]
    node_types = ["processus", "processus"]
    reps = _minimal_reps_from_labels(node_labels, node_types)
    assert len(reps) == 2
    pipeline.forward(reps, "test")
    enriched = pipeline.get_enriched_vectors()
    assert enriched is not None
    assert enriched.shape == (2, pipeline.vocabulary.d_clause)
    logits = pipeline.decoder.forward_decode(enriched)
    assert logits.ndim >= 1


# --- Test 3 : cohérence d'échelle des gradients ---

def test_gradient_scale_coherence():
    """Après fix double norm, d_node_embs et grads doivent être à la même échelle."""
    sv = SurfaceVocabulary()
    sv.build(["a b c d"])
    dec = TrainableDecoder(sv, d_hidden=8, d_in=4, seed=0)
    rng = np.random.default_rng(42)
    vecs = rng.random((3, 4)).astype(np.float32)
    gold = np.array(dec.vocab.encode("a b c"), dtype=np.int64)

    logits = dec.forward_decode(vecs, gold)
    loss, d_logits = dec.loss_decode(logits, gold)
    d_node_embs, grads, d_attn = dec.backward_decode(d_logits)

    norm_dnode = float(np.linalg.norm(d_node_embs))
    norm_grad = float(np.linalg.norm(grads[0][0]))
    if norm_grad > 1e-9:
        ratio = norm_dnode / norm_grad
        assert ratio < 50.0, \
            f"d_node_embs et grads à des échelles très différentes (ratio={ratio:.1f})"


# --- Test 4 : decode() idempotent ---

def test_decode_idempotent():
    """Deux appels identiques à decode() retournent la même chaîne."""
    pipeline = _make_pipeline_with_decoder()
    reps = [_make_rep("baisser"), _make_rep("augmenter")]
    pipeline.forward(reps, "test")
    enriched = pipeline.get_enriched_vectors()
    assert enriched is not None
    result1 = pipeline.decoder.decode(enriched)
    result2 = pipeline.decoder.decode(enriched)
    assert result1 == result2, f"decode() non-idempotent : '{result1}' vs '{result2}'"


# --- Test 5 : engine.verbalize() avec décodeur ---

def test_engine_verbalize_neural():
    """engine.verbalize(use_neural=True) retourne {'cir': dict, 'text': str}."""
    from gcn_python.engine import GCNEngine
    from gcn_python.layer1.representation import UDRepresentation
    from gcn_python.frontend.bridge import GCNBridgeParser

    class _MockBridgeParser(GCNBridgeParser):
        def __init__(self):
            pass

        def parse(self, text):
            rep = UDRepresentation(
                tokens=[{"lemma": "baisser", "pos": "VERB", "dep_rel": "root", "morph": {}}],
                root_lemma="baisser", root_pos="VERB", root_dep_rel="root",
                root_morph={}, subject_pos=None,
                has_object=False, has_advcl=False, has_temporal_obl=False,
                token_span=(0, 1),
            )
            return [rep], []

    pipeline = _make_pipeline_with_decoder()
    engine = GCNEngine(pipeline, text_parser=_MockBridgeParser())
    result = engine.verbalize("test phrase", use_neural=True)
    assert isinstance(result, dict)
    assert "cir" in result and "text" in result
    assert isinstance(result["text"], str)
    assert "nodes" in result["cir"]


# --- Test 6 : engine.verbalize() repli template sans décodeur ---

def test_engine_verbalize_fallback():
    """Sans décodeur, engine.verbalize() utilise le template sans crash."""
    from gcn_python.engine import GCNEngine
    from gcn_python.layer1.representation import UDRepresentation

    class _MockBridgeParser:
        def parse(self, text):
            rep = UDRepresentation(
                tokens=[{"lemma": "baisser", "pos": "VERB", "dep_rel": "root", "morph": {}}],
                root_lemma="baisser", root_pos="VERB", root_dep_rel="root",
                root_morph={}, subject_pos=None,
                has_object=False, has_advcl=False, has_temporal_obl=False,
                token_span=(0, 1),
            )
            return [rep], []

    vocab = FeatureVocabulary()
    D = vocab.d_clause
    encoder = MLPEncoder(d_clause=D, d_edge=vocab.d_edge_closed_loop(D, 7), seed=0)
    graph = RGCNLayer(d_in=D, d_out=D, seed=0)
    pipeline_no_dec = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)
    engine = GCNEngine(pipeline_no_dec, text_parser=_MockBridgeParser())

    result = engine.verbalize("test", use_neural=True)
    assert isinstance(result, dict)
    assert "cir" in result and "text" in result
    assert isinstance(result["text"], str)


# --- Test 7 : B3 — backward_accumulate propage S11 ---

def test_backward_accumulate_s11():
    """backward_accumulate doit propager d_node_embs vers R-GCN (fix B3)."""
    pipeline = _make_pipeline_with_decoder()
    reps = [_make_rep("baisser"), _make_rep("augmenter")]
    pipeline.forward(reps, "test")
    nl = pipeline._cached_node_logits
    el = pipeline._cached_edge_logits
    gn = np.zeros(len(nl), dtype=np.int64)
    ge = np.zeros(len(el), dtype=np.int64) if el is not None else None

    sv_gold = np.array(pipeline.decoder.vocab.encode("parce que"), dtype=np.int64)
    _, dn, de = pipeline.loss(nl, el, gn, ge, gold_surface=sv_gold)

    graph = pipeline._graph_layers[0]
    wr_before = graph.W_r.copy()

    pipeline.backward_accumulate(dn, de)
    pipeline.apply_accumulated_gradients(lr=0.1)

    assert not np.allclose(graph.W_r, wr_before), \
        "backward_accumulate ne propage pas S11 vers R-GCN"
