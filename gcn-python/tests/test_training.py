"""Tests de la boucle d'entraînement Phase 2b."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from gcn_python.constants import NODE_TYPES, RELATION_TYPES
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline, _cross_entropy


# ---------------------------------------------------------------------------
# Fixture : pipeline minimal sans spaCy
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline() -> CGNPipeline:
    vocab = FeatureVocabulary()
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge, seed=0)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause, seed=0)
    return CGNPipeline(encoder=encoder, graph=graph, lang="fr", vocabulary=vocab)


# ---------------------------------------------------------------------------
# Tests loss — cross-entropie
# ---------------------------------------------------------------------------

def test_loss_cross_entropy_returns_float_and_two_arrays():
    logits = np.random.randn(3, 7).astype(np.float32)
    labels = np.array([0, 2, 5], dtype=np.int64)
    loss, d = _cross_entropy(logits, labels)
    assert isinstance(loss, float)
    assert d.shape == (3, 7)


def test_loss_gradient_shape():
    n_nodes, n_edges = 4, 3
    node_logits = np.random.randn(n_nodes, len(NODE_TYPES)).astype(np.float32)
    edge_logits = np.random.randn(n_edges, len(RELATION_TYPES)).astype(np.float32)
    gold_node = np.zeros(n_nodes, dtype=np.int64)
    gold_edge = np.zeros(n_edges, dtype=np.int64)

    node_loss, d_node = _cross_entropy(node_logits, gold_node)
    edge_loss, d_edge = _cross_entropy(edge_logits, gold_edge)

    assert d_node.shape == (n_nodes, len(NODE_TYPES))
    assert d_edge.shape == (n_edges, len(RELATION_TYPES))
    assert isinstance(node_loss + edge_loss, float)


def test_loss_gradient_sums_near_zero():
    """Le gradient cross-entropie sum ≈ 0 (softmax - one_hot, moyenné)."""
    logits = np.random.randn(5, 7).astype(np.float32)
    labels = np.array([1, 0, 3, 6, 2], dtype=np.int64)
    _, d = _cross_entropy(logits, labels)
    # Somme des gradients ≈ 0 (la softmax somme à 1, one_hot somme à 1)
    assert abs(d.sum()) < 1e-5


# ---------------------------------------------------------------------------
# Tests RGCNLayer backward
# ---------------------------------------------------------------------------

def test_rgcn_backward_shape():
    vocab = FeatureVocabulary()
    D = vocab.d_clause
    N, E = 4, 3
    rng = np.random.default_rng(42)

    graph = RGCNLayer(d_in=D, d_out=D, seed=0)
    node_features = rng.random((N, D)).astype(np.float32)
    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    edge_types = np.array([0, 1, 0], dtype=np.int64)

    _ = graph.message_pass(node_features, edge_index, edge_types)
    d_output = rng.random((N, D)).astype(np.float32)
    d_input, grads = graph.backward_message_pass(d_output)

    assert d_input.shape == (N, D)
    assert grads[0].shape == graph.W_r.shape   # (n_relations, D_out, D_in)
    assert grads[1].shape == graph.W_0.shape   # (D_out, D_in)


def test_rgcn_update_changes_weights():
    vocab = FeatureVocabulary()
    D = vocab.d_clause
    rng = np.random.default_rng(1)

    graph = RGCNLayer(d_in=D, d_out=D, seed=1)
    W_r_before = graph.W_r.copy()
    W_0_before = graph.W_0.copy()

    node_features = rng.random((3, D)).astype(np.float32)
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    edge_types = np.array([0, 0], dtype=np.int64)

    graph.message_pass(node_features, edge_index, edge_types)
    d_output = rng.random((3, D)).astype(np.float32)
    _, grads = graph.backward_message_pass(d_output)
    graph.update(grads, lr=0.1)

    assert not np.allclose(graph.W_r, W_r_before)
    assert not np.allclose(graph.W_0, W_0_before)


# ---------------------------------------------------------------------------
# Tests pipeline backward
# ---------------------------------------------------------------------------

def test_backward_updates_encoder_weights(pipeline: CGNPipeline):
    from gcn_python.layer1.representation import UDRepresentation
    params_before = [p.copy() for p in pipeline.encoder.parameters()]
    reps = [
        UDRepresentation(
            tokens=[{"lemma": "baisser", "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma="baisser", root_pos="VERB", root_dep_rel="root",
            root_morph={"Tense": "Pres"}, subject_pos="NOUN",
            has_object=False, has_advcl=False, has_temporal_obl=False,
            token_span=(1, 2), lang="fr",
        ),
        UDRepresentation(
            tokens=[{"lemma": "augmenter", "pos": "VERB", "dep_rel": "advcl", "morph": {}}],
            root_lemma="augmenter", root_pos="VERB", root_dep_rel="advcl",
            root_morph={"Tense": "Pres"}, subject_pos="NOUN",
            has_object=False, has_advcl=False, has_temporal_obl=False,
            token_span=(4, 5), lang="fr",
        ),
    ]
    pipeline.forward(reps, "Les ventes baissent parce que les coûts augmentent.")
    node_logits = pipeline._cached_node_logits
    assert node_logits is not None and len(node_logits) > 0

    edge_logits = pipeline._cached_edge_logits
    gold_node = np.zeros(len(node_logits), dtype=np.int64)
    gold_edge = (np.zeros(len(edge_logits), dtype=np.int64)
                 if edge_logits is not None else None)
    _, d_node, d_edge = pipeline.loss(node_logits, edge_logits, gold_node, gold_edge)
    pipeline.backward(d_node, d_edge, lr=0.1)

    params_after = pipeline.encoder.parameters()
    assert any(not np.allclose(b, a) for b, a in zip(params_before, params_after))


def test_loss_decreases_over_epochs(pipeline: CGNPipeline):
    """10 epochs sur un seul exemple : la loss doit décroître."""
    from gcn_python.layer1.representation import UDRepresentation
    reps = [
        UDRepresentation(
            tokens=[{"lemma": "baisser", "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma="baisser", root_pos="VERB", root_dep_rel="root",
            root_morph={"Tense": "Pres"}, subject_pos="NOUN",
            has_object=False, has_advcl=False, has_temporal_obl=False,
            token_span=(1, 2), lang="fr",
        ),
        UDRepresentation(
            tokens=[{"lemma": "augmenter", "pos": "VERB", "dep_rel": "advcl", "morph": {}}],
            root_lemma="augmenter", root_pos="VERB", root_dep_rel="advcl",
            root_morph={"Tense": "Pres"}, subject_pos="NOUN",
            has_object=False, has_advcl=False, has_temporal_obl=False,
            token_span=(4, 5), lang="fr",
        ),
    ]
    losses = []

    for _ in range(10):
        pipeline.forward(reps, "Les ventes baissent parce que les coûts augmentent.")
        node_logits = pipeline._cached_node_logits
        assert node_logits is not None and len(node_logits) > 0
        edge_logits = pipeline._cached_edge_logits
        gold_node = np.zeros(len(node_logits), dtype=np.int64)
        gold_edge = (np.zeros(len(edge_logits), dtype=np.int64)
                     if edge_logits is not None else None)
        loss_val, d_node, d_edge = pipeline.loss(node_logits, edge_logits, gold_node, gold_edge)
        losses.append(loss_val)
        pipeline.backward(d_node, d_edge, lr=0.01)

    assert losses[-1] < losses[0], f"Loss non décroissante : {losses[0]:.4f} → {losses[-1]:.4f}"


# ---------------------------------------------------------------------------
# Test checkpoint
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip(tmp_path: Path, pipeline: CGNPipeline):
    from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer

    params_orig = [p.copy() for p in pipeline.encoder.parameters()]

    ckpt = tmp_path / "model.npz"
    save_checkpoint(pipeline, ckpt)

    # Nouveau pipeline avec seed différent (poids différents)
    vocab = FeatureVocabulary()
    enc2 = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge, seed=99)
    gr2 = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause, seed=99)
    p2 = CGNPipeline(enc2, gr2, pipeline.lang, vocab)

    load_checkpoint(p2, ckpt)
    params_loaded = p2.encoder.parameters()

    for orig, loaded in zip(params_orig, params_loaded):
        assert np.allclose(orig, loaded), "Poids non restaurés correctement"


# ---------------------------------------------------------------------------
# Test DataLoader
# ---------------------------------------------------------------------------

def test_dataloader_yields_batches(paper_examples_yaml: Path):
    from gcn_python.data.loader import GCNDataLoader
    loader = GCNDataLoader(paper_examples_yaml.parent, lang="fr")
    samples = list(loader)
    assert len(samples) > 0
    for s in samples:
        assert s.gold_node_labels.dtype == np.int64
        assert isinstance(s.edge_map, dict)


def test_edge_map_alignment():
    """edge_map mappe les node_ids YAML → indices de clauses, pas l'ordre d'insertion."""
    import warnings
    from gcn_python.data.loader import GCNDataLoader
    from gcn_python.data.schema import (
        SentenceRecord, ClauseRecord, EdgeRecord, TokenRecord
    )
    from gcn_python.constants import RELATION_TYPES

    clauses = [
        ClauseRecord(node_id="n001", node_type="etat", label="A",
                     token_span=(1, 1), scope="specific", temporal_index=0, origin="explicit"),
        ClauseRecord(node_id="n002", node_type="action", label="B",
                     token_span=(2, 2), scope="specific", temporal_index=1, origin="explicit"),
        ClauseRecord(node_id="n003", node_type="processus", label="C",
                     token_span=(3, 3), scope="specific", temporal_index=2, origin="explicit"),
    ]
    # Arête non-consécutive : n001 → n003 (saute n002)
    edges = [
        EdgeRecord(source="n001", target="n003", relation="cause",
                   confidence=1.0, explicit=True, negated=False, marker_token=None),
    ]
    rec = SentenceRecord(id="s1", text="test", lang="fr",
                         tokens=[], clauses=clauses, edges=edges)

    loader = GCNDataLoader.__new__(GCNDataLoader)

    # Une arête longue distance doit émettre un UserWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sample = loader._to_sample(rec)
    assert any(issubclass(x.category, UserWarning) and "longue distance" in str(x.message) for x in w)

    # Arête longue distance non insérée dans edge_map (m3)
    assert (0, 2) not in sample.edge_map
    assert (2, 0) not in sample.edge_map
    # Les paires consécutives (0,1) et (1,2) restent sans gold label
    assert (0, 1) not in sample.edge_map
    assert (1, 2) not in sample.edge_map


def test_reps_from_sentence_alignment():
    """Les valid_indices doivent aligner reps et gold_node_labels sans décalage."""
    from gcn_python.data.loader import reps_from_sentence
    from gcn_python.data.schema import SentenceRecord, ClauseRecord, TokenRecord, EdgeRecord

    tokens = [
        TokenRecord(id=1, form="Les", lemma="le", pos="DET", dep_rel="det", dep_head=2),
        TokenRecord(id=2, form="ventes", lemma="vente", pos="NOUN", dep_rel="nsubj", dep_head=3),
        TokenRecord(id=3, form="baissent", lemma="baisser", pos="VERB", dep_rel="root", dep_head=0),
        TokenRecord(id=5, form="augmentent", lemma="augmenter", pos="VERB", dep_rel="advcl", dep_head=3),
    ]
    clauses = [
        ClauseRecord(node_id="n001", node_type="action", label="baisser(vente)",
                     token_span=(99, 100),  # span vide — aucun token id 99/100
                     scope="specific", temporal_index=0, origin="explicit"),
        ClauseRecord(node_id="n002", node_type="etat", label="baisser(vente)",
                     token_span=(1, 3),
                     scope="specific", temporal_index=1, origin="explicit"),
        ClauseRecord(node_id="n003", node_type="processus", label="augmenter(?)",
                     token_span=(5, 5),
                     scope="specific", temporal_index=2, origin="explicit"),
    ]
    rec = SentenceRecord(id="s1", text="test", lang="fr", tokens=tokens,
                         clauses=clauses, edges=[])

    reps, valid_indices, _ = reps_from_sentence(rec)

    # La clause n001 a un span vide → filtrée
    assert len(reps) == 2
    assert valid_indices == [1, 2]

    # Les gold labels indexés par valid_indices correspondent aux bonnes clauses
    from gcn_python.data.loader import GCNDataLoader
    from gcn_python.constants import NODE_TYPES
    gold_all = np.array(
        [NODE_TYPES.index(c.node_type) if c.node_type in NODE_TYPES else 0
         for c in clauses],
        dtype=np.int64,
    )
    gold_aligned = gold_all[np.array(valid_indices, dtype=np.int64)]

    # gold_aligned[0] doit correspondre au node_type de n002 ("etat"), pas n001 ("action")
    assert gold_aligned[0] == NODE_TYPES.index("etat")
    assert gold_aligned[1] == NODE_TYPES.index("processus")
    assert len(gold_aligned) == len(reps)


# ---------------------------------------------------------------------------
# Nouveaux tests — audit corrections
# ---------------------------------------------------------------------------

def test_invalid_node_type_warns_not_crashes():
    """C1 : node_type invalide émet un warning et ne crashe pas l'itération."""
    import warnings
    from gcn_python.data.loader import GCNDataLoader
    from gcn_python.data.schema import SentenceRecord, ClauseRecord, EdgeRecord

    clauses = [
        ClauseRecord(node_id="n001", node_type="evenement", label="A",
                     token_span=(1, 1), scope="specific", temporal_index=0, origin="explicit"),
    ]
    rec = SentenceRecord(id="s_bad", text="test", lang="fr", tokens=[], clauses=clauses, edges=[])

    loader = GCNDataLoader.__new__(GCNDataLoader)
    loader._records = [rec]
    loader.repeat = False

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        samples = list(loader)

    assert len(samples) == 0
    assert any(issubclass(x.category, UserWarning) and "s_bad" in str(x.message) for x in w)


def test_backward_edge_not_supervised():
    """M1 : arête gold backward (src > tgt) non supervisée — lookup strict retourne -1."""
    import warnings
    from gcn_python.data.loader import GCNDataLoader
    from gcn_python.data.schema import SentenceRecord, ClauseRecord, EdgeRecord
    from gcn_python.constants import RELATION_TYPES

    clauses = [
        ClauseRecord(node_id="n001", node_type="etat", label="A",
                     token_span=(1, 2), scope="specific", temporal_index=0, origin="explicit"),
        ClauseRecord(node_id="n002", node_type="action", label="B",
                     token_span=(4, 5), scope="specific", temporal_index=1, origin="explicit"),
    ]
    # Gold : n002 → n001 (direction inverse, src_idx=1 > tgt_idx=0)
    edges = [
        EdgeRecord(source="n002", target="n001", relation="cause",
                   confidence=1.0, explicit=True, negated=False, marker_token=None),
    ]
    rec = SentenceRecord(id="s_bwd", text="test", lang="fr", tokens=[], clauses=clauses, edges=edges)
    loader = GCNDataLoader.__new__(GCNDataLoader)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sample = loader._to_sample(rec)

    # L'arête backward (1, 0) n'est pas insérée dans edge_map
    assert (1, 0) not in sample.edge_map
    assert (0, 1) not in sample.edge_map
    # Un warning signale l'arête non-supervisable
    assert any(issubclass(x.category, UserWarning) and "direction inverse" in str(x.message) for x in w)


def test_checkpoint_dimension_mismatch_raises(tmp_path: Path, pipeline: CGNPipeline):
    """M3 : load_checkpoint avec poids de forme incompatible lève ValueError."""
    import pytest
    from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.layer1.features import FeatureVocabulary

    ckpt = tmp_path / "model.npz"
    save_checkpoint(pipeline, ckpt)

    # Pipeline avec des dimensions différentes (liste UPOS raccourcie pour forcer mismatch)
    vocab2 = FeatureVocabulary(upos_tags=["NOUN", "VERB", "ADJ"])
    enc2 = MLPEncoder(d_clause=vocab2.d_clause, d_edge=vocab2.d_edge, seed=1)
    gr2 = RGCNLayer(d_in=vocab2.d_clause, d_out=vocab2.d_clause, seed=1)
    p2 = CGNPipeline(enc2, gr2, pipeline.lang, vocab2)

    with pytest.raises(ValueError, match="Incompatibilité"):
        load_checkpoint(p2, ckpt)
