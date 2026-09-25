# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
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
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7), seed=0)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause, seed=0)
    return CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)


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
    N, _E = 4, 3
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
            token_span=(1, 2),
        ),
        UDRepresentation(
            tokens=[{"lemma": "augmenter", "pos": "VERB", "dep_rel": "advcl", "morph": {}}],
            root_lemma="augmenter", root_pos="VERB", root_dep_rel="advcl",
            root_morph={"Tense": "Pres"}, subject_pos="NOUN",
            has_object=False, has_advcl=False, has_temporal_obl=False,
            token_span=(4, 5),
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
    assert any(not np.allclose(b, a) for b, a in zip(params_before, params_after, strict=False))


def test_loss_decreases_over_epochs(pipeline: CGNPipeline):
    """10 epochs sur un seul exemple : la loss doit décroître."""
    from gcn_python.layer1.representation import UDRepresentation
    reps = [
        UDRepresentation(
            tokens=[{"lemma": "baisser", "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma="baisser", root_pos="VERB", root_dep_rel="root",
            root_morph={"Tense": "Pres"}, subject_pos="NOUN",
            has_object=False, has_advcl=False, has_temporal_obl=False,
            token_span=(1, 2),
        ),
        UDRepresentation(
            tokens=[{"lemma": "augmenter", "pos": "VERB", "dep_rel": "advcl", "morph": {}}],
            root_lemma="augmenter", root_pos="VERB", root_dep_rel="advcl",
            root_morph={"Tense": "Pres"}, subject_pos="NOUN",
            has_object=False, has_advcl=False, has_temporal_obl=False,
            token_span=(4, 5),
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
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.training.checkpoint import load_checkpoint, save_checkpoint

    params_orig = [p.copy() for p in pipeline.encoder.parameters()]

    ckpt = tmp_path / "model.npz"
    save_checkpoint(pipeline, ckpt)

    # Nouveau pipeline avec seed différent (poids différents)
    vocab = FeatureVocabulary()
    enc2 = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7), seed=99)
    gr2 = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause, seed=99)
    p2 = CGNPipeline(enc2, gr2, vocab)

    load_checkpoint(p2, ckpt, trusted=True)
    params_loaded = p2.encoder.parameters()

    for orig, loaded in zip(params_orig, params_loaded, strict=False):
        assert np.allclose(orig, loaded), "Poids non restaurés correctement"


# ---------------------------------------------------------------------------
# Test DataLoader
# ---------------------------------------------------------------------------

def test_dataloader_yields_batches(paper_examples_json: Path):  # L5 : renommé
    from gcn_python.data.loader import GCNDataLoader
    loader = GCNDataLoader(paper_examples_json.parent)
    samples = list(loader)
    assert len(samples) > 0
    for s in samples:
        assert s.gold_node_labels.dtype == np.int64
        assert isinstance(s.edge_map, dict)


def test_edge_map_alignment():
    """edge_map mappe les node_ids → indices de clauses, pas l'ordre d'insertion.

    S4 : avec all_pairs=False (défaut), les arêtes gap>1 sont filtrées.
    Avec all_pairs=True, elles sont insérées dans edge_map.
    """
    import warnings

    from gcn_python.constants import RELATION_TYPES
    from gcn_python.data.loader import GCNDataLoader
    from gcn_python.data.schema import ClauseRecord, EdgeRecord, SentenceRecord

    clauses = [
        ClauseRecord(node_id="n001", node_type="etat", label="A",
                     token_span=(1, 1), scope="specific", temporal_index=0, origin="explicit"),
        ClauseRecord(node_id="n002", node_type="action", label="B",
                     token_span=(2, 2), scope="specific", temporal_index=1, origin="explicit"),
        ClauseRecord(node_id="n003", node_type="processus", label="C",
                     token_span=(3, 3), scope="specific", temporal_index=2, origin="explicit"),
    ]
    # Arête non-consécutive : n001 → n003 (gap=2)
    edges = [
        EdgeRecord(source="n001", target="n003", relation="cause",
                   confidence=1.0, explicit=True, negated=False, marker_token=None),
    ]
    rec = SentenceRecord(id="s1", text="test", lang="fr",
                         tokens=[], clauses=clauses, edges=edges)

    # Mode défaut (all_pairs=False) : arête longue distance filtrée + warning
    loader_default = GCNDataLoader.__new__(GCNDataLoader)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sample_default = loader_default._to_sample(rec)
    assert any(issubclass(x.category, UserWarning) and "longue distance" in str(x.message) for x in w)
    assert (0, 2) not in sample_default.edge_map
    assert (2, 0) not in sample_default.edge_map

    # Mode all_pairs=True (S4) : arête longue distance présente dans edge_map
    loader_all = GCNDataLoader.__new__(GCNDataLoader)
    loader_all.all_pairs = True
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        sample_all = loader_all._to_sample(rec)
    assert not any("longue distance" in str(x.message) for x in w2), \
        "Pas de warning long-distance en mode all_pairs=True"
    cause_idx = RELATION_TYPES.index("cause")
    assert sample_all.edge_map.get((0, 2)) == cause_idx, \
        "L'arête gap=2 doit être dans edge_map quand all_pairs=True"


def test_reps_from_sentence_alignment():
    """Les valid_indices doivent aligner reps et gold_node_labels sans décalage."""
    from gcn_python.data.loader import reps_from_sentence
    from gcn_python.data.schema import (
        ClauseRecord,
        SentenceRecord,
        TokenRecord,
    )

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
    from gcn_python.data.schema import ClauseRecord, SentenceRecord

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
    """M1 : arête asymétrique backward (cause/enable/prevent, src > tgt) ignorée depuis v2.5.1.

    Comportement antérieur (< v2.5.1) : l'arête était remappée comme (tgt, src) — supervision inversée.
    Comportement actuel : l'arête est rejetée (edge_map vide) et un warning est émis.
    """
    import warnings

    from gcn_python.data.loader import GCNDataLoader
    from gcn_python.data.schema import ClauseRecord, EdgeRecord, SentenceRecord

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

    # L'arête asymétrique backward est rejetée — edge_map vide
    assert (0, 1) not in sample.edge_map
    assert (1, 0) not in sample.edge_map
    assert sample.edge_map == {}
    # Un warning signale le rejet
    assert any(
        issubclass(x.category, UserWarning) and "asymétrique ignorée" in str(x.message)
        for x in w
    )


def test_train_cmd_cli(tmp_path: Path):
    """M7 : test d'intégration CLI — gcn-train s'exécute sans erreur sur un dataset minimal."""
    import json

    from click.testing import CliRunner

    from gcn_python.training.train import train_cmd

    # Dataset minimal au format document
    dataset = {
        "document": {
            "lang": "fr",
            "sentences": [
                {
                    "id": "s1",
                    "text": "Les ventes baissent parce que les prix augmentent.",
                    "tokens": [
                        {"id": 1, "form": "Les", "lemma": "le", "pos": "DET",
                         "dep_rel": "det", "dep_head": 2, "morph": {}},
                        {"id": 2, "form": "ventes", "lemma": "vente", "pos": "NOUN",
                         "dep_rel": "nsubj", "dep_head": 3, "morph": {}},
                        {"id": 3, "form": "baissent", "lemma": "baisser", "pos": "VERB",
                         "dep_rel": "root", "dep_head": 0, "morph": {}},
                        {"id": 4, "form": "augmentent", "lemma": "augmenter", "pos": "VERB",
                         "dep_rel": "advcl", "dep_head": 3, "morph": {}},
                    ],
                    "cir": {
                        "nodes": [
                            {"id": "n1", "type": "processus", "label": "baisse ventes",
                             "token_span": [1, 3]},
                            {"id": "n2", "type": "etat", "label": "hausse prix",
                             "token_span": [4, 4]},
                        ],
                        "edges": [
                            {"source": "n2", "target": "n1", "relation": "cause",
                             "attributes": {"confidence": 1.0, "explicit": True,
                                           "negated": False}},
                        ],
                    },
                }
            ],
        }
    }
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.json").write_text(
        json.dumps(dataset), encoding="utf-8"
    )
    output_path = tmp_path / "model.npz"

    runner = CliRunner()
    result = runner.invoke(train_cmd, [
        "--data-dir", str(data_dir),
        "--epochs", "2",
        "--output", str(output_path),
    ])
    assert result.exit_code == 0, f"gcn-train a échoué :\n{result.output}\n{result.exception}"
    assert output_path.exists(), "Checkpoint non créé"


def test_checkpoint_dimension_mismatch_raises(tmp_path: Path, pipeline: CGNPipeline):
    """M3 : load_checkpoint avec poids de forme incompatible lève ValueError."""
    import pytest

    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.training.checkpoint import load_checkpoint, save_checkpoint

    ckpt = tmp_path / "model.npz"
    save_checkpoint(pipeline, ckpt)

    # Pipeline avec des dimensions différentes (liste UPOS raccourcie pour forcer mismatch)
    vocab2 = FeatureVocabulary(upos_tags=["NOUN", "VERB", "ADJ"])
    enc2 = MLPEncoder(d_clause=vocab2.d_clause, d_edge=vocab2.d_edge, seed=1)
    gr2 = RGCNLayer(d_in=vocab2.d_clause, d_out=vocab2.d_clause, seed=1)
    p2 = CGNPipeline(enc2, gr2, vocab2)

    with pytest.raises(ValueError, match="Incompatibilité"):
        load_checkpoint(p2, ckpt, trusted=True)


# ---------------------------------------------------------------------------
# Item 5 : validation lr / isfinite / gradient clipping / weight_decay
# ---------------------------------------------------------------------------

def _make_reps_2():
    from gcn_python.layer1.representation import UDRepresentation
    return [
        UDRepresentation(
            tokens=[{"lemma": "a", "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma="a", root_pos="VERB", root_dep_rel="root",
            root_morph={}, subject_pos="NOUN",
            has_object=False, has_advcl=False, has_temporal_obl=False,
            token_span=(0, 1),
        ),
        UDRepresentation(
            tokens=[{"lemma": "b", "pos": "VERB", "dep_rel": "advcl", "morph": {}}],
            root_lemma="b", root_pos="VERB", root_dep_rel="advcl",
            root_morph={}, subject_pos="NOUN",
            has_object=False, has_advcl=False, has_temporal_obl=False,
            token_span=(2, 3),
        ),
    ]


def test_backward_lr_validation(pipeline: CGNPipeline):
    reps = _make_reps_2()
    pipeline.forward(reps, "a b")
    nl = pipeline._cached_node_logits
    el = pipeline._cached_edge_logits
    gn = np.zeros(len(nl), dtype=np.int64)
    ge = np.zeros(len(el), dtype=np.int64) if el is not None else None
    _, dn, de = pipeline.loss(nl, el, gn, ge)

    with pytest.raises(ValueError, match="lr"):
        pipeline.backward(dn, de, lr=0.0)
    with pytest.raises(ValueError, match="lr"):
        pipeline.backward(dn, de, lr=-1.0)
    with pytest.raises(ValueError, match="lr"):
        pipeline.backward(dn, de, lr=float("nan"))


def test_backward_nonfinite_grad_warns(pipeline: CGNPipeline):
    from gcn_python.layer1.representation import UDRepresentation
    reps = [
        UDRepresentation(
            tokens=[{"lemma": "a", "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma="a", root_pos="VERB", root_dep_rel="root",
            root_morph={}, subject_pos="NOUN",
            has_object=False, has_advcl=False, has_temporal_obl=False,
            token_span=(0, 1),
        ),
    ]
    pipeline.forward(reps, "a")
    nl = pipeline._cached_node_logits
    d_nan = np.full_like(nl, float("nan"))
    d_edge = np.zeros((0, len(RELATION_TYPES)), dtype=np.float32)

    with pytest.warns(UserWarning, match="non finis"):
        pipeline.backward(d_nan, d_edge, lr=0.01)


def test_backward_grad_clip(pipeline: CGNPipeline):
    reps = _make_reps_2()
    pipeline.forward(reps, "x y")
    nl = pipeline._cached_node_logits
    el = pipeline._cached_edge_logits
    gn = np.zeros(len(nl), dtype=np.int64)
    ge = np.zeros(len(el), dtype=np.int64) if el is not None else None
    _, dn, de = pipeline.loss(nl, el, gn, ge)

    params_before = [p.copy() for p in pipeline.encoder.parameters()]
    pipeline.backward(dn, de, lr=0.01, max_grad_norm=1e-6)
    params_after = pipeline.encoder.parameters()
    assert any(not np.allclose(b, a) for b, a in zip(params_before, params_after, strict=False))


def test_backward_weight_decay_changes_rgcn(pipeline: CGNPipeline):
    reps = _make_reps_2()
    pipeline.forward(reps, "p q")
    nl = pipeline._cached_node_logits
    el = pipeline._cached_edge_logits
    gn = np.zeros(len(nl), dtype=np.int64)
    ge = np.zeros(len(el), dtype=np.int64) if el is not None else None
    _, dn, de = pipeline.loss(nl, el, gn, ge)

    graph = pipeline._graph_layers[0]
    pipeline.backward(dn, de, lr=0.01, weight_decay=0.1)

    vocab = pipeline.vocabulary
    pipeline2 = CGNPipeline(
        encoder=MLPEncoder(d_clause=vocab.d_clause,
                           d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7), seed=0),
        graph=RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause, seed=0),
        vocabulary=vocab,
    )
    pipeline2.forward(reps, "p q")
    nl2 = pipeline2._cached_node_logits
    el2 = pipeline2._cached_edge_logits
    gn2 = np.zeros(len(nl2), dtype=np.int64)
    ge2 = np.zeros(len(el2), dtype=np.int64) if el2 is not None else None
    _, dn2, de2 = pipeline2.loss(nl2, el2, gn2, ge2)
    pipeline2.backward(dn2, de2, lr=0.01, weight_decay=0.0)

    assert not np.allclose(graph.W_r, pipeline2._graph_layers[0].W_r), \
        "weight_decay=0.1 doit produire une mise à jour différente de weight_decay=0.0"


# ---------------------------------------------------------------------------
# Item 6 : backward bidirectionnel
# ---------------------------------------------------------------------------

def test_backward_bidirectional_no_crash():
    """backward() avec bidirectional=True ne doit pas crasher."""
    vocab = FeatureVocabulary()
    D = vocab.d_clause
    encoder = MLPEncoder(d_clause=D, d_edge=vocab.d_edge_closed_loop(D, 7), seed=0)
    graph = RGCNLayer(d_in=D, d_out=D, seed=0)
    p = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab, bidirectional=True)

    reps = _make_reps_2()
    p.forward(reps, "r s")
    nl = p._cached_node_logits
    el = p._cached_edge_logits
    gn = np.zeros(len(nl), dtype=np.int64)
    ge = np.zeros(len(el), dtype=np.int64) if el is not None else None
    _, dn, de = p.loss(nl, el, gn, ge)
    p.backward(dn, de, lr=0.01)


def test_apply_accumulated_lr_validation(pipeline: CGNPipeline):
    with pytest.raises(ValueError, match="lr"):
        pipeline.apply_accumulated_gradients(lr=0.0)
    with pytest.raises(ValueError, match="lr"):
        pipeline.apply_accumulated_gradients(lr=-0.1)


# ── Amélioration F — Pondération silver ───────────────────────────────────────

def _silver_json(tmp_path: Path, methode: str | None) -> Path:
    sent: dict = {
        "id": "s0001",
        "text": "Le chat dort car il est fatigué.",
        "tokens": [
            {"id": 1, "form": "Le", "lemma": "le", "pos": "DET", "dep_rel": "det", "dep_head": 2},
            {"id": 2, "form": "chat", "lemma": "chat", "pos": "NOUN", "dep_rel": "nsubj", "dep_head": 3},
            {"id": 3, "form": "dort", "lemma": "dormir", "pos": "VERB", "dep_rel": "root", "dep_head": 0},
            {"id": 4, "form": "car", "lemma": "car", "pos": "SCONJ", "dep_rel": "mark", "dep_head": 6},
            {"id": 5, "form": "il", "lemma": "il", "pos": "PRON", "dep_rel": "nsubj", "dep_head": 6},
            {"id": 6, "form": "fatigué", "lemma": "fatigué", "pos": "ADJ", "dep_rel": "advcl", "dep_head": 3},
        ],
        "cir": {
            "nodes": [
                {"id": "n001", "type": "entite", "label": "chat", "token_span": [1, 3],
                 "scope": "specific", "temporal_index": 0, "origin": "explicit"},
                {"id": "n002", "type": "processus", "label": "fatigué", "token_span": [5, 6],
                 "scope": "specific", "temporal_index": 0, "origin": "explicit"},
            ],
            "edges": [
                {"sources": ["n001"], "target": "n002", "relation": "cause"},
            ],
        },
    }
    if methode is not None:
        sent["_methode"] = methode
    doc = {"document": {"lang": "fr", "sentences": [sent]}}
    p = tmp_path / "train.json"
    import json as _json
    p.write_text(_json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def test_silver_weight_0_7_reduces_edge_loss(tmp_path: Path):
    """F : phrase _methode=silver-* → weight=0.7 ; loss arêtes réduite vs gold."""
    from gcn_python.data.loader import GCNDataLoader
    d = _silver_json(tmp_path, "silver-fr-moi-types")
    loader = GCNDataLoader(d, silver_weight=0.7)
    samples = list(loader)
    assert len(samples) == 1
    assert samples[0].sentence.weight == 0.7

    _d_gold = _silver_json(tmp_path, None)  # réécrit sans _methode → gold
    # gold : _methode absente → 1.0
    loader_gold = GCNDataLoader(tmp_path)
    assert next(iter(loader_gold)).sentence.weight == 1.0

    # loss : même phrase, poids 0.7 < poids 1.0 sur la partie arêtes
    vocab = FeatureVocabulary()
    encoder = MLPEncoder(d_clause=vocab.d_clause,
                         d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7), seed=0)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause, seed=0)
    pipe = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)
    rng = np.random.default_rng(1)
    nl = rng.normal(0, 1, (2, 7)).astype(np.float32)
    el = rng.normal(0, 1, (1, 11)).astype(np.float32)
    loss_gold, _, _ = pipe.loss(nl, el, np.array([0, 1]), np.array([3]),
                                sample_weight=1.0)
    loss_silver, _, _ = pipe.loss(nl, el, np.array([0, 1]), np.array([3]),
                                  sample_weight=0.7)
    assert loss_silver < loss_gold
