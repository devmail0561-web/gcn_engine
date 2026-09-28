# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests D10 ETUDE — word_embedding obligatoire dans le moteur."""
import numpy as np
import pytest

from conftest import make_test_pipeline, make_word_embedding
from gcn_python.constants import NODE_TYPES
from gcn_python.layer1.embedding import WordEmbedding
from gcn_python.layer1.features import FeatureVocabulary, vectorize_clause
from gcn_python.layer1.representation import UDRepresentation
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline

# ---------------------------------------------------------------------------
# CGNPipeline — word_embedding obligatoire
# ---------------------------------------------------------------------------

def test_cgn_pipeline_requires_word_embedding():
    """ValueError si word_embedding=None (D10 ETUDE)."""
    vocab = FeatureVocabulary()
    # Sans embedding, d_eff = d_clause = 106 → dimensions cohérentes
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=50)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    with pytest.raises(ValueError, match="word_embedding"):
        CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                    word_embedding=None)


def test_cgn_pipeline_valid_with_embedding():
    """make_test_pipeline() crée un pipeline valide sans erreur."""
    pipeline = make_test_pipeline()
    assert pipeline.word_embedding is not None
    assert pipeline.word_embedding.d_emb == 4


def test_d_eff_includes_embedding():
    """d_clause_effective(d_emb) > d_clause — le vecteur inclut les embeddings."""
    vocab = FeatureVocabulary()
    we = make_word_embedding(d_emb=4)
    assert vocab.d_clause_effective(we.d_emb) == vocab.d_clause + we.d_emb


def test_pipeline_dimensions_consistent():
    """RGCNLayer.d_out = d_clause_effective — pas de mismatch."""
    vocab = FeatureVocabulary()
    we = make_word_embedding(d_emb=4)
    d_eff = vocab.d_clause_effective(we.d_emb)
    d_edge = vocab.d_edge_closed_loop(d_eff, len(NODE_TYPES), we.d_emb)
    encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge)
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff)
    # Ne doit pas lever ValueError
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                           word_embedding=we)
    assert pipeline is not None


# ---------------------------------------------------------------------------
# vectorize_clause — word_embedding obligatoire
# ---------------------------------------------------------------------------

def _make_rep() -> UDRepresentation:
    return UDRepresentation(
        tokens=[{"lemma": "test", "pos": "VERB", "dep_rel": "root",
                 "morph": {"Mood": "Ind"}, "id": 0}],
        root_lemma="test", root_pos="VERB", root_dep_rel="root",
        root_morph={"Mood": "Ind"}, subject_pos=None,
        has_object=False, has_advcl=False, has_temporal_obl=False,
        token_span=(0, 1),
    )


def test_vectorize_clause_requires_word_embedding():
    """ValueError si word_embedding=None dans vectorize_clause."""
    vocab = FeatureVocabulary()
    rep = _make_rep()
    with pytest.raises(ValueError, match="word_embedding"):
        vectorize_clause(rep, vocab, word_embedding=None)


def test_vectorize_clause_with_embedding_shape():
    """vectorize_clause produit un vecteur de taille d_clause_effective."""
    vocab = FeatureVocabulary()
    we = make_word_embedding(d_emb=4)
    rep = _make_rep()
    vec = vectorize_clause(rep, vocab, word_embedding=we)
    expected = vocab.d_clause_effective(we.d_emb)
    assert vec.shape == (expected,), f"expected ({expected},) got {vec.shape}"


# ---------------------------------------------------------------------------
# Checkpoint — word_emb_E obligatoire
# ---------------------------------------------------------------------------

def test_checkpoint_saves_word_emb_E(tmp_path):
    """save_checkpoint inclut word_emb_E (D10 — paramètre appris)."""
    import numpy as _np

    from gcn_python.training.checkpoint import save_checkpoint

    pipeline = make_test_pipeline()
    ckpt = tmp_path / "model.npz"
    save_checkpoint(pipeline, ckpt)

    data = _np.load(ckpt, allow_pickle=False)
    assert "word_emb_E" in data.files, "word_emb_E doit être dans le checkpoint"
    assert "_word_emb_vocab_json" in data.files


def test_checkpoint_fails_without_embedding(tmp_path):
    """save_checkpoint lève ValueError si word_embedding absent."""
    from gcn_python.training.checkpoint import save_checkpoint

    vocab = FeatureVocabulary()
    we = make_word_embedding(d_emb=4)
    vocab.d_clause_effective(we.d_emb)
    pipeline = make_test_pipeline()
    # Forcer word_embedding à None après construction (contournement pour tester checkpoint)
    pipeline.word_embedding = None
    with pytest.raises(ValueError, match="word_embedding"):
        save_checkpoint(pipeline, tmp_path / "bad.npz")


# ---------------------------------------------------------------------------
# train.py — --embedding-dim obligatoire
# ---------------------------------------------------------------------------

def test_train_requires_embedding_dim(tmp_path):
    """ClickException si --embedding-dim 0 (D10 ETUDE)."""
    import json

    from click.testing import CliRunner

    from gcn_python.training.train import train_cmd

    # Dataset minimal
    doc = {"document": {"sentences": [
        {"id": "s1", "text": "A cause B.", "cir": {
            "nodes": [
                {"id": "n1", "type": "processus", "label": "A", "token_span": [0, 1]},
                {"id": "n2", "type": "processus", "label": "B", "token_span": [2, 3]},
            ],
            "edges": [{"sources": ["n1"], "target": "n2", "relation": "cause"}]
        }}
    ]}}
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.json").write_text(json.dumps(doc), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(train_cmd, [
        "--data-dir", str(data_dir),
        "--output", str(tmp_path / "model.npz"),
        "--epochs", "1",
        "--embedding-dim", "0",
    ])
    assert result.exit_code != 0
    assert "obligatoire" in (result.output or "") or "obligatoire" in str(result.exception or "")


# ---------------------------------------------------------------------------
# Connecteur — embedding appris (D10)
# ---------------------------------------------------------------------------

def test_connector_embedding_in_edge_vec():
    """vectorize_connector utilise word_embedding pour le connecteur (D10)."""
    from gcn_python.layer1.features import vectorize_connector

    vocab = FeatureVocabulary()
    we = WordEmbedding(d_emb=4, seed=0)
    we.add_lemma("si")
    we.add_lemma("bien_que")

    conn_si = UDRepresentation(
        tokens=[], root_lemma="si", root_pos="SCONJ", root_dep_rel="mark",
        root_morph={}, subject_pos=None, has_object=False,
        has_advcl=False, has_temporal_obl=False, token_span=(0, 1),
    )
    conn_bien = UDRepresentation(
        tokens=[], root_lemma="bien_que", root_pos="SCONJ", root_dep_rel="mark",
        root_morph={}, subject_pos=None, has_object=False,
        has_advcl=False, has_temporal_obl=False, token_span=(0, 1),
    )

    v_si   = vectorize_connector(conn_si,   0, 1, 3, vocab, we)
    v_bien = vectorize_connector(conn_bien, 0, 1, 3, vocab, we)

    assert v_si.shape == (vocab.d_conn_effective(we.d_emb),)
    assert not np.array_equal(v_si, v_bien), (
        "Des connecteurs différents doivent produire des vecteurs différents"
    )
