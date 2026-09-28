# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests Éq.6 — tête d'intention MLP apprise."""
import io
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gcn_python.constants import INTENT_TYPES
from gcn_python.layer2.reference import MLPEncoder


N_INTENT = len(INTENT_TYPES)   # 21
D_CLAUSE = 106
D_EDGE   = 50


# ---------------------------------------------------------------------------
# n_intent_types=0 — désactivé, aucune tête créée
# ---------------------------------------------------------------------------

def test_n_intent_types_zero_disabled():
    enc = MLPEncoder(d_clause=D_CLAUSE, d_edge=D_EDGE, n_intent_types=0)
    assert not hasattr(enc, '_intent_layers')
    assert enc.n_intent_types == 0


def test_n_intent_types_zero_no_forward():
    enc = MLPEncoder(d_clause=D_CLAUSE, d_edge=D_EDGE, n_intent_types=0)
    assert not hasattr(enc, 'forward_intent') or not hasattr(enc, '_intent_layers')


# ---------------------------------------------------------------------------
# forward_intent — shape correcte
# ---------------------------------------------------------------------------

def test_forward_intent_shape():
    enc = MLPEncoder(d_clause=D_CLAUSE, d_edge=D_EDGE, n_intent_types=N_INTENT)
    x = np.random.randn(D_CLAUSE).astype(np.float32)
    logit = enc.forward_intent(x)
    assert logit.shape == (N_INTENT,), f"attendu ({N_INTENT},) — reçu {logit.shape}"


def test_forward_intent_finite():
    enc = MLPEncoder(d_clause=D_CLAUSE, d_edge=D_EDGE, n_intent_types=N_INTENT)
    x = np.random.randn(D_CLAUSE).astype(np.float32)
    logit = enc.forward_intent(x)
    assert np.all(np.isfinite(logit))


def test_forward_intent_different_from_forward_node():
    enc = MLPEncoder(d_clause=D_CLAUSE, d_edge=D_EDGE, n_intent_types=N_INTENT)
    x = np.random.randn(D_CLAUSE).astype(np.float32)
    node_logit = enc.forward_node(x)
    intent_logit = enc.forward_intent(x)
    assert node_logit.shape != intent_logit.shape or not np.allclose(node_logit, intent_logit[:len(node_logit)])


# ---------------------------------------------------------------------------
# backward_intent — gradients corrects
# ---------------------------------------------------------------------------

def test_backward_intent_grads_shape():
    enc = MLPEncoder(d_clause=D_CLAUSE, d_edge=D_EDGE, n_intent_types=N_INTENT)
    x = np.random.randn(D_CLAUSE).astype(np.float32)
    enc.forward_intent(x)
    d_logits = np.ones(N_INTENT, dtype=np.float32)
    grads = enc.backward_intent(d_logits)
    assert len(grads) == len(enc._intent_layers)
    for dW, db in grads:
        assert np.all(np.isfinite(dW))
        assert np.all(np.isfinite(db))


def test_backward_intent_updates_weights():
    enc = MLPEncoder(d_clause=D_CLAUSE, d_edge=D_EDGE, n_intent_types=N_INTENT, seed=0)
    x = np.random.randn(D_CLAUSE).astype(np.float32)
    W_before = enc._intent_layers[0].W.copy()
    enc.forward_intent(x)
    grads = enc.backward_intent(np.ones(N_INTENT, dtype=np.float32))
    enc.update_intent(grads, lr=0.1)
    assert not np.allclose(enc._intent_layers[0].W, W_before), "update_intent n'a pas modifié les poids"


def test_update_intent_does_not_touch_node_weights():
    enc = MLPEncoder(d_clause=D_CLAUSE, d_edge=D_EDGE, n_intent_types=N_INTENT, seed=0)
    W_node_before = enc._node_layers[0].W.copy()
    x = np.random.randn(D_CLAUSE).astype(np.float32)
    enc.forward_intent(x)
    grads = enc.backward_intent(np.ones(N_INTENT, dtype=np.float32))
    enc.update_intent(grads, lr=0.1)
    assert np.allclose(enc._node_layers[0].W, W_node_before), "update_intent a touché les poids nœuds"


# ---------------------------------------------------------------------------
# Checkpoint roundtrip
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_intent_weights():
    from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.layer1.features import FeatureVocabulary

    vocab = FeatureVocabulary()
    d_eff = vocab.d_clause
    encoder = MLPEncoder(d_clause=d_eff, d_edge=50, n_intent_types=N_INTENT, seed=42)
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=1)
    pipeline = CGNPipeline(
        encoder=encoder, graph=graph, vocabulary=vocab,
        n_intent_types=N_INTENT,
    )

    W_before = encoder._intent_layers[0].W.copy()

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = Path(f.name)

    try:
        save_checkpoint(pipeline, path)

        # Modifier les poids intent
        encoder._intent_layers[0].W[:] = 0.0

        load_checkpoint(pipeline, path, trusted=True)

        assert np.allclose(encoder._intent_layers[0].W, W_before, atol=1e-6), (
            "Roundtrip checkpoint : poids intent non restaurés"
        )
    finally:
        path.unlink(missing_ok=True)


def test_checkpoint_without_intent_loads_cleanly():
    """Un checkpoint sans intent_layer_* doit charger sans erreur (rétrocompat)."""
    from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.layer1.features import FeatureVocabulary

    vocab = FeatureVocabulary()
    d_eff = vocab.d_clause

    # Sauvegarder SANS tête intent
    enc_no_intent = MLPEncoder(d_clause=d_eff, d_edge=50, n_intent_types=0, seed=1)
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=1)
    pipeline_no = CGNPipeline(encoder=enc_no_intent, graph=graph, vocabulary=vocab)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = Path(f.name)

    try:
        save_checkpoint(pipeline_no, path)

        # Charger dans pipeline AVEC tête intent
        enc_with = MLPEncoder(d_clause=d_eff, d_edge=50, n_intent_types=N_INTENT, seed=2)
        pipeline_with = CGNPipeline(
            encoder=enc_with, graph=RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=1),
            vocabulary=vocab, n_intent_types=N_INTENT,
        )
        W_init = enc_with._intent_layers[0].W.copy()

        load_checkpoint(pipeline_with, path, trusted=True)

        # Les poids intent doivent rester à leur valeur initiale (non touchés)
        assert np.allclose(enc_with._intent_layers[0].W, W_init), (
            "Chargement d'un checkpoint sans intent a modifié les poids intent"
        )
    finally:
        path.unlink(missing_ok=True)
