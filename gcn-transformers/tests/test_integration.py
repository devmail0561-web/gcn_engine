# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Tests d'intégration avec CGNPipeline complet.
"""
import numpy as np

from gcn_transformers import XLMRobertaEncoder


def _pipeline_v4(vocab):
    """Pipeline contrat v4 (D10 : WordEmbedding requis, 19 relations + no-edge).

    Les tests écrivaient le contrat v2 (sans embedding, 11 relations) —
    incompatible avec gcn-python>=4.0.0 (floor pyproject).
    """
    from gcn_python.layer1.embedding import WordEmbedding
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline

    we = WordEmbedding(d_emb=8)
    d_eff = vocab.d_clause_effective(8, False)
    d_edge = vocab.d_edge_closed_loop(d_eff, 8, 8, False)
    encoder = XLMRobertaEncoder(d_clause=d_eff, d_edge=d_edge)
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=20)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                           word_embedding=we)
    return pipeline, encoder


def test_pipeline_forward(vocab, dimensions, sample_ud_reps):
    """Test forward avec CGNPipeline."""
    pipeline, encoder = _pipeline_v4(vocab)

    # Forward
    result = pipeline.forward(sample_ud_reps, "La pluie cause l'inondation.")

    # Vérifications
    assert "nodes" in result
    assert len(result["nodes"]) == 2
    assert "edges" in result


def test_pipeline_forward_backward(vocab, dimensions, sample_ud_reps, sample_gold_labels):
    """Test forward+backward avec CGNPipeline (API correcte)."""
    pipeline, encoder = _pipeline_v4(vocab)

    # Forward
    result = pipeline.forward(sample_ud_reps, "La pluie cause l'inondation.")

    # Récupérer logits
    node_logits = pipeline._cached_node_logits
    edge_logits = pipeline._cached_edge_logits

    # Filtrer arêtes gold (consécutives)
    n_clauses = len(node_logits)
    pairs = [(i, i+1) for i in range(n_clauses-1)]
    gold_edge_full = np.array([sample_gold_labels["edges"].get(p, -1) for p in pairs], dtype=np.int64)
    valid_mask = gold_edge_full >= 0
    gold_edge = gold_edge_full[valid_mask] if valid_mask.any() else None
    edge_logits_filtered = edge_logits[valid_mask] if edge_logits is not None and valid_mask.any() else None

    # Backward (API CORRECTE : loss() puis backward())
    loss, d_node, d_edge_filtered = pipeline.loss(
        node_logits, edge_logits_filtered,
        sample_gold_labels["nodes"], gold_edge
    )

    assert loss > 0, f"Loss should be > 0, got {loss}"

    # Backward avec gradients
    d_edge_full = None
    if d_edge_filtered is not None and edge_logits is not None:
        d_edge_full = np.zeros_like(edge_logits)
        d_edge_full[valid_mask] = d_edge_filtered
    pipeline.backward(d_node, d_edge_full, lr=1e-5)


def test_pipeline_multiple_epochs(vocab, dimensions, sample_ud_reps, sample_gold_labels):
    """Test entraînement sur plusieurs epochs (vérifier accumulation gradients)."""
    pipeline, encoder = _pipeline_v4(vocab)

    losses = []
    for epoch in range(3):
        # Forward
        result = pipeline.forward(sample_ud_reps, "La pluie cause l'inondation.")

        # Récupérer logits
        node_logits = pipeline._cached_node_logits
        edge_logits = pipeline._cached_edge_logits

        # Filtrer arêtes gold
        n_clauses = len(node_logits)
        pairs = [(i, i+1) for i in range(n_clauses-1)]
        gold_edge_full = np.array([sample_gold_labels["edges"].get(p, -1) for p in pairs], dtype=np.int64)
        valid_mask = gold_edge_full >= 0
        gold_edge = gold_edge_full[valid_mask] if valid_mask.any() else None
        edge_logits_filtered = edge_logits[valid_mask] if edge_logits is not None and valid_mask.any() else None

        # Loss
        loss, d_node, d_edge_filtered = pipeline.loss(
            node_logits, edge_logits_filtered,
            sample_gold_labels["nodes"], gold_edge
        )
        losses.append(loss)

        # Backward
        d_edge_full = None
        if d_edge_filtered is not None and edge_logits is not None:
            d_edge_full = np.zeros_like(edge_logits)
            d_edge_full[valid_mask] = d_edge_filtered
        pipeline.backward(d_node, d_edge_full, lr=1e-5)

    # Vérifier que la loss a bougé (pas forcément descendu sur 3 epochs)
    assert losses[0] != losses[-1], "Loss devrait changer après entraînement"


def test_pipeline_with_eval_mode(vocab, dimensions, sample_ud_reps):
    """Test eval mode (dropout désactivé)."""
    pipeline, encoder = _pipeline_v4(vocab)

    # Mode eval
    encoder.eval()

    # Forward (2 fois pour vérifier déterminisme)
    result1 = pipeline.forward(sample_ud_reps, "La pluie cause l'inondation.")
    logits1 = pipeline._cached_node_logits.copy()

    result2 = pipeline.forward(sample_ud_reps, "La pluie cause l'inondation.")
    logits2 = pipeline._cached_node_logits.copy()

    # En mode eval, forward devrait être déterministe
    np.testing.assert_allclose(
        logits1[0],
        logits2[0],
        rtol=1e-5,
        err_msg="Forward en eval mode devrait être déterministe"
    )
